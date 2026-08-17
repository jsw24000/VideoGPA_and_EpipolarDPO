from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
VIDEOGPA_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = VIDEOGPA_ROOT.parent
WAN_PATH = VIDEOGPA_ROOT / "Wan2.2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WAN_PATH) not in sys.path:
    sys.path.insert(0, str(WAN_PATH))

from vgm_common.config import resolve_experiment_config, write_resolved_config  # noqa: E402
from vgm_common.paths import get_dl3dv_root  # noqa: E402
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.image2video import WanI2V  # noqa: E402
from wan.text2video import WanT2V  # noqa: E402

FIRST_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def safe_id(value: str) -> str:
    return value.strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


def relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_config(config_path: Path | None, run_dir: Path | None, model_path: str | None) -> dict[str, Any]:
    if config_path is None:
        raise ValueError("--config is required so paths can be resolved through the active VGM profile")
    cfg = resolve_experiment_config(config_path, run_dir, model_path_override=model_path)
    cfg.setdefault("generation", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    task = str(cfg.get("project", {}).get("task", "")).lower()
    if task not in {"t2v", "i2v"}:
        raise ValueError(f"A14B generation requires project.task=t2v or i2v, got {task!r}")
    wan_task_key = str(cfg["model"].get("wan_task_key", f"{task}-A14B"))
    if wan_task_key not in WAN_CONFIGS:
        raise ValueError(f"Unknown WAN task key {wan_task_key!r}; available={sorted(WAN_CONFIGS)}")
    if not wan_task_key.endswith("A14B"):
        raise ValueError(f"This entrypoint is A14B only, got wan_task_key={wan_task_key!r}")
    cfg["project"]["task"] = task
    return cfg


def parse_seed_list(text: str | None, cfg: dict[str, Any]) -> list[int]:
    if text:
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    seeds = cfg.get("data", {}).get("candidate_seeds", [1001, 1002, 1003])
    return [int(seed) for seed in seeds]


def parse_size(size: str) -> tuple[int, int]:
    if size not in SIZE_CONFIGS:
        if "*" not in size:
            raise ValueError(f"Unsupported size {size!r}")
        width, height = size.split("*", 1)
        return int(width), int(height)
    return SIZE_CONFIGS[size]


def parse_guide_scale(value: Any) -> float | tuple[float, float]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"A14B guide_scale list must contain two values, got {value!r}")
        return (float(value[0]), float(value[1]))
    if isinstance(value, str) and "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"A14B guide_scale string must contain two comma-separated values: {value!r}")
        return (float(parts[0]), float(parts[1]))
    return float(value)


def guide_scale_json(value: float | tuple[float, float]) -> float | list[float]:
    if isinstance(value, tuple):
        return [float(value[0]), float(value[1])]
    return float(value)


@lru_cache(maxsize=1)
def available_ffmpeg_video_encoders() -> set[str]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    encoders = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def select_ffmpeg_video_encoder() -> tuple[str, list[str]]:
    requested = os.environ.get("FFMPEG_VIDEO_ENCODER")
    presets = {
        "libx264": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23"],
        "libopenh264": ["-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "10M"],
        "mpeg4": ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "4"],
    }
    if requested:
        return requested, presets.get(requested, ["-c:v", requested, "-pix_fmt", "yuv420p"])
    encoders = available_ffmpeg_video_encoders()
    for name in ["libx264", "libopenh264", "mpeg4"]:
        if name in encoders:
            return name, presets[name]
    return "libx264", presets["libx264"]


def save_video_ffmpeg(video_tensor: torch.Tensor, output_path: Path, fps: int = 16, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing video: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_data = video_tensor.permute(1, 2, 3, 0).cpu().numpy().clip(-1, 1)
    video_data = ((video_data + 1) * 127.5).astype(np.uint8)
    frames, height, width, _channels = video_data.shape
    encoder_name, encoder_args = select_ffmpeg_video_encoder()
    cmd = [
        "ffmpeg",
        "-y" if force else "-n",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        *encoder_args,
        str(output_path),
    ]
    print(f"Running ffmpeg with encoder={encoder_name}:", " ".join(cmd[:-1]), str(output_path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(video_data.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {proc.returncode}\n"
            f"STDOUT:\n{stdout.decode(errors='replace')}\n"
            f"STDERR:\n{stderr.decode(errors='replace')}"
        )
    print(f"Saved {frames} frames to {output_path}")


def ffprobe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()}
    try:
        data = json.loads(proc.stdout)
        stream = data.get("streams", [{}])[0]
        return {
            "ok": True,
            "frames": int(stream.get("nb_read_frames") or 0),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "r_frame_rate": stream.get("r_frame_rate"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"ffprobe parse failed: {exc}"}


def existing_video_ok(path: Path, expected_frames: int) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    meta = ffprobe_video(path)
    return bool(meta.get("ok") and int(meta.get("frames", 0)) == expected_frames)


def quarantine_invalid_video(path: Path) -> Path:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.stem}.invalid_{suffix}{path.suffix}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.stem}.invalid_{suffix}_{counter}{path.suffix}")
        counter += 1
    path.replace(target)
    return target


def infer_source_split(scene_uid: str, bucket: str | None = None) -> str:
    source_bucket = (bucket or scene_uid.split("/", 1)[0]).upper()
    return "test" if source_bucket == "1K" else "train"


def normalize_sample_defaults(item: dict[str, Any], task: str) -> dict[str, Any]:
    scene_uid = str(item.get("scene_uid", ""))
    if scene_uid and "/" in scene_uid:
        bucket, scene_id = scene_uid.split("/", 1)
        item.setdefault("scene_id", scene_id)
        item.setdefault("source_bucket", bucket.lower())
        item.setdefault("source_split", infer_source_split(scene_uid, bucket))
    elif item.get("source_bucket"):
        item.setdefault("source_split", infer_source_split(scene_uid, str(item["source_bucket"])))
    item.setdefault("task", task)
    return item


def load_samples(input_json: Path, task: str) -> list[dict[str, Any]]:
    payload = read_json(input_json)
    if isinstance(payload, dict):
        samples = payload.get("samples") or payload.get("groups")
        if samples is None:
            samples = []
            for scene_uid, item in payload.items():
                if not isinstance(item, dict):
                    continue
                sample = dict(item)
                sample.setdefault("scene_uid", scene_uid)
                sample.setdefault("group_id", safe_id(scene_uid))
                samples.append(normalize_sample_defaults(sample, task))
    elif isinstance(payload, list):
        samples = [normalize_sample_defaults(dict(item), task) for item in payload]
    else:
        raise ValueError(f"Unsupported prompt JSON format: {input_json}")
    clean = []
    for idx, item in enumerate(samples):
        item = normalize_sample_defaults(dict(item), task)
        prompt = str(item.get("text_prompt", item.get("prompt", ""))).strip()
        if not prompt:
            raise ValueError(f"Empty prompt at sample index {idx}")
        item_task = str(item.get("task", task)).lower()
        if item_task and item_task != task:
            raise ValueError(f"Input sample task={item_task!r} does not match config task={task!r}: {item.get('group_id', idx)}")
        has_image = any(key in item for key in ["image_path", "image_prompt", "input_image_path", "first_frame_path", "first_frame_relpath"])
        if task == "t2v" and has_image:
            raise ValueError(f"T2V input must not include image fields: {item.get('group_id', idx)}")
        if task == "i2v" and not has_image and not (item.get("scene_id") and item.get("source_bucket")):
            raise ValueError(f"I2V input must include an image field or resolvable scene metadata: index {idx}")
        clean.append(item)
    return clean


def find_first_frame(first_frames_root: Path, split: str, bucket: str, scene_id: str) -> Path | None:
    scene_dir = first_frames_root / split / bucket.upper() / scene_id
    for ext in FIRST_FRAME_EXTENSIONS:
        candidate = scene_dir / f"first_frame{ext}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_image_path(item: dict[str, Any], cfg: dict[str, Any], run_dir: Path) -> Path:
    project_root = Path(cfg["project"]["project_root"])
    data_root = get_dl3dv_root()
    first_frames_root = Path(cfg["paths"]["first_frames_root"])
    raw_values = [
        item.get("image_path"),
        item.get("image_prompt"),
        item.get("input_image_path"),
        item.get("first_frame_path"),
        item.get("first_frame_relpath"),
    ]
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        resolved = path.expanduser()
        if resolved not in candidates:
            candidates.append(resolved)

    for value in raw_values:
        if not value:
            continue
        raw = Path(str(value)).expanduser()
        if raw.is_absolute():
            add_candidate(raw)
        else:
            add_candidate(run_dir / raw)
            add_candidate(project_root / raw)
            add_candidate(data_root / raw)
            add_candidate(first_frames_root / raw)
        parts = raw.parts
        if "first_frames" in parts:
            first_frames_idx = parts.index("first_frames")
            tail = Path(*parts[first_frames_idx + 1 :])
            add_candidate(first_frames_root / tail)
            add_candidate(data_root / "first_frames" / tail)
        elif parts and parts[0] in {"train", "test"}:
            add_candidate(first_frames_root / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    scene_id = item.get("scene_id")
    bucket = item.get("source_bucket")
    split = item.get("source_split", "train")
    if scene_id and bucket:
        found = find_first_frame(first_frames_root, str(split), str(bucket), str(scene_id))
        if found is not None:
            return found
    tried = "\n".join(str(path) for path in candidates) or "(no explicit image path candidates)"
    raise FileNotFoundError(f"Could not resolve first-frame image for {item.get('group_id', item.get('scene_uid'))}:\n{tried}")


def init_distributed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed(state: dict[str, Any]) -> None:
    if not state["distributed"]:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def mount_dual_lora(engine: Any, lora_path: Path | None, lora_weight: float) -> bool:
    if not lora_path:
        return False
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    missing = []
    for expert in ("low_noise_model", "high_noise_model"):
        expert_root = lora_path / expert
        if not (expert_root / "adapter_config.json").is_file():
            missing.append(f"{expert}/adapter_config.json")
        if not any((expert_root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            missing.append(f"{expert}/adapter_model")
    if missing:
        raise FileNotFoundError(f"A14B LoRA checkpoint must contain dual expert adapters: {', '.join(missing)}")

    for expert in ("low_noise_model", "high_noise_model"):
        model = getattr(engine, expert)
        adapter_root = lora_path / expert
        print(f"Mounting {expert} LoRA: {adapter_root} (weight={lora_weight})")
        peft_model = PeftModel.from_pretrained(
            model,
            str(adapter_root),
            adapter_name="default",
            torch_dtype=torch.bfloat16,
        )
        if lora_weight != 1.0:
            for module in peft_model.modules():
                if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                    for adapter in module.scaling:
                        module.scaling[adapter] *= lora_weight
        setattr(engine, expert, peft_model.merge_and_unload())
    print("Dual-expert LoRA merged.")
    return True


def build_engine(
    task: str,
    wan_cfg: Any,
    model_path: Path,
    device_id: int,
    rank: int,
    args: argparse.Namespace,
    generation_cfg: dict[str, Any],
    distributed: bool,
) -> WanT2V | WanI2V:
    t5_cpu = bool(generation_cfg.get("t5_cpu", True)) if args.t5_cpu is None else args.t5_cpu
    convert_model_dtype = (
        bool(generation_cfg.get("convert_model_dtype", True))
        if args.convert_model_dtype is None
        else args.convert_model_dtype
    )
    t5_fsdp = bool(generation_cfg.get("t5_fsdp", False)) if args.t5_fsdp is None else args.t5_fsdp
    dit_default = distributed
    dit_fsdp = bool(generation_cfg.get("dit_fsdp", dit_default)) if args.dit_fsdp is None else args.dit_fsdp
    use_sp = bool(generation_cfg.get("use_sp", False)) if args.use_sp is None else args.use_sp
    common = {
        "config": wan_cfg,
        "checkpoint_dir": str(model_path),
        "device_id": device_id,
        "rank": rank,
        "t5_fsdp": t5_fsdp,
        "dit_fsdp": dit_fsdp,
        "use_sp": use_sp,
        "t5_cpu": t5_cpu,
        "convert_model_dtype": convert_model_dtype,
    }
    if task == "t2v":
        return WanT2V(**common)
    return WanI2V(**common)


def generate(args: argparse.Namespace) -> None:
    dist_state = init_distributed()
    try:
        cfg = resolve_config(
            args.config if args.config else None,
            Path(args.run_dir).expanduser().resolve() if args.run_dir else None,
            args.model_path,
        )
        task = str(cfg["project"]["task"])
        wan_task_key = str(cfg["model"]["wan_task_key"])
        wan_cfg = WAN_CONFIGS[wan_task_key]
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else Path(cfg["paths"].get("run_dir", ".")).resolve()
        input_json = Path(args.input_json or run_dir / "manifests/input_subset.json").expanduser().resolve()
        output_dir = Path(args.output_dir or run_dir / "candidates").expanduser().resolve()
        output_manifest = Path(args.candidate_groups_json or run_dir / "manifests/candidate_groups.json").expanduser().resolve()
        model_path = Path(cfg["paths"]["wan_model_path"]).resolve()
        lora_path = Path(args.lora_path).expanduser().resolve() if args.lora_path else None
        seeds = parse_seed_list(args.candidate_seeds, cfg)
        if args.candidates_per_prompt:
            seeds = seeds[: args.candidates_per_prompt]

        generation_cfg = cfg.get("generation", {})
        frame_num = int(args.frame_num or generation_cfg.get("frame_num", wan_cfg.get("sample_frames", 81)))
        size_text = str(args.size or generation_cfg.get("size", "1280*720"))
        size = parse_size(size_text)
        max_area = int(args.max_area or generation_cfg.get("max_area", MAX_AREA_CONFIGS.get(size_text, size[0] * size[1])))
        sampling_steps = int(args.sampling_steps or generation_cfg.get("sampling_steps", wan_cfg.sample_steps))
        guide_scale = parse_guide_scale(args.guide_scale if args.guide_scale is not None else generation_cfg.get("guide_scale", wan_cfg.sample_guide_scale))
        shift = float(args.shift or generation_cfg.get("sample_shift", wan_cfg.sample_shift))
        sample_solver = str(args.sample_solver or generation_cfg.get("sample_solver", "unipc"))
        fps = int(args.fps or generation_cfg.get("fps", 16))
        offload_model = bool(generation_cfg.get("offload_model", True)) if args.offload_model is None else args.offload_model

        rank = int(dist_state["rank"])
        is_main = bool(dist_state["is_main"])
        device_id = int(dist_state["local_rank"]) if dist_state["distributed"] else int(args.gpu_id)

        def log(*items: object) -> None:
            if is_main:
                print(*items)

        log("Resolved paths:")
        log(f"  project_root={cfg['project']['project_root']}")
        log(f"  run_dir={run_dir}")
        log(f"  model_path={model_path}")
        log(f"  input_json={input_json}")
        log(f"  output_dir={output_dir}")
        log(f"  output_manifest={output_manifest}")
        log(f"  lora_path={lora_path}")
        log("Generation args:")
        log(
            f"  task={task} wan_task_key={wan_task_key} size={size_text} max_area={max_area} "
            f"frame_num={frame_num} steps={sampling_steps} shift={shift} guide_scale={guide_scale_json(guide_scale)} seeds={seeds}"
        )
        log(
            f"  distributed={dist_state['distributed']} world_size={dist_state['world_size']} "
            f"rank={rank} device_id={device_id}"
        )

        if not input_json.exists():
            raise FileNotFoundError(input_json)
        if task == "t2v" and ("test_i2v" in str(input_json) or "train_i2v" in str(input_json)):
            raise ValueError(f"Refusing non-T2V input manifest: {input_json}")
        if task == "i2v" and ("test_t2v" in str(input_json) or "train_t2v" in str(input_json)):
            raise ValueError(f"Refusing non-I2V input manifest: {input_json}")
        samples = load_samples(input_json, task)
        if args.num_prompts:
            samples = samples[: args.num_prompts]
        if args.num_shards < 1:
            raise ValueError("--num_shards must be positive")
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
        if args.num_shards > 1:
            samples = [item for idx, item in enumerate(samples) if idx % args.num_shards == args.shard_index]
        if not samples:
            raise RuntimeError("No prompts to generate")

        if not torch.cuda.is_available():
            raise RuntimeError("WAN2.2 A14B generation requires CUDA; preflight should report GPU availability first.")
        torch.cuda.set_device(device_id)
        engine = build_engine(task, wan_cfg, model_path, device_id, rank, args, generation_cfg, bool(dist_state["distributed"]))
        lora_loaded = mount_dual_lora(engine, lora_path, args.lora_weight)
        engine.low_noise_model.eval()
        engine.high_noise_model.eval()

        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        groups = []
        for idx, item in enumerate(samples):
            group_id = safe_id(str(item.get("group_id", item.get("scene_uid", idx))))
            prompt = str(item.get("text_prompt", item.get("prompt", ""))).strip()
            assert prompt, f"Empty prompt for {group_id}"
            image_path = resolve_image_path(item, cfg, run_dir) if task == "i2v" else None
            group_dir = output_dir / group_id
            if is_main:
                group_dir.mkdir(parents=True, exist_ok=True)
            videos = []
            for seed in seeds:
                video_path = group_dir / f"seed_{seed}.mp4"
                should_generate = True
                if is_main:
                    should_generate = not existing_video_ok(video_path, frame_num) or args.force
                    if not should_generate:
                        print(f"[{idx + 1}/{len(samples)}] Skip valid existing {video_path}")
                    elif video_path.exists() and not args.force:
                        quarantined = quarantine_invalid_video(video_path)
                        print(
                            f"[{idx + 1}/{len(samples)}] Regenerating invalid/incomplete video {video_path}; "
                            f"moved old file to {quarantined}"
                        )
                if dist_state["distributed"]:
                    import torch.distributed as dist

                    flag = torch.tensor([1 if should_generate else 0], device=f"cuda:{device_id}")
                    dist.broadcast(flag, src=0)
                    should_generate = bool(flag.item())
                try:
                    video_tensor = None
                    if should_generate:
                        with torch.inference_mode():
                            if task == "t2v":
                                print(f"[rank {rank}] T2V A14B generating group={group_id} seed={seed}")
                                video_tensor = engine.generate(
                                    input_prompt=prompt,
                                    size=size,
                                    frame_num=frame_num,
                                    shift=shift,
                                    sample_solver=sample_solver,
                                    sampling_steps=sampling_steps,
                                    guide_scale=guide_scale,
                                    seed=seed,
                                    offload_model=offload_model,
                                )
                            else:
                                assert image_path is not None
                                print(f"[rank {rank}] I2V A14B generating group={group_id} seed={seed} image={image_path}")
                                with Image.open(image_path) as raw_image:
                                    image = raw_image.convert("RGB")
                                video_tensor = engine.generate(
                                    input_prompt=prompt,
                                    img=image,
                                    max_area=max_area,
                                    frame_num=frame_num,
                                    shift=shift,
                                    sample_solver=sample_solver,
                                    sampling_steps=sampling_steps,
                                    guide_scale=guide_scale,
                                    seed=seed,
                                    offload_model=offload_model,
                                )
                    if is_main and should_generate:
                        if video_tensor is None:
                            raise RuntimeError("Rank 0 did not receive generated frames")
                        save_video_ffmpeg(video_tensor, video_path, fps=fps, force=args.force)
                except Exception:
                    traceback.print_exc()
                    raise
                finally:
                    torch.cuda.empty_cache()
                if is_main:
                    probe = ffprobe_video(video_path)
                    if not probe.get("ok"):
                        raise RuntimeError(f"Generated video is not decodable: {video_path}: {probe}")
                    if int(probe.get("frames", 0)) != frame_num:
                        raise RuntimeError(f"Frame count mismatch for {video_path}: {probe}")
                    video_record = {
                        "generation_id": f"seed_{seed}",
                        "seed": seed,
                        "video_path": relative_or_abs(video_path, run_dir),
                        "frame_num": frame_num,
                        "size": size_text,
                        "max_area": max_area if task == "i2v" else None,
                        "fps": fps,
                        "lora_loaded": lora_loaded,
                        "ffprobe": probe,
                    }
                    if video_record["max_area"] is None:
                        del video_record["max_area"]
                    videos.append(video_record)
            if is_main:
                group = {
                    "group_id": group_id,
                    "scene_uid": item.get("scene_uid"),
                    "scene_id": item.get("scene_id"),
                    "source_split": item.get("source_split", "train"),
                    "source_bucket": item.get("source_bucket", "8k"),
                    "text_prompt": prompt,
                    "task": task,
                    "model": model_path.name,
                    "wan_task_key": wan_task_key,
                    "videos": videos,
                }
                if task == "i2v":
                    assert image_path is not None
                    group.update(
                        {
                            "camera_motion": item.get("camera_motion"),
                            "image_path": str(image_path),
                            "image_prompt": str(image_path),
                            "image_conditioned": True,
                        }
                    )
                groups.append(group)

        if is_main:
            payload = {
                "task": task,
                "text_only_branch": task == "t2v",
                "image_conditioned": task == "i2v",
                "base_path": str(run_dir),
                "model": model_path.name,
                "model_path": str(model_path),
                "wan_task_key": wan_task_key,
                "architecture": "dual_expert_a14b",
                "lora_path": str(lora_path) if lora_path else None,
                "generation_args": {
                    "frame_num": frame_num,
                    "size": size_text,
                    "max_area": max_area if task == "i2v" else None,
                    "sampling_steps": sampling_steps,
                    "guide_scale": guide_scale_json(guide_scale),
                    "sample_shift": shift,
                    "sample_solver": sample_solver,
                    "fps": fps,
                    "offload_model": offload_model,
                    "seeds": seeds,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "distributed": dist_state,
                },
                "groups": groups,
            }
            if payload["generation_args"]["max_area"] is None:
                del payload["generation_args"]["max_area"]
            write_json(output_manifest, payload)
            write_json(output_manifest.parent / "generation_args.json", payload["generation_args"])
            write_resolved_config(run_dir, cfg)
            print(f"Wrote candidate manifest: {output_manifest}")
    finally:
        cleanup_distributed(dist_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2 A14B T2V/I2V generation for VideoGPA")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", "--run-dir", dest="run_dir", default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_json", "--prompt_json", dest="input_json", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--candidate_groups_json", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--candidate_seeds", type=str, default=None)
    parser.add_argument("--candidates_per_prompt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Compatibility alias for one candidate seed")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--size", type=str, default=None)
    parser.add_argument("--max_area", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--guide_scale", default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--sample_solver", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--offload_model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--convert_model_dtype", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--t5_fsdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dit_fsdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_sp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.candidate_seeds is None:
        args.candidate_seeds = str(args.seed)
        args.candidates_per_prompt = 1
    generate(args)


if __name__ == "__main__":
    main()
