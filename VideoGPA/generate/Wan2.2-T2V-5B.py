from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel

CURRENT_DIR = Path(__file__).resolve().parent
VIDEOGPA_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = VIDEOGPA_ROOT.parent
WAN_PATH = VIDEOGPA_ROOT / "Wan2.2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WAN_PATH) not in sys.path:
    sys.path.insert(0, str(WAN_PATH))

from vgm_common.config import resolve_experiment_config, write_resolved_config  # noqa: E402
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402

WAN_CONFIG = WAN_CONFIGS["ti2v-5B"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(project_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def iter_dirs_limited(root: Path, max_depth: int = 5):
    root = root.resolve()
    if not root.exists():
        return
    for parent, dirs, _files in os.walk(root):
        parent_path = Path(parent)
        try:
            depth = len(parent_path.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            dirs[:] = []
            continue
        yield parent_path
        if depth == max_depth:
            dirs[:] = []


def find_unique_model_dir(models_root: Path) -> Path:
    candidates = sorted(
        {
            p.resolve()
            for p in iter_dirs_limited(models_root, max_depth=5)
            if p.is_dir()
            and p.name == "Wan2.2-TI2V-5B"
            and (p / "Wan2.2_VAE.pth").is_file()
            and (p / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
            and (p / "diffusion_pytorch_model.safetensors.index.json").is_file()
        }
    )
    if not candidates:
        raise FileNotFoundError(f"No Wan2.2-TI2V-5B model found under {models_root}")
    if len(candidates) > 1:
        raise RuntimeError("Multiple WAN candidates found:\n" + "\n".join(str(p) for p in candidates))
    return candidates[0]


def resolve_config(config_path: Path | None, run_dir: Path | None, model_path: str | None) -> dict[str, Any]:
    if config_path is None:
        raise ValueError("--config is required so paths can be resolved through the active VGM profile")
    cfg = resolve_experiment_config(config_path, run_dir, model_path_override=model_path)
    cfg.setdefault("generation", {})
    cfg.setdefault("data", {})
    cfg["project"]["task"] = "t2v"
    return cfg


def safe_id(value: str) -> str:
    return value.strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


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


def save_video_ffmpeg(video_tensor: torch.Tensor, output_path: Path, fps: int = 24, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing video: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_data = video_tensor.permute(1, 2, 3, 0).cpu().numpy().clip(-1, 1)
    video_data = ((video_data + 1) * 127.5).astype(np.uint8)
    frames, height, width, _ = video_data.shape
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
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        str(output_path),
    ]
    print("Running ffmpeg:", " ".join(cmd[:-1]), str(output_path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(video_data.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {proc.returncode}\nSTDOUT:\n{stdout.decode(errors='replace')}\nSTDERR:\n{stderr.decode(errors='replace')}"
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


def load_samples(input_json: Path) -> list[dict[str, Any]]:
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
                sample.setdefault("scene_id", scene_uid.split("/", 1)[-1])
                sample.setdefault("source_split", "train")
                sample.setdefault("source_bucket", scene_uid.split("/", 1)[0].lower())
                sample.setdefault("task", "t2v")
                samples.append(sample)
    elif isinstance(payload, list):
        samples = payload
    else:
        raise ValueError(f"Unsupported prompt JSON format: {input_json}")
    clean = []
    for idx, item in enumerate(samples):
        prompt = str(item.get("text_prompt", item.get("prompt", ""))).strip()
        if not prompt:
            raise ValueError(f"Empty prompt at sample index {idx}")
        if any(key in item for key in ["image_path", "image_prompt", "input_image_path"]):
            raise ValueError(f"T2V input must not include image fields: {item.get('group_id', idx)}")
        clean.append(item)
    return clean


def mount_lora(engine: WanTI2V, lora_path: Path | None, lora_weight: float) -> bool:
    if not lora_path:
        return False
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    print(f"Mounting LoRA: {lora_path} (weight={lora_weight})")
    engine.model = PeftModel.from_pretrained(
        engine.model,
        str(lora_path),
        adapter_name="default",
        torch_dtype=torch.bfloat16,
    )
    if lora_weight != 1.0:
        for module in engine.model.modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                for adapter in module.scaling:
                    module.scaling[adapter] *= lora_weight
    engine.model.merge_and_unload()
    print("LoRA merged.")
    return True


def generate(args: argparse.Namespace) -> None:
    cfg = resolve_config(
        args.config if args.config else None,
        Path(args.run_dir).expanduser().resolve() if args.run_dir else None,
        args.model_path,
    )
    assert cfg["project"].get("task") == "t2v", "This entrypoint is T2V only"
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else Path(cfg["paths"].get("run_dir", ".")).resolve()
    input_json = Path(args.input_json or run_dir / "manifests/input_subset.json").expanduser().resolve()
    output_dir = Path(args.output_dir or run_dir / "candidates").expanduser().resolve()
    output_manifest = Path(args.candidate_groups_json or run_dir / "manifests/candidate_groups.json").expanduser().resolve()
    model_path = Path(cfg["paths"]["wan_model_path"]).resolve()
    lora_path = Path(args.lora_path).expanduser().resolve() if args.lora_path else None
    seeds = parse_seed_list(args.candidate_seeds, cfg)
    if args.candidates_per_prompt:
        seeds = seeds[: args.candidates_per_prompt]
    num_prompts = args.num_prompts

    generation_cfg = cfg.get("generation", {})
    frame_num = int(args.frame_num or generation_cfg.get("frame_num", 81))
    size_text = str(args.size or generation_cfg.get("size", "1280*704"))
    size = parse_size(size_text)
    sampling_steps = int(args.sampling_steps or generation_cfg.get("sampling_steps", 50))
    guide_scale = float(args.guide_scale or generation_cfg.get("guide_scale", 5.0))
    shift = float(args.shift or generation_cfg.get("sample_shift", 5.0))
    sample_solver = str(args.sample_solver or generation_cfg.get("sample_solver", "unipc"))
    fps = int(args.fps or generation_cfg.get("fps", 24))
    offload_model = bool(generation_cfg.get("offload_model", True)) if args.offload_model is None else args.offload_model
    t5_cpu = bool(generation_cfg.get("t5_cpu", True)) if args.t5_cpu is None else args.t5_cpu
    convert_model_dtype = (
        bool(generation_cfg.get("convert_model_dtype", True))
        if args.convert_model_dtype is None
        else args.convert_model_dtype
    )

    print("Resolved paths:")
    print(f"  project_root={cfg['project']['project_root']}")
    print(f"  run_dir={run_dir}")
    print(f"  model_path={model_path}")
    print(f"  input_json={input_json}")
    print(f"  output_dir={output_dir}")
    print(f"  output_manifest={output_manifest}")
    print(f"  lora_path={lora_path}")
    print("Generation args:")
    print(
        f"  task=t2v img=None size={size_text} frame_num={frame_num} steps={sampling_steps} "
        f"shift={shift} guide_scale={guide_scale} seeds={seeds}"
    )

    if not input_json.exists():
        raise FileNotFoundError(input_json)
    if "test_t2v" in str(input_json) or "test_i2v" in str(input_json) or "train_i2v" in str(input_json):
        raise ValueError(f"Refusing non-T2V-train input manifest: {input_json}")
    samples = load_samples(input_json)
    if num_prompts:
        samples = samples[:num_prompts]
    if not samples:
        raise RuntimeError("No prompts to generate")

    if not torch.cuda.is_available():
        raise RuntimeError("WAN2.2 generation smoke requires CUDA; preflight should report GPU availability first.")
    torch.cuda.set_device(args.gpu_id)
    engine = WanTI2V(
        config=WAN_CONFIG,
        checkpoint_dir=str(model_path),
        device_id=args.gpu_id,
        rank=0,
        t5_cpu=t5_cpu,
        convert_model_dtype=convert_model_dtype,
    )
    lora_loaded = mount_lora(engine, lora_path, args.lora_weight)
    engine.model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    for idx, item in enumerate(samples):
        group_id = safe_id(str(item.get("group_id", item.get("scene_uid", idx))))
        prompt = str(item.get("text_prompt", item.get("prompt", ""))).strip()
        assert prompt, f"Empty prompt for {group_id}"
        group_dir = output_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        videos = []
        for seed in seeds:
            video_path = group_dir / f"seed_{seed}.mp4"
            if existing_video_ok(video_path, frame_num) and not args.force:
                print(f"[{idx + 1}/{len(samples)}] Skip valid existing {video_path}")
            else:
                if video_path.exists() and not args.force:
                    quarantined = quarantine_invalid_video(video_path)
                    print(
                        f"[{idx + 1}/{len(samples)}] Regenerating invalid/incomplete video {video_path}; "
                        f"moved old file to {quarantined}"
                    )
                print(f"[{idx + 1}/{len(samples)}] T2V generating group={group_id} seed={seed}")
                try:
                    with torch.inference_mode():
                        # Official WAN2.2 TI2V uses img=None to enter its text-only T2V branch.
                        video_tensor = engine.generate(
                            input_prompt=prompt,
                            img=None,
                            size=size,
                            frame_num=frame_num,
                            shift=shift,
                            sample_solver=sample_solver,
                            sampling_steps=sampling_steps,
                            guide_scale=guide_scale,
                            seed=seed,
                            offload_model=offload_model,
                        )
                    save_video_ffmpeg(video_tensor, video_path, fps=fps, force=args.force)
                except Exception:
                    traceback.print_exc()
                    raise
                finally:
                    torch.cuda.empty_cache()
            probe = ffprobe_video(video_path)
            if not probe.get("ok"):
                raise RuntimeError(f"Generated video is not decodable: {video_path}: {probe}")
            if int(probe.get("frames", 0)) != frame_num:
                raise RuntimeError(f"Frame count mismatch for {video_path}: {probe}")
            videos.append(
                {
                    "generation_id": f"seed_{seed}",
                    "seed": seed,
                    "video_path": str(video_path.relative_to(run_dir)),
                    "frame_num": frame_num,
                    "size": size_text,
                    "fps": fps,
                    "lora_loaded": lora_loaded,
                    "ffprobe": probe,
                }
            )
        groups.append(
            {
                "group_id": group_id,
                "scene_uid": item.get("scene_uid"),
                "scene_id": item.get("scene_id"),
                "source_split": item.get("source_split", "train"),
                "source_bucket": item.get("source_bucket", "8k"),
                "text_prompt": prompt,
                "task": "t2v",
                "model": "Wan2.2-TI2V-5B",
                "videos": videos,
            }
        )

    payload = {
        "task": "t2v",
        "text_only_branch": True,
        "image_conditioned": False,
        "base_path": str(run_dir),
        "model": "Wan2.2-TI2V-5B",
        "model_path": str(model_path),
        "lora_path": str(lora_path) if lora_path else None,
        "generation_args": {
            "frame_num": frame_num,
            "size": size_text,
            "sampling_steps": sampling_steps,
            "guide_scale": guide_scale,
            "sample_shift": shift,
            "sample_solver": sample_solver,
            "fps": fps,
            "offload_model": offload_model,
            "t5_cpu": t5_cpu,
            "convert_model_dtype": convert_model_dtype,
            "seeds": seeds,
        },
        "groups": groups,
    }
    write_json(output_manifest, payload)
    write_json(output_manifest.parent / "generation_args.json", payload["generation_args"])
    write_resolved_config(run_dir, cfg)
    print(f"Wrote candidate manifest: {output_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B text-only T2V generation for VideoGPA")
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
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--size", type=str, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--guide_scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--sample_solver", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--offload_model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--convert_model_dtype", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.candidate_seeds is None:
        args.candidate_seeds = str(args.seed)
        args.candidates_per_prompt = 1
    generate(args)


if __name__ == "__main__":
    main()
