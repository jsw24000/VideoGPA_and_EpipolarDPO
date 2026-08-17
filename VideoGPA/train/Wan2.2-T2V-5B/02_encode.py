"""
WAN2.2 VideoGPA DPO - Step 2: encode text/image conditions and video latents.

Task handling:
- 5B T2V keeps the historical text-only condition files.
- 5B I2V adds the TI2V first-frame ``image_latent`` condition.
- A14B T2V uses Wan2.1 VAE latents.
- A14B I2V adds the WanI2V ``i2v_y`` conditioning tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
VIDEOGPA_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = VIDEOGPA_ROOT.parent
TRAIN_DIR = VIDEOGPA_ROOT / "train"
WAN_PATH = VIDEOGPA_ROOT / "Wan2.2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for path in [TRAIN_DIR, VIDEOGPA_ROOT, WAN_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vgm_common.config import resolve_experiment_config, write_resolved_config  # noqa: E402
from vgm_common.paths import get_dl3dv_root  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # noqa: E402
from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402

FIRST_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


def resolve_config(config_path: Path | None, run_dir: Path, model_path: str | None) -> dict[str, Any]:
    if config_path is None:
        raise ValueError("--config is required so paths can be resolved through the active VGM profile")
    cfg = resolve_experiment_config(config_path, run_dir, model_path_override=model_path)
    cfg.setdefault("encoding", {})
    cfg.setdefault("model", {})
    cfg["model"].setdefault("wan_task_key", "ti2v-5B")
    cfg["model"].setdefault("architecture", "single_ti2v_5b")
    cfg["model"].setdefault("vae_version", "wan2_2")
    return cfg


def safe_id(value: str) -> str:
    return value.strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_meta(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "path": str(path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def condition_valid(path: Path, *, task: str, architecture: str) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    data = torch_load(path)
    if not isinstance(data, dict):
        return False
    emb = data.get("encoder_hidden_states")
    if not isinstance(emb, torch.Tensor) or emb.numel() <= 0 or not torch.isfinite(emb).all().item():
        return False
    if task == "t2v":
        return "image_latent" not in data and "i2v_y" not in data
    if architecture == "single_ti2v_5b":
        image_latent = data.get("image_latent")
        return isinstance(image_latent, torch.Tensor) and image_latent.ndim == 4 and image_latent.numel() > 0
    if architecture == "dual_expert_a14b":
        i2v_y = data.get("i2v_y")
        return isinstance(i2v_y, torch.Tensor) and i2v_y.ndim == 4 and i2v_y.numel() > 0
    return False


def latent_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    latent = torch_load(path)
    return isinstance(latent, torch.Tensor) and latent.ndim == 4 and torch.isfinite(latent).all().item()


def load_video_frames(video_path: Path, num_frames: int, device: torch.device) -> torch.Tensor:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) <= 0:
        raise ValueError(f"No decodable frames: {video_path}")
    indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0
    return frames.permute(3, 0, 1, 2).contiguous().to(device)


def load_pairs(input_path: Path, expected_task: str) -> tuple[list[dict[str, Any]], Path]:
    payload = read_json(input_path)
    pairs = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(pairs, list):
        raise ValueError(f"Expected list of preference pairs in {input_path}")
    if not pairs:
        raise ValueError(f"No preference pairs in {input_path}")
    if isinstance(payload, dict) and payload.get("base_path"):
        base_path = Path(payload["base_path"])
    else:
        base_path = input_path.parents[1] if input_path.parent.name == "manifests" else input_path.parent
    for pair in pairs:
        task = str(pair.get("task", expected_task)).lower()
        if task != expected_task:
            raise ValueError(f"Refusing {task!r} pair in {expected_task!r} encoder: {pair.get('pair_id')}")
        if pair.get("source_split", "train") != "train":
            raise ValueError(f"Refusing non-train pair: {pair.get('pair_id')}")
    return pairs, base_path.expanduser().resolve()


def relative_or_text(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_video_path(video_value: str, candidate_base_path: Path) -> Path:
    video_path = Path(video_value).expanduser()
    if not video_path.is_absolute():
        video_path = candidate_base_path / video_path
    return video_path.resolve()


def find_first_frame(first_frames_root: Path, split: str, bucket: str, scene_id: str) -> Path | None:
    scene_dir = first_frames_root / split / bucket.upper() / scene_id
    for ext in FIRST_FRAME_EXTENSIONS:
        candidate = scene_dir / f"first_frame{ext}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_image_path(pair: dict[str, Any], cfg: dict[str, Any], run_dir: Path) -> Path:
    project_root = Path(cfg["project"]["project_root"])
    data_root = get_dl3dv_root()
    first_frames_root = Path(cfg["paths"]["first_frames_root"])
    raw_values = [
        pair.get("image_path"),
        pair.get("image_prompt"),
        pair.get("input_image_path"),
        pair.get("first_frame_path"),
        pair.get("first_frame_relpath"),
    ]
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        expanded = path.expanduser()
        if expanded not in candidates:
            candidates.append(expanded)

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
            idx = parts.index("first_frames")
            tail = Path(*parts[idx + 1 :])
            add_candidate(first_frames_root / tail)
            add_candidate(data_root / "first_frames" / tail)
        elif parts and parts[0] in {"train", "test"}:
            add_candidate(first_frames_root / raw)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    scene_id = pair.get("scene_id")
    bucket = pair.get("source_bucket")
    split = pair.get("source_split", "train")
    if scene_id and bucket:
        found = find_first_frame(first_frames_root, str(split), str(bucket), str(scene_id))
        if found is not None:
            return found
    tried = "\n".join(str(path) for path in candidates) or "(no explicit image path candidates)"
    raise FileNotFoundError(f"Could not resolve first-frame image for {pair.get('pair_id', pair.get('group_id'))}:\n{tried}")


def load_image_tensor_5b(image_path: Path, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    scale = max(target_w / img.width, target_h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1 = (img.width - target_w) // 2
    y1 = (img.height - target_h) // 2
    img = img.crop((x1, y1, x1 + target_w, y1 + target_h))
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5)
    return img_tensor.unsqueeze(1).to(device)


def build_a14b_i2v_y(
    image_path: Path,
    *,
    vae: Any,
    latent_shape: tuple[int, int, int, int],
    frame_num: int,
    vae_stride: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    _channels, latent_f, latent_h, latent_w = latent_shape
    target_h = latent_h * vae_stride[1]
    target_w = latent_w * vae_stride[2]
    img = Image.open(image_path).convert("RGB")
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

    msk = torch.ones(1, frame_num, latent_h, latent_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    if msk.shape[1] < latent_f * 4:
        pad = torch.zeros(1, latent_f * 4 - msk.shape[1], latent_h, latent_w, device=device)
        msk = torch.cat([msk, pad], dim=1)
    msk = msk[:, : latent_f * 4]
    msk = msk.view(1, latent_f, 4, latent_h, latent_w).transpose(1, 2)[0]

    image_video = torch.concat(
        [
            torch.nn.functional.interpolate(
                img_tensor[None].cpu(),
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            ).transpose(0, 1),
            torch.zeros(3, frame_num - 1, target_h, target_w),
        ],
        dim=1,
    ).to(device)
    with torch.no_grad():
        image_latent = vae.encode([image_video])[0]
    return torch.concat([msk, image_latent]).detach().cpu()


def make_vae(config: Any, vae_version: str, model_path: Path, device: torch.device) -> Any:
    if vae_version == "wan2_1":
        return Wan2_1_VAE(vae_pth=str(model_path / config.vae_checkpoint), device=device)
    if vae_version == "wan2_2":
        return Wan2_2_VAE(vae_pth=str(model_path / config.vae_checkpoint), device=device)
    raise ValueError(f"Unsupported model.vae_version={vae_version!r}")


def encode(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(args.config if args.config else None, run_dir, args.model_path)
    task = str(cfg.get("project", {}).get("task", "t2v")).lower()
    architecture = str(cfg.get("model", {}).get("architecture", "single_ti2v_5b"))
    wan_task_key = str(cfg.get("model", {}).get("wan_task_key", "ti2v-5B"))
    vae_version = str(cfg.get("model", {}).get("vae_version", "wan2_2"))
    model_path = Path(cfg["paths"]["wan_model_path"]).resolve()
    input_json = Path(args.input_json or run_dir / "manifests/preference_pairs.json").expanduser().resolve()
    output_json = Path(args.output_json or run_dir / "manifests/encoded_pairs.json").expanduser().resolve()
    encoded_root = Path(args.encoded_root or run_dir / "encoded").expanduser().resolve()
    num_frames = int(args.num_frames or cfg.get("encoding", {}).get("num_frames", cfg.get("generation", {}).get("frame_num", 81)))
    force = bool(args.force)

    if "test_t2v" in str(input_json) or "test_i2v" in str(input_json):
        raise ValueError(f"Refusing test manifest input: {input_json}")
    pairs, candidate_base_path = load_pairs(input_json, task)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        pairs = [pair for idx, pair in enumerate(pairs) if idx % args.num_shards == args.shard_index]
    if not pairs:
        raise RuntimeError(f"No preference pairs assigned to shard {args.shard_index}/{args.num_shards}")

    print("Resolved paths:")
    print(f"  task={task}")
    print(f"  architecture={architecture}")
    print(f"  wan_task_key={wan_task_key}")
    print(f"  vae_version={vae_version}")
    print(f"  run_dir={run_dir}")
    print(f"  candidate_base_path={candidate_base_path}")
    print(f"  model_path={model_path}")
    print(f"  input_json={input_json}")
    print(f"  output_json={output_json}")
    print(f"  encoded_root={encoded_root}")
    print(f"  num_frames={num_frames}")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = WAN_CONFIGS[wan_task_key]
    vae_stride = tuple(int(value) for value in config.vae_stride)
    t5 = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=str(model_path / config.t5_checkpoint),
        tokenizer_path=str(model_path / config.t5_tokenizer),
    )
    vae = make_vae(config, vae_version, model_path, device)

    cond_dir = encoded_root / "conditions"
    win_dir = encoded_root / "winners"
    lose_dir = encoded_root / "losers"
    for path in [cond_dir, win_dir, lose_dir]:
        path.mkdir(parents=True, exist_ok=True)

    encoded_pairs: list[dict[str, Any]] = []
    groups_for_dataset: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        pair_id = safe_id(pair.get("pair_id") or f"pair_{idx:06d}")
        group_id = safe_id(pair.get("group_id") or pair_id)
        prompt = str(pair.get("text_prompt", pair.get("prompt", ""))).strip()
        if not prompt:
            raise ValueError(f"Empty prompt for {pair_id}")
        winner = dict(pair["winner"])
        loser = dict(pair["loser"])

        latent_entries: dict[str, dict[str, Any]] = {}
        for role, entry, out_dir in [("winner", winner, win_dir), ("loser", loser, lose_dir)]:
            video_rel = entry.get("video_path")
            if not video_rel:
                raise ValueError(f"Missing {role} video_path for {pair_id}")
            video_path = resolve_video_path(str(video_rel), candidate_base_path)
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            latent_path = out_dir / f"latent_{pair_id}_{role}.pt"
            if latent_path.exists() and latent_valid(latent_path) and not force:
                latent = torch_load(latent_path)
            else:
                video_tensor = load_video_frames(video_path, num_frames, device)
                with torch.no_grad():
                    latent_list = vae.encode([video_tensor])
                if latent_list is None:
                    raise RuntimeError(f"VAE encode returned None for {pair_id} {role}")
                latent = latent_list[0].detach().cpu()
                if not torch.isfinite(latent).all().item():
                    raise RuntimeError(f"Non-finite latent for {pair_id} {role}")
                torch.save(latent, latent_path)
            latent_entries[role] = {"video_path": video_path, "latent_path": latent_path, "latent": latent}

        win_lat = latent_entries["winner"]["latent"]
        lose_lat = latent_entries["loser"]["latent"]
        if tuple(win_lat.shape) != tuple(lose_lat.shape):
            raise RuntimeError(f"Winner/loser latent shape mismatch in {pair_id}: {win_lat.shape} vs {lose_lat.shape}")

        cond_path = cond_dir / f"cond_{group_id}.pt"
        if cond_path.exists() and condition_valid(cond_path, task=task, architecture=architecture) and not force:
            condition = torch_load(cond_path)
            prompt_emb = condition["encoder_hidden_states"]
        else:
            condition: dict[str, Any] = {}
            with torch.no_grad():
                context = t5([prompt], device)
            prompt_emb = context[0].detach().cpu()
            if not torch.isfinite(prompt_emb).all().item() or prompt_emb.numel() == 0:
                raise RuntimeError(f"Invalid prompt embedding for {pair_id}")
            condition["encoder_hidden_states"] = prompt_emb
            if task == "i2v":
                image_path = resolve_image_path(pair, cfg, run_dir)
                if architecture == "single_ti2v_5b":
                    target_h = int(win_lat.shape[2]) * vae_stride[1]
                    target_w = int(win_lat.shape[3]) * vae_stride[2]
                    img_tensor = load_image_tensor_5b(image_path, target_h, target_w, device)
                    with torch.no_grad():
                        image_latent = vae.encode([img_tensor])[0].detach().cpu()
                    condition["image_latent"] = image_latent
                elif architecture == "dual_expert_a14b":
                    condition["i2v_y"] = build_a14b_i2v_y(
                        image_path,
                        vae=vae,
                        latent_shape=tuple(int(v) for v in win_lat.shape),
                        frame_num=num_frames,
                        vae_stride=vae_stride,
                        device=device,
                    )
                else:
                    raise ValueError(f"Unsupported I2V architecture: {architecture}")
                condition["image_path"] = str(image_path)
            torch.save(condition, cond_path)

        condition_payload = torch_load(cond_path)
        condition_keys = sorted(key for key in condition_payload.keys() if not key.endswith("_path"))
        contains_image_condition = task == "i2v"
        winner["latent_path"] = str(latent_entries["winner"]["latent_path"].relative_to(run_dir))
        winner["condition_path"] = str(cond_path.relative_to(run_dir))
        loser["latent_path"] = str(latent_entries["loser"]["latent_path"].relative_to(run_dir))
        loser["condition_path"] = str(cond_path.relative_to(run_dir))

        encoded_pair = {
            "pair_id": pair_id,
            "group_id": group_id,
            "scene_uid": pair.get("scene_uid"),
            "scene_id": pair.get("scene_id"),
            "prompt": prompt,
            "text_prompt": prompt,
            "condition_path": str(cond_path.relative_to(run_dir)),
            "winner_latent_path": winner["latent_path"],
            "loser_latent_path": loser["latent_path"],
            "winner_video_path": relative_or_text(latent_entries["winner"]["video_path"], run_dir),
            "loser_video_path": relative_or_text(latent_entries["loser"]["video_path"], run_dir),
            "winner_score": pair.get("winner_score"),
            "loser_score": pair.get("loser_score"),
            "score_gap": pair.get("score_gap"),
            "task": task,
            "architecture": architecture,
            "wan_task_key": wan_task_key,
            "contains_image_condition": contains_image_condition,
            "condition_keys": condition_keys,
            "text_embedding_shape": list(prompt_emb.shape),
            "video_latent_shape": list(win_lat.shape),
            "condition_meta": tensor_meta(cond_path, prompt_emb),
            "winner_latent_meta": tensor_meta(latent_entries["winner"]["latent_path"], win_lat),
            "loser_latent_meta": tensor_meta(latent_entries["loser"]["latent_path"], lose_lat),
        }
        for key in ("image_path", "image_prompt", "first_frame_relpath", "camera_motion"):
            if pair.get(key) is not None:
                encoded_pair[key] = pair[key]
        encoded_pairs.append(encoded_pair)
        groups_for_dataset.append(
            {
                "group_id": group_id,
                "text_prompt": prompt,
                "task": task,
                "architecture": architecture,
                "contains_image_condition": contains_image_condition,
                "videos": [winner, loser],
            }
        )
        print(f"Encoded {idx + 1}/{len(pairs)}: {pair_id}")

    payload = {
        "task": task,
        "architecture": architecture,
        "wan_task_key": wan_task_key,
        "vae_version": vae_version,
        "contains_image_condition": task == "i2v",
        "base_path": str(run_dir),
        "candidate_base_path": str(candidate_base_path),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pairs": encoded_pairs,
        "groups": groups_for_dataset,
    }
    write_json(output_json, payload)
    write_resolved_config(run_dir, cfg)
    print(f"Wrote encoded manifest: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2 VideoGPA: encode text/image conditions and winner/loser video latents")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", "--run-dir", dest="run_dir", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--input_json", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--encoded_root", default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    encode(args)


if __name__ == "__main__":
    main()
