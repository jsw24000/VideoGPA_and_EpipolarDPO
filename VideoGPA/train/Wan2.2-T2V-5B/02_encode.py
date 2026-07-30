"""
Wan2.2 TI2V-5B text-only VideoGPA DPO - Step 2: encode text conditions and video latents.

This file is a low-intrusion T2V sibling of ``Wan2.2-TI2V-5B/02_encode.py``.
The WAN T5 tokenizer/text encoder, VAE, video normalization, and latent layout
are preserved. The TI2V-only first-frame image condition is intentionally absent:
no image path is read and no ``image_latent`` key is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from decord import VideoReader, cpu

CURRENT_DIR = Path(__file__).resolve().parent
VIDEOGPA_ROOT = CURRENT_DIR.parents[1]
PROJECT_ROOT = VIDEOGPA_ROOT.parent
TRAIN_DIR = VIDEOGPA_ROOT / "train"
WAN_PATH = VIDEOGPA_ROOT / "Wan2.2"
for path in [TRAIN_DIR, VIDEOGPA_ROOT, WAN_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


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


def find_unique_wan_model(models_root: Path) -> Path:
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
        raise FileNotFoundError(f"No WAN2.2-TI2V-5B model found under {models_root}")
    if len(candidates) > 1:
        raise RuntimeError("Multiple WAN candidates found:\n" + "\n".join(str(p) for p in candidates))
    return candidates[0]


def resolve_config(config_path: Path | None, run_dir: Path, model_path: str | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if config_path:
        cfg = read_yaml(config_path)
    cfg.setdefault("project", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("encoding", {})
    project_root = Path(os.environ.get("PROJECT_ROOT", PROJECT_ROOT)).expanduser().resolve()
    cfg["project"]["project_root"] = str(project_root)
    cfg["project"]["task"] = "t2v"
    cfg["paths"]["run_dir"] = str(run_dir.resolve())
    cfg["paths"]["videogpa_root"] = str(resolve_path(project_root, cfg["paths"].get("videogpa_root", "VideoGPA")))
    if model_path:
        cfg["paths"]["wan_model_path"] = str(resolve_path(project_root, model_path))
    elif os.environ.get("WAN22_5B_MODEL_PATH"):
        cfg["paths"]["wan_model_path"] = str(resolve_path(project_root, os.environ["WAN22_5B_MODEL_PATH"]))
    elif cfg["paths"].get("wan_model_path", "auto") == "auto":
        cfg["paths"]["wan_model_path"] = str(find_unique_wan_model(project_root / "models"))
    else:
        cfg["paths"]["wan_model_path"] = str(resolve_path(project_root, cfg["paths"]["wan_model_path"]))
    return cfg


def safe_id(value: str) -> str:
    return value.strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def condition_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    data = torch_load(path)
    if not isinstance(data, dict):
        return False
    if "image_latent" in data:
        return False
    emb = data.get("encoder_hidden_states")
    return isinstance(emb, torch.Tensor) and emb.numel() > 0 and torch.isfinite(emb).all().item()


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


def load_pairs(input_path: Path) -> list[dict[str, Any]]:
    payload = read_json(input_path)
    pairs = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(pairs, list):
        raise ValueError(f"Expected list of preference pairs in {input_path}")
    if not pairs:
        raise ValueError(f"No preference pairs in {input_path}")
    for pair in pairs:
        if pair.get("task") != "t2v":
            raise ValueError(f"Refusing non-T2V pair: {pair.get('pair_id')}")
        if pair.get("source_split", "train") != "train":
            raise ValueError(f"Refusing non-train pair: {pair.get('pair_id')}")
    return pairs


def encode(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config).expanduser().resolve() if args.config else None, run_dir, args.model_path)
    assert cfg["project"].get("task") == "t2v"
    model_path = Path(cfg["paths"]["wan_model_path"]).resolve()
    input_json = Path(args.input_json or run_dir / "manifests/preference_pairs.json").expanduser().resolve()
    output_json = Path(args.output_json or run_dir / "manifests/encoded_pairs.json").expanduser().resolve()
    encoded_root = Path(args.encoded_root or run_dir / "encoded").expanduser().resolve()
    num_frames = int(args.num_frames or cfg.get("encoding", {}).get("num_frames", cfg.get("generation", {}).get("frame_num", 81)))
    force = bool(args.force)

    if "test_t2v" in str(input_json) or "test_i2v" in str(input_json):
        raise ValueError(f"Refusing test manifest input: {input_json}")
    pairs = load_pairs(input_json)

    print("Resolved paths:")
    print(f"  run_dir={run_dir}")
    print(f"  model_path={model_path}")
    print(f"  input_json={input_json}")
    print(f"  output_json={output_json}")
    print(f"  encoded_root={encoded_root}")
    print(f"  num_frames={num_frames}")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = WAN_CONFIGS["ti2v-5B"]
    t5 = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=str(model_path / config.t5_checkpoint),
        tokenizer_path=str(model_path / config.t5_tokenizer),
    )
    vae = Wan2_2_VAE(vae_pth=str(model_path / config.vae_checkpoint), device=device)

    cond_dir = encoded_root / "conditions"
    win_dir = encoded_root / "winners"
    lose_dir = encoded_root / "losers"
    for path in [cond_dir, win_dir, lose_dir]:
        path.mkdir(parents=True, exist_ok=True)

    encoded_pairs = []
    groups_for_dataset = []
    for idx, pair in enumerate(pairs):
        pair_id = safe_id(pair.get("pair_id") or f"pair_{idx:06d}")
        group_id = safe_id(pair.get("group_id") or pair_id)
        prompt = str(pair.get("text_prompt", pair.get("prompt", ""))).strip()
        if not prompt:
            raise ValueError(f"Empty prompt for {pair_id}")
        winner = dict(pair["winner"])
        loser = dict(pair["loser"])
        for role, entry in [("winner", winner), ("loser", loser)]:
            if any(key in entry for key in ["image_path", "image_prompt", "input_image_path"]):
                raise ValueError(f"{role} contains image condition fields in {pair_id}")

        cond_path = cond_dir / f"cond_{group_id}.pt"
        if cond_path.exists() and condition_valid(cond_path) and not force:
            condition = torch_load(cond_path)
            prompt_emb = condition["encoder_hidden_states"]
        else:
            condition = {}
            with torch.no_grad():
                context = t5([prompt], device)
            prompt_emb = context[0].detach().cpu()
            if not torch.isfinite(prompt_emb).all().item() or prompt_emb.numel() == 0:
                raise RuntimeError(f"Invalid prompt embedding for {pair_id}")
            condition["encoder_hidden_states"] = prompt_emb
            assert "image_latent" not in condition
            torch.save(condition, cond_path)

        latent_entries = {}
        for role, entry, out_dir in [("winner", winner, win_dir), ("loser", loser, lose_dir)]:
            video_rel = entry.get("video_path")
            if not video_rel:
                raise ValueError(f"Missing {role} video_path for {pair_id}")
            video_path = Path(video_rel)
            if not video_path.is_absolute():
                video_path = run_dir / video_path
            video_path = video_path.resolve()
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            latent_path = out_dir / f"latent_{pair_id}_{role}.pt"
            if latent_path.exists() and latent_valid(latent_path) and not force:
                latent = torch_load(latent_path)
            else:
                video_tensor = load_video_frames(video_path, num_frames, device)
                with torch.no_grad():
                    latent_list = vae.encode([video_tensor])
                latent = latent_list[0].detach().cpu()
                if not torch.isfinite(latent).all().item():
                    raise RuntimeError(f"Non-finite latent for {pair_id} {role}")
                torch.save(latent, latent_path)
            latent_entries[role] = {
                "video_path": video_path,
                "latent_path": latent_path,
                "latent": latent,
            }

        win_lat = latent_entries["winner"]["latent"]
        lose_lat = latent_entries["loser"]["latent"]
        if tuple(win_lat.shape) != tuple(lose_lat.shape):
            raise RuntimeError(f"Winner/loser latent shape mismatch in {pair_id}: {win_lat.shape} vs {lose_lat.shape}")
        if win_lat.shape[1] != lose_lat.shape[1]:
            raise RuntimeError(f"Temporal latent length mismatch in {pair_id}")

        cond_loaded = torch_load(cond_path)
        assert "image_latent" not in cond_loaded, "T2V condition must not contain image_latent"
        assert "encoder_hidden_states" in cond_loaded

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
            "winner_video_path": str(latent_entries["winner"]["video_path"].relative_to(run_dir)),
            "loser_video_path": str(latent_entries["loser"]["video_path"].relative_to(run_dir)),
            "winner_score": pair.get("winner_score"),
            "loser_score": pair.get("loser_score"),
            "score_gap": pair.get("score_gap"),
            "task": "t2v",
            "contains_image_condition": False,
            "text_embedding_shape": list(prompt_emb.shape),
            "video_latent_shape": list(win_lat.shape),
            "condition_meta": tensor_meta(cond_path, prompt_emb),
            "winner_latent_meta": tensor_meta(latent_entries["winner"]["latent_path"], win_lat),
            "loser_latent_meta": tensor_meta(latent_entries["loser"]["latent_path"], lose_lat),
        }
        encoded_pairs.append(encoded_pair)
        groups_for_dataset.append(
            {
                "group_id": group_id,
                "text_prompt": prompt,
                "task": "t2v",
                "contains_image_condition": False,
                "videos": [winner, loser],
            }
        )
        print(f"Encoded {idx + 1}/{len(pairs)}: {pair_id}")

    payload = {
        "task": "t2v",
        "contains_image_condition": False,
        "base_path": str(run_dir),
        "pairs": encoded_pairs,
        "groups": groups_for_dataset,
    }
    write_json(output_json, payload)
    write_yaml(run_dir / "config/resolved_config.yaml", cfg)
    print(f"Wrote encoded manifest: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.2 T2V: encode text conditions and winner/loser video latents")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_dir", "--run-dir", dest="run_dir", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--input_json", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--encoded_root", default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    encode(args)


if __name__ == "__main__":
    main()
