from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


class ResumeError(RuntimeError):
    """Base class for unsafe or incompatible training resume requests."""


class ResumeConfigError(ResumeError):
    """Raised when checkpoint and current training configs are incompatible."""


STRICT_TRAINING_KEYS = (
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "learning_rate",
    "warmup_steps",
    "max_steps",
    "batch_size",
    "accumulate_grad_batches",
    "beta",
    "num_train_timesteps",
    "shift",
    "gradient_clip_val",
)

PATH_FIELDS_ALLOWED_TO_DRIFT = (
    "config_path",
    "project_root",
    "videogpa_root",
    "wan_source_root",
    "wan_model_path",
    "vggt_model_path",
    "train_manifest",
    "manifest_root",
    "first_frames_root",
    "output_root",
    "profile_output_root",
    "run_dir",
)

_TRAINING_ALIASES = {
    "max_steps": ("max_steps", "max_train_steps"),
    "batch_size": ("batch_size", "batch_size_per_gpu"),
    "accumulate_grad_batches": ("accumulate_grad_batches", "gradient_accumulation_steps"),
    "beta": ("beta", "dpo_beta"),
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def parse_checkpoint_step(checkpoint_dir: str | Path) -> int:
    name = Path(checkpoint_dir).name
    match = re.fullmatch(r"step_(\d{6,})", name)
    if not match:
        raise ValueError(f"Checkpoint directory must be named step_XXXXXX, got: {name}")
    return int(match.group(1))


def discover_checkpoint_files(checkpoint_dir: str | Path) -> dict[str, Path | None]:
    root = Path(checkpoint_dir).expanduser().resolve(strict=False)
    adapter_model = None
    for candidate in ("adapter_model.safetensors", "adapter_model.bin"):
        path = root / candidate
        if path.is_file():
            adapter_model = path
            break
    return {
        "checkpoint_dir": root,
        "adapter_model": adapter_model,
        "adapter_config": root / "adapter_config.json",
        "optimizer": root / "optimizer.pt",
        "scheduler": root / "scheduler.pt",
        "trainer_state": root / "trainer_state.json",
        "config_resolved": root / "config_resolved.yaml",
        "rng_state": root / "rng_state.pt",
    }


def validate_checkpoint_manifest(checkpoint_dir: str | Path) -> dict[str, Path | None]:
    files = discover_checkpoint_files(checkpoint_dir)
    root = files["checkpoint_dir"]
    assert isinstance(root, Path)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")

    required = {
        "adapter_model": files["adapter_model"],
        "adapter_config": files["adapter_config"],
        "optimizer": files["optimizer"],
        "scheduler": files["scheduler"],
        "trainer_state": files["trainer_state"],
        "config_resolved": files["config_resolved"],
    }
    missing = []
    for label, path in required.items():
        if path is None or not Path(path).is_file() or Path(path).stat().st_size <= 0:
            missing.append(label)
    if missing:
        raise FileNotFoundError(f"Checkpoint {root} is missing required file(s): {', '.join(missing)}")
    return files


def discover_latest_checkpoint(checkpoint_root: str | Path) -> Path:
    root = Path(checkpoint_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            step = parse_checkpoint_step(child)
            validate_checkpoint_manifest(child)
        except (ValueError, FileNotFoundError):
            continue
        candidates.append((step, child))
    if not candidates:
        raise FileNotFoundError(f"No complete step_* checkpoints found under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def ensure_checkpoint_save_target_safe(checkpoint_dir: str | Path) -> None:
    path = Path(checkpoint_dir).expanduser().resolve(strict=False)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty checkpoint directory: {path}")


def _training_section(config: dict[str, Any]) -> dict[str, Any]:
    if any(key in config for key in STRICT_TRAINING_KEYS):
        return config
    if isinstance(config.get("training_resolved"), dict):
        return config["training_resolved"]
    if isinstance(config.get("training"), dict):
        return config["training"]
    return {}


def normalize_training_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _training_section(config)
    normalized: dict[str, Any] = {}
    for key in STRICT_TRAINING_KEYS:
        aliases = _TRAINING_ALIASES.get(key, (key,))
        for alias in aliases:
            if alias in raw:
                normalized[key] = raw[alias]
                break
    if "lora_target_modules" in normalized:
        normalized["lora_target_modules"] = list(normalized["lora_target_modules"])
    return normalized


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


def compare_path_fields(current_config: dict[str, Any], checkpoint_config: dict[str, Any]) -> list[dict[str, str]]:
    current_paths = current_config.get("paths", {}) if isinstance(current_config.get("paths"), dict) else {}
    checkpoint_paths = checkpoint_config.get("paths", {}) if isinstance(checkpoint_config.get("paths"), dict) else {}
    diffs = []
    for key in PATH_FIELDS_ALLOWED_TO_DRIFT:
        current_value = current_paths.get(key)
        checkpoint_value = checkpoint_paths.get(key)
        if current_value is not None and checkpoint_value is not None and current_value != checkpoint_value:
            diffs.append({"field": f"paths.{key}", "checkpoint": str(checkpoint_value), "current": str(current_value)})
    return diffs


def validate_resume_config(current_config: dict[str, Any], checkpoint_config: dict[str, Any]) -> dict[str, Any]:
    current = normalize_training_config(current_config)
    checkpoint = normalize_training_config(checkpoint_config)
    mismatches = []
    missing = []
    for key in STRICT_TRAINING_KEYS:
        if key not in current or key not in checkpoint:
            missing.append(key)
            continue
        if not _values_match(current[key], checkpoint[key]):
            mismatches.append({"field": key, "checkpoint": checkpoint[key], "current": current[key]})
    if missing or mismatches:
        message = []
        if missing:
            message.append(f"missing strict field(s): {', '.join(missing)}")
        if mismatches:
            rendered = ", ".join(f"{item['field']} checkpoint={item['checkpoint']!r} current={item['current']!r}" for item in mismatches)
            message.append(f"mismatched field(s): {rendered}")
        raise ResumeConfigError("; ".join(message))
    return {
        "strict_fields": list(STRICT_TRAINING_KEYS),
        "path_differences": compare_path_fields(current_config, checkpoint_config),
    }


def validate_resume_metadata(checkpoint_dir: str | Path, trainer_state: dict[str, Any], scheduler_state: dict[str, Any]) -> int:
    resume_step = parse_checkpoint_step(checkpoint_dir)
    trainer_step = trainer_state.get("step")
    if int(trainer_step) != resume_step:
        raise ResumeError(f"trainer_state step mismatch: checkpoint dir={resume_step}, trainer_state.step={trainer_step}")
    scheduler_epoch = scheduler_state.get("last_epoch")
    if int(scheduler_epoch) != resume_step:
        raise ResumeError(f"scheduler step mismatch: checkpoint dir={resume_step}, scheduler.last_epoch={scheduler_epoch}")
    return resume_step


def compute_resume_data_cursor(resume_step: int, accumulate_grad_batches: int, dataloader_len: int) -> dict[str, int]:
    if resume_step < 0:
        raise ValueError("resume_step must be non-negative")
    if accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be positive")
    if dataloader_len <= 0:
        raise ValueError("dataloader_len must be positive")
    consumed_microbatches = resume_step * accumulate_grad_batches
    return {
        "resume_step": resume_step,
        "consumed_microbatches": consumed_microbatches,
        "dataloader_len": dataloader_len,
        "data_epoch": consumed_microbatches // dataloader_len,
        "data_offset": consumed_microbatches % dataloader_len,
    }
