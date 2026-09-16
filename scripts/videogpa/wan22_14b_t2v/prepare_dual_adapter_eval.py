from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


EXPERT_DIR = {"high": "high_noise_model", "low": "low_noise_model"}
COMPATIBILITY_KEYS = (
    "task",
    "architecture",
    "wan_task_key",
    "reference_mode",
    "timestep_mode",
    "shift",
    "loss_strategy",
    "dpo_beta",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
)
ADAPTER_COMPATIBILITY_KEYS = (
    "peft_type",
    "task_type",
    "r",
    "target_modules",
    "lora_alpha",
    "lora_dropout",
    "fan_in_fan_out",
    "bias",
    "use_rslora",
    "use_dora",
    "rank_pattern",
    "alpha_pattern",
    "modules_to_save",
)
ORDER_INSENSITIVE_ADAPTER_KEYS = {"target_modules", "modules_to_save"}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_adapter_value(key: str, value: Any) -> Any:
    if key in ORDER_INSENSITIVE_ADAPTER_KEYS and isinstance(value, (list, tuple, set)):
        return sorted(value)
    return value


def adapter_files(adapter_dir: Path) -> tuple[Path, Path]:
    config = adapter_dir / "adapter_config.json"
    candidates = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    weights = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
    if not config.is_file() or weights is None:
        raise FileNotFoundError(f"Incomplete adapter directory: {adapter_dir}")
    return config, weights


def checkpoint_for(run_dir: Path, step: int, mode: str) -> dict[str, Any]:
    checkpoint = (run_dir / "checkpoints" / f"step_{step:06d}").resolve(strict=True)
    state_path = checkpoint / "trainer_state.json"
    state = read_json(state_path)
    config = state.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing config object in {state_path}")
    if state.get("step") != step:
        raise ValueError(f"Checkpoint step mismatch in {state_path}: expected {step}, got {state.get('step')}")
    if config.get("expert_mode") != mode:
        raise ValueError(
            f"Expert mismatch in {state_path}: expected {mode}, got {config.get('expert_mode')}"
        )
    expert_name = EXPERT_DIR[mode]
    adapter_dir = checkpoint / expert_name
    config_path, weights_path = adapter_files(adapter_dir)
    other_name = EXPERT_DIR["low" if mode == "high" else "high"]
    if (checkpoint / other_name).exists():
        raise ValueError(f"Single-expert checkpoint unexpectedly contains {other_name}: {checkpoint}")
    return {
        "mode": mode,
        "checkpoint": checkpoint,
        "state_path": state_path,
        "state": state,
        "config": config,
        "adapter_dir": adapter_dir.resolve(strict=True),
        "adapter_config": config_path.resolve(strict=True),
        "adapter_weights": weights_path.resolve(strict=True),
    }


def validate_pair(high: dict[str, Any], low: dict[str, Any], step: int) -> None:
    high_sha = high["state"].get("metadata_sha256")
    low_sha = low["state"].get("metadata_sha256")
    if not high_sha or high_sha != low_sha:
        raise ValueError(f"Encoded-pair metadata SHA256 mismatch at step {step}: high={high_sha}, low={low_sha}")
    mismatches = [
        key
        for key in COMPATIBILITY_KEYS
        if high["config"].get(key) != low["config"].get(key)
    ]
    if mismatches:
        raise ValueError(f"High/low training config mismatch at step {step}: {', '.join(mismatches)}")
    high_adapter = read_json(high["adapter_config"])
    low_adapter = read_json(low["adapter_config"])
    adapter_mismatches = [
        key
        for key in ADAPTER_COMPATIBILITY_KEYS
        if normalized_adapter_value(key, high_adapter.get(key))
        != normalized_adapter_value(key, low_adapter.get(key))
    ]
    if adapter_mismatches:
        raise ValueError(
            f"High/low adapter configuration mismatch at step {step}: {', '.join(adapter_mismatches)}"
        )


def source_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": str(item["checkpoint"]),
        "trainer_state": str(item["state_path"]),
        "adapter_dir": str(item["adapter_dir"]),
        "adapter_config": {
            "path": str(item["adapter_config"]),
            "size": item["adapter_config"].stat().st_size,
            "sha256": sha256_file(item["adapter_config"]),
        },
        "adapter_weights": {
            "path": str(item["adapter_weights"]),
            "size": item["adapter_weights"].stat().st_size,
            "sha256": sha256_file(item["adapter_weights"]),
        },
    }


def create_bundle(output_root: Path, step: int, high: dict[str, Any], low: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"step_{step:06d}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing bundle: {target}")
    temp = output_root / f".{target.name}.tmp.{os.getpid()}"
    if temp.exists():
        raise FileExistsError(f"Temporary path already exists: {temp}")
    try:
        temp.mkdir(parents=False)
        os.symlink(low["adapter_dir"], temp / "low_noise_model", target_is_directory=True)
        os.symlink(high["adapter_dir"], temp / "high_noise_model", target_is_directory=True)
        provenance = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "step": step,
            "metadata_sha256": high["state"]["metadata_sha256"],
            "lora_inference_weight": 1.0,
            "layout": "dual_expert_a14b_symlink_bundle",
            "sources": {
                "high_noise_model": source_record(high),
                "low_noise_model": source_record(low),
            },
        }
        (temp / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp.rename(target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair separate WAN2.2 A14B high/low checkpoints into immutable evaluation layouts"
    )
    parser.add_argument("--high-run", required=True)
    parser.add_argument("--low-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()

    high_run = Path(args.high_run).expanduser().resolve(strict=True)
    low_run = Path(args.low_run).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    steps = list(dict.fromkeys(args.steps))
    if any(step <= 0 for step in steps):
        raise ValueError("All --steps values must be positive")

    records = []
    for step in steps:
        high = checkpoint_for(high_run, step, "high")
        low = checkpoint_for(low_run, step, "low")
        validate_pair(high, low, step)
        records.append((step, high, low))

    output_root.mkdir(parents=True, exist_ok=True)
    for step, _, _ in records:
        target = output_root / f"step_{step:06d}"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing bundle: {target}")

    for step, high, low in records:
        target = create_bundle(output_root, step, high, low)
        print(f"Prepared dual-expert evaluation bundle: {target}")


if __name__ == "__main__":
    main()
