from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "VideoGPA" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from resume_utils import (  # noqa: E402
    ResumeConfigError,
    ResumeError,
    compute_resume_data_cursor,
    ensure_checkpoint_save_target_safe,
    parse_checkpoint_step,
    validate_checkpoint_manifest,
    validate_resume_config,
    validate_resume_metadata,
)


def write_file(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_checkpoint(root: Path, name: str = "step_005000") -> Path:
    checkpoint = root / name
    checkpoint.mkdir(parents=True)
    write_file(checkpoint / "adapter_model.safetensors")
    write_file(checkpoint / "adapter_config.json", "{}")
    write_file(checkpoint / "optimizer.pt")
    write_file(checkpoint / "scheduler.pt")
    write_file(checkpoint / "trainer_state.json", '{"step": 5000}')
    with (checkpoint / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"training_resolved": base_training_config()}, handle)
    return checkpoint


def base_training_config() -> dict[str, object]:
    return {
        "lora_rank": 64,
        "lora_alpha": 128.0,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q", "k", "v", "o"],
        "learning_rate": 5e-6,
        "warmup_steps": 500,
        "max_steps": 10000,
        "batch_size": 1,
        "accumulate_grad_batches": 2,
        "beta": 1.0,
        "num_train_timesteps": 1000,
        "shift": 5.0,
        "gradient_clip_val": 1.0,
    }


def test_parse_checkpoint_step() -> None:
    assert parse_checkpoint_step("step_005000") == 5000
    with pytest.raises(ValueError):
        parse_checkpoint_step("checkpoint_005000")


def test_validate_checkpoint_manifest_requires_optimizer_and_scheduler(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    validate_checkpoint_manifest(checkpoint)

    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match="optimizer"):
        validate_checkpoint_manifest(checkpoint)

    write_file(checkpoint / "optimizer.pt")
    (checkpoint / "scheduler.pt").unlink()
    with pytest.raises(FileNotFoundError, match="scheduler"):
        validate_checkpoint_manifest(checkpoint)


def test_validate_resume_metadata_rejects_step_mismatch(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    validate_resume_metadata(checkpoint, {"step": 5000}, {"last_epoch": 5000})

    with pytest.raises(ResumeError, match="trainer_state step mismatch"):
        validate_resume_metadata(checkpoint, {"step": 4999}, {"last_epoch": 5000})
    with pytest.raises(ResumeError, match="scheduler step mismatch"):
        validate_resume_metadata(checkpoint, {"step": 5000}, {"last_epoch": 4999})


def test_validate_resume_config_rejects_lora_mismatch() -> None:
    current = {"training_resolved": base_training_config()}
    checkpoint = {"training_resolved": {**base_training_config(), "lora_rank": 32}}
    with pytest.raises(ResumeConfigError, match="lora_rank"):
        validate_resume_config(current, checkpoint)


def test_validate_resume_config_reports_path_drift() -> None:
    current = {"training_resolved": base_training_config(), "paths": {"run_dir": "/new/root/run"}}
    checkpoint = {"training_resolved": base_training_config(), "paths": {"run_dir": "/old/root/run"}}
    report = validate_resume_config(current, checkpoint)
    assert report["path_differences"] == [
        {"field": "paths.run_dir", "checkpoint": "/old/root/run", "current": "/new/root/run"}
    ]


def test_compute_resume_data_cursor() -> None:
    cursor = compute_resume_data_cursor(resume_step=5000, accumulate_grad_batches=2, dataloader_len=435)
    assert cursor == {
        "resume_step": 5000,
        "consumed_microbatches": 10000,
        "dataloader_len": 435,
        "data_epoch": 22,
        "data_offset": 430,
    }


def test_checkpoint_save_target_refuses_nonempty_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_001000"
    checkpoint.mkdir()
    ensure_checkpoint_save_target_safe(checkpoint)

    write_file(checkpoint / "adapter_model.safetensors")
    with pytest.raises(FileExistsError):
        ensure_checkpoint_save_target_safe(checkpoint)
