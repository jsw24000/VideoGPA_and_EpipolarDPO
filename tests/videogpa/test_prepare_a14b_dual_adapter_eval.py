from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/videogpa/wan22_14b_t2v/prepare_dual_adapter_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_dual_adapter_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_checkpoint(
    run: Path,
    step: int,
    mode: str,
    metadata_sha: str = "abc",
    rank: int = 64,
    target_modules: list[str] | None = None,
) -> Path:
    expert = f"{mode}_noise_model"
    checkpoint = run / "checkpoints" / f"step_{step:06d}"
    adapter = checkpoint / expert
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": f"/models/{mode}_noise_model",
                "r": rank,
                "lora_alpha": 128,
                "target_modules": target_modules or ["q", "k", "v", "o"],
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(f"{mode}-{step}".encode())
    config = {
        "task": "t2v",
        "architecture": "dual_expert_a14b",
        "wan_task_key": "t2v-A14B",
        "expert_mode": mode,
        "reference_mode": "shared_base",
        "timestep_mode": "shifted_scheduler",
        "shift": 5.0,
        "loss_strategy": "dpo",
        "dpo_beta": 1.0,
        "lora_rank": 64,
        "lora_alpha": 128.0,
        "lora_dropout": 0.0,
        "lora_target_modules": ["q", "k", "v", "o"],
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": step, "config": config, "metadata_sha256": metadata_sha}),
        encoding="utf-8",
    )
    return checkpoint


def test_create_dual_bundle_uses_validated_symlinks_and_provenance(tmp_path: Path) -> None:
    module = load_module()
    high_run = tmp_path / "high"
    low_run = tmp_path / "low"
    make_checkpoint(high_run, 330, "high")
    make_checkpoint(low_run, 330, "low")

    high = module.checkpoint_for(high_run, 330, "high")
    low = module.checkpoint_for(low_run, 330, "low")
    module.validate_pair(high, low, 330)
    bundle = module.create_bundle(tmp_path / "bundles", 330, high, low)

    assert (bundle / "high_noise_model").is_symlink()
    assert (bundle / "low_noise_model").is_symlink()
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["step"] == 330
    assert provenance["metadata_sha256"] == "abc"
    assert provenance["lora_inference_weight"] == 1.0
    assert provenance["sources"]["high_noise_model"]["adapter_weights"]["sha256"]


def test_validate_pair_rejects_different_encoded_pair_inputs(tmp_path: Path) -> None:
    module = load_module()
    high_run = tmp_path / "high"
    low_run = tmp_path / "low"
    make_checkpoint(high_run, 330, "high", metadata_sha="high-sha")
    make_checkpoint(low_run, 330, "low", metadata_sha="low-sha")

    high = module.checkpoint_for(high_run, 330, "high")
    low = module.checkpoint_for(low_run, 330, "low")
    with pytest.raises(ValueError, match="metadata SHA256 mismatch"):
        module.validate_pair(high, low, 330)


def test_create_bundle_refuses_existing_target(tmp_path: Path) -> None:
    module = load_module()
    high_run = tmp_path / "high"
    low_run = tmp_path / "low"
    make_checkpoint(high_run, 110, "high")
    make_checkpoint(low_run, 110, "low")
    high = module.checkpoint_for(high_run, 110, "high")
    low = module.checkpoint_for(low_run, 110, "low")
    output_root = tmp_path / "bundles"
    module.create_bundle(output_root, 110, high, low)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        module.create_bundle(output_root, 110, high, low)


def test_validate_pair_rejects_incompatible_adapter_rank(tmp_path: Path) -> None:
    module = load_module()
    high_run = tmp_path / "high"
    low_run = tmp_path / "low"
    make_checkpoint(high_run, 220, "high", rank=64)
    make_checkpoint(low_run, 220, "low", rank=32)
    high = module.checkpoint_for(high_run, 220, "high")
    low = module.checkpoint_for(low_run, 220, "low")

    with pytest.raises(ValueError, match="adapter configuration mismatch"):
        module.validate_pair(high, low, 220)


def test_validate_pair_accepts_different_target_module_order(tmp_path: Path) -> None:
    module = load_module()
    high_run = tmp_path / "high"
    low_run = tmp_path / "low"
    make_checkpoint(high_run, 110, "high", target_modules=["q", "k", "v", "o"])
    make_checkpoint(low_run, 110, "low", target_modules=["o", "v", "k", "q"])
    high = module.checkpoint_for(high_run, 110, "high")
    low = module.checkpoint_for(low_run, 110, "low")

    module.validate_pair(high, low, 110)


def test_validate_pair_rejects_different_target_module_members(tmp_path: Path) -> None:
    module = load_module()
    high_run = tmp_path / "high"
    low_run = tmp_path / "low"
    make_checkpoint(high_run, 110, "high", target_modules=["q", "k", "v", "o"])
    make_checkpoint(low_run, 110, "low", target_modules=["q", "k", "v"])
    high = module.checkpoint_for(high_run, 110, "high")
    low = module.checkpoint_for(low_run, 110, "low")

    with pytest.raises(ValueError, match="target_modules"):
        module.validate_pair(high, low, 110)
