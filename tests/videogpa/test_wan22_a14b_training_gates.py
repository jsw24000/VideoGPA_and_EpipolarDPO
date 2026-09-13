from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "VideoGPA" / "train" / "Wan2.2-T2V-5B" / "03_train.py"
LAUNCHER = REPO_ROOT / "scripts" / "videogpa" / "wan22_14b_t2v" / "run_memory_gate.sh"
SUMMARIZER = REPO_ROOT / "scripts" / "videogpa" / "wan22_14b_t2v" / "summarize_memory_gate.py"


def test_trainer_exposes_memory_safe_a14b_gate_modes() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert '--expert-mode", choices=("both", "high", "low")' in source
    assert '--reference-mode", choices=("separate", "shared_base")' in source
    assert '"--memory-probe"' in source
    assert '"--timestep-mode"' in source
    assert '"--training-shift"' in source
    assert "with shared_base_reference(transformer)" in source
    assert 'mode == "high"' in source
    assert 'mode == "low"' in source


def test_memory_gate_isolated_output_and_formal_latents() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PROBE_DIR must be separate from RUN_DIR" in source
    assert 'Refusing non-empty PROBE_DIR' in source
    assert '--metadata_path "${RUN_DIR}/manifests/encoded_pairs.json"' in source
    assert '--output_dir "${PROBE_DIR}"' in source
    assert '--memory-probe' in source
    assert '--timestep-mode shifted_scheduler' in source
    assert '--training-shift "${TRAINING_SHIFT}"' in source
    assert "expandable_segments:True" in source


def test_memory_gate_summarizer_reports_headroom_and_reference_difference() -> None:
    source = SUMMARIZER.read_text(encoding="utf-8")
    assert "peak_reserved" in source
    assert "headroom" in source
    assert "first_step_policy_reference_max_abs_diff" in source
    assert "first_step_model_timestep_range" in source
