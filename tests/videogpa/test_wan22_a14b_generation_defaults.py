from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_14b_t2v_formal_uses_480p_81_frame_generation() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs/videogpa/wan22_14b_t2v_formal.yaml").read_text(encoding="utf-8"))

    assert cfg["model"]["wan_task_key"] == "t2v-A14B"
    assert cfg["generation"]["size"] == "832*480"
    assert cfg["generation"]["frame_num"] == 81


def test_14b_t2v_multigpu_defaults_enable_fsdp_and_sequence_parallel() -> None:
    launcher = (REPO_ROOT / "scripts/videogpa/wan22_14b_t2v/02_generate_candidates.sh").read_text(encoding="utf-8")

    assert "${DIT_FSDP:-1}" in launcher
    assert "${T5_FSDP:-1}" in launcher
    assert "${USE_SP:-1}" in launcher
    assert "--dit_fsdp" in launcher
    assert "--t5_fsdp" in launcher
    assert "--use_sp" in launcher


def test_a14b_eval_multigpu_defaults_enable_sequence_parallel() -> None:
    launcher = (REPO_ROOT / "scripts/videogpa/wan22_5b_eval/run_eval.sh").read_text(encoding="utf-8")

    assert "${T5_FSDP:-1}" in launcher
    assert "${USE_SP:-1}" in launcher
