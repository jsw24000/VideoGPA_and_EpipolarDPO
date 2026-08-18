from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_14b_t2v_formal_uses_480p_81_frame_generation() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs/videogpa/wan22_14b_t2v_formal.yaml").read_text(encoding="utf-8"))

    assert cfg["model"]["wan_task_key"] == "t2v-A14B"
    assert cfg["generation"]["size"] == "832*480"
    assert cfg["generation"]["frame_num"] == 81
    assert cfg["generation"]["offload_model"] is False
    assert cfg["generation"]["cache_text_embeddings"] is True


def test_14b_t2v_multigpu_defaults_enable_fsdp_and_sequence_parallel() -> None:
    launcher = (REPO_ROOT / "scripts/videogpa/wan22_14b_t2v/02_generate_candidates.sh").read_text(encoding="utf-8")

    assert "${DIT_FSDP:-1}" in launcher
    assert "${T5_FSDP:-1}" in launcher
    assert "${USE_SP:-1}" in launcher
    assert "--dit_fsdp" in launcher
    assert "--t5_fsdp" in launcher
    assert "--ulysses_size" in launcher
    assert "ULYSSES_SIZE:-${#GPU_LIST[@]}" in launcher


def test_14b_t2v_throughput_mode_runs_one_shard_per_gpu() -> None:
    launcher = (REPO_ROOT / "scripts/videogpa/wan22_14b_t2v/02_generate_candidates.sh").read_text(encoding="utf-8")

    assert "A14B_PARALLEL_MODE" in launcher
    assert "candidate_groups.shard_${shard_index}.json" in launcher
    assert "generation.shard_${shard_index}.log" in launcher
    assert "--shard_index" in launcher
    assert "--num_shards" in launcher
    assert "merge_shards.py\" groups" in launcher
    assert "PYTHONUNBUFFERED=1" in launcher
    assert "--no-capture-output" in launcher


def test_14b_t2v_text_cache_and_timing_are_exposed() -> None:
    entrypoint = (REPO_ROOT / "VideoGPA/generate/Wan2.2-A14B.py").read_text(encoding="utf-8")
    native_t2v = (REPO_ROOT / "third_party/Wan2.2/wan/text2video.py").read_text(encoding="utf-8")

    assert "--cache_text_embeddings" in entrypoint
    assert "attention_backend=" in entrypoint
    assert "[timing]" in entrypoint
    assert "def encode_prompt(" in native_t2v
    assert "[WanT2V transfer]" in native_t2v
    assert "[WanT2V timing]" in native_t2v


def test_a14b_eval_multigpu_defaults_enable_sequence_parallel() -> None:
    launcher = (REPO_ROOT / "scripts/videogpa/wan22_5b_eval/run_eval.sh").read_text(encoding="utf-8")

    assert "${T5_FSDP:-1}" in launcher
    assert "${USE_SP:-1}" in launcher
    assert "--ulysses_size" in launcher
