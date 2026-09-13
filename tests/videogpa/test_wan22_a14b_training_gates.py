from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "VideoGPA" / "train" / "Wan2.2-T2V-5B" / "03_train.py"
LAUNCHER = REPO_ROOT / "scripts" / "videogpa" / "wan22_14b_t2v" / "run_memory_gate.sh"
SUMMARIZER = REPO_ROOT / "scripts" / "videogpa" / "wan22_14b_t2v" / "summarize_memory_gate.py"
EXPERT_LAUNCHER = REPO_ROOT / "scripts" / "videogpa" / "wan22_14b_t2v" / "run_expert_training.sh"


def test_trainer_exposes_memory_safe_a14b_gate_modes() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    assert '--expert-mode", choices=("both", "high", "low")' in source
    assert '--reference-mode", choices=("separate", "shared_base")' in source
    assert '"--memory-probe"' in source
    assert '"--timestep-mode"' in source
    assert '"--training-shift"' in source
    assert '"--distributed-strategy"' in source
    assert '"--backward-mode"' in source
    assert '"--pair-score-mode"' in source
    assert "wrap_fsdp_full_shard" in source
    assert "ShardingStrategy.FULL_SHARD" in source
    assert "transformer_layer_cls={WanAttentionBlock}" in source
    assert "use_orig_params=True" in source
    assert "model.clip_grad_norm_" in source
    assert "ignored_states=replicated_trainable" in source
    assert "param.data = param.data.to(device=device)" in source
    assert "sync_replicated_trainable_gradients(transformer, dist_state)" in source
    assert "dist.all_reduce(param.grad" in source
    assert "fsdp_lora_state_dict" in source
    assert 'rng_state.rank_{rank}.pt' in source
    assert 'memory_callback("policy_winner_backward_complete")' in source
    assert 'memory_callback("policy_loser_backward_complete")' in source
    assert "torch.autograd.grad(loss_out.loss" in source
    assert "if not debug[\"backward_performed\"]" in source
    assert "winner_recompute_max_abs_diff" in source
    assert "loser_recompute_max_abs_diff" in source
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
    assert "DISTRIBUTED_STRATEGY" in source
    assert "--skip-checkpoint" in source
    assert "BACKWARD_MODE" in source
    assert "PAIR_SCORE_MODE" in source


def test_memory_gate_summarizer_reports_headroom_and_reference_difference() -> None:
    source = SUMMARIZER.read_text(encoding="utf-8")
    assert "peak_reserved" in source
    assert "headroom" in source
    assert "first_step_policy_reference_max_abs_diff" in source
    assert "first_step_model_timestep_range" in source
    assert "read_probe_config" in source
    assert "last_label=" in source
    assert "completed_step_allocated_growth" in source
    assert "statistics.median" in source


def test_expert_launcher_isolated_fsdp_training_contract() -> None:
    source = EXPERT_LAUNCHER.read_text(encoding="utf-8")
    assert "OUTPUT_DIR must be separate from SOURCE_RUN_DIR" in source
    assert "Refusing non-empty OUTPUT_DIR without RESUME=1" in source
    assert "--distributed-strategy fsdp_full_shard" in source
    assert "--backward-mode sequential_recompute" in source
    assert "--reference-mode shared_base" in source
    assert "--timestep-mode shifted_scheduler" in source
    assert 'ARGS+=(--resume)' in source
