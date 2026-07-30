from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from common import read_json, resolve_config, write_json


def load_optional(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return read_json(path)


def file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def video_paths_from_groups(payload: dict) -> list[str]:
    paths = []
    for group in payload.get("groups", []):
        for video in group.get("videos", []):
            if video.get("video_path"):
                paths.append(video["video_path"])
    return paths


def group_signatures(payload: dict) -> list[dict[str, Any]]:
    signatures = []
    for group in payload.get("groups", []):
        signatures.append(
            {
                "group_id": group.get("group_id"),
                "prompt": group.get("text_prompt", group.get("prompt")),
                "seeds": [video.get("seed") for video in group.get("videos", [])],
            }
        )
    return signatures


def write_comparison_manifest(run_dir: Path) -> None:
    base = load_optional(run_dir / "comparisons/base_manifest.json", {})
    lora = load_optional(run_dir / "comparisons/lora_manifest.json", {})
    base_paths = video_paths_from_groups(base)
    lora_paths = video_paths_from_groups(lora)
    base_sig = group_signatures(base)
    lora_sig = group_signatures(lora)
    manifest = {
        "task": "t2v",
        "base_manifest": "comparisons/base_manifest.json",
        "lora_manifest": "comparisons/lora_manifest.json",
        "base_videos": base_paths,
        "lora_videos": lora_paths,
        "base_signature": base_sig,
        "lora_signature": lora_sig,
        "same_prompt_seed": bool(base_sig and base_sig == lora_sig),
        "adapter_loaded": bool(lora.get("lora_path"))
        and all(video.get("lora_loaded") for group in lora.get("groups", []) for video in group.get("videos", [])),
    }
    write_json(run_dir / "comparisons/comparison_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write WAN2.2 T2V smoke summary")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-comparison-only", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    write_comparison_manifest(run_dir)
    if args.write_comparison_only:
        return

    subset = load_optional(run_dir / "manifests/input_subset.json", {})
    candidates = load_optional(run_dir / "manifests/candidate_groups.json", {})
    scored = load_optional(run_dir / "manifests/scored_candidates.json", {})
    pairs = load_optional(run_dir / "manifests/preference_pairs.json", {})
    encoded = load_optional(run_dir / "manifests/encoded_pairs.json", {})
    training = load_optional(run_dir / "reports/training_summary.json", {})
    comparison = load_optional(run_dir / "comparisons/comparison_manifest.json", {})

    candidate_videos = video_paths_from_groups(candidates)
    pair_rows = pairs.get("pairs", [])
    filtered_rows = pairs.get("filtered", [])
    encoded_pairs = encoded.get("pairs", [])
    training_metrics = training.get("metrics", [])
    checkpoint_path = training.get("checkpoint_path")

    status = "PASS"
    last_stage = "comparison"
    if not comparison.get("adapter_loaded"):
        status = "PARTIAL"
        last_stage = "training"
    if not training.get("checkpoint_reloaded"):
        status = "PARTIAL"
        last_stage = "encoding"
    if len(encoded_pairs) < 2:
        status = "FAIL"
        last_stage = "scoring"
    if len(pair_rows) < 2:
        status = "FAIL"
        last_stage = "generation"

    lines = [
        "# WAN2.2 5B T2V VideoGPA Smoke Summary",
        "",
        "## Summary",
        "",
        f"- Status: {status}",
        f"- Last successful stage: {last_stage}",
        f"- LoRA checkpoint saved: {bool(checkpoint_path)}",
        f"- LoRA checkpoint reloaded: {bool(training.get('checkpoint_reloaded'))}",
        "",
        "## Data",
        "",
        f"- Scene IDs: {subset.get('selected_scene_ids', [])}",
        f"- All selected from 8K train: {subset.get('required_bucket') == '8k' and subset.get('required_split') == 'train'}",
        f"- Prompt count: {len(subset.get('samples', []))}",
        f"- Candidate count: {len(candidate_videos)}",
        f"- Preference pair count: {len(pair_rows)}",
        f"- Debug fallback used: {pairs.get('debug_fallback_used', False)}",
        f"- Filtered group count: {len(filtered_rows)}",
        "",
        "## Model",
        "",
        f"- WAN weights: `{cfg['paths']['wan_model_path']}`",
        f"- VGGT weights: `{cfg['paths']['vggt_model_path']}`",
        f"- WAN config summary: `ti2v-5B`, VAE stride `(4, 16, 16)`, patch size `(1, 2, 2)`",
        f"- LoRA target modules/rank/alpha: `{cfg['training']['lora_target_modules']}`, `{cfg['training']['lora_rank']}`, `{cfg['training']['lora_alpha']}`",
        f"- Trainable parameters: {training.get('trainable_stats', {}).get('trainable_params')}",
        "",
        "## Generation",
        "",
        f"- frame_num: {cfg['generation']['frame_num']}",
        f"- size: {cfg['generation']['size']}",
        f"- sampling_steps: {cfg['generation']['sampling_steps']}",
        f"- seeds: {cfg['data']['candidate_seeds']}",
        f"- videos: {candidate_videos}",
        "",
        "## Scoring",
        "",
    ]
    for group in scored.get("groups", []):
        lines.append(f"- {group.get('group_id')}:")
        for video in group.get("videos", []):
            lines.append(
                f"  - seed={video.get('seed')} score={video.get('consistency_score')} motion={video.get('motion_norm')} path={video.get('video_path')}"
            )
    if pair_rows:
        lines.append("")
        lines.append("Preference pairs:")
        for pair in pair_rows:
            lines.append(
                f"- {pair.get('pair_id')}: winner_seed={pair.get('winner', {}).get('seed')} "
                f"loser_seed={pair.get('loser', {}).get('seed')} gap={pair.get('score_gap')} "
                f"fallback={pair.get('debug_fallback')}"
            )
    if filtered_rows:
        lines.append("")
        lines.append("Filtered groups:")
        for row in filtered_rows:
            lines.append(f"- {row.get('group_id')}: {row.get('reason')}")
    lines.extend(["", "## Encoding", ""])
    if encoded_pairs:
        first = encoded_pairs[0]
        lines.extend(
            [
                f"- text_embedding_shape: {first.get('text_embedding_shape')}",
                f"- video_latent_shape: {first.get('video_latent_shape')}",
                "- contains image_latent: false",
            ]
        )
    lines.extend(["", "## Training", ""])
    lines.extend(
        [
            f"- steps: {training.get('steps')}",
            f"- checkpoint: `{checkpoint_path}`",
            f"- LoRA updated L1 delta: {training.get('lora_delta_l1')}",
            f"- grad nonzero: {training.get('grad_nonzero')}",
            f"- base parameters changed: {training.get('base_parameters_changed')}",
            f"- reference parameters changed: {training.get('reference_parameters_changed')}",
        ]
    )
    for row in training_metrics:
        lines.append(
            f"- step {row.get('step')}: loss={row.get('total_loss')} margin={row.get('implicit_reward_margin')} grad_norm={row.get('grad_norm')} mem_reserved_gb={row.get('gpu_reserved_gb')}"
        )
    lines.extend(
        [
            "",
            "## Reload Generation",
            "",
            f"- base videos: {comparison.get('base_videos', [])}",
            f"- lora videos: {comparison.get('lora_videos', [])}",
            f"- same prompt/seed: {comparison.get('same_prompt_seed')}",
            f"- adapter loaded: {comparison.get('adapter_loaded')}",
            f"- base signature: {comparison.get('base_signature', [])}",
            f"- lora signature: {comparison.get('lora_signature', [])}",
            "",
            "## Official Deviations",
            "",
            "- T2V required: no first-frame image, no image latent, no clean first temporal latent, no TI2V timestep mask.",
            "- Smoke shrink: 4 train 8K prompts, 3 seeds each, 5 optimizer steps.",
            "- Orchestration: direct PyTorch smoke loop instead of official Lightning/W&B because those optional packages are absent locally.",
            "",
            "## Next Step",
            "",
            "Formal training command template, not executed:",
            "",
            "```bash",
            "# Create a formal YAML beside the smoke config with:",
            "# experiment.output_subdir: videogpa/wan22_5b_t2v/formal",
            "# data.manifest_relpath: manifests/videogpa_protocol/train_t2v.json",
            "# data.required_split: train",
            "# data.required_bucket: 8k",
            "# scoring.smoke_fallback_if_no_pairs: false",
            "# training.max_train_steps restored to the intended formal value",
            "source scripts/env/activate_profile.sh local",
            "VIDEOGPA_CONDA_ENV=wan22_videogpa \\",
            "bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh \\",
            "  --config configs/videogpa/wan22_5b_t2v_formal.yaml",
            "```",
            "",
            "Before formal use, remove smoke limits in a new formal config, keep the full `train_t2v.json` train split, and continue to forbid `test_t2v.json`.",
            "",
            f"Checkpoint adapter SHA256: {file_sha(Path(checkpoint_path) / 'adapter_model.safetensors') if checkpoint_path else None}",
        ]
    )
    out = run_dir / "reports/smoke_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
