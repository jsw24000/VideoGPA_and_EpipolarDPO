from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_probe_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.rstrip("\n").partition("=")
            if separator:
                values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a WAN2.2 A14B memory gate")
    parser.add_argument("probe_dir")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir).expanduser().resolve()
    traces = sorted((probe_dir / "reports").glob("memory_trace.rank_*.jsonl"))
    if not traces:
        raise SystemExit(f"No memory traces found under {probe_dir / 'reports'}")

    summary_path = probe_dir / "reports" / "training_summary.json"
    training = read_json(summary_path) if summary_path.is_file() else {}
    probe_config = read_probe_config(probe_dir / "probe_config.txt")
    print(f"probe_dir: {probe_dir}")
    print(f"status: {training.get('status', 'INCOMPLETE_OR_FAILED')}")
    print(f"expert_mode: {training.get('expert_mode', probe_config.get('EXPERT_MODE', 'unknown'))}")
    print(f"reference_mode: {training.get('reference_mode', probe_config.get('REFERENCE_MODE', 'unknown'))}")
    print(f"timestep_mode: {training.get('timestep_mode', probe_config.get('TIMESTEP_MODE', 'unknown'))}")
    print(f"training_shift: {training.get('training_shift', probe_config.get('TRAINING_SHIFT', 'unknown'))}")
    print(
        "distributed_strategy: "
        f"{training.get('distributed_strategy', probe_config.get('DISTRIBUTED_STRATEGY', 'unknown'))}"
    )
    print(f"backward_mode: {training.get('backward_mode', probe_config.get('BACKWARD_MODE', 'unknown'))}")
    print(f"pair_score_mode: {training.get('pair_score_mode', probe_config.get('PAIR_SCORE_MODE', 'unknown'))}")
    print("rank memory:")

    minimum_headroom = float("inf")
    maximum_reserved = 0.0
    for trace in traces:
        rows = read_jsonl(trace)
        if not rows:
            continue
        rank = rows[0].get("rank")
        peak_allocated = max(float(row.get("max_allocated_gb", 0.0)) for row in rows)
        peak_reserved = max(float(row.get("max_reserved_gb", 0.0)) for row in rows)
        total = max(float(row.get("total_memory_gb", 0.0)) for row in rows)
        headroom = total - peak_reserved if total else float("nan")
        minimum_headroom = min(minimum_headroom, headroom)
        maximum_reserved = max(maximum_reserved, peak_reserved)
        print(
            f"  rank={rank} peak_allocated={peak_allocated:.2f}GB "
            f"peak_reserved={peak_reserved:.2f}GB total={total:.2f}GB headroom={headroom:.2f}GB "
            f"last_label={rows[-1].get('label', 'unknown')}"
        )
        completed = [row for row in rows if str(row.get("label", "")).endswith("_complete") and str(row.get("label", "")).startswith("step_")]
        if len(completed) >= 2:
            growth = float(completed[-1].get("allocated_gb", 0.0)) - float(completed[0].get("allocated_gb", 0.0))
            print(f"    completed_step_allocated_growth={growth:+.3f}GB over {len(completed)} steps")

    metrics = training.get("metrics", [])
    if metrics:
        first = metrics[0]
        last = metrics[-1]
        debug = first.get("debug_shapes", {})
        print(f"first_step_policy_reference_max_abs_diff: {first.get('policy_reference_max_abs_diff')}")
        print(
            "first_step_model_timestep_range: "
            f"[{debug.get('model_timestep_min')}, {debug.get('model_timestep_max')}]"
        )
        print(f"first_step_grad_norm: {first.get('grad_norm')}")
        print(f"first_step_time_sec: {first.get('step_time_sec')}")
        print(f"first_step_winner_recompute_max_abs_diff: {debug.get('winner_recompute_max_abs_diff')}")
        print(f"first_step_loser_recompute_max_abs_diff: {debug.get('loser_recompute_max_abs_diff')}")
        step_times = [float(row["step_time_sec"]) for row in metrics if row.get("step_time_sec") is not None]
        print(f"recorded_steps: {len(metrics)}")
        if step_times:
            print(
                "step_time_sec: "
                f"median={statistics.median(step_times):.3f} min={min(step_times):.3f} max={max(step_times):.3f}"
            )
        print(f"loss_first_last: {first.get('total_loss')} -> {last.get('total_loss')}")
        print(f"grad_norm_first_last: {first.get('grad_norm')} -> {last.get('grad_norm')}")
        recompute_diffs = [
            float(value)
            for row in metrics
            for value in (
                row.get("debug_shapes", {}).get("winner_recompute_max_abs_diff"),
                row.get("debug_shapes", {}).get("loser_recompute_max_abs_diff"),
            )
            if value is not None
        ]
        if recompute_diffs:
            print(f"maximum_recompute_max_abs_diff: {max(recompute_diffs)}")

    if not training:
        verdict = "FAILED_OR_INCOMPLETE: inspect logs/gate.log and the last memory trace label"
    elif minimum_headroom < 5.0:
        verdict = "UNSAFE_MARGIN: use single-expert FSDP or reduce activation memory"
    elif minimum_headroom < 10.0:
        verdict = "BORDERLINE: run 10 steps before deciding; FSDP may be needed"
    else:
        verdict = "MEMORY_PASS: run the 10-step and 8-GPU gates"
    print(f"verdict: {verdict}")
    print(f"maximum_reserved_gb: {maximum_reserved:.2f}")


if __name__ == "__main__":
    main()
