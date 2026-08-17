#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRIC_DIRECTIONS = {
    "mse": "lower",
    "psnr": "higher",
    "ssim": "higher",
    "lpips": "lower",
    "mvcs": "higher",
    "3dcs": "lower",
    "consistency_score": "lower",
    "epipolar": "lower",
    "sampson_error": "lower",
    "motion_score": "report",
}


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def collect_by_prompt(rows, metric):
    out = {}
    grouped = defaultdict(list)
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not prompt_id:
            continue
        grouped[prompt_id].append(parse_float(row.get(metric)))
    for prompt_id, values in grouped.items():
        out[prompt_id] = mean(values)
    return out


def format_value(value):
    return "" if value is None else f"{value:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--lora_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    baseline_rows = load_rows(args.baseline_csv)
    lora_rows = load_rows(args.lora_csv)
    summary_rows = []

    for metric, direction in METRIC_DIRECTIONS.items():
        baseline_values = [parse_float(row.get(metric)) for row in baseline_rows]
        lora_values = [parse_float(row.get(metric)) for row in lora_rows]
        baseline_mean = mean(baseline_values)
        lora_mean = mean(lora_values)
        diff = None if baseline_mean is None or lora_mean is None else lora_mean - baseline_mean

        baseline_by_prompt = collect_by_prompt(baseline_rows, metric)
        lora_by_prompt = collect_by_prompt(lora_rows, metric)
        paired_diffs = []
        for prompt_id, base_value in baseline_by_prompt.items():
            lora_value = lora_by_prompt.get(prompt_id)
            if base_value is not None and lora_value is not None:
                paired_diffs.append(lora_value - base_value)
        paired_diff = mean(paired_diffs)

        if direction == "higher":
            improvement = None if paired_diff is None else paired_diff > 0
        elif direction == "lower":
            improvement = None if paired_diff is None else paired_diff < 0
        else:
            improvement = None

        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline_mean": baseline_mean,
                "lora_mean": lora_mean,
                "mean_diff_lora_minus_baseline": diff,
                "paired_prompt_count": len(paired_diffs),
                "paired_mean_diff_lora_minus_baseline": paired_diff,
                "paired_improved": "" if improvement is None else str(improvement).lower(),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: format_value(value) if isinstance(value, float) else value for key, value in row.items()})

    output_md = Path(args.output_md)
    lines = [
        "# WAN2.2 Baseline vs VideoGPA LoRA Summary",
        "",
        f"- Baseline videos: {len(baseline_rows)}",
        f"- LoRA videos: {len(lora_rows)}",
        "",
        "| Metric | Direction | Baseline Mean | LoRA Mean | Paired Diff | Improved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {metric} | {direction} | {baseline} | {lora} | {paired} | {improved} |".format(
                metric=row["metric"],
                direction=row["direction"],
                baseline=format_value(row["baseline_mean"]),
                lora=format_value(row["lora_mean"]),
                paired=format_value(row["paired_mean_diff_lora_minus_baseline"]),
                improved=row["paired_improved"],
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
