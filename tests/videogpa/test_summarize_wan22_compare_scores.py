from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/summarize_wan22_compare_scores.py"


def write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_id", "consistency_score"])
        writer.writeheader()
        writer.writerows(rows)


def test_summary_reports_paired_ci_and_win_rate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    lora = tmp_path / "lora.csv"
    output_csv = tmp_path / "summary.csv"
    output_md = tmp_path / "summary.md"
    write_scores(
        baseline,
        [
            {"prompt_id": "a", "consistency_score": 4.0},
            {"prompt_id": "b", "consistency_score": 4.0},
            {"prompt_id": "c", "consistency_score": 4.0},
        ],
    )
    write_scores(
        lora,
        [
            {"prompt_id": "a", "consistency_score": 3.0},
            {"prompt_id": "b", "consistency_score": 4.0},
            {"prompt_id": "c", "consistency_score": 5.0},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline_csv",
            str(baseline),
            "--lora_csv",
            str(lora),
            "--output_csv",
            str(output_csv),
            "--output_md",
            str(output_md),
            "--bootstrap_samples",
            "200",
        ],
        check=True,
    )
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    result = next(row for row in rows if row["metric"] == "consistency_score")
    assert result["paired_prompt_count"] == "3"
    assert abs(float(result["paired_win_rate"]) - 1 / 3) < 1e-6
    assert abs(float(result["paired_tie_rate"]) - 1 / 3) < 1e-6
    assert result["paired_mean_diff_ci95_low"]
    assert result["paired_mean_diff_ci95_high"]
    assert "95% CI" in output_md.read_text(encoding="utf-8")
