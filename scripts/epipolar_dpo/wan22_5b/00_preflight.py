from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import resolve_config, source_manifest_path, source_run_path, write_json, write_run_config, write_text


def run_text(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def run_required(cmd: list[str], cwd: Path, label: str) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        output = exc.output.strip()
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}\n{output}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight for WAN2.2 5B Epipolar-DPO sibling pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    project_root = Path(cfg["project"]["project_root"])
    source_run = source_run_path(cfg)
    source_manifest = source_manifest_path(cfg)
    output_root = Path(cfg["paths"]["output_root"]).resolve()

    if source_run == run_dir or source_run in run_dir.parents:
        raise RuntimeError(f"Epipolar run_dir must not be inside the read-only source run: {run_dir}")
    if run_dir == source_run or run_dir in source_run.parents:
        raise RuntimeError(f"Source run must not be inside the Epipolar run_dir: {source_run}")

    missing_repo_paths = [
        rel
        for rel in [
            "VideoGPA/train/Wan2.2-T2V-5B/02_encode.py",
            "VideoGPA/train/Wan2.2-T2V-5B/03_train.py",
            "VideoGPA/train/dataset.py",
            "VideoGPA/train/loss.py",
            "Epipolar-DPO/metrics/video_evaluation/epipolar.py",
            "Epipolar-DPO/metrics/video_evaluation/dynamics.py",
            "Epipolar-DPO/model_training/reward_lora/loss.py",
            "scripts/epipolar_dpo/wan22_5b/02_score_epipolar.py",
        ]
        if not (project_root / rel).is_file()
    ]
    if missing_repo_paths:
        raise FileNotFoundError("Missing required repo files: " + ", ".join(missing_repo_paths))
    if not source_run.is_dir():
        raise FileNotFoundError(f"Configured source run does not exist: {source_run}")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Configured source candidate manifest does not exist: {source_manifest}")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe is required for the upstream candidate provenance check")

    scoring_runtime_import_check = run_required(
        [
            sys.executable,
            str(project_root / "scripts/epipolar_dpo/wan22_5b/02_score_epipolar.py"),
            "--config",
            str(Path(args.config).expanduser().resolve()),
            "--run-dir",
            str(run_dir),
            "--check-runtime-imports",
        ],
        project_root,
        "Epipolar scoring runtime import check",
    )

    report = {
        "status": "PASS",
        "method": "epipolar_dpo",
        "task": cfg["project"].get("task"),
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "source_run": str(source_run),
        "source_run_relpath": cfg.get("source", {}).get("run_relpath"),
        "source_candidate_manifest": str(source_manifest),
        "source_candidate_manifest_relpath": cfg.get("source", {}).get("candidate_manifest_relpath"),
        "source_writes_allowed": False,
        "generation_stage": "absent",
        "candidate_reuse": "source VideoGPA MP4s are referenced in-place",
        "latent_provenance": cfg.get("encoding", {}).get("latent_provenance", "posthoc_mp4_vae"),
        "condition_schema": cfg.get("encoding", {}).get("condition_schema"),
        "metric_name": cfg.get("scoring", {}).get("metric_name"),
        "metric_mode": cfg.get("scoring", {}).get("metric_mode"),
        "motion_metric_name": cfg.get("motion_filter", {}).get("metric_name", "motion_dynamics"),
        "scoring_runtime_import_check": scoring_runtime_import_check,
        "git_commit": run_text(["git", "rev-parse", "HEAD"], project_root),
        "git_status_short": run_text(["git", "status", "--short"], project_root),
        "python": sys.executable,
    }
    write_json(run_dir / "preflight" / "epipolar_preflight.json", report)
    write_text(
        run_dir / "reports" / "preflight.md",
        "\n".join(
            [
                f"# Epipolar-DPO Preflight ({str(report['task']).upper()})",
                "",
                "Status: PASS",
                "",
                f"- run_dir: `{run_dir}`",
                f"- source_run: `{source_run}`",
                f"- source_candidate_manifest: `{source_manifest}`",
                "- generation_stage: `absent`",
                "- candidate_reuse: `source VideoGPA MP4s are referenced in-place`",
                f"- metric_name: `{report['metric_name']}`",
                f"- metric_mode: `{report['metric_mode']}`",
                f"- latent_provenance: `{report['latent_provenance']}`",
                f"- condition_schema: `{report['condition_schema']}`",
                f"- scoring_runtime_import_check: `{report['scoring_runtime_import_check']}`",
                "",
            ]
        ),
    )
    write_run_config(run_dir, cfg)
    print(f"Preflight PASS: {run_dir}")


if __name__ == "__main__":
    main()
