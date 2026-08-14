from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
I2V_CONFIG = "configs/videogpa/wan22_5b_i2v_formal.yaml"
I2V_CONFIG_ABS = REPO_ROOT / I2V_CONFIG


def run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_i2v_formal_output_roots_are_t2v_siblings() -> None:
    local = run_bash(
        "source scripts/env/activate_profile.sh local >/dev/null && "
        f"python -m vgm_common.config --config {I2V_CONFIG} --print output_root"
    )
    assert local.returncode == 0, local.stdout + local.stderr
    assert local.stdout.strip() == str(REPO_ROOT / "outputs" / "videogpa" / "wan22_5b_i2v" / "formal")

    cluster = run_bash(
        "source scripts/env/activate_profile.sh cluster_zk >/dev/null && "
        f"python -m vgm_common.config --config {shlex.quote(str(I2V_CONFIG_ABS))} --print output_root"
    )
    assert cluster.returncode == 0, cluster.stdout + cluster.stderr
    assert (
        cluster.stdout.strip()
        == "/data/pbq/system/peibaoqi/project_a/zk/outputs/VideoGPA_and_EpipolarDPO/videogpa/wan22_5b_i2v/formal"
    )


def test_i2v_subset_resolves_profile_first_frames(tmp_path: Path) -> None:
    run_dir = tmp_path / "i2v_run"
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        "python scripts/videogpa/wan22_5b_i2v/01_make_train_subset.py "
        f"--config {I2V_CONFIG} --run-dir {shlex.quote(str(run_dir))} --subset-size 2"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads((run_dir / "manifests" / "input_subset.json").read_text(encoding="utf-8"))
    assert payload["task"] == "i2v"
    assert payload["image_conditioned"] is True
    assert payload["selected_count"] == 2
    assert payload["required_buckets"] == ["8k", "9k", "10k", "11k"]
    assert payload["first_frames_root"] == str(REPO_ROOT / "data" / "first_frames")

    for sample in payload["samples"]:
        image_path = Path(sample["image_path"])
        assert sample["task"] == "i2v"
        assert sample["image_conditioned"] is True
        assert "Camera motion:" in sample["text_prompt"]
        assert sample["camera_motion"]
        assert image_path.is_file()
        assert image_path.is_relative_to(REPO_ROOT / "data" / "first_frames")
