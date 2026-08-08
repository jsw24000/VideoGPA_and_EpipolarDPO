from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-lc", command], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_local_profile_check_paths_from_repo() -> None:
    proc = run_bash("source scripts/env/activate_profile.sh local >/dev/null && python scripts/env/check_paths.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"VGM_REPO_ROOT={REPO_ROOT}" in proc.stdout
    assert f"VGM_DL3DV_ROOT={REPO_ROOT / 'data'}" in proc.stdout


def test_local_profile_check_paths_from_other_cwd() -> None:
    command = f"cd /tmp && source {REPO_ROOT}/scripts/env/activate_profile.sh local >/dev/null && python {REPO_ROOT}/scripts/env/check_paths.py"
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"VGM_OUTPUT_ROOT={REPO_ROOT / 'outputs'}" in proc.stdout


def test_cluster_profile_non_strict_resolves_without_existing_remote_dirs() -> None:
    proc = run_bash("source scripts/env/activate_profile.sh cluster_zk >/dev/null && python scripts/env/check_paths.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VGM_PROFILE=cluster_zk" in proc.stdout
    assert "VGM_REPO_ROOT=/data/pbq/system/peibaoqi/project_a/zk/repos/VideoGPA_and_EpipolarDPO" in proc.stdout


def test_smoke_yaml_resolves_to_local_roots() -> None:
    command = (
        "source scripts/env/activate_profile.sh local >/dev/null && "
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "from vgm_common.config import resolve_experiment_config\n"
        "cfg = resolve_experiment_config('configs/videogpa/wan22_5b_t2v_smoke.yaml')\n"
        "print(cfg['paths']['wan_model_path'])\n"
        "print(cfg['paths']['train_manifest'])\n"
        "print(cfg['paths']['first_frames_root'])\n"
        "print(cfg['paths']['output_root'])\n"
        "PY"
    )
    proc = run_bash(command)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == str(REPO_ROOT / "models" / "wan" / "Wan2.2-TI2V-5B")
    assert lines[1] == str(REPO_ROOT / "data" / "manifests" / "videogpa_protocol" / "train_t2v.json")
    assert lines[2] == str(REPO_ROOT / "data" / "first_frames")
    assert lines[3] == str(REPO_ROOT / "outputs" / "videogpa" / "wan22_5b_t2v" / "smoke")
