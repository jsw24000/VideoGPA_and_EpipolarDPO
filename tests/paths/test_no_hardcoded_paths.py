from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("/home/", "/data/pbq/", "Desktop/3DVGM")
ALLOW_CLUSTER_ROOT = Path("configs/paths/cluster_zk.sh")


def tracked_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--cached"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    files = []
    for line in proc.stdout.splitlines():
        path = Path(line)
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.parts and path.parts[0] in {"configs", "scripts", "vgm_common"}:
            files.append(path)
        if path.match("VideoGPA/generate/Wan2.2-*.py") or path.match("VideoGPA/train/Wan2.2-T2V-5B/*.py"):
            files.append(path)
    return files


def test_no_host_absolute_paths_in_business_files() -> None:
    offenders: list[str] = []
    for relpath in tracked_files():
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN:
            if pattern not in text:
                continue
            if relpath == ALLOW_CLUSTER_ROOT and pattern == "/data/pbq/":
                continue
            offenders.append(f"{relpath}: {pattern}")
    assert not offenders, "\n".join(offenders)
