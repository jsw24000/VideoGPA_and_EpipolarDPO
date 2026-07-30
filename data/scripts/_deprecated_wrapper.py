from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main(target: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts" / "data"
    print(f"WARNING: data/scripts/{target} is deprecated; use scripts/data/{target}.", file=sys.stderr)
    sys.path.insert(0, str(scripts_dir))
    runpy.run_path(str(scripts_dir / target), run_name="__main__")
