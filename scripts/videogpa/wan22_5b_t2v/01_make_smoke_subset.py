from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import deterministic_sample, read_json, resolve_config, safe_id, sha256_file, write_json
from vgm_common.config import write_resolved_config
from vgm_common.paths import get_manifest_root


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_optional_master(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {row.get("scene_uid"): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic 8K train smoke subset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    project_root = Path(cfg["project"]["project_root"])
    train_manifest = Path(cfg["paths"]["train_manifest"])
    subset_size = args.subset_size or int(cfg["data"]["subset_size"])
    seed = args.seed or int(cfg["data"]["subset_seed"])
    required_bucket = str(cfg["data"].get("required_bucket", "8k")).upper()
    required_split = str(cfg["data"].get("required_split", "train"))

    manifest = read_json(train_manifest)
    if not isinstance(manifest, dict):
        raise ValueError("train_t2v manifest must be a dict")

    master_path = get_manifest_root() / "master_all.jsonl"
    master_by_uid = load_optional_master(master_path)

    candidates = []
    for scene_uid, item in manifest.items():
        bucket = scene_uid.split("/", 1)[0]
        scene_id = scene_uid.split("/", 1)[-1]
        prompt = item.get("text_prompt", "") if isinstance(item, dict) else ""
        master = master_by_uid.get(scene_uid)
        if bucket.upper() != required_bucket:
            continue
        if master_by_uid:
            if not master or master.get("split") != required_split:
                continue
            if master.get("source_subset", "").upper() != required_bucket:
                continue
        if not prompt.strip():
            continue
        candidates.append(
            {
                "group_id": safe_id(scene_uid),
                "scene_uid": scene_uid,
                "scene_id": scene_id,
                "source_split": required_split,
                "source_bucket": required_bucket.lower(),
                "text_prompt": prompt.strip(),
                "task": "t2v",
            }
        )

    if not candidates:
        raise RuntimeError(f"No {required_bucket} {required_split} T2V samples found")

    selected = deterministic_sample(candidates, min(subset_size, len(candidates)), seed)
    scene_ids = [item["scene_id"] for item in selected]
    if len(scene_ids) != len(set(scene_ids)):
        raise RuntimeError("Selected duplicate scene IDs")

    out = {
        "task": "t2v",
        "source_manifest": str(train_manifest),
        "source_manifest_sha256": sha256_file(train_manifest),
        "master_manifest": str(master_path) if master_by_uid else None,
        "master_manifest_used": bool(master_by_uid),
        "required_split": required_split,
        "required_bucket": required_bucket.lower(),
        "subset_seed": seed,
        "requested_subset_size": subset_size,
        "available_8k_train": len(candidates),
        "selected_scene_ids": scene_ids,
        "samples": selected,
        "dpo_smoke_eligible": len(selected) >= 2,
    }
    out_path = run_dir / "manifests/input_subset.json"
    write_json(out_path, out)
    write_resolved_config(run_dir, cfg)
    print(f"Wrote {out_path}")
    print(f"Selected {len(selected)} scenes: {', '.join(scene_ids)}")
    if len(selected) < 2:
        raise SystemExit("At least 2 scenes are required for a DPO dataset smoke")


if __name__ == "__main__":
    main()
