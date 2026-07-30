from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import deterministic_sample, read_json, resolve_config, safe_id, sha256_file, write_json, write_yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_buckets(data_cfg: dict[str, Any]) -> list[str]:
    buckets = data_cfg.get("required_buckets")
    if buckets is None:
        buckets = [data_cfg.get("required_bucket", "8k")]
    if isinstance(buckets, str):
        buckets = [item.strip() for item in buckets.split(",") if item.strip()]
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("data.required_buckets must be a non-empty list or comma-separated string")
    return [str(bucket).upper() for bucket in buckets]


def parse_subset_size(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "all":
        return None
    size = int(value)
    if size <= 0:
        raise ValueError("subset_size must be positive or 'all'")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description="Create full/formal train T2V subset manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subset-size", default=None, help="Positive integer or 'all'")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    project_root = Path(cfg["project"]["project_root"])
    train_manifest = Path(cfg["paths"]["train_manifest"])
    data_cfg = cfg.get("data", {})
    required_split = str(data_cfg.get("required_split", "train"))
    required_buckets = parse_buckets(data_cfg)
    subset_size = parse_subset_size(args.subset_size if args.subset_size is not None else data_cfg.get("subset_size", "all"))
    seed = args.seed if args.seed is not None else int(data_cfg.get("subset_seed", 2026))

    if "test_t2v" in str(train_manifest) or "test_i2v" in str(train_manifest):
        raise ValueError(f"Formal T2V training must not read test manifests: {train_manifest}")
    manifest = read_json(train_manifest)
    if not isinstance(manifest, dict):
        raise ValueError("train_t2v manifest must be a dict")

    master_rows = load_jsonl(project_root / "data/manifests/master_all.jsonl")
    master_by_uid = {row.get("scene_uid"): row for row in master_rows}

    candidates = []
    skipped = Counter()
    for scene_uid, item in manifest.items():
        bucket = scene_uid.split("/", 1)[0]
        scene_id = scene_uid.split("/", 1)[-1]
        prompt = item.get("text_prompt", "") if isinstance(item, dict) else ""
        master = master_by_uid.get(scene_uid)
        if bucket.upper() not in required_buckets:
            skipped["bucket"] += 1
            continue
        if not master or master.get("split") != required_split:
            skipped["split"] += 1
            continue
        if master.get("source_subset", "").upper() != bucket.upper():
            skipped["source_subset"] += 1
            continue
        if not prompt.strip():
            skipped["empty_prompt"] += 1
            continue
        candidates.append(
            {
                "group_id": safe_id(scene_uid),
                "scene_uid": scene_uid,
                "scene_id": scene_id,
                "source_split": required_split,
                "source_bucket": bucket.lower(),
                "text_prompt": prompt.strip(),
                "task": "t2v",
            }
        )

    if not candidates:
        raise RuntimeError(f"No {required_buckets} {required_split} T2V samples found")

    selected = deterministic_sample(candidates, subset_size, seed) if subset_size is not None else list(candidates)
    scene_uids = [item["scene_uid"] for item in selected]
    scene_ids = [item["scene_id"] for item in selected]
    if len(scene_uids) != len(set(scene_uids)):
        raise RuntimeError("Selected duplicate scene UIDs")
    counts_by_bucket = dict(Counter(item["source_bucket"] for item in selected))

    out = {
        "task": "t2v",
        "run_type": cfg.get("project", {}).get("run_type", "formal"),
        "source_manifest": str(train_manifest),
        "source_manifest_sha256": sha256_file(train_manifest),
        "required_split": required_split,
        "required_buckets": [bucket.lower() for bucket in required_buckets],
        "subset_seed": seed,
        "requested_subset_size": "all" if subset_size is None else subset_size,
        "available_train": len(candidates),
        "selected_count": len(selected),
        "counts_by_bucket": counts_by_bucket,
        "skipped_counts": dict(skipped),
        "selected_scene_ids": scene_ids,
        "samples": selected,
        "dpo_train_eligible": len(selected) >= 2,
    }
    out_path = run_dir / "manifests/input_subset.json"
    write_json(out_path, out)
    write_yaml(run_dir / "config/resolved_config.yaml", cfg)
    print(f"Wrote {out_path}")
    print(f"Selected {len(selected)} train T2V scenes across buckets {counts_by_bucket}")
    if len(selected) < 2:
        raise SystemExit("At least 2 scenes are required for DPO training")


if __name__ == "__main__":
    main()
