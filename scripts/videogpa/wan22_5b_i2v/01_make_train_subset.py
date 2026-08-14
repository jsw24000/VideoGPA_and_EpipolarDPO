from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
T2V_SCRIPT_DIR = CURRENT_DIR.parent / "wan22_5b_t2v"
if str(T2V_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(T2V_SCRIPT_DIR))

from common import deterministic_sample, read_json, resolve_config, safe_id, sha256_file, write_json  # noqa: E402
from vgm_common.config import write_resolved_config  # noqa: E402
from vgm_common.paths import get_manifest_root  # noqa: E402


FIRST_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_optional_master(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {row.get("scene_uid"): row for row in rows}


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


def find_first_frame(first_frames_root: Path, split: str, bucket: str, scene_id: str) -> Path | None:
    scene_dir = first_frames_root / split / bucket.upper() / scene_id
    for ext in FIRST_FRAME_EXTENSIONS:
        candidate = scene_dir / f"first_frame{ext}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def ensure_camera_prompt(
    *,
    scene_uid: str,
    prompt: str,
    camera_motion: str,
    seed: int,
    project_root: Path,
) -> tuple[str, str, int | None]:
    prompt = prompt.strip()
    camera_motion = camera_motion.strip()
    if prompt and "camera motion:" in prompt.lower() and camera_motion:
        return prompt, camera_motion, None

    from scripts.data.dl3dv_conditions.common import generate_official_i2v_prompt

    generated = generate_official_i2v_prompt(scene_uid, seed, project_root)
    generated_motion = str(generated["scripted_camera_motion"]).strip()
    generated_prompt = str(generated["i2v_train_text_prompt"]).strip()
    return generated_prompt, generated_motion, int(generated["scripted_camera_seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create full/formal train I2V subset manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--subset-size", default=None, help="Positive integer or 'all'")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    cfg["project"]["task"] = "i2v"
    project_root = Path(cfg["project"]["project_root"])
    train_manifest = Path(cfg["paths"]["train_manifest"])
    first_frames_root = Path(cfg["paths"]["first_frames_root"])
    data_cfg = cfg.get("data", {})
    required_split = str(data_cfg.get("required_split", "train"))
    required_buckets = parse_buckets(data_cfg)
    subset_size = parse_subset_size(args.subset_size if args.subset_size is not None else data_cfg.get("subset_size", "all"))
    seed = args.seed if args.seed is not None else int(data_cfg.get("subset_seed", 2026))
    camera_prompt_seed = int(data_cfg.get("camera_prompt_seed", seed))

    if "test_t2v" in str(train_manifest) or "test_i2v" in str(train_manifest) or "train_t2v" in str(train_manifest):
        raise ValueError(f"Formal I2V training must read train_i2v, not T2V/test manifests: {train_manifest}")
    manifest = read_json(train_manifest)
    if not isinstance(manifest, dict):
        raise ValueError("train_i2v manifest must be a dict")

    master_path = get_manifest_root() / "master_all.jsonl"
    master_by_uid = load_optional_master(master_path)

    candidates: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    regenerated_prompt_count = 0
    for scene_uid, item in manifest.items():
        if not isinstance(item, dict):
            skipped["invalid_item"] += 1
            continue
        bucket = scene_uid.split("/", 1)[0]
        scene_id = scene_uid.split("/", 1)[-1]
        master = master_by_uid.get(scene_uid)
        if bucket.upper() not in required_buckets:
            skipped["bucket"] += 1
            continue
        if master_by_uid:
            if not master or master.get("split") != required_split:
                skipped["split"] += 1
                continue
            if master.get("source_subset", "").upper() != bucket.upper():
                skipped["source_subset"] += 1
                continue
        first_frame = find_first_frame(first_frames_root, required_split, bucket, scene_id)
        if first_frame is None:
            skipped["missing_first_frame"] += 1
            continue

        prompt = str(item.get("text_prompt", item.get("prompt", "")))
        camera_motion = str(item.get("camera_motion", ""))
        prompt, camera_motion, scripted_seed = ensure_camera_prompt(
            scene_uid=scene_uid,
            prompt=prompt,
            camera_motion=camera_motion,
            seed=camera_prompt_seed,
            project_root=project_root,
        )
        if scripted_seed is not None:
            regenerated_prompt_count += 1
        if not prompt.strip() or not camera_motion.strip():
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
                "camera_motion": camera_motion.strip(),
                "image_path": str(first_frame),
                "image_prompt": str(first_frame),
                "task": "i2v",
                "image_conditioned": True,
                "scripted_camera_seed": scripted_seed,
            }
        )

    if not candidates:
        raise RuntimeError(f"No {required_buckets} {required_split} I2V samples found")

    selected = deterministic_sample(candidates, subset_size, seed) if subset_size is not None else list(candidates)
    scene_uids = [item["scene_uid"] for item in selected]
    scene_ids = [item["scene_id"] for item in selected]
    if len(scene_uids) != len(set(scene_uids)):
        raise RuntimeError("Selected duplicate scene UIDs")
    counts_by_bucket = dict(Counter(item["source_bucket"] for item in selected))
    expected_train_prompts = cfg.get("formal_requirements", {}).get("expected_train_prompts")
    if expected_train_prompts is not None and subset_size is None and len(selected) != int(expected_train_prompts):
        raise RuntimeError(
            "Selected formal train sample count "
            f"{len(selected)} != expected {int(expected_train_prompts)}. "
            "Check train_i2v.json, master_all.jsonl, and first_frames consistency."
        )

    out = {
        "task": "i2v",
        "run_type": cfg.get("project", {}).get("run_type", "formal"),
        "image_conditioned": True,
        "source_manifest": str(train_manifest),
        "source_manifest_sha256": sha256_file(train_manifest),
        "master_manifest": str(master_path) if master_by_uid else None,
        "master_manifest_used": bool(master_by_uid),
        "first_frames_root": str(first_frames_root),
        "required_split": required_split,
        "required_buckets": [bucket.lower() for bucket in required_buckets],
        "subset_seed": seed,
        "camera_prompt_seed": camera_prompt_seed,
        "requested_subset_size": "all" if subset_size is None else subset_size,
        "available_train": len(candidates),
        "selected_count": len(selected),
        "expected_train_prompts": int(expected_train_prompts) if expected_train_prompts is not None else None,
        "counts_by_bucket": counts_by_bucket,
        "skipped_counts": dict(skipped),
        "regenerated_camera_prompt_count": regenerated_prompt_count,
        "selected_scene_ids": scene_ids,
        "samples": selected,
        "dpo_train_eligible": len(selected) >= 2,
    }
    out_path = run_dir / "manifests/input_subset.json"
    write_json(out_path, out)
    write_resolved_config(run_dir, cfg)
    print(f"Wrote {out_path}")
    print(f"Selected {len(selected)} train I2V scenes across buckets {counts_by_bucket}")
    if regenerated_prompt_count:
        print(f"Regenerated {regenerated_prompt_count} missing/incomplete camera prompts")
    if len(selected) < 2:
        raise SystemExit("At least 2 scenes are required for DPO training")


if __name__ == "__main__":
    main()
