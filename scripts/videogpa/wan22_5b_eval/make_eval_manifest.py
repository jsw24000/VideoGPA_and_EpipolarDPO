from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vgm_common.paths import ensure_profile, get_dl3dv_root, get_manifest_root


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_id(value: str) -> str:
    return value.strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


def assert_no_absolute_paths(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_absolute_paths(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            assert_no_absolute_paths(child, f"{path}[{idx}]")
    elif isinstance(value, str) and value.startswith("/"):
        raise ValueError(f"Manifest contains an absolute path at {path}: {value}")


def source_relpath(path: Path, manifest_root: Path) -> str:
    return path.resolve().relative_to(manifest_root.resolve()).as_posix()


def parse_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    limit = int(value)
    if limit <= 0:
        raise ValueError("--limit must be positive or 'all'")
    return limit


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    ensure_profile()
    manifest_root = get_manifest_root()
    data_root = get_dl3dv_root()
    t2v_path = manifest_root / "videogpa_protocol/test_t2v.json"
    i2v_path = manifest_root / "videogpa_protocol/test_i2v.json"
    master_path = manifest_root / "master_test.jsonl"
    for path in [t2v_path, i2v_path, master_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    test_t2v = read_json(t2v_path)
    test_i2v = read_json(i2v_path)
    if not isinstance(test_t2v, dict) or not isinstance(test_i2v, dict):
        raise ValueError("test_t2v.json and test_i2v.json must both be dict manifests")
    if list(test_t2v) != list(test_i2v):
        raise RuntimeError("test_t2v.json and test_i2v.json must have identical scene order")

    master_rows = read_jsonl(master_path)
    master_by_uid = {row["scene_uid"]: row for row in master_rows}
    limit = parse_limit(args.limit)
    samples: list[dict[str, Any]] = []
    for index, (scene_uid, t2v_item) in enumerate(test_t2v.items()):
        if limit is not None and len(samples) >= limit:
            break
        i2v_item = test_i2v[scene_uid]
        t2v_prompt = str(t2v_item.get("text_prompt", "")).strip()
        i2v_prompt = str(i2v_item.get("text_prompt", "")).strip()
        if not t2v_prompt:
            raise RuntimeError(f"Empty T2V prompt for {scene_uid}")
        if t2v_prompt != i2v_prompt:
            raise RuntimeError(f"T2V/I2V prompt mismatch for {scene_uid}")
        master = master_by_uid.get(scene_uid)
        if master is None:
            raise RuntimeError(f"Missing master_test row for {scene_uid}")
        if master.get("split") != "test" or master.get("source_subset") != "1K":
            raise RuntimeError(f"Unexpected split/subset for {scene_uid}: {master.get('split')} {master.get('source_subset')}")
        first_frame_relpath = str(master.get("first_frame_relpath") or "")
        if not first_frame_relpath:
            scene_id = scene_uid.split("/", 1)[1]
            first_frame_relpath = f"first_frames/test/1K/{scene_id}/first_frame.png"
        first_frame_path = data_root / first_frame_relpath
        if not first_frame_path.is_file():
            raise FileNotFoundError(f"Missing first frame for {scene_uid}: {first_frame_relpath}")
        scene_id = str(master.get("scene_id") or scene_uid.split("/", 1)[1])
        samples.append(
            {
                "index": index,
                "scene_uid": scene_uid,
                "scene_id": scene_id,
                "group_id": safe_id(scene_uid),
                "source_split": "test",
                "source_bucket": "1K",
                "text_prompt": t2v_prompt,
                "seed": int(args.seed),
                "first_frame_relpath": first_frame_relpath,
                "first_frame_sha256": master.get("first_frame_sha256"),
                "caption_source": master.get("caption_source", "VideoGPA CogVLM caption"),
                "caption_source_file": master.get("caption_source_file"),
                "caption_source_key": master.get("caption_source_key"),
            }
        )

    if limit is None and len(samples) != 1000:
        raise RuntimeError(f"Expected 1000 DL3DV-1K eval samples, got {len(samples)}")

    payload = {
        "schema_version": 1,
        "protocol": "wan22_5b_dl3dv1k_eval_v1",
        "tasks": ["t2v", "i2v"],
        "split": "test",
        "source_subset": "1K",
        "prompt": "CogVLM2 natural caption",
        "i2v_extra_input": "same-scene first frame",
        "seed": int(args.seed),
        "seeds": [int(args.seed)],
        "num_samples": len(samples),
        "generation_settings": {
            "frame_num": 81,
            "size": "1280*704",
            "sampling_steps": 50,
            "sample_shift": 5.0,
            "guide_scale": 5.0,
            "sample_solver": "unipc",
            "fps": 24,
            "lora_weight_primary": 0.2,
        },
        "evaluator_settings": {
            "geometry_backbone": "DA3-Large",
            "num_sampled_frames": 10,
            "frame_sampling": "uniform",
            "confidence_threshold": 0,
            "epipolar_matcher": "lightglue",
            "metrics": ["psnr", "ssim", "lpips", "mvcs", "3dcs", "epipolar", "sampson_error"],
        },
        "source_manifests": {
            "test_t2v": {
                "relpath": source_relpath(t2v_path, manifest_root),
                "sha256": sha256_file(t2v_path),
            },
            "test_i2v": {
                "relpath": source_relpath(i2v_path, manifest_root),
                "sha256": sha256_file(i2v_path),
            },
            "master_test": {
                "relpath": source_relpath(master_path, manifest_root),
                "sha256": sha256_file(master_path),
            },
        },
        "path_policy": {
            "contains_absolute_paths": False,
            "first_frame_relpath_root": "VGM_DL3DV_ROOT",
            "first_frames_are_not_copied": True,
        },
        "samples": samples,
    }
    assert_no_absolute_paths(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical DL3DV-1K WAN2.2 T2V/I2V eval manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--limit", default="all", help="Positive integer for a smoke subset, or 'all'")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    payload = build_manifest(args)
    write_json(output, payload)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    output.with_name(f"{output.stem}.content_sha256.txt").write_text(sha256_text(json.dumps(payload, sort_keys=True)) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Samples: {payload['num_samples']}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
