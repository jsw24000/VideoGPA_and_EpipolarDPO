from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from make_eval_manifest import assert_no_absolute_paths, read_json, write_json


COMMON_SAMPLE_KEYS = [
    "index",
    "scene_uid",
    "scene_id",
    "group_id",
    "source_split",
    "source_bucket",
    "text_prompt",
    "seed",
    "caption_source",
    "caption_source_file",
    "caption_source_key",
]


def project_sample(sample: dict[str, Any], task: str) -> dict[str, Any]:
    projected = {key: sample[key] for key in COMMON_SAMPLE_KEYS if key in sample}
    projected["task"] = task
    if task == "i2v":
        projected["first_frame_relpath"] = sample["first_frame_relpath"]
        if "first_frame_sha256" in sample:
            projected["first_frame_sha256"] = sample["first_frame_sha256"]
        projected["image_conditioned"] = True
    else:
        projected["image_conditioned"] = False
    return projected


def build_task_manifest(canonical: dict[str, Any], task: str, source_name: str = "") -> dict[str, Any]:
    if task not in {"t2v", "i2v"}:
        raise ValueError("--task must be t2v or i2v")
    samples = canonical.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Canonical eval manifest must contain a samples list")

    payload = {
        "schema_version": 1,
        "protocol": canonical["protocol"],
        "task": task,
        "split": canonical["split"],
        "source_subset": canonical["source_subset"],
        "prompt": canonical["prompt"],
        "seed": canonical["seed"],
        "seeds": canonical["seeds"],
        "num_samples": canonical["num_samples"],
        "generation_settings": canonical["generation_settings"],
        "source_eval_manifest": {
            "filename": source_name,
        },
        "samples": [project_sample(dict(sample), task) for sample in samples],
    }
    if task == "i2v":
        payload["i2v_extra_input"] = canonical["i2v_extra_input"]
        payload["path_policy"] = canonical["path_policy"]
    else:
        payload["path_policy"] = {
            "contains_absolute_paths": False,
            "first_frame_fields_present": False,
        }
    assert_no_absolute_paths(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive task-specific WAN2.2 eval input from the canonical DL3DV-1K manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True, choices=["t2v", "i2v"])
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    canonical = read_json(input_path)
    payload = build_task_manifest(canonical, args.task, source_name=input_path.name)
    write_json(output_path, payload)
    print(f"Wrote {args.task} task manifest: {output_path}")
    print(f"Samples: {payload['num_samples']}")


if __name__ == "__main__":
    main()
