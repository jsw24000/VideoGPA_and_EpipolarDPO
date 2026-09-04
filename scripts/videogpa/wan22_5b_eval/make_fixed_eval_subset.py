from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from make_eval_manifest import build_manifest, read_jsonl, sha256_file, sha256_text, write_json
from vgm_common.paths import ensure_profile, get_manifest_root


DEFAULT_LIMIT = 500
DEFAULT_SAMPLING_SEED = 456
DEFAULT_PER_PROMPT_SEED_BASE = 100000


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def prompt_id(index: int) -> str:
    return f"prompt_{index:06d}"


def motion_family(text: str | None) -> str:
    value = (text or "unknown").strip().lower()
    value = re.split(r",| then ", value, maxsplit=1)[0].strip()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unknown"


def caption_length_bounds(word_counts: list[int], bins: int) -> list[int]:
    if bins <= 1:
        return []
    ordered = sorted(word_counts)
    return [ordered[min(len(ordered) - 1, (len(ordered) * idx) // bins)] for idx in range(1, bins)]


def caption_length_bin(word_count: int, bounds: list[int]) -> int:
    bucket = 0
    for bound in bounds:
        if word_count >= bound:
            bucket += 1
    return bucket


def allocate_quotas(strata: dict[str, list[int]], limit: int, total: int) -> dict[str, int]:
    exact = {key: limit * len(indices) / total for key, indices in strata.items()}
    quotas = {key: min(len(strata[key]), int(value)) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda key: (exact[key] - int(exact[key]), len(strata[key]), key),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate all requested samples across strata")
    return quotas


def select_stratified_indices(strata: dict[str, list[int]], limit: int, seed: int, total: int) -> list[int]:
    quotas = allocate_quotas(strata, limit, total)
    selected: list[int] = []
    for key, indices in sorted(strata.items()):
        quota = quotas[key]
        if quota <= 0:
            continue
        rng = random.Random(stable_int(f"{seed}:{key}"))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected.extend(sorted(shuffled[:quota]))
    return sorted(selected)


def add_fixed_eval_fields(
    sample: dict[str, Any],
    master_row: dict[str, Any],
    word_count: int,
    length_bin: int,
    seed_base: int,
) -> dict[str, Any]:
    index = int(sample["index"])
    fixed_id = prompt_id(index)
    updated = dict(sample)
    updated["prompt_id"] = fixed_id
    updated["source_group_id"] = str(sample["group_id"])
    updated["group_id"] = fixed_id
    updated["seed"] = seed_base + index
    updated["seed_source"] = "per_prompt_seed_base_plus_source_index"
    updated["stratum"] = {
        "motion_family": motion_family(master_row.get("scripted_camera_motion")),
        "caption_word_count": word_count,
        "caption_length_bin": length_bin,
    }
    if master_row.get("scripted_camera_motion") is not None:
        updated["scripted_camera_motion"] = master_row["scripted_camera_motion"]
    return updated


def build_fixed_subset(args: argparse.Namespace) -> dict[str, Any]:
    ensure_profile()
    source_args = argparse.Namespace(seed=args.source_seed, limit="all")
    full = build_manifest(source_args)
    samples = list(full["samples"])
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.limit > len(samples):
        raise ValueError(f"--limit={args.limit} exceeds source samples={len(samples)}")

    manifest_root = get_manifest_root()
    master_rows = read_jsonl(manifest_root / "master_test.jsonl")
    master_by_uid = {row["scene_uid"]: row for row in master_rows}
    word_counts = [len(str(master_by_uid[sample["scene_uid"]].get("vlm_caption") or sample["text_prompt"]).split()) for sample in samples]
    bounds = caption_length_bounds(word_counts, args.caption_length_bins)

    strata: dict[str, list[int]] = defaultdict(list)
    sample_meta: dict[int, tuple[dict[str, Any], int, int]] = {}
    for position, sample in enumerate(samples):
        master_row = master_by_uid[sample["scene_uid"]]
        word_count = len(str(master_row.get("vlm_caption") or sample["text_prompt"]).split())
        length_bin = caption_length_bin(word_count, bounds)
        key = f"motion={motion_family(master_row.get('scripted_camera_motion'))}|caption_len_bin={length_bin}"
        strata[key].append(position)
        sample_meta[position] = (master_row, word_count, length_bin)

    selected_positions = select_stratified_indices(strata, args.limit, args.sampling_seed, len(samples))
    selected_samples = []
    for position in selected_positions:
        master_row, word_count, length_bin = sample_meta[position]
        selected_samples.append(
            add_fixed_eval_fields(
                samples[position],
                master_row,
                word_count,
                length_bin,
                args.per_prompt_seed_base,
            )
        )

    selected_by_key: dict[str, int] = defaultdict(int)
    for sample in selected_samples:
        stratum = sample["stratum"]
        key = f"motion={stratum['motion_family']}|caption_len_bin={stratum['caption_length_bin']}"
        selected_by_key[key] += 1

    payload = dict(full)
    payload["protocol"] = "wan22_dl3dv1k_fixed_subset_eval_v1"
    payload["num_samples"] = len(selected_samples)
    payload["seed"] = int(args.sampling_seed)
    payload["seeds"] = [int(sample["seed"]) for sample in selected_samples]
    payload["selection"] = {
        "strategy": "stratified_proportional_without_replacement",
        "sampling_seed": int(args.sampling_seed),
        "requested_limit": int(args.limit),
        "source_protocol": full["protocol"],
        "source_size": len(samples),
        "stratify_by": ["scripted_camera_motion_family", "caption_length_bin"],
        "caption_length_bins": int(args.caption_length_bins),
        "caption_length_bounds": bounds,
        "strata_total": len(strata),
        "strata_selected_counts": dict(sorted(selected_by_key.items())),
        "selected_source_indices": [int(sample["index"]) for sample in selected_samples],
    }
    payload["generation_settings"] = dict(full["generation_settings"])
    payload["generation_settings"].update(
        {
            "seed_policy": "per_prompt_seed",
            "per_prompt_seed_base": int(args.per_prompt_seed_base),
            "per_prompt_seed_formula": "seed = per_prompt_seed_base + original_1k_index",
            "prompt_id_format": "prompt_{original_1k_index:06d}",
        }
    )
    payload["samples"] = selected_samples
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed stratified DL3DV-1K subset for WAN2.2 evaluation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--source-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--per-prompt-seed-base", type=int, default=DEFAULT_PER_PROMPT_SEED_BASE)
    parser.add_argument("--caption-length-bins", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    payload = build_fixed_subset(args)
    write_json(output, payload)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    output.with_name(f"{output.stem}.content_sha256.txt").write_text(
        sha256_text(json.dumps(payload, sort_keys=True)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote fixed eval subset: {output}")
    print(f"Samples: {payload['num_samples']}")
    print(f"Selection: {payload['selection']['strategy']}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
