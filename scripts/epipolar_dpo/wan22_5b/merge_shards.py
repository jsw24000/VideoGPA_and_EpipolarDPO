from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import group_order, read_json, sort_groups, write_json


def merge_scored_payloads(inputs: list[Path], order_payload: Any | None = None) -> dict[str, Any]:
    if not inputs:
        raise ValueError("No shard inputs provided")
    payloads = [read_json(path) for path in inputs]
    merged = dict(payloads[0])
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in payloads:
        for group in payload.get("groups", []):
            group_id = str(group.get("group_id"))
            if group_id in seen:
                raise ValueError(f"Duplicate group_id across score shards: {group_id}")
            seen.add(group_id)
            groups.append(group)
    order = group_order(order_payload) if order_payload is not None else {}
    merged["groups"] = sort_groups(groups, order)
    merged["shards"] = [str(path) for path in inputs]
    merged["shard_count"] = len(inputs)
    merged["shard_indices"] = [payload.get("shard_index") for payload in payloads]
    merged["score_summary"] = {
        "groups": len(merged["groups"]),
        "candidates": sum(len(group.get("videos", [])) for group in merged["groups"]),
        "scored_now": sum(int(payload.get("score_summary", {}).get("scored_now", 0)) for payload in payloads),
        "reused": sum(int(payload.get("score_summary", {}).get("reused", 0)) for payload in payloads),
        "invalid_or_failed": sum(int(payload.get("score_summary", {}).get("invalid_or_failed", 0)) for payload in payloads),
    }
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Epipolar-DPO shard manifests")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    scored = subparsers.add_parser("scored")
    scored.add_argument("--output", required=True)
    scored.add_argument("--order-json", default=None)
    scored.add_argument("inputs", nargs="+")

    args = parser.parse_args()
    if args.mode == "scored":
        order_payload = read_json(Path(args.order_json).expanduser().resolve()) if args.order_json else None
        merged = merge_scored_payloads([Path(path).expanduser().resolve() for path in args.inputs], order_payload)
        write_json(Path(args.output).expanduser().resolve(), merged)
        print(f"Wrote {len(merged['groups'])} scored groups to {args.output}")


if __name__ == "__main__":
    main()
