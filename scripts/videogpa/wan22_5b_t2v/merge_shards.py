from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def group_order(order_json: Path | None) -> dict[str, int]:
    if order_json is None or not order_json.exists():
        return {}
    payload = read_json(order_json)
    samples = payload.get("samples", []) if isinstance(payload, dict) else payload
    order = {}
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        group_id = sample.get("group_id") or sample.get("scene_uid")
        if group_id is not None:
            order[str(group_id).strip().replace("/", "__").replace("\\", "__").replace(" ", "_")] = idx
    return order


def sort_groups(groups: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    if not order:
        return groups
    return sorted(groups, key=lambda item: order.get(str(item.get("group_id")), 10**12))


def merge_group_payloads(inputs: list[Path], output: Path, order_json: Path | None = None) -> None:
    payloads = [read_json(path) for path in inputs]
    merged = dict(payloads[0]) if payloads else {}
    groups: list[dict[str, Any]] = []
    for payload in payloads:
        groups.extend(payload.get("groups", []))
    merged["groups"] = sort_groups(groups, group_order(order_json))
    merged["shards"] = [str(path) for path in inputs]
    merged["shard_count"] = len(inputs)
    write_json(output, merged)
    print(f"Wrote {len(groups)} groups to {output}")


def merge_scored_pairs(
    scored_inputs: list[Path],
    pair_inputs: list[Path],
    scored_output: Path,
    pairs_output: Path,
    order_json: Path | None = None,
) -> None:
    scored_payloads = [read_json(path) for path in scored_inputs]
    scored = dict(scored_payloads[0]) if scored_payloads else {"task": "t2v"}
    groups: list[dict[str, Any]] = []
    for payload in scored_payloads:
        groups.extend(payload.get("groups", []))
    scored["groups"] = sort_groups(groups, group_order(order_json))
    scored["shards"] = [str(path) for path in scored_inputs]
    scored["shard_count"] = len(scored_inputs)
    write_json(scored_output, scored)

    pairs: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    debug_fallback_used = False
    for path in pair_inputs:
        payload = read_json(path)
        pairs.extend(payload.get("pairs", []))
        filtered.extend(payload.get("filtered", []))
        debug_fallback_used = debug_fallback_used or bool(payload.get("debug_fallback_used"))

    pair_payload = {
        "task": "t2v",
        "base_path": scored.get("base_path"),
        "metric_name": scored_payloads[0].get("metric_name", "consistency_score") if scored_payloads else "consistency_score",
        "metric_mode": scored_payloads[0].get("metric_mode", "min") if scored_payloads else "min",
        "debug_fallback_used": debug_fallback_used,
        "debug_only": "DEBUG_ONLY_NOT_COMPARABLE" if debug_fallback_used else False,
        "pairs": pairs,
        "filtered": filtered,
        "shards": [str(path) for path in pair_inputs],
        "shard_count": len(pair_inputs),
    }
    if len(pairs) < 2:
        pair_payload["status"] = "INSUFFICIENT_PAIRS"
        write_json(pairs_output, pair_payload)
        raise SystemExit(f"Fewer than 2 merged preference pairs: {len(pairs)}")
    write_json(pairs_output, pair_payload)
    print(f"Wrote {len(groups)} scored groups to {scored_output}")
    print(f"Wrote {len(pairs)} preference pairs to {pairs_output}")


def merge_encoded(inputs: list[Path], output: Path, order_json: Path | None = None) -> None:
    payloads = [read_json(path) for path in inputs]
    merged = dict(payloads[0]) if payloads else {"task": "t2v"}
    pairs: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    order = group_order(order_json)
    for payload in payloads:
        pairs.extend(payload.get("pairs", []))
        groups.extend(payload.get("groups", []))
    if order:
        pairs = sorted(pairs, key=lambda item: order.get(str(item.get("group_id")), 10**12))
    merged["pairs"] = pairs
    merged["groups"] = sort_groups(groups, order)
    merged["shards"] = [str(path) for path in inputs]
    merged["shard_count"] = len(inputs)
    write_json(output, merged)
    print(f"Wrote {len(pairs)} encoded pairs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge WAN2.2 T2V formal shard manifests.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    groups = subparsers.add_parser("groups")
    groups.add_argument("--output", required=True)
    groups.add_argument("--order-json", default=None)
    groups.add_argument("inputs", nargs="+")

    scored_pairs = subparsers.add_parser("scored-pairs")
    scored_pairs.add_argument("--scored-output", required=True)
    scored_pairs.add_argument("--pairs-output", required=True)
    scored_pairs.add_argument("--order-json", default=None)
    scored_pairs.add_argument("--scored-inputs", nargs="+", required=True)
    scored_pairs.add_argument("--pair-inputs", nargs="+", required=True)

    encoded = subparsers.add_parser("encoded")
    encoded.add_argument("--output", required=True)
    encoded.add_argument("--order-json", default=None)
    encoded.add_argument("inputs", nargs="+")

    args = parser.parse_args()
    if args.mode == "groups":
        merge_group_payloads(
            [Path(path).expanduser().resolve() for path in args.inputs],
            Path(args.output).expanduser().resolve(),
            Path(args.order_json).expanduser().resolve() if args.order_json else None,
        )
    elif args.mode == "scored-pairs":
        merge_scored_pairs(
            [Path(path).expanduser().resolve() for path in args.scored_inputs],
            [Path(path).expanduser().resolve() for path in args.pair_inputs],
            Path(args.scored_output).expanduser().resolve(),
            Path(args.pairs_output).expanduser().resolve(),
            Path(args.order_json).expanduser().resolve() if args.order_json else None,
        )
    elif args.mode == "encoded":
        merge_encoded(
            [Path(path).expanduser().resolve() for path in args.inputs],
            Path(args.output).expanduser().resolve(),
            Path(args.order_json).expanduser().resolve() if args.order_json else None,
        )


if __name__ == "__main__":
    main()
