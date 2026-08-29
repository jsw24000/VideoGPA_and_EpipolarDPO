from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, resolve_config, select_preference_pairs, source_run_path, write_json, write_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Epipolar-DPO best-vs-worst preference pairs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--allow-insufficient-pairs", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    source_run = source_run_path(cfg)
    input_json = Path(args.input_json).expanduser().resolve() if args.input_json else run_dir / "manifests" / "scored_candidates.json"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else run_dir / "manifests" / "preference_pairs.json"
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else run_dir / "manifests" / "pair_summary.json"

    scored_payload = read_json(input_json)
    pair_payload, summary = select_preference_pairs(scored_payload, cfg, source_run)
    write_json(output_json, pair_payload)
    write_json(summary_json, summary)
    write_run_config(run_dir, cfg)
    if len(pair_payload["pairs"]) < 2 and not args.allow_insufficient_pairs:
        raise SystemExit(f"Fewer than 2 Epipolar-DPO pairs ({len(pair_payload['pairs'])}); see {summary_json}")
    print(f"Wrote {len(pair_payload['pairs'])} Epipolar-DPO pairs to {output_json}")


if __name__ == "__main__":
    main()
