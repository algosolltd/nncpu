#!/usr/bin/env python3
"""Materialize the predeclared speculative-decoding prompt split."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


HUMAN_EVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
SPLITS = {"development": [0, 5], "holdout": [10, 25, 50, 100]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path, compressed: bool = False) -> list[dict]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed = {
        "human_eval": sha256(args.human_eval),
        "gsm8k": sha256(args.gsm8k),
    }
    expected = {"human_eval": HUMAN_EVAL_SHA256, "gsm8k": GSM8K_SHA256}
    if observed != expected:
        raise SystemExit(f"dataset hash mismatch: expected {expected}, observed {observed}")

    human_eval = load_jsonl(args.human_eval, compressed=True)
    gsm8k = load_jsonl(args.gsm8k)
    prompts = []
    for split, indices in SPLITS.items():
        for index in indices:
            prompts.append(
                {
                    "id": f"humaneval-{index}",
                    "dataset": "HumanEval",
                    "source_index": index,
                    "split": split,
                    "prompt": human_eval[index]["prompt"],
                }
            )
            prompts.append(
                {
                    "id": f"gsm8k-{index}",
                    "dataset": "GSM8K",
                    "source_index": index,
                    "split": split,
                    "prompt": f"Question: {gsm8k[index]['question']}\nAnswer:",
                }
            )

    artifact = {
        "schema_version": 1,
        "selection_frozen_before_inference": "2026-08-27",
        "sources": {
            "human_eval": {
                "repository": "https://github.com/openai/human-eval",
                "commit": "6d43fb980f9fee3c892a914eda09951f772ad10d",
                "sha256": HUMAN_EVAL_SHA256,
            },
            "gsm8k": {
                "repository": "https://github.com/openai/grade-school-math",
                "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
                "sha256": GSM8K_SHA256,
            },
        },
        "prompts": prompts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
