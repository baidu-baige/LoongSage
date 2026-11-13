#!/usr/bin/env python3
"""Turn openai/gsm8k parquet files into a short-prompt, answer-only RL dataset.

Each row becomes ``{"prompt": [{"role": "user", "content": ...}], "reward_model": {"ground_truth": ...}}``
where the prompt asks for the final number only (no reasoning), so both prefill and decode stay short.
Questions longer than --max-question-chars are dropped.

  hf download openai/gsm8k --repo-type=dataset --local-dir /root/gsm8k
  python coda/custom/dsv41_4090/tools/prepare_gsm8k_answer_only.py --src /root/gsm8k/main --dst /root/gsm8k_answer_only
"""
import argparse
import json
import os

import pyarrow.parquet as pq

INSTRUCTION = "\nReply with only the final number."


def convert(src: str, dst: str, max_chars: int) -> int:
    rows = pq.read_table(src).to_pylist()
    kept = 0
    with open(dst, "w") as f:
        for row in rows:
            question = row["question"].strip()
            if max_chars and len(question) > max_chars:
                continue
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            record = {
                "prompt": [{"role": "user", "content": question + INSTRUCTION}],
                "reward_model": {"ground_truth": answer},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding train-*.parquet / test-*.parquet")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-question-chars", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    for split in ("train", "test"):
        files = sorted(f for f in os.listdir(args.src) if f.startswith(split) and f.endswith(".parquet"))
        assert files, f"no {split}*.parquet under {args.src}"
        n = convert(os.path.join(args.src, files[0]), os.path.join(args.dst, f"{split}.jsonl"), args.max_question_chars)
        print(f"{split}: {n} rows")


if __name__ == "__main__":
    main()
