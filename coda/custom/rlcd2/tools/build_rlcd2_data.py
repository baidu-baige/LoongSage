"""Build RLCD² RL data: render the inference prompts, read p0 with the frozen scorer, keep max p0 < gate.

Input: DecisionPivot-style items ({id, state, questions: {q: {type, instructions, criteria}}, gold, soft, group,
meta}), one tasks/<task>/items.jsonl per task. Prompts are rendered with the official inference package (its
checkout given by $RLCD2_INFER_DIR), so they are byte-identical to the inference path.

Output (coda jsonl, one prompt per line):
    {"prompt": [system, user],                       # the one-pass / thinking messages
     "label":  {id, task, type, k, option_ids, target, p0, p0_max, think_base},
     "metadata": {id, task, group}}
"think_base" is the thinking prompt without the forced prefix; the scorer appends prefix + thought + CLOSE/CUT to it.

Files written to --out:
    p0_all.jsonl       every scored item with p0 (label only, no think_base), for analysis
    train.jsonl        max p0 < gate, minus the eval split
    eval.jsonl         max p0 < gate, --eval-groups groups held out by group id
    eval_small.jsonl   --eval-small items of eval.jsonl in hash order (all tasks mixed), for the in-training eval
    summary.json       counts per task

NOTE: DecisionPivot is licensed for evaluation only. Data built from it is for validating the training pipeline,
not for training a model that is used or published.

Usage:
    RLCD2_INFER_DIR=<inference checkout> python3 coda/custom/rlcd2/tools/build_rlcd2_data.py \
        --bench <DecisionPivot dir> --model <model dir> \
        --scorer-url http://127.0.0.1:30106,http://127.0.0.1:30107 --out <data dir>
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                                     # coda/custom/rlcd2
INFER_DIR = os.environ.get("RLCD2_INFER_DIR")                    # checkout of the official inference package
if not INFER_DIR or not os.path.isdir(INFER_DIR):
    sys.exit("set RLCD2_INFER_DIR to the checkout of the official inference package")
sys.path.insert(0, INFER_DIR)

from startlux_decide.prompt import (MAX_OPTIONS, check_tokenizer, count_tokens, from_systemone,  # noqa: E402
                                    messages, plain_prompt, think_prompt)
from transformers import AutoTokenizer  # noqa: E402

# coda/custom/__init__.py imports every custom module; import the scorer module file directly instead.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rlcd2_scorer", os.path.join(PKG, "scorer.py"))
scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer_mod)


def target_vector(item: dict, option_ids: list[str], qtype: str) -> list[float] | None:
    """Target distribution over the options in prompt order: one-hot gold, or the soft label."""
    if item.get("soft"):
        soft = {str(k): float(v) for k, v in item["soft"].items()}
        t = [soft.get(i, 0.0) for i in option_ids]
        s = sum(t)
        return [v / s for v in t] if s > 0 else None
    gold = item.get("gold")
    if gold is None:
        return None
    gold = str(gold)
    if qtype == "noul":
        gold = {"yes": "true", "no": "false"}.get(gold.lower(), gold.lower())
    if gold not in option_ids:
        return None
    return [1.0 if i == gold else 0.0 for i in option_ids]


def render(item: dict, tok) -> dict | str:
    """Rendered question or the reason it is skipped."""
    q = (item.get("questions") or {}).get("q")
    if not isinstance(q, dict):
        return "no_question"
    spec = dict(q)
    # DecisionPivot writes yes/no questions with the keys yes/no; the package reads true/false.
    if spec.get("type") == "noul" and isinstance(spec.get("criteria"), dict):
        c = spec["criteria"]
        spec["criteria"] = {"true": c.get("yes", c.get("true")), "false": c.get("no", c.get("false"))}
    crit = spec.get("criteria", spec.get("options"))
    if spec.get("type", "choice") == "choice" and crit is not None and len(crit) > MAX_OPTIONS:
        return "knockout"
    try:
        row = from_systemone(item.get("state"), spec)
        msgs = messages(row)
    except ValueError:
        return "invalid"
    option_ids = [o["id"] for o in row["options"]]
    t = target_vector(item, option_ids, row["type"])
    if t is None:
        return "no_target"
    return {"row": row, "msgs": msgs, "plain": plain_prompt(tok, msgs), "think_base": think_prompt(tok, msgs),
            "option_ids": option_ids, "target": t}


async def score_all(client, jobs: list[dict], log_every: int = 2000) -> None:
    done = 0
    t0 = time.time()

    async def one(j):
        nonlocal done
        j["p0"] = await client.read(j["plain"], len(j["option_ids"]), j["row"]["type"])
        done += 1
        if done % log_every == 0:
            print(f"  scored {done}/{len(jobs)}  {done / (time.time() - t0):.1f}/s", flush=True)

    await asyncio.gather(*(one(j) for j in jobs))
    await client.close()


def write_eval_small(out: str, n: int, seed: str) -> None:
    """eval_small.jsonl: the first n items of eval.jsonl by hash of the id (eval.jsonl is in task order)."""
    with open(os.path.join(out, "eval.jsonl")) as f:
        lines = f.readlines()
    lines.sort(key=lambda line: hashlib.sha1((seed + json.loads(line)["label"]["id"]).encode()).hexdigest())
    with open(os.path.join(out, "eval_small.jsonl"), "w") as f:
        f.writelines(lines[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="DecisionPivot folder (holds tasks/<task>/items.jsonl)")
    ap.add_argument("--model", required=True, help="model folder (tokenizer, decision_config.json)")
    ap.add_argument("--scorer-url", required=True, help="comma-separated SGLang urls (launch_scorer.sh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    ap.add_argument("--gate", type=float, default=0.9)
    ap.add_argument("--max-prompt-tokens", type=int, default=12288,
                    help="skip items whose one-pass prompt is longer (context = prompt + 4096 thought)")
    ap.add_argument("--max-per-task", type=int, default=0, help="cap items per task (0: no cap)")
    ap.add_argument("--eval-groups", type=int, default=640, help="number of groups held out for eval")
    ap.add_argument("--eval-small", type=int, default=128, help="items of eval_small.jsonl")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--seed", default="rlcd2")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    decision = scorer_mod.DecisionConfig(a.model)
    if check_tokenizer(tok) != decision.letter_ids:
        raise SystemExit("tokenizer letter ids differ from decision_config.json")

    tasks_dir = os.path.join(a.bench, "tasks")
    tasks = a.tasks.split(",") if a.tasks else sorted(
        t for t in os.listdir(tasks_dir) if os.path.isfile(os.path.join(tasks_dir, t, "items.jsonl")))

    skipped = collections.Counter()
    jobs = []
    for task in tasks:
        with open(os.path.join(tasks_dir, task, "items.jsonl")) as f:
            items = [json.loads(line) for line in f]
        if a.max_per_task:
            # deterministic subset
            items.sort(key=lambda it: hashlib.sha1((a.seed + it["id"]).encode()).hexdigest())
            items = items[:a.max_per_task]
        for item in items:
            r = render(item, tok)
            if isinstance(r, str):
                skipped[f"{task}:{r}"] += 1
                continue
            n = count_tokens(tok, r["plain"])
            if n > a.max_prompt_tokens:
                skipped[f"{task}:too_long"] += 1
                continue
            r.update(id=item["id"], task=task, group=str(item.get("group") or item["id"]), n_prompt=n)
            jobs.append(r)
    print(f"rendered {len(jobs)} items from {len(tasks)} tasks; skipped {sum(skipped.values())}", flush=True)

    client = scorer_mod.ScorerClient(a.scorer_url, decision, max_concurrency=a.concurrency)
    asyncio.run(score_all(client, jobs))

    # Held-out groups: the lowest hashes among groups that pass the gate.
    passing = [j for j in jobs if max(j["p0"]) < a.gate]
    groups = sorted({j["group"] for j in passing}, key=lambda g: hashlib.sha1((a.seed + g).encode()).hexdigest())
    eval_groups = set(groups[:a.eval_groups])

    os.makedirs(a.out, exist_ok=True)
    per_task = collections.defaultdict(lambda: collections.Counter())
    with open(os.path.join(a.out, "p0_all.jsonl"), "w") as f_all, \
            open(os.path.join(a.out, "train.jsonl"), "w") as f_tr, \
            open(os.path.join(a.out, "eval.jsonl"), "w") as f_ev:
        for j in jobs:
            p0_max = max(j["p0"])
            label = {"id": j["id"], "task": j["task"], "type": j["row"]["type"], "k": len(j["option_ids"]),
                     "option_ids": j["option_ids"], "target": j["target"], "p0": j["p0"], "p0_max": p0_max}
            correct = j["target"].index(max(j["target"])) == j["p0"].index(p0_max)
            f_all.write(json.dumps({**label, "group": j["group"], "n_prompt": j["n_prompt"],
                                    "p0_correct": correct}, ensure_ascii=False) + "\n")
            c = per_task[j["task"]]
            c["scored"] += 1
            c["p0_correct"] += correct
            if p0_max >= a.gate:
                continue
            c["below_gate"] += 1
            rec = {"prompt": j["msgs"], "label": {**label, "think_base": j["think_base"]},
                   "metadata": {"id": j["id"], "task": j["task"], "group": j["group"]}}
            split = "eval" if j["group"] in eval_groups else "train"
            c[split] += 1
            (f_ev if split == "eval" else f_tr).write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = collections.Counter()
    for c in per_task.values():
        total.update(c)
    summary = {"gate": a.gate, "model": a.model, "total": dict(total), "skipped": dict(skipped),
               "per_task": {t: dict(c) for t, c in sorted(per_task.items())}}
    write_eval_small(a.out, a.eval_small, a.seed)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["total"]), flush=True)


if __name__ == "__main__":
    main()
