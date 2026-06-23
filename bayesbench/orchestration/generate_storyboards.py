#!/usr/bin/env python3
"""
generate_storyboards.py — unified storyboard generator for all storyboard tasks.

Pipeline (shared across tasks): source dataset -> task extraction prompt ->
OpenAI API -> validate -> write one {id}.json into the task's storyboards/ dir.

Only the *task spec* differs per task — its dataset loader, its
`storyboard_prompt` module (the extraction prompt + `validate_storyboard`),
how the model response is parsed, and how the final record is assembled. Those
live in the TASKS registry at the bottom of this file; the engine above is
generic. To add a task, register one more TaskSpec — no engine changes.

The storyboards shipped in each task's storyboards/ dir are the fixed benchmark
inputs; you do NOT need this script to run the benchmarks. Use it only to
rebuild storyboards from scratch, swap the extraction model, or extend the set.
It depends only on `openai` + `datasets` (it loads no models), so it
runs anywhere with an API key.

Setup:
    pip install openai datasets
    export OPENAI_API_KEY=sk-...

Usage (from the repo root):
    # Triage — reproduce the shipped set (the exact case IDs already present)
    python -m bayesbench.orchestration.generate_storyboards --task medical_triage --from-existing

    # Triage — fresh balanced sample, 25 cases per urgency tier
    python -m bayesbench.orchestration.generate_storyboards --task medical_triage --balanced --per-tier 25 --seed 42

    # AITA — reproduce the shipped post IDs
    python -m bayesbench.orchestration.generate_storyboards --task social_judgment --from-existing

    # AITA — fresh sample of 100 posts
    python -m bayesbench.orchestration.generate_storyboards --task social_judgment --sample 100 --seed 42

    # Either task — specific IDs, overwrite, different model
    python -m bayesbench.orchestration.generate_storyboards --task medical_triage --ids 4,12,1016 --overwrite --model gpt-4o

(Installed, the console script `bayesbench-storyboards` is equivalent to
`python -m bayesbench.orchestration.generate_storyboards`.)
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

DEFAULT_MODEL = "gpt-4o"
# This module lives in bayesbench/orchestration/; the package root (where the
# task packages and their storyboards/ dirs live) is one level up.
PKG_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────── task spec ──────────────────────────────────

@dataclass
class TaskSpec:
    """Everything task-specific the engine needs. The engine is otherwise generic."""
    name: str
    storyboard_dir: Path
    load_cases: Callable[["argparse.Namespace"], dict]   # -> {str_id: case}
    build_messages: Callable[[dict], list]               # case -> chat messages
    validate: Callable[[list], dict]                     # aspects -> {valid, errors, ...}
    parse_response: Callable[[str], tuple]               # text -> (aspects, extra_fields)
    assemble_record: Callable[[dict, list, dict], dict]  # case, aspects, extra -> record
    use_json_object: bool                                # OpenAI response_format object mode
    balance_field: Optional[str] = None                  # case key for --balanced
    balance_values: Optional[list] = None                # tier ordering for --balanced


# A `case` (the unit the engine passes around) is a dict with:
#   "id"           str   — output filename stem
#   "prompt_input" dict  — passed to the task's build_extraction_messages
#   "meta"         dict  — source metadata carried into the record
#   "balance_key"  any   — value of balance_field (optional)


# ─────────────────────────── response parsing ───────────────────────────────

def _extract_json(text: str, opener: str, closer: str):
    """Pull the first balanced JSON object/array out of a model response."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t[:4].lower() == "json":
            t = t[4:]
        if "```" in t:
            t = t.rsplit("```", 1)[0]
    start = t.find(opener)
    end = t.rfind(closer)
    if start == -1 or end <= start:
        raise ValueError(f"no JSON {opener}{closer} found in response")
    return json.loads(t[start:end + 1])


def parse_object_response(text: str) -> tuple:
    """Triage: model returns an object {self_diagnosis, storyboard:[...]}."""
    obj = _extract_json(text, "{", "}")
    return obj.get("storyboard", []), {"self_diagnosis": obj.get("self_diagnosis", "")}


def parse_array_response(text: str) -> tuple:
    """AITA: model returns a bare array of aspect objects."""
    return _extract_json(text, "[", "]"), {}


# ──────────────────────────────── engine ────────────────────────────────────

def generate_one(client, model: str, case: dict, spec: TaskSpec, max_retries: int) -> dict:
    """Run extraction for one case; validate and re-prompt on failure."""
    messages = spec.build_messages(case)
    last_errors = None
    last_raw = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            messages = messages + [
                {"role": "assistant", "content": last_raw},
                {"role": "user", "content":
                    "The storyboard failed validation with these errors:\n"
                    + "\n".join(f"- {e}" for e in last_errors)
                    + "\nReturn a corrected response in the exact same format."},
            ]

        kwargs = dict(model=model, messages=messages, temperature=0.3)
        if spec.use_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        last_raw = resp.choices[0].message.content

        try:
            aspects, extra = spec.parse_response(last_raw)
        except Exception as e:
            last_errors = [f"unparseable response: {e}"]
            continue

        result = spec.validate(aspects)
        if result["valid"]:
            return spec.assemble_record(case, aspects, extra)
        last_errors = result["errors"]

    raise ValueError(
        f"{spec.name} {case['id']}: invalid after {max_retries} retries: {last_errors}")


def select_ids(args, cases: dict, spec: TaskSpec) -> list:
    """Resolve --ids / --from-existing / --balanced / --sample to a list of str IDs."""
    if args.ids:
        return [x.strip() for x in args.ids.split(",") if x.strip()]

    if args.from_existing:
        existing = sorted(p.stem for p in spec.storyboard_dir.glob("*.json"))
        if not existing:
            sys.exit(f"--from-existing: no storyboards in {spec.storyboard_dir}")
        return existing

    if args.balanced:
        if not spec.balance_field:
            sys.exit(f"--balanced is not supported for task '{spec.name}'. Use --sample.")
        rng = random.Random(args.seed)
        by = defaultdict(list)
        for cid, c in cases.items():
            by[c.get("balance_key")].append(cid)
        picked = []
        for key in (spec.balance_values or sorted(by)):
            pool = sorted(by.get(key, []))
            if len(pool) < args.per_tier:
                print(f"  warning: '{key}' has {len(pool)} cases (< {args.per_tier}); taking all")
            rng.shuffle(pool)
            picked.extend(pool[:args.per_tier])
        return sorted(picked)

    if args.sample:
        rng = random.Random(args.seed)
        allids = sorted(cases)
        rng.shuffle(allids)
        return sorted(allids[:args.sample])

    sys.exit("Pick a selection mode: --from-existing, --balanced, --sample, or --ids")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS),
                    help="which storyboard task to generate")
    sel = ap.add_argument_group("case selection (pick one)")
    sel.add_argument("--from-existing", action="store_true",
                     help="regenerate the IDs already in the task's storyboards/")
    sel.add_argument("--balanced", action="store_true",
                     help="balanced sample of --per-tier per balance value (if supported)")
    sel.add_argument("--sample", type=int, help="flat random sample of N cases")
    sel.add_argument("--ids", type=str, help="comma-separated IDs")

    ap.add_argument("--per-tier", type=int, default=25, help="cases per value for --balanced")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed")
    ap.add_argument("--pool-size", type=int, default=2000,
                    help="how many source rows to load when indexing by ID (AITA)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default {DEFAULT_MODEL})")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="re-prompt this many times if validation fails")
    ap.add_argument("--overwrite", action="store_true", help="regenerate existing files")
    ap.add_argument("--limit", type=int, help="cap the number generated")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai SDK not installed. Run: pip install openai")

    spec = TASKS[args.task]()          # build the spec (lazy task imports happen here)
    client = OpenAI()
    spec.storyboard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{spec.name}] loading source cases ...")
    cases = spec.load_cases(args)
    ids = select_ids(args, cases, spec)

    if not args.overwrite:
        ids = [i for i in ids if not (spec.storyboard_dir / f"{i}.json").exists()]
    if args.limit:
        ids = ids[:args.limit]

    print(f"[{spec.name}] generating {len(ids)} storyboard(s) with {args.model} "
          f"-> {spec.storyboard_dir}/")

    n_ok = n_fail = 0
    for i, cid in enumerate(ids, 1):
        case = cases.get(cid)
        if case is None:
            print(f"  [{i}/{len(ids)}] {cid}: NOT in source, skipping")
            n_fail += 1
            continue
        try:
            record = generate_one(client, args.model, case, spec, args.max_retries)
        except Exception as e:
            print(f"  [{i}/{len(ids)}] {cid}: FAILED — {e}")
            n_fail += 1
            continue
        with open(spec.storyboard_dir / f"{cid}.json", "w") as f:
            json.dump(record, f, indent=2)
        n_ok += 1
        print(f"  [{i}/{len(ids)}] {cid}: {len(record['storyboard'])} aspects")
        time.sleep(0.1)

    print(f"\n[{spec.name}] done: {n_ok} written, {n_fail} failed.")


# ════════════════════════════ task registry ═════════════════════════════════
# Each entry is a zero-arg factory returning a TaskSpec. Imports are done inside
# the factory so selecting one task never imports the other task's deps.

def _medical_triage_spec() -> TaskSpec:
    from datasets import load_dataset
    from bayesbench.medical_triage.storyboard_prompt import build_extraction_messages, validate_storyboard

    storyboard_dir = PKG_ROOT / "medical_triage" / "storyboards"
    TIERS = ["Emergency", "Urgent", "Observation", "Routine"]

    def load_cases(args) -> dict:
        ds = load_dataset("sweatSmile/medical-symptom-triage", split="train")
        cases = {}
        for row in ds:
            try:
                cid = str(int(row["id"]))
            except (TypeError, ValueError):
                continue
            cases[cid] = {
                "id": cid,
                "prompt_input": {"input": row["input"]},
                "meta": {"true_urgency": row["urgency"], "specialty": row["specialty"]},
                "balance_key": row["urgency"],
            }
        return cases

    def assemble(case, aspects, extra) -> dict:
        return {
            "case_id": int(case["id"]),
            "true_urgency": case["meta"]["true_urgency"],
            "specialty": case["meta"]["specialty"],
            "self_diagnosis": extra.get("self_diagnosis", ""),
            "storyboard": aspects,
        }

    return TaskSpec(
        name="medical_triage",
        storyboard_dir=storyboard_dir,
        load_cases=load_cases,
        build_messages=lambda case: build_extraction_messages(case["prompt_input"]),
        validate=validate_storyboard,
        parse_response=parse_object_response,   # object: {self_diagnosis, storyboard}
        assemble_record=assemble,
        use_json_object=True,
        balance_field="true_urgency",
        balance_values=TIERS,
    )


def _social_judgment_spec() -> TaskSpec:
    from bayesbench.social_judgment.storyboard_prompt import build_extraction_messages, validate_storyboard
    from bayesbench.social_judgment.data.load_aita import load_aita_dataset

    storyboard_dir = PKG_ROOT / "social_judgment" / "storyboards"

    def load_cases(args) -> dict:
        posts = load_aita_dataset(mode="binary", max_samples=args.pool_size, seed=args.seed)
        cases = {}
        for p in posts:
            pid = str(p["id"])
            cases[pid] = {
                "id": pid,
                "prompt_input": {"title": p["title"], "text": p["text"]},
                "meta": {"id": pid, "title": p["title"], "text": p["text"],
                         "verdict": p["verdict"], "is_yta": p["is_yta"]},
                "balance_key": p["verdict"],
            }
        return cases

    def assemble(case, aspects, extra) -> dict:
        m = case["meta"]
        return {
            "id": m["id"], "title": m["title"], "text": m["text"],
            "verdict": m["verdict"], "is_yta": m["is_yta"],
            "storyboard": aspects,
        }

    return TaskSpec(
        name="social_judgment",
        storyboard_dir=storyboard_dir,
        load_cases=load_cases,
        build_messages=lambda case: build_extraction_messages(case["prompt_input"]),
        validate=validate_storyboard,
        parse_response=parse_array_response,    # bare array of aspects
        assemble_record=assemble,
        use_json_object=False,                  # array isn't a JSON object
        balance_field=None,                     # AITA ships a natural sample, not balanced
    )


TASKS = {
    "medical_triage": _medical_triage_spec,
    "social_judgment": _social_judgment_spec,
}


if __name__ == "__main__":
    main()
