#!/usr/bin/env python3
"""
Analyze recommender-system experiment results across all result JSON files.
Computes summary statistics grouped by (model, condition, pop_info) and per-model.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

# Default to the experiments/ directory next to this file; override with the
# CF_EXPERIMENTS_DIR environment variable or a path argument on the CLI.
EXPERIMENTS_DIR = os.environ.get(
    "CF_EXPERIMENTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"),
)

# ── Metrics to extract ──────────────────────────────────────────────────────
METRIC_KEYS = [
    "mae_from_bayesian",
    "cross_item_transfer_score",
    "genre_transfer_score",
    "correlation_with_bayesian",
    "type_inference_correct",
    "mean_rating_mass",
    "mean_scale_bias",
    "mean_update_magnitude",
    "prior_expected_rating",
    "mean_kl_divergence",
    "mean_wasserstein",
    # Type elicitation metrics
    "type_posterior_kl",
    "type_accuracy",
    "cot_type_accuracy",
    "conditioned_mae_from_bayesian",
    "conditioning_lift",
    # CoT follow-up MCQ distributional metrics
    "cot_type_posterior_kl",
    "mean_cot_type_mass",
    "mean_cot_type_scale_bias",
]

# ── Model display order (small → large) ────────────────────────────────────
MODEL_ORDER = ["llama3b", "qwen3b", "qwen7b", "llama8b", "qwen14b", "gptoss20b", "gptoss120b"]
CONDITION_ORDER = ["single_turn", "multi_turn_minimal", "multi_turn_actual"]
POP_ORDER = ["zero_shot", "explicit_types", "anonymized"]


def load_all_results(directory):
    """Load every JSON file in *directory* and return a list of (config, metrics) tuples."""
    records = []
    skipped = 0
    for fname in os.listdir(directory):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            with open(fpath) as fh:
                data = json.load(fh)
            cfg = data["config"]
            met = data["metrics"]
            records.append((cfg, met))
        except Exception as e:
            skipped += 1
    return records, skipped


def group_records(records):
    """Group records by (model, condition, pop_info) and by model alone."""
    by_triple = defaultdict(lambda: defaultdict(list))
    by_model  = defaultdict(lambda: defaultdict(list))

    for cfg, met in records:
        model = cfg["model_name"]
        cond  = cfg["condition"]
        pop   = cfg["pop_info"]
        key   = (model, cond, pop)

        for mk in METRIC_KEYS:
            val = met.get(mk)
            if val is not None:
                # type_inference_correct is bool → convert to 0/1
                if mk == "type_inference_correct":
                    val = int(val)
                by_triple[key][mk].append(float(val))
                by_model[model][mk].append(float(val))

    return by_triple, by_model


def fmt(val, width=8):
    """Format a float nicely."""
    if val is None:
        return " " * width
    if abs(val) < 0.001:
        return f"{val:>{width}.4f}"
    return f"{val:>{width}.3f}"


def print_divider(width=160):
    print("-" * width)


def print_grouped_table(by_triple):
    """Print a detailed table grouped by (model, condition, pop_info)."""
    # Determine which triples exist
    existing_models = sorted(set(k[0] for k in by_triple), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)
    existing_conds  = sorted(set(k[1] for k in by_triple), key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 999)
    existing_pops   = sorted(set(k[2] for k in by_triple), key=lambda p: POP_ORDER.index(p) if p in POP_ORDER else 999)

    header_cols = [
        ("MAE(Bayes)", 18),
        ("CrossItem", 18),
        ("Corr(Bayes)", 10),
        ("TypeAcc", 8),
        ("RatingMass", 10),
        ("ScaleBias", 10),
        ("UpdateMag", 10),
        ("TypeElicAcc", 10),
        ("CoTAcc", 8),
        ("CondLift", 8),
        ("CoTMass", 8),
        ("CoTKL", 8),
        ("N", 5),
    ]

    header_line = f"{'Model':<14} {'Condition':<22} {'PopInfo':<18}"
    for name, w in header_cols:
        header_line += f" {name:>{w}}"

    print()
    print("=" * len(header_line))
    print("  DETAILED RESULTS: grouped by (Model, Condition, PopInfo)")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    for model in existing_models:
        first_model_row = True
        for cond in existing_conds:
            for pop in existing_pops:
                key = (model, cond, pop)
                if key not in by_triple:
                    continue
                d = by_triple[key]
                n = len(d.get("mae_from_bayesian", []))
                if n == 0:
                    continue

                mae_mean = np.mean(d["mae_from_bayesian"])
                mae_std  = np.std(d["mae_from_bayesian"])
                cit_mean = np.mean(d["cross_item_transfer_score"])
                cit_std  = np.std(d["cross_item_transfer_score"])
                corr_mean = np.mean(d["correlation_with_bayesian"])
                tacc     = np.mean(d["type_inference_correct"])
                rmass    = np.mean(d["mean_rating_mass"])
                sbias    = np.mean(d["mean_scale_bias"])
                umag     = np.mean(d["mean_update_magnitude"])

                # Type elicitation metrics (may be absent for zero_shot)
                te_vals = d.get("type_accuracy", [])
                te_acc = np.mean(te_vals) if te_vals else None
                cot_vals = d.get("cot_type_accuracy", [])
                cot_acc = np.mean(cot_vals) if cot_vals else None
                cl_vals = d.get("conditioning_lift", [])
                cond_lift = np.mean(cl_vals) if cl_vals else None
                cot_mass_vals = d.get("mean_cot_type_mass", [])
                cot_mass = np.mean(cot_mass_vals) if cot_mass_vals else None
                cot_kl_vals = d.get("cot_type_posterior_kl", [])
                cot_kl = np.mean(cot_kl_vals) if cot_kl_vals else None

                model_str = model if first_model_row else ""
                first_model_row = False

                row = f"{model_str:<14} {cond:<22} {pop:<18}"
                row += f" {mae_mean:>7.3f}+-{mae_std:<7.3f}"
                row += f"  {cit_mean:>7.3f}+-{cit_std:<7.3f}"
                row += f"  {corr_mean:>8.3f}"
                row += f"  {tacc:>6.1%}"
                row += f"  {rmass:>8.3f}"
                row += f"  {sbias:>+8.3f}"
                row += f"  {umag:>8.3f}"
                row += f"  {te_acc:>8.1%}" if te_acc is not None else f"  {'N/A':>10}"
                row += f"  {cot_acc:>6.1%}" if cot_acc is not None else f"  {'N/A':>8}"
                row += f"  {cond_lift:>+6.3f}" if cond_lift is not None else f"  {'N/A':>8}"
                row += f"  {cot_mass:>6.3f}" if cot_mass is not None else f"  {'N/A':>8}"
                row += f"  {cot_kl:>6.3f}" if cot_kl is not None else f"  {'N/A':>8}"
                row += f"  {n:>4d}"
                print(row)
        print("-" * len(header_line))

    print()


def print_model_table(by_model):
    """Print a summary table with one row per model."""
    existing_models = sorted(by_model.keys(), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)

    header = (f"{'Model':<14} {'MAE(Bayes)':>18}  {'CrossItem':>18}  {'Corr(B)':>8}"
              f"  {'TypeAcc':>8}  {'RtgMass':>8}  {'ScaleBias':>10}  {'UpdMag':>8}  {'N':>5}")
    print()
    print("=" * len(header))
    print("  PER-MODEL AVERAGES (across all conditions & pop_info)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model in existing_models:
        d = by_model[model]
        n = len(d.get("mae_from_bayesian", []))
        if n == 0:
            continue

        mae_mean = np.mean(d["mae_from_bayesian"])
        mae_std  = np.std(d["mae_from_bayesian"])
        cit_mean = np.mean(d["cross_item_transfer_score"])
        cit_std  = np.std(d["cross_item_transfer_score"])
        corr_mean = np.mean(d["correlation_with_bayesian"])
        tacc     = np.mean(d["type_inference_correct"])
        rmass    = np.mean(d["mean_rating_mass"])
        sbias    = np.mean(d["mean_scale_bias"])
        umag     = np.mean(d["mean_update_magnitude"])

        row = f"{model:<14} {mae_mean:>7.3f}+-{mae_std:<7.3f}"
        row += f"  {cit_mean:>7.3f}+-{cit_std:<7.3f}"
        row += f"  {corr_mean:>8.3f}"
        row += f"  {tacc:>6.1%}"
        row += f"  {rmass:>8.3f}"
        row += f"  {sbias:>+10.3f}"
        row += f"  {umag:>8.3f}"
        row += f"  {n:>5d}"
        print(row)

    print("-" * len(header))
    print()


def print_condition_table(by_triple):
    """Print a summary table with one row per condition (across all models)."""
    by_cond = defaultdict(lambda: defaultdict(list))
    for (model, cond, pop), d in by_triple.items():
        for mk in METRIC_KEYS:
            if mk in d:
                by_cond[cond][mk].extend(d[mk])

    existing_conds = sorted(by_cond.keys(), key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 999)

    header = (f"{'Condition':<24} {'MAE(Bayes)':>18}  {'CrossItem':>18}  {'Corr(B)':>8}"
              f"  {'TypeAcc':>8}  {'RtgMass':>8}  {'ScaleBias':>10}  {'UpdMag':>8}  {'N':>5}")
    print()
    print("=" * len(header))
    print("  PER-CONDITION AVERAGES (across all models & pop_info)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cond in existing_conds:
        d = by_cond[cond]
        n = len(d.get("mae_from_bayesian", []))
        mae_mean = np.mean(d["mae_from_bayesian"])
        mae_std  = np.std(d["mae_from_bayesian"])
        cit_mean = np.mean(d["cross_item_transfer_score"])
        cit_std  = np.std(d["cross_item_transfer_score"])
        corr_mean = np.mean(d["correlation_with_bayesian"])
        tacc     = np.mean(d["type_inference_correct"])
        rmass    = np.mean(d["mean_rating_mass"])
        sbias    = np.mean(d["mean_scale_bias"])
        umag     = np.mean(d["mean_update_magnitude"])

        row = f"{cond:<24} {mae_mean:>7.3f}+-{mae_std:<7.3f}"
        row += f"  {cit_mean:>7.3f}+-{cit_std:<7.3f}"
        row += f"  {corr_mean:>8.3f}"
        row += f"  {tacc:>6.1%}"
        row += f"  {rmass:>8.3f}"
        row += f"  {sbias:>+10.3f}"
        row += f"  {umag:>8.3f}"
        row += f"  {n:>5d}"
        print(row)

    print("-" * len(header))
    print()


def print_popinfo_table(by_triple):
    """Print a summary table with one row per pop_info (across all models)."""
    by_pop = defaultdict(lambda: defaultdict(list))
    for (model, cond, pop), d in by_triple.items():
        for mk in METRIC_KEYS:
            if mk in d:
                by_pop[pop][mk].extend(d[mk])

    existing_pops = sorted(by_pop.keys(), key=lambda p: POP_ORDER.index(p) if p in POP_ORDER else 999)

    header = (f"{'PopInfo':<24} {'MAE(Bayes)':>18}  {'CrossItem':>18}  {'Corr(B)':>8}"
              f"  {'TypeAcc':>8}  {'RtgMass':>8}  {'ScaleBias':>10}  {'UpdMag':>8}  {'N':>5}")
    print()
    print("=" * len(header))
    print("  PER-POP_INFO AVERAGES (across all models & conditions)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for pop in existing_pops:
        d = by_pop[pop]
        n = len(d.get("mae_from_bayesian", []))
        mae_mean = np.mean(d["mae_from_bayesian"])
        mae_std  = np.std(d["mae_from_bayesian"])
        cit_mean = np.mean(d["cross_item_transfer_score"])
        cit_std  = np.std(d["cross_item_transfer_score"])
        corr_mean = np.mean(d["correlation_with_bayesian"])
        tacc     = np.mean(d["type_inference_correct"])
        rmass    = np.mean(d["mean_rating_mass"])
        sbias    = np.mean(d["mean_scale_bias"])
        umag     = np.mean(d["mean_update_magnitude"])

        row = f"{pop:<24} {mae_mean:>7.3f}+-{mae_std:<7.3f}"
        row += f"  {cit_mean:>7.3f}+-{cit_std:<7.3f}"
        row += f"  {corr_mean:>8.3f}"
        row += f"  {tacc:>6.1%}"
        row += f"  {rmass:>8.3f}"
        row += f"  {sbias:>+10.3f}"
        row += f"  {umag:>8.3f}"
        row += f"  {n:>5d}"
        print(row)

    print("-" * len(header))
    print()


def print_type_table(records):
    """Print a breakdown by true_type across all models, showing which types are easiest/hardest."""
    by_type = defaultdict(lambda: defaultdict(list))

    for cfg, met in records:
        true_type = cfg.get("true_type")
        if true_type is None:
            continue
        for mk in METRIC_KEYS:
            val = met.get(mk)
            if val is not None:
                if mk == "type_inference_correct":
                    val = int(val)
                by_type[true_type][mk].append(float(val))

    existing_types = sorted(by_type.keys())

    header = (f"{'Type':<8} {'MAE(Bayes)':>18}  {'CrossItem':>18}  {'GenreXfer':>18}"
              f"  {'Corr(B)':>8}  {'TypeAcc':>8}  {'KL':>8}  {'W1':>8}  {'N':>5}")
    print()
    print("=" * len(header))
    print("  PER-TYPE AVERAGES (profile-level analysis)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for t in existing_types:
        d = by_type[t]
        n = len(d.get("mae_from_bayesian", []))
        if n == 0:
            continue

        mae_mean = np.mean(d["mae_from_bayesian"])
        mae_std  = np.std(d["mae_from_bayesian"])
        cit_mean = np.mean(d["cross_item_transfer_score"])
        cit_std  = np.std(d["cross_item_transfer_score"])

        gt_vals = d.get("genre_transfer_score", [])
        gt_vals = [v for v in gt_vals if v is not None]
        gt_mean = np.mean(gt_vals) if gt_vals else float('nan')
        gt_std = np.std(gt_vals) if gt_vals else float('nan')

        corr_mean = np.mean(d["correlation_with_bayesian"])
        tacc     = np.mean(d["type_inference_correct"])

        kl_vals = d.get("mean_kl_divergence", [])
        kl_vals = [v for v in kl_vals if v is not None]
        kl_mean = np.mean(kl_vals) if kl_vals else float('nan')

        w1_vals = d.get("mean_wasserstein", [])
        w1_vals = [v for v in w1_vals if v is not None]
        w1_mean = np.mean(w1_vals) if w1_vals else float('nan')

        row = f"Type {t:<4} {mae_mean:>7.3f}+-{mae_std:<7.3f}"
        row += f"  {cit_mean:>7.3f}+-{cit_std:<7.3f}"
        if np.isnan(gt_mean):
            row += f"  {'N/A':>18}"
        else:
            row += f"  {gt_mean:>7.3f}+-{gt_std:<7.3f}"
        row += f"  {corr_mean:>8.3f}"
        row += f"  {tacc:>6.1%}"
        if np.isnan(kl_mean):
            row += f"  {'N/A':>8}"
        else:
            row += f"  {kl_mean:>8.4f}"
        if np.isnan(w1_mean):
            row += f"  {'N/A':>8}"
        else:
            row += f"  {w1_mean:>8.4f}"
        row += f"  {n:>5d}"
        print(row)

    print("-" * len(header))
    print()


def print_model_type_table(records):
    """Print per-(model, type) breakdown to see which models handle which types best."""
    by_model_type = defaultdict(lambda: defaultdict(list))

    for cfg, met in records:
        model = cfg["model_name"]
        true_type = cfg.get("true_type")
        if true_type is None:
            continue
        key = (model, true_type)
        for mk in ["mae_from_bayesian", "cross_item_transfer_score", "type_inference_correct"]:
            val = met.get(mk)
            if val is not None:
                if mk == "type_inference_correct":
                    val = int(val)
                by_model_type[key][mk].append(float(val))

    existing_models = sorted(set(k[0] for k in by_model_type),
                              key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)
    existing_types = sorted(set(k[1] for k in by_model_type))

    header = f"{'Model':<14}"
    for t in existing_types:
        header += f" {'Type'+str(t)+' MAE':>10} {'TypeAcc':>8}"
    print()
    print("=" * len(header))
    print("  PER-(MODEL, TYPE) BREAKDOWN")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model in existing_models:
        row = f"{model:<14}"
        for t in existing_types:
            key = (model, t)
            d = by_model_type.get(key, {})
            if d.get("mae_from_bayesian"):
                mae = np.mean(d["mae_from_bayesian"])
                tacc = np.mean(d["type_inference_correct"])
                row += f" {mae:>10.3f} {tacc:>6.1%}  "
            else:
                row += f" {'N/A':>10} {'N/A':>8}"
        print(row)

    print("-" * len(header))
    print()


def main():
    experiments_dir = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENTS_DIR
    print(f"Loading results from: {experiments_dir}")
    records, skipped = load_all_results(experiments_dir)
    print(f"Loaded {len(records)} result files ({skipped} skipped/errored).")

    by_triple, by_model = group_records(records)
    print(f"Found {len(by_triple)} unique (model, condition, pop_info) groups.")
    print(f"Found {len(by_model)} unique models.")

    # ── Print all tables ───────────────────────────────────────────────────
    print_model_table(by_model)
    print_condition_table(by_triple)
    print_popinfo_table(by_triple)
    print_type_table(records)
    print_model_type_table(records)
    print_grouped_table(by_triple)


if __name__ == "__main__":
    main()
