#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fuzzing study plots (v3) for code benchmarks (e.g., BigO-Bench, HumanEval):

- SCTD + DCTD boxes with per-problem colored points (red↑ green↓ vs fuzz0).
- Paired delta violins using **fuzz0 - fuzzX** ("0-x change") for SCTD/DCTD.
- Failure/pass rate per condition.
- Logprob spread over ALL solutions (pass + fail).
- Naive entropy over 5 sampled solutions per problem (proxy UQ).

Inputs (3 conditions, same order):
  --sctd-csvs  sctd_fuzz0.csv sctd_fuzz0.5.csv sctd_fuzz0.9.csv
  --dctd-csvs  dctd_fuzz0.csv dctd_fuzz0.5.csv dctd_fuzz0.9.csv
  --jsonls     unittested_bigo_wf.jsonl unittested_bigo_f_0.5.jsonl unittested_bigo_f_0.9.jsonl

Outputs in --out-dir.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, DefaultDict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Metric CSV loading
# -------------------------
def load_metric_csv(path: Path, jsd_pct_key, jsd_raw_key, tau_pct_key, tau_raw_key):
    """
    Generic loader for SCTD/DCTD CSVs.

    Returns:
      pids: List[str]
      jsd_pct: List[float]
      tau_pct: List[float]
    """
    pids, jsd_pct, tau_pct = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row.get("problem_id", "")).strip()
            if not pid:
                continue

            def _read_pct(pct_key, raw_key):
                if row.get(pct_key) not in ("", None, "NaN", "nan"):
                    try:
                        return float(row[pct_key])
                    except Exception:
                        return None
                if row.get(raw_key) not in ("", None, "NaN", "nan"):
                    try:
                        return 100.0 * float(row[raw_key])
                    except Exception:
                        return None
                return None

            jsd = _read_pct(jsd_pct_key, jsd_raw_key)
            tau = _read_pct(tau_pct_key, tau_raw_key)
            if jsd is None or tau is None:
                continue

            pids.append(pid)
            jsd_pct.append(jsd)
            tau_pct.append(tau)

    return pids, jsd_pct, tau_pct


def _make_pid_to_val(pids: List[str], vals: List[float]) -> Dict[str, float]:
    return {pid: v for pid, v in zip(pids, vals)}


def plot_box_with_colored_points(
    values_by_label,
    pids_by_label,
    baseline_label,
    ylabel,
    title,
    out_path,
    out_map_dir,
    map_prefix,
    jitter=0.08,
    label_fontsize=6,
):
    """
    Boxplot + per-problem points.
    - Baseline (fuzz0) points black.
    - Other labels: red if increased vs baseline for that pid, green if decreased.
    - Saves point_id_map_<map_prefix>_<label>.csv
    """
    labels = list(values_by_label.keys())
    base_map = _make_pid_to_val(
        pids_by_label[baseline_label], values_by_label[baseline_label]
    )

    data, pid_lists = [], []
    for lab in labels:
        vals, pids = values_by_label[lab], pids_by_label[lab]
        cleaned_vals, cleaned_pids = [], []
        for v, pid in zip(vals, pids):
            if isinstance(v, (int, float)) and not math.isnan(v):
                cleaned_vals.append(v)
                cleaned_pids.append(pid)
        data.append(cleaned_vals)
        pid_lists.append(cleaned_pids)

    plt.figure(figsize=(8.8, 5.0))
    plt.boxplot(data, labels=labels, showfliers=False)

    for i, (lab, vals, pids) in enumerate(zip(labels, data, pid_lists), start=1):
        sort_idx = np.argsort(pids)
        pids_sorted = [pids[j] for j in sort_idx]
        vals_sorted = [vals[j] for j in sort_idx]

        map_path = out_map_dir / f"point_id_map_{map_prefix}_{lab}.csv"
        with map_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["point_id", "problem_id"])
            w.writeheader()
            for k, pid in enumerate(pids_sorted, start=1):
                w.writerow({"point_id": k, "problem_id": pid})

        xs = np.random.normal(loc=i, scale=jitter, size=len(vals_sorted))

        colors = []
        for pid, v in zip(pids_sorted, vals_sorted):
            if lab == baseline_label:
                colors.append("black")
            else:
                b = base_map.get(pid)
                if b is None or math.isnan(b):
                    colors.append("gray")
                elif v > b:
                    colors.append("red")
                elif v < b:
                    colors.append("green")
                else:
                    colors.append("gray")

        plt.scatter(xs, vals_sorted, s=14, alpha=0.8, c=colors)

        for k, (x, y) in enumerate(zip(xs, vals_sorted), start=1):
            plt.text(
                x,
                y,
                str(k),
                fontsize=label_fontsize,
                ha="center",
                va="bottom",
                alpha=0.85,
            )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_delta_violin(
    values_by_label,
    pids_by_label,
    baseline_label,
    ylabel,
    title,
    out_path,
):
    """
    For each non-baseline condition:
      delta(pid) = baseline - condition  (fuzz0 - fuzzX)
    Plot violins of deltas.

    X-ticks labeled as "fuzz0-fuzzX".
    """
    labels = list(values_by_label.keys())
    base_map = _make_pid_to_val(
        pids_by_label[baseline_label], values_by_label[baseline_label]
    )

    delta_data = []
    delta_labels = []
    for lab in labels:
        if lab == baseline_label:
            continue

        deltas = []
        for pid, v in zip(pids_by_label[lab], values_by_label[lab]):
            b = base_map.get(pid)
            if b is None or math.isnan(b) or math.isnan(v):
                continue
            deltas.append(b - v)  # 0 - x change

        delta_data.append(deltas)
        delta_labels.append(f"{baseline_label}-{lab}")

    plt.figure(figsize=(8.8, 5.0))
    plt.violinplot(delta_data, showmeans=True, showextrema=False)
    plt.axhline(0, linestyle="--", linewidth=1.0)
    plt.xticks(range(1, len(delta_labels) + 1), delta_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -------------------------
# Pass/fail + logprobs
# -------------------------
def is_row_pass(d: dict) -> bool:
    if str(d.get("success", "")).lower() == "pass":
        return True
    if str(d.get("status", "")).lower() == "pass":
        return True
    return False


def load_norm_seq_logprobs_by_problem(jsonl_path: Path) -> DefaultDict[str, List[float]]:
    """
    ALL solutions (pass + fail):
      norm_seq_lp = mean ln(p_chosen_tokens)
    """
    by_pid: DefaultDict[str, List[float]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue

            pid = str(d.get("problem_id", "")).strip()
            if not pid:
                continue

            tp = d.get("token_probs")
            if not tp or not isinstance(tp, list):
                continue

            lps = []
            for obj in tp:
                if not isinstance(obj, dict):
                    continue
                p = obj.get("prob")
                try:
                    p = float(p)
                except Exception:
                    continue
                if p <= 0 or math.isnan(p):
                    continue
                lps.append(math.log(p))

            if lps:
                by_pid[pid].append(float(np.mean(lps)))

    return by_pid


def compute_spread_stats(by_pid):
    stats = {}
    for pid, vals in by_pid.items():
        v = np.array([x for x in vals if not math.isnan(x)], dtype=float)
        if len(v) == 0:
            continue
        std = float(np.std(v, ddof=0))
        q75, q25 = np.percentile(v, [75, 25])
        iqr = float(q75 - q25)
        mean = float(np.mean(v))
        stats[pid] = {"std": std, "iqr": iqr, "mean": mean, "n": len(v)}
    return stats


def compute_naive_entropy(by_pid):
    """
    Naive entropy over the 5 sampled solutions per problem.
    Uses softmax over mean_logp as a proxy weight.
    """
    ent = {}
    for pid, vals in by_pid.items():
        v = np.array(vals, dtype=float)
        if len(v) < 2:
            continue
        m = np.max(v)
        w = np.exp(v - m)
        p = w / np.sum(w)
        h = float(-np.sum(p * np.log(p + 1e-12)))
        ent[pid] = h
    return ent


def load_passrate_by_problem(jsonl_path: Path):
    by_pid = defaultdict(lambda: {"pass": 0, "total": 0})
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("problem_id", "")).strip()
            if not pid:
                continue
            by_pid[pid]["total"] += 1
            if is_row_pass(d):
                by_pid[pid]["pass"] += 1
    return {
        pid: v["pass"] / v["total"]
        for pid, v in by_pid.items()
        if v["total"] > 0
    }


def plot_spread_trajectories(
    spreads_by_label, labels, x_vals, ylabel, title, out_path
):
    pid_sets = [set(spreads_by_label[l].keys()) for l in labels]
    common = sorted(set.intersection(*pid_sets)) if pid_sets else []

    plt.figure(figsize=(8.8, 5.0))

    for pid in common:
        ys = [spreads_by_label[l][pid] for l in labels]
        plt.plot(x_vals, ys, alpha=0.35, linewidth=1.0)

    mean_ys = []
    for l in labels:
        vals = [spreads_by_label[l][pid] for pid in common]
        mean_ys.append(float(np.mean(vals)) if vals else float("nan"))
    plt.plot(x_vals, mean_ys, linewidth=2.5, label="mean", color="black")

    plt.xticks(x_vals, labels)
    plt.xlabel("Fuzzing condition")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_passrate_bar(passrates_by_label, labels, out_path):
    means = [np.mean(list(passrates_by_label[l].values())) for l in labels]
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, means)
    plt.ylabel("Pass rate (fraction passing public tests)")
    plt.title("Mean public-test pass rate by fuzzing condition")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bigobench", "humaneval"], default="bigobench")
    ap.add_argument("--sctd-csvs", nargs=3, required=True)
    ap.add_argument("--dctd-csvs", nargs=3, required=True)
    ap.add_argument("--jsonls", nargs=3, required=True)
    ap.add_argument("--labels", nargs=3, default=["fuzz0", "fuzz0.5", "fuzz0.9"])
    ap.add_argument("--x-vals", nargs=3, type=float, default=[0.0, 0.5, 0.9])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dataset = args.dataset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels
    x_vals = args.x_vals
    baseline = labels[0]

    # ---- SCTD ----
    sctd_jsd_by, sctd_tau_by, sctd_pids_by = {}, {}, {}
    for lab, path in zip(labels, args.sctd_csvs):
        pids, jsd, tau = load_metric_csv(
            Path(path),
            "SCTD_JSD_pct", "SCTD_JSD",
            "SCTD_TAU_pct", "SCTD_TAU",
        )
        sctd_pids_by[lab] = pids
        sctd_jsd_by[lab] = jsd
        sctd_tau_by[lab] = tau

    plot_box_with_colored_points(
        sctd_jsd_by, sctd_pids_by, baseline,
        "SCTD_JSD (%)",
        "SCTD_JSD across conditions (red↑ green↓ vs fuzz0)",
        out_dir / "sctd_jsd_box_points_colored.png",
        out_dir, "sctd_jsd",
    )
    plot_box_with_colored_points(
        sctd_tau_by, sctd_pids_by, baseline,
        "SCTD_TAU (%)",
        "SCTD_TAU across conditions (red↑ green↓ vs fuzz0)",
        out_dir / "sctd_tau_box_points_colored.png",
        out_dir, "sctd_tau",
    )
    plot_delta_violin(
        sctd_jsd_by, sctd_pids_by, baseline,
        "Δ SCTD_JSD (%) = fuzz0 − fuzzX",
        "Per-problem Δ SCTD_JSD (fuzz0 − fuzzX)",
        out_dir / "sctd_jsd_delta_violin_fuzz0_minus_fuzzx.png",
    )
    plot_delta_violin(
        sctd_tau_by, sctd_pids_by, baseline,
        "Δ SCTD_TAU (%) = fuzz0 − fuzzX",
        "Per-problem Δ SCTD_TAU (fuzz0 − fuzzX)",
        out_dir / "sctd_tau_delta_violin_fuzz0_minus_fuzzx.png",
    )

    # ---- DCTD ----
    dctd_jsd_by, dctd_tau_by, dctd_pids_by = {}, {}, {}
    for lab, path in zip(labels, args.dctd_csvs):
        pids, jsd, tau = load_metric_csv(
            Path(path),
            "DCTD_JSD_pct", "DCTD_JSD",
            "DCTD_TAU_pct", "DCTD_TAU",
        )
        dctd_pids_by[lab] = pids
        dctd_jsd_by[lab] = jsd
        dctd_tau_by[lab] = tau

    plot_box_with_colored_points(
        dctd_jsd_by, dctd_pids_by, baseline,
        "DCTD_JSD (%)",
        "DCTD_JSD across conditions (red↑ green↓ vs fuzz0)",
        out_dir / "dctd_jsd_box_points_colored.png",
        out_dir, "dctd_jsd",
    )
    plot_box_with_colored_points(
        dctd_tau_by, dctd_pids_by, baseline,
        "DCTD_TAU (%)",
        "DCTD_TAU across conditions (red↑ green↓ vs fuzz0)",
        out_dir / "dctd_tau_box_points_colored.png",
        out_dir, "dctd_tau",
    )
    plot_delta_violin(
        dctd_jsd_by, dctd_pids_by, baseline,
        "Δ DCTD_JSD (%) = fuzz0 − fuzzX",
        "Per-problem Δ DCTD_JSD (fuzz0 − fuzzX)",
        out_dir / "dctd_jsd_delta_violin_fuzz0_minus_fuzzx.png",
    )
    plot_delta_violin(
        dctd_tau_by, dctd_pids_by, baseline,
        "Δ DCTD_TAU (%) = fuzz0 − fuzzX",
        "Per-problem Δ DCTD_TAU (fuzz0 − fuzzX)",
        out_dir / "dctd_tau_delta_violin_fuzz0_minus_fuzzx.png",
    )

    # ---- Pass rate ----
    passrates_by = {}
    for lab, jp in zip(labels, args.jsonls):
        passrates_by[lab] = load_passrate_by_problem(Path(jp))
        print(
            f"[{lab}] mean pass rate = {np.mean(list(passrates_by[lab].values())):.3f}"
        )
    plot_passrate_bar(passrates_by, labels, out_dir / "passrate_bar.png")

    # ---- Logprob stats (ALL solutions) ----
    spread_std_by, spread_iqr_by, spread_mean_by, ent_by = {}, {}, {}, {}
    spread_rows = []
    for lab, jp in zip(labels, args.jsonls):
        by_pid = load_norm_seq_logprobs_by_problem(Path(jp))
        stats = compute_spread_stats(by_pid)
        ent = compute_naive_entropy(by_pid)

        spread_std_by[lab] = {pid: s["std"] for pid, s in stats.items()}
        spread_iqr_by[lab] = {pid: s["iqr"] for pid, s in stats.items()}
        spread_mean_by[lab] = {pid: s["mean"] for pid, s in stats.items()}
        ent_by[lab] = ent

        for pid, s in stats.items():
            spread_rows.append(
                dict(
                    condition=lab,
                    problem_id=pid,
                    mean_norm_seq_lp=s["mean"],
                    std_norm_seq_lp=s["std"],
                    iqr_norm_seq_lp=s["iqr"],
                    naive_entropy=ent.get(pid, float("nan")),
                    n_completions=s["n"],
                )
            )

    plot_spread_trajectories(
        spread_std_by, labels, x_vals,
        "Std of norm seq ln(p) across 5 solutions",
        "Intra-problem logprob spread (STD)",
        out_dir / "logprob_spread_std_traj.png",
    )
    plot_spread_trajectories(
        spread_iqr_by, labels, x_vals,
        "IQR of norm seq ln(p) across 5 solutions",
        "Intra-problem logprob spread (IQR)",
        out_dir / "logprob_spread_iqr_traj.png",
    )
    plot_spread_trajectories(
        spread_mean_by, labels, x_vals,
        "Mean norm seq ln(p)",
        "Mean normalized sequence logprob per problem",
        out_dir / "logprob_mean_traj.png",
    )
    plot_spread_trajectories(
        ent_by, labels, x_vals,
        "Naive entropy over 5 solutions",
        "Naive entropy uncertainty per problem",
        out_dir / "logprob_naive_entropy_traj.png",
    )

    out_csv = out_dir / "per_problem_logprob_spread.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "problem_id",
                "mean_norm_seq_lp",
                "std_norm_seq_lp",
                "iqr_norm_seq_lp",
                "naive_entropy",
                "n_completions",
            ],
        )
        w.writeheader()
        w.writerows(spread_rows)

    print("\nWrote all plots to:", out_dir)


if __name__ == "__main__":
    main()