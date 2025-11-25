#!/usr/bin/env python3
"""
Bounded-spread metrics for collections of Python solutions
==========================================================

Metrics implemented
-------------------
1. **Mean Jensen–Shannon divergence** of *structure* and *cost* opcode
   distributions, combined with weight `ALPHA`.
2. **Normalised trace-of-covariance** (“tau”) of the same two families,
   also combined with `ALPHA`.

Both scores are ∈ [0, 1]; higher ⇒ **more spread / less stability**.

"""
from __future__ import annotations

import ast
import csv
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dis
import numpy as np

# --- START: Populated Opcode Time Complexity Dictionary (Updated for Python 3.12) ---
# Scoring System:
# Bracket 1: O(1), O(log n)     -> 1
# Bracket 2: O(n), O(n log n)   -> 10
# Bracket 3: O(n^2), O(n^3)     -> 100

opcode_time_complexity = {
    # --- Bracket 3: O(n^2) and O(n^3) ---
    "BINARY_MATRIX_MULTIPLY": 100,
    "INPLACE_MATRIX_MULTIPLY": 100,
    # --- Bracket 2: O(n) and O(n log n) ---
    "UNPACK_SEQUENCE": 10,
    "UNPACK_EX": 10,
    "BUILD_LIST": 10,
    "BUILD_TUPLE": 10,
    "BUILD_SET": 10,
    "BUILD_MAP": 10,
    "BUILD_STRING": 10,
    "LIST_EXTEND": 10,
    "SET_UPDATE": 10,
    "DICT_MERGE": 10,
    "DICT_UPDATE": 10,
    "FOR_ITER": 10,
    "CONTAINS_OP": 10,
    "BINARY_SLICE": 10,
    "STORE_SLICE": 10,
    "BINARY_OP": 10,
    "INSTRUMENTED_FOR_ITER": 10,
    # --- Bracket 1: O(1) and O(log n) ---
    "CACHE": 1,
    "POP_TOP": 1,
    "PUSH_NULL": 1,
    "INTERPRETER_EXIT": 1,
    "END_FOR": 1,
    "END_SEND": 1,
    "NOP": 1,
    "UNARY_NEGATIVE": 1,
    "UNARY_NOT": 1,
    "UNARY_INVERT": 1,
    "BINARY_SUBSCR": 1,
    "GET_LEN": 1,
    "MATCH_MAPPING": 1,
    "MATCH_SEQUENCE": 1,
    "MATCH_KEYS": 1,
    "PUSH_EXC_INFO": 1,
    "CHECK_EXC_MATCH": 1,
    "CHECK_EG_MATCH": 1,
    "WITH_EXCEPT_START": 1,
    "GET_AITER": 1,
    "GET_ANEXT": 1,
    "BEFORE_ASYNC_WITH": 1,
    "BEFORE_WITH": 1,
    "END_ASYNC_FOR": 1,
    "CLEANUP_THROW": 1,
    "STORE_SUBSCR": 1,
    "DELETE_SUBSCR": 1,
    "GET_ITER": 1,
    "GET_YIELD_FROM_ITER": 1,
    "LOAD_BUILD_CLASS": 1,
    "LOAD_ASSERTION_ERROR": 1,
    "RETURN_GENERATOR": 1,
    "RETURN_VALUE": 1,
    "SETUP_ANNOTATIONS": 1,
    "LOAD_LOCALS": 1,
    "POP_EXCEPT": 1,
    "STORE_NAME": 1,
    "DELETE_NAME": 1,
    "STORE_ATTR": 1,
    "DELETE_ATTR": 1,
    "STORE_GLOBAL": 1,
    "DELETE_GLOBAL": 1,
    "SWAP": 1,
    "LOAD_CONST": 1,
    "LOAD_NAME": 1,
    "LOAD_ATTR": 1,
    "COMPARE_OP": 1,
    "IMPORT_NAME": 1,
    "IMPORT_FROM": 1,
    "JUMP_FORWARD": 1,
    "POP_JUMP_IF_FALSE": 1,
    "POP_JUMP_IF_TRUE": 1,
    "LOAD_GLOBAL": 1,
    "IS_OP": 1,
    "RERAISE": 1,
    "COPY": 1,
    "RETURN_CONST": 1,
    "SEND": 1,
    "LOAD_FAST": 1,
    "STORE_FAST": 1,
    "DELETE_FAST": 1,
    "LOAD_FAST_CHECK": 1,
    "POP_JUMP_IF_NOT_NONE": 1,
    "POP_JUMP_IF_NONE": 1,
    "RAISE_VARARGS": 1,
    "GET_AWAITABLE": 1,
    "MAKE_FUNCTION": 1,
    "BUILD_SLICE": 1,
    "JUMP_BACKWARD_NO_INTERRUPT": 1,
    "MAKE_CELL": 1,
    "LOAD_CLOSURE": 1,
    "LOAD_DEREF": 1,
    "STORE_DEREF": 1,
    "DELETE_DEREF": 1,
    "JUMP_BACKWARD": 1,
    "LOAD_SUPER_ATTR": 1,
    "CALL_FUNCTION_EX": 1,
    "LOAD_FAST_AND_CLEAR": 1,
    "EXTENDED_ARG": 1,
    "LIST_APPEND": 1,
    "SET_ADD": 1,
    "MAP_ADD": 1,
    "COPY_FREE_VARS": 1,
    "YIELD_VALUE": 1,
    "RESUME": 1,
    "MATCH_CLASS": 1,
    "FORMAT_VALUE": 1,
    "BUILD_CONST_KEY_MAP": 1,
    "CALL": 1,
    "KW_NAMES": 1,
    "CALL_INTRINSIC_1": 1,
    "CALL_INTRINSIC_2": 1,
    "LOAD_FROM_DICT_OR_GLOBALS": 1,
    "LOAD_FROM_DICT_OR_DEREF": 1,
    "SETUP_FINALLY": 1,
    "SETUP_CLEANUP": 1,
    "SETUP_WITH": 1,
    "POP_BLOCK": 1,
    "JUMP": 1,
    "JUMP_NO_INTERRUPT": 1,
    "LOAD_METHOD": 1,
    "LOAD_SUPER_METHOD": 1,
    "LOAD_ZERO_SUPER_METHOD": 1,
    "LOAD_ZERO_SUPER_ATTR": 1,
    "STORE_FAST_MAYBE_NULL": 1,
    "INSTRUMENTED_LOAD_SUPER_ATTR": 1,
    "INSTRUMENTED_POP_JUMP_IF_NONE": 1,
    "INSTRUMENTED_POP_JUMP_IF_NOT_NONE": 1,
    "INSTRUMENTED_RESUME": 1,
    "INSTRUMENTED_CALL": 1,
    "INSTRUMENTED_RETURN_VALUE": 1,
    "INSTRUMENTED_YIELD_VALUE": 1,
    "INSTRUMENTED_CALL_FUNCTION_EX": 1,
    "INSTRUMENTED_JUMP_FORWARD": 1,
    "INSTRUMENTED_JUMP_BACKWARD": 1,
    "INSTRUMENTED_RETURN_CONST": 1,
    "INSTRUMENTED_POP_JUMP_IF_FALSE": 1,
    "INSTRUMENTED_POP_JUMP_IF_TRUE": 1,
    "INSTRUMENTED_END_FOR": 1,
    "INSTRUMENTED_END_SEND": 1,
}
# --- END: Populated Opcode Time Complexity Dictionary ---

# If an opcode is missing, default complexity is 1
DEFAULT_WEIGHT = 1

# Jensen–Shannon mixture weight between structure and cost distributions
ALPHA = 0.5

# Epsilon for probability smoothing
EPS = 1e-9

# ──────────────────────────────────────────────────────────────── UTILITIES ────


def compile_source_string(src: str, filename: str = "<string>"):
    """Parse a source code string into an AST and compile to a code object."""
    tree = ast.parse(src, filename=filename)
    return compile(tree, filename=filename, mode="exec")


def collect_unique_opcodes(code_objects: Sequence):
    """Return a sorted list of opcode integers appearing in the code objects."""
    seen: set[int] = set()
    queue = list(code_objects)
    visited = {id(co) for co in queue}

    while queue:
        co = queue.pop(0)
        for const in co.co_consts:
            if hasattr(const, "co_code") and id(const) not in visited:
                queue.append(const)
                visited.add(id(const))
        for instr in dis.get_instructions(co):
            seen.add(instr.opcode)

    return sorted(seen)


def build_opcode_index_map(opcodes: Sequence[int]) -> Dict[int, int]:
    """Map opcode integer → contiguous index 0…d-1."""
    return {op: i for i, op in enumerate(opcodes)}


def count_opcodes(code_obj, index_map: Dict[int, int], d: int) -> np.ndarray:
    """Return length-d vector of opcode counts for *one* solution."""
    vec = np.zeros(d, dtype=np.int32)
    queue = [code_obj]
    visited = {id(code_obj)}

    while queue:
        co = queue.pop(0)
        for const in co.co_consts:
            if hasattr(const, "co_code") and id(const) not in visited:
                queue.append(const)
                visited.add(id(const))
        for instr in dis.get_instructions(co):
            idx = index_map.get(instr.opcode)
            if idx is not None:
                vec[idx] += 1
    return vec


# ───────────────────────────────────────────────────────── MATHS / METRICS ─────


def _pmf_from_counts(
    counts: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert raw opcode counts to a probability mass function (PMF).

    If `weights` is provided, it computes a cost-weighted PMF:
        q_i = (w_i * c_i) / Σ(w_j * c_j)
    Otherwise, it computes a structural PMF:
        p_i = c_i / Σ(c_j)

    Args:
        counts: A 1D numpy array of raw opcode counts.
        weights: An optional 1D numpy array of the same dimension as `counts`
                 containing cost weights for each opcode.

    Returns:
        The resulting probability mass function as a 1D numpy array.
    """
    if weights is None:
        num = counts.astype(float)
    else:
        num = weights * counts
    den = num.sum()
    if den == 0:
        # This case is unlikely with valid compiled code but is handled for safety.
        num = np.full_like(num, 1.0)
        den = num.sum()
    pmf = num / den
    # Smooth probabilities to avoid issues with log(0) in divergence calculations.
    pmf = np.clip(pmf, EPS, 1.0)
    pmf /= pmf.sum()
    return pmf


def js_divergence(p: np.ndarray, q: np.ndarray, pi: float = 0.5) -> float:
    """
    Calculates the Jensen-Shannon Divergence (JSD) between two distributions.

    This is a generalized JSD, controlled by the weight parameter `pi`. The standard,
    symmetric JSD (bounded in [0, 1]) is recovered with the default `pi = 0.5`.

    Args:
        p: A probability mass function (PMF) as a numpy array.
        q: Another PMF of the same dimension as p.
        pi: The weight for the mixture distribution `m = pi*p + (1-pi)*q`.
            Defaults to 0.5.

    Returns:
        The JSD score (using base-2 logarithm).
    """
    m = pi * p + (1 - pi) * q
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return pi * kl_pm + (1 - pi) * kl_qm


def mean_pairwise_js(values: List[np.ndarray]) -> float:
    """
    Computes the average pairwise Jensen-Shannon Divergence for a list of PMFs.

    This corresponds to the formula term: 2/(m*(m-1)) * Σ JSD(p_s, p_t)
    for all pairs s < t.

    Args:
        values: A list of `m` probability mass functions (PMFs).

    Returns:
        The mean pairwise JSD score, or 0.0 if there are fewer than 2 PMFs.
    """
    if len(values) < 2:
        return 0.0
    # The default pi=0.5 in js_divergence ensures the standard, symmetric JSD is used.
    pairs = itertools.combinations(values, 2)
    return np.mean([js_divergence(a, b) for a, b in pairs])


def normalised_cov_trace(matrix: np.ndarray) -> float:
    """
    Computes the normalized total variance (τ), a covariance-based metric.

    The formula is:
        τ(X) = tr(Σ) / (1 - ||μ||²)
    where μ is the mean PMF vector and Σ is the covariance matrix. This score
    is bounded in [0, 1] and measures the dispersion of the PMFs around their mean.

    Args:
        matrix: A numpy array of shape (m, d), where m is the number of solutions
                and d is the number of distinct opcodes. Each row is a PMF.

    Returns:
        The normalized trace-of-covariance score (τ), or NaN if there are
        fewer than 2 solutions.
    """
    m, _d = matrix.shape
    if m < 2:
        return float("nan")

    mu = matrix.mean(axis=0)
    centred = matrix - mu
    trace_cov = np.sum(centred * centred) / m

    upper_bound = 1.0 - np.sum(mu**2)
    if upper_bound <= 0:
        return 0.0
    return trace_cov / upper_bound


# ───────────────────────────────────────────────────────────── IO HELPERS ──────


def load_jsonl_grouped(path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Read *path* and group {"problem_id": [...(solution_code, category)...]}.

    Skips lines with JSON errors or missing fields.
    """
    groups: defaultdict[str, List[Tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                pid = data.get("problem_id")
                code = data.get("generated_solution")
                cat = data.get("category")
                if pid and code and cat is not None:
                    groups[pid].append((code, cat))
            except json.JSONDecodeError:
                print(f"[warn] JSON error in line {i}, skipping.")
    return groups


# ──────────────────────────────────────────────────────────── MAIN ANALYSIS ────


def analyse_problem(
    records: List[Tuple[str, str]],
    alpha: float = ALPHA,
) -> Tuple[float, float]:
    """
    Computes the Static Canonical Trace Divergence (SCTD) for a single problem.

    This function calculates two versions of SCTD:
    1.  SCTD_JSD: Based on the average pairwise Jensen-Shannon Divergence.
    2.  SCTD_τ: Based on the normalized total variance (τ).

    The final score for each metric is a convex combination of the divergence
    from structural PMFs (P) and cost-weighted PMFs (Q), controlled by `alpha`.

    SCTD_JSD = α * [avg_JSD(P)] + (1-α) * [avg_JSD(Q)]
    SCTD_τ   = α * τ(P) + (1-α) * τ(Q)

    Args:
        records: A list of (source_code, category) tuples for a given problem.
        alpha: The hyperparameter to weigh the structural and cost-weighted PMFs.
               Defaults to 0.5.

    Returns:
        A tuple containing (sctd_jsd_score, sctd_tau_score).
        Returns (0.0, 0.0) if fewer than two solutions can be compiled.
    """
    # compile
    code_objects = []
    for idx, (src, _cat) in enumerate(records):
        try:
            code_objects.append(
                compile_source_string(src, filename=f"<solution_{idx}>")
            )
        except (SyntaxError, ValueError, TypeError):
            pass

    if len(code_objects) < 2:
        return 0.0, 0.0

    # dynamic opcode universe for this problem
    opcodes = collect_unique_opcodes(code_objects)
    d = len(opcodes)
    index_map = build_opcode_index_map(opcodes)
    weights = np.array(
        [opcode_time_complexity.get(dis.opname[op], DEFAULT_WEIGHT) for op in opcodes],
        dtype=float,
    )

    # build probability vectors
    p_list: List[np.ndarray] = []  # structure
    q_list: List[np.ndarray] = []  # cost
    for co in code_objects:
        counts = count_opcodes(co, index_map, d)
        p_list.append(_pmf_from_counts(counts))
        q_list.append(_pmf_from_counts(counts, weights))

    # Jensen–Shannon
    js_p = mean_pairwise_js(p_list)
    js_q = mean_pairwise_js(q_list)
    js_metric = alpha * js_p + (1.0 - alpha) * js_q

    # Normalised trace-of-covariance
    P = np.vstack(p_list)
    Q = np.vstack(q_list)
    tau_p = normalised_cov_trace(P)
    tau_q = normalised_cov_trace(Q)
    cov_metric = alpha * tau_p + (1.0 - alpha) * tau_q

    return js_metric, cov_metric


def main(input_jsonl: Path, output_csv: Path):
    groups = load_jsonl_grouped(input_jsonl)
    print(f"Loaded {len(groups)} unique problem IDs.")

    fieldnames = [
        "problem_id",
        "SCTD_JSD",
        "SCTD_TAU",  # raw 0-1 metrics
        "SCTD_JSD_pct",
        "SCTD_TAU_pct",  # 0-100 rescale  ← NEW
        "categories",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pid, recs in groups.items():
            js_score, cov_score = analyse_problem(recs)
            cats = [cat for _code, cat in recs]
            writer.writerow(
                {
                    "problem_id": pid,
                    "SCTD_JSD": js_score,
                    "SCTD_JSD_pct": 100 * js_score,
                    "SCTD_TAU": cov_score,
                    "SCTD_TAU_pct": 100 * cov_score,
                    "categories": cats,
                }
            )
            print(
                f"[{pid}]  JSD={js_score:.4f}   τ={cov_score:.4f}   "
                f"(solutions={len(recs)})"
            )
    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    INFILE = Path(sys.argv[1])
    OUTFILE = Path(sys.argv[2])

    if not INFILE.is_file():
        sys.exit(f"Input file not found: {INFILE}")

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    main(INFILE, OUTFILE)
