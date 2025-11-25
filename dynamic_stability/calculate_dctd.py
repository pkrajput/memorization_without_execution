#!/usr/bin/env python3
"""
Bounded-spread metrics for dynamic opcode profiles (DCTD)
==========================================================

This script calculates dynamic stability metrics based on opcode traces
generated from executing multiple solutions against multiple unit tests.

Metrics implemented
-------------------
1.  **DCTD_JSD (Dynamic Canonical Trace Divergence - JSD):**
    Averages the Jensen-Shannon divergence of opcode distributions
    across all unit tests.

2.  **DCTD_TAU (Dynamic Canonical Trace Divergence - Tau):**
    Averages the normalised trace-of-covariance of opcode distributions
    across all unit tests.

Both scores are ∈ [0, 1]; higher ⇒ **more spread / less stability**.

Input format:
    A JSONL file where each line contains a record for one solution on
    one test case, including an "opcodes" dictionary.
    Example: {"problem_id": "1", "solution_id": "A", "test_case_index": 0, "opcodes": {"LOAD_CONST": 5}}

"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict, Any

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

# Mixture weight between structure and cost distributions
ALPHA = 0.5

# Epsilon for probability smoothing to avoid log(0)
EPS = 1e-9

# Type aliases for clarity
OpcodeCounts = Dict[str, int]
# Data structure: {problem_id: {solution_id: {test_idx: OpcodeCounts}}}
ProblemData = DefaultDict[str, DefaultDict[str, Dict[int, OpcodeCounts]]]
# Data structure: {problem_id: {solution_id: {metadata_key: value}}}
ProblemMetadata = DefaultDict[str, DefaultDict[str, Dict[str, Any]]]

# ───────────────────────────────────────────────────────── MATHS / METRICS ─────


def _pmf_from_counts(
    counts: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert raw opcode counts to a probability distribution.
    If *weights* is given, compute w_i*c_i / Σ(w_j*c_j) (cost distribution).
    Otherwise, compute c_i / Σ(c_j) (structure distribution).
    """
    if weights is None:
        num = counts.astype(float)
    else:
        num = weights * counts
    den = num.sum()
    if den == 0:
        # A valid trace with zero opcodes is impossible, but for safety:
        return np.full_like(num, 1.0 / len(num) if len(num) > 0 else 1.0)
    pmf = num / den
    pmf = np.clip(pmf, EPS, 1.0)  # Smooth to avoid log(0) later
    pmf /= pmf.sum()
    return pmf


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (base-2, ∈[0,1])."""
    m = 0.5 * (p + q)
    # Use np.log2 for base-2, as specified in the paper
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return 0.5 * (kl_pm + kl_qm)


def mean_pairwise_js(values: List[np.ndarray]) -> float:
    """Calculates the average JSD over all unique pairs in a list of vectors."""
    if len(values) < 2:
        return 0.0
    pairs = itertools.combinations(values, 2)
    return np.mean([js_divergence(a, b) for a, b in pairs])


def normalised_cov_trace(matrix: np.ndarray) -> float:
    """
    Calculates the normalised trace of the covariance matrix: τ = tr(Cov) / (1 - ||μ||²).
    The input matrix has shape (m, d), where m is the number of samples (solutions)
    and d is the number of features (opcodes).
    """
    m, _d = matrix.shape
    if m < 2:
        return 0.0

    mu = matrix.mean(axis=0)
    # The trace of the sample covariance matrix is E[||X - μ||²]
    trace_cov = np.sum((matrix - mu) ** 2) / m

    # Denominator is the sharp upper bound on the trace
    upper = 1.0 - np.sum(mu**2)
    if upper <= 1e-9:  # Avoid division by zero or tiny numbers
        return 0.0
    return trace_cov / upper


# ───────────────────────────────────────────────────────────── IO HELPERS ──────


def load_dynamic_traces(path: Path) -> Tuple[ProblemData, ProblemMetadata]:
    """
    Read a JSONL file of dynamic traces and group by problem, solution, and test.
    Returns two dictionaries: one for opcode data and one for metadata.
    """
    problem_data: ProblemData = defaultdict(lambda: defaultdict(dict))
    problem_metadata: ProblemMetadata = defaultdict(lambda: defaultdict(dict))

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                pid = data.get("problem_id")
                sid = data.get("solution_id")
                test_idx = data.get("test_case_index")
                opcodes = data.get("opcodes")
                status = data.get("status")

                if not all([pid, sid, opcodes is not None, test_idx is not None]):
                    print(f"[warn] Skipping line {i} due to missing required fields.")
                    continue

                # Only process successful traces for metric calculation
                if status:
                    problem_data[pid][sid][test_idx] = opcodes

                # Store metadata only once per solution
                if sid not in problem_metadata[pid]:
                    problem_metadata[pid][sid] = {
                        "category": data.get("original_category", "N/A"),
                    }

            except json.JSONDecodeError:
                print(f"[warn] JSON error in line {i}, skipping.")

    return problem_data, problem_metadata


# ──────────────────────────────────────────────────────────── MAIN ANALYSIS ────


def analyse_problem_dynamically(
    solutions_data: Dict[str, Dict[int, OpcodeCounts]],
    alpha: float = ALPHA,
) -> Tuple[float, float]:
    """
    Compute (DCTD_JSD, DCTD_TAU) for a single problem's dynamic traces.

    This function implements the logic described in the paper:
    1. For each test, calculate the divergence/dispersion across solutions.
    2. Average these per-test scores to get the final metric.
    """
    solution_ids = sorted(solutions_data.keys())
    m = len(solution_ids)
    if m < 2:
        return 0.0, 0.0

    # 1. Determine the universe of opcodes and tests for this problem
    all_opcodes = set()
    all_test_indices = set()
    for sid in solution_ids:
        all_test_indices.update(solutions_data[sid].keys())
        for test_idx in solutions_data[sid]:
            all_opcodes.update(solutions_data[sid][test_idx].keys())

    if not all_opcodes or not all_test_indices:
        return 0.0, 0.0

    sorted_opcodes = sorted(list(all_opcodes))
    d = len(sorted_opcodes)
    opcode_map = {op: i for i, op in enumerate(sorted_opcodes)}
    weights = np.array(
        [opcode_time_complexity.get(op, DEFAULT_WEIGHT) for op in sorted_opcodes],
        dtype=float,
    )

    # 2. Iterate through each test case to compute per-test metrics
    per_test_js_p, per_test_js_q = [], []
    per_test_tau_p, per_test_tau_q = [], []

    for test_idx in sorted(list(all_test_indices)):
        # For this test, collect PMFs for all solutions that have a trace
        p_list_test, q_list_test = [], []

        for sid in solution_ids:
            trace = solutions_data[sid].get(test_idx)
            # Only include solutions that ran successfully for this test
            if trace is not None:
                # Convert opcode dict to a count vector
                counts_vec = np.zeros(d, dtype=np.int32)
                for op, count in trace.items():
                    if op in opcode_map:  # Should always be true
                        counts_vec[opcode_map[op]] = count

                # Create structural (p) and cost-weighted (q) PMFs
                p_list_test.append(_pmf_from_counts(counts_vec))
                q_list_test.append(_pmf_from_counts(counts_vec, weights))

        # We need at least 2 solutions for this test to calculate divergence
        if len(p_list_test) < 2:
            continue

        # Calculate JSD-based score for this test
        per_test_js_p.append(mean_pairwise_js(p_list_test))
        per_test_js_q.append(mean_pairwise_js(q_list_test))

        # Calculate Tau-based score for this test
        P_test = np.vstack(p_list_test)
        Q_test = np.vstack(q_list_test)
        per_test_tau_p.append(normalised_cov_trace(P_test))
        per_test_tau_q.append(normalised_cov_trace(Q_test))

    # 3. Average the per-test scores
    if not per_test_js_p:  # No tests had >= 2 valid traces
        return 0.0, 0.0

    # Final DCTD_JSD is the blended average of per-test JSD scores
    mean_js_p = np.mean(per_test_js_p)
    mean_js_q = np.mean(per_test_js_q)
    dctd_jsd = alpha * mean_js_p + (1.0 - alpha) * mean_js_q

    # Final DCTD_TAU is the blended average of per-test Tau scores
    mean_tau_p = np.mean(per_test_tau_p)
    mean_tau_q = np.mean(per_test_tau_q)
    dctd_tau = alpha * mean_tau_p + (1.0 - alpha) * mean_tau_q

    return dctd_jsd, dctd_tau


def main(input_jsonl: Path, output_csv: Path):
    problem_data, problem_metadata = load_dynamic_traces(input_jsonl)
    print(f"Loaded traces for {len(problem_data)} unique problem IDs.")

    fieldnames = [
        "problem_id",
        "solution_ids",
        "DCTD_JSD",
        "DCTD_JSD_pct",
        "DCTD_TAU",
        "DCTD_TAU_pct",
        "original_categories",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pid, solutions_data in problem_data.items():
            js_score, cov_score = analyse_problem_dynamically(solutions_data)

            sids = sorted(solutions_data.keys())
            if not sids:
                continue

            # Prepare metadata for CSV output
            meta = problem_metadata[pid]
            categories = [meta[sid]["category"] for sid in sids if sid in meta]

            writer.writerow(
                {
                    "problem_id": pid,
                    "solution_ids": sids,
                    "DCTD_JSD": js_score,
                    "DCTD_JSD_pct": 100 * js_score,
                    "DCTD_TAU": cov_score,
                    "DCTD_TAU_pct": 100 * cov_score,
                    "original_categories": categories,
                }
            )
            print(
                f"[{pid:<6}]  DCTD_JSD={js_score:.4f}   DCTD_TAU={cov_score:.4f}   "
                f"(solutions={len(sids)})"
            )
    print(f"\nResults written to {output_csv}")


# ──────────────────────────────────────────────────────────── ENTRY POINT ──────


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_traces.jsonl> <output_metrics.csv>")
        sys.exit(1)

    INFILE = Path(sys.argv[1])
    OUTFILE = Path(sys.argv[2])

    if not INFILE.is_file():
        sys.exit(f"Input file not found: {INFILE}")

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    main(INFILE, OUTFILE)
