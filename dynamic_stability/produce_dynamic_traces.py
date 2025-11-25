#!/usr/bin/env python3
"""
Fail-safe Evaluator (Watchdog Version) — Unified Driver
-------------------------------------------------------

This is the main WATCHDOG script. It must not hang.
It spawns a separate, disposable worker process for each solution.

Supported datasets:
  - bigo         : BigOBench-style, tests in a separate JSONL file.
  - codecontests : CodeContests-style, private_tests embedded in solutions.
  - humaneval    : HumanEval-style, private_tests embedded in solutions.

• For each solution, it runs the appropriate worker (`worker_bigo.py` or `worker_cc.py`)
  as an isolated subprocess.
• Enforces a hard OS-level timeout on the worker process.
• If the worker hangs, it is forcefully terminated, and the watchdog moves on.
• Collects results from the worker's stdout and writes them to the final output file.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import traceback

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def load_jsonl_grouped(path: str, key: str) -> dict:
    grouped = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            grouped.setdefault(str(data[key]), []).append(data)
    return grouped


def load_solutions(path: str) -> dict:
    sols = load_jsonl_grouped(path, "problem_id")
    print(
        f"Loaded {sum(len(v) for v in sols.values())} solutions for "
        f"{len(sols)} unique problems."
    )
    return sols


def load_tests(path: str) -> dict:
    """
    For BigOBench-style datasets: tests in a separate JSONL file.
    Each line is expected to have a 'problem_id' and a 'tests' field.
    """
    tests = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            tests[str(data["problem_id"])] = data
    print(f"Loaded test data for {len(tests)} unique problems.")
    return tests


def get_worker_script_path(dataset: str) -> Path:
    """
    Map dataset name to the appropriate worker script.
    """
    base = Path(__file__).parent
    if dataset == "bigo":
        return base / "worker_bigo.py"
    elif dataset in ("codecontests", "humaneval"):
        # HumanEval currently uses the same worker as CodeContests.
        return base / "worker_cc.py"
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")


def run_worker_for_solution(
    dataset: str,
    sol_data: dict,
    test_data: dict,
    timeout: int,
) -> list:
    """
    Spawns, manages, and kills one worker subprocess for a single solution.

    Parameters
    ----------
    dataset : str
        One of {"bigo", "codecontests", "humaneval"}; selects worker script.
    sol_data : dict
        Single solution dict (must include 'problem_id' and optionally 'solution_id').
    test_data : dict
        Dict with keys: {"problem_id": ..., "tests": {"private_tests": [...]}}.
    timeout : int
        Per-test timeout in seconds (for the worker).

    Returns
    -------
    list[dict]
        List of records (one per test) emitted by the worker.
    """
    worker_input = json.dumps(
        {"solution": sol_data, "tests": test_data, "timeout_per_test": timeout}
    )

    worker_script_path = get_worker_script_path(dataset)
    cmd = [sys.executable, str(worker_script_path)]

    num_tests = len(test_data.get("tests", {}).get("private_tests", [])[:10])
    if num_tests == 0:
        num_tests = 1
    process_timeout = (timeout * num_tests) + 15

    try:
        proc = subprocess.run(
            cmd,
            input=worker_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=process_timeout,
        )

        if proc.returncode != 0:
            return [
                {
                    "problem_id": sol_data["problem_id"],
                    "solution_id": sol_data.get("solution_id"),
                    "status": "fail",
                    "reason": "worker_crash",
                    "traceback": (
                        f"Worker process exited with code {proc.returncode}.\n"
                        f"STDERR:\n{proc.stderr}"
                    ),
                }
            ]

        if not proc.stdout.strip():
            # No output lines — treat as a worker error.
            return [
                {
                    "problem_id": sol_data["problem_id"],
                    "solution_id": sol_data.get("solution_id"),
                    "status": "fail",
                    "reason": "worker_empty_output",
                    "traceback": (
                        "Worker process produced no JSONL output on stdout.\n"
                        f"STDERR:\n{proc.stderr}"
                    ),
                }
            ]

        results = [json.loads(line) for line in proc.stdout.strip().splitlines()]
        return results

    except subprocess.TimeoutExpired:
        return [
            {
                "problem_id": sol_data["problem_id"],
                "solution_id": sol_data.get("solution_id"),
                "status": "fail",
                "reason": "timeout_process_terminated",
                "test_case_index": -1,
                "traceback": (
                    f"Worker process exceeded hard timeout of {process_timeout}s "
                    "and was terminated."
                ),
                "opcodes": {},
            }
        ]
    except Exception as e:
        return [
            {
                "problem_id": sol_data["problem_id"],
                "solution_id": sol_data.get("solution_id"),
                "status": "fail",
                "reason": "watchdog_error",
                "test_case_index": -1,
                "traceback": (
                    f"Watchdog failed to run worker: {e}\n"
                    f"{traceback.format_exc()}"
                ),
            }
        ]


# ---------------------------------------------------------------------------
# Dataset-specific drivers
# ---------------------------------------------------------------------------

def run_bigo(
    solutions_path: str,
    tests_path: str,
    output_path: str,
    timeout: int,
    workers: int,
) -> None:
    """
    BigOBench-style driver:
    - Solutions JSONL keyed by 'problem_id'.
    - Tests in a separate JSONL file keyed by 'problem_id'.
    """
    dataset = "bigo"
    solutions = load_solutions(solutions_path)
    tests = load_tests(tests_path)

    output_file = Path(output_path)
    output_file.write_text("")

    tasks = [
        (sol, tests[pid])
        for pid, sols_for_pid in solutions.items()
        if pid in tests
        for sol in sols_for_pid
    ]

    print(
        f"Evaluating {len(tasks)} solutions for dataset '{dataset}' "
        f"using {workers} parallel worker processes..."
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_worker_for_solution, dataset, sol, tdata, timeout): (
                sol,
                tdata,
            )
            for sol, tdata in tasks
        }

        with output_file.open("a", encoding="utf-8") as fh:
            for future in tqdm(
                as_completed(futures),
                total=len(tasks),
                desc="Evaluating Solutions",
                unit="solution",
                file=sys.stderr,
            ):
                try:
                    list_of_records = future.result()
                    for record in list_of_records:
                        fh.write(json.dumps(record) + "\n")
                except Exception as e:
                    sol_data, _ = futures[future]
                    print(
                        f"\nFATAL: Watchdog thread for solution "
                        f"{sol_data.get('solution_id')} crashed: {e}",
                        file=sys.stderr,
                    )

    print(f"\nDone. Results saved to {output_path}")


def _build_tasks_from_embedded_tests(
    dataset: str,
    solutions: dict,
) -> list:
    """
    Shared helper for codecontests / humaneval where tests are embedded as
    'private_tests' inside each solution.
    """
    tasks = []
    for pid, sols_for_pid in solutions.items():
        for sol_data in sols_for_pid:
            if "private_tests" in sol_data and sol_data["private_tests"]:
                test_data_for_sol = {
                    "problem_id": sol_data["problem_id"],
                    "tests": {"private_tests": sol_data["private_tests"]},
                }
                tasks.append((sol_data, test_data_for_sol))
            else:
                sol_id = sol_data.get("solution_id", "N/A")
                print(
                    f"WARNING[{dataset}]: Skipping solution_id '{sol_id}' "
                    f"for problem_id '{pid}' because 'private_tests' key "
                    f"is missing or empty.",
                    file=sys.stderr,
                )
    return tasks


def run_codecontests(
    solutions_path: str,
    output_path: str,
    timeout: int,
    workers: int,
) -> None:
    """
    CodeContests-style driver:
    - Solutions JSONL keyed by 'problem_id'.
    - Each solution contains 'private_tests' list.
    """
    dataset = "codecontests"
    solutions = load_solutions(solutions_path)
    output_file = Path(output_path)
    output_file.write_text("")

    tasks = _build_tasks_from_embedded_tests(dataset, solutions)

    print(
        f"Evaluating {len(tasks)} solutions for dataset '{dataset}' "
        f"using {workers} parallel worker processes..."
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_worker_for_solution, dataset, sol, tdata, timeout): (
                sol,
                tdata,
            )
            for sol, tdata in tasks
        }

        with output_file.open("a", encoding="utf-8") as fh:
            for future in tqdm(
                as_completed(futures),
                total=len(tasks),
                desc="Evaluating Solutions",
                unit="solution",
                file=sys.stderr,
            ):
                try:
                    list_of_records = future.result()
                    for record in list_of_records:
                        fh.write(json.dumps(record) + "\n")
                except Exception as e:
                    sol_data, _ = futures[future]
                    print(
                        f"\nFATAL: Watchdog thread for solution "
                        f"{sol_data.get('solution_id')} crashed: {e}",
                        file=sys.stderr,
                    )

    print(f"\nDone. Results saved to {output_path}")


def run_humaneval(
    solutions_path: str,
    output_path: str,
    timeout: int,
    workers: int,
) -> None:
    """
    HumanEval-style driver:
    - Solutions JSONL keyed by 'problem_id'.
    - Each solution contains 'private_tests' list (constructed upstream).
    """
    dataset = "humaneval"
    solutions = load_solutions(solutions_path)
    output_file = Path(output_path)
    output_file.write_text("")

    tasks = _build_tasks_from_embedded_tests(dataset, solutions)

    print(
        f"Evaluating {len(tasks)} solutions for dataset '{dataset}' "
        f"using {workers} parallel worker processes..."
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_worker_for_solution, dataset, sol, tdata, timeout): (
                sol,
                tdata,
            )
            for sol, tdata in tasks
        }

        with output_file.open("a", encoding="utf-8") as fh:
            for future in tqdm(
                as_completed(futures),
                total=len(tasks),
                desc="Evaluating Solutions",
                unit="solution",
                file=sys.stderr,
            ):
                try:
                    list_of_records = future.result()
                    for record in list_of_records:
                        fh.write(json.dumps(record) + "\n")
                except Exception as e:
                    sol_data, _ = futures[future]
                    print(
                        f"\nFATAL: Watchdog thread for solution "
                        f"{sol_data.get('solution_id')} crashed: {e}",
                        file=sys.stderr,
                    )

    print(f"\nDone. Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified robust watchdog evaluator.\n\n"
            "Datasets:\n"
            "  - bigo         : BigOBench-style, tests in separate JSONL (--tests required).\n"
            "  - codecontests : CodeContests-style, private_tests embedded in solutions.\n"
            "  - humaneval    : HumanEval-style, private_tests embedded in solutions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["bigo", "codecontests", "humaneval"],
        help="Which dataset format to use.",
    )
    parser.add_argument(
        "--solutions",
        required=True,
        help="Path to JSONL with generated solutions (and possibly private_tests).",
    )
    parser.add_argument(
        "--tests",
        help=(
            "Path to JSONL with test cases (BigOBench-style). "
            "Required when --dataset bigo; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write results (JSONL).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Per-test timeout in seconds (for the worker). "
            "If omitted, defaults to 20 for 'bigo' and 10 for others."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes to run.",
    )

    args = parser.parse_args()

    # Dataset-specific requirements / defaults
    if args.dataset == "bigo" and not args.tests:
        parser.error("--tests is required when --dataset bigo")

    if args.timeout is None:
        # Preserve old per-dataset defaults
        args.timeout = 20 if args.dataset == "bigo" else 10

    return args


def main() -> None:
    args = parse_args()

    if args.dataset == "bigo":
        run_bigo(
            solutions_path=args.solutions,
            tests_path=args.tests,
            output_path=args.output,
            timeout=args.timeout,
            workers=args.workers,
        )
    elif args.dataset == "codecontests":
        run_codecontests(
            solutions_path=args.solutions,
            output_path=args.output,
            timeout=args.timeout,
            workers=args.workers,
        )
    elif args.dataset == "humaneval":
        run_humaneval(
            solutions_path=args.solutions,
            output_path=args.output,
            timeout=args.timeout,
            workers=args.workers,
        )
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}'")


if __name__ == "__main__":
    main()