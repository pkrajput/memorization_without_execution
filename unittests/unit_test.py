#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Public unit tester for generated solutions.

Datasets:
- CodeContests / BigO-Bench:
    - Runs ONLY public_tests[] from each JSONL row (input/output pairs).
- HumanEval:
    - Uses the `test` code and `entry_point` to call check(candidate).

Adds a result row per solution with category:
    success | test_failure | execution_error | timeout | empty_code_extracted | no_response

Safe against `if __name__ == "__main__":` in generated code.
Supports both solve(input_lines) and solve() that reads stdin for non-HumanEval.

Usage:
  python unit_test.py \
    --path_to_file IN.jsonl \
    --output_file OUT.jsonl \
    --timeout 10 \
    [--dataset auto|codecontests|bigobench|humaneval]
"""

import argparse
import json
import multiprocessing
import traceback
import uuid
from tqdm import tqdm


def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping line {line_num}: JSON decode error - {e}")
    return items


def _worker_standard(generated_solution: str, public_tests, conn):
    """
    Child process for CodeContests/BigO-Bench style: input/output public_tests.
    Sends dict back on conn.
    """
    import io, sys, inspect

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf

    try:
        globs = {"__name__": "__unit_test__"}

        exec(generated_solution, globs)

        solve = globs.get("solve", None)
        if solve is None or not callable(solve):
            conn.send({
                "category": "execution_error",
                "error": "No callable 'solve' found after exec().",
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
            })
            return

        try:
            sig = inspect.signature(solve)
            n_params = len(sig.parameters)
        except Exception:
            n_params = None

        for i, case in enumerate(public_tests):
            input_str = case.get("input", "")
            expected = case.get("output", "")

            input_str_norm = str(input_str)
            expected_norm = str(expected).strip()

            input_lines = input_str_norm.strip("\n").split("\n")

            out_text = None
            err_here = None

            # attempt 1: solve(input_lines)
            try:
                if n_params is None or n_params >= 1:
                    res = solve(input_lines)
                    out_text = "" if res is None else str(res)
                else:
                    raise TypeError("solve has 0 params; try stdin mode")
            except Exception as e1:
                err_here = e1

            # attempt 2: stdin-mode solve()
            if out_text is None:
                try:
                    sys.stdin = io.StringIO(input_str_norm)
                    call_stdout = io.StringIO()
                    sys.stdout = call_stdout
                    res = solve()
                    printed = call_stdout.getvalue()
                    out_text = printed if res is None else str(res)
                except Exception as e2:
                    conn.send({
                        "category": "execution_error",
                        "error": (
                            f"Failed running solve on public test {i}.\n"
                            f"arg-mode err: {repr(err_here)}\n"
                            f"stdin-mode err: {repr(e2)}"
                        ),
                        "traceback": traceback.format_exc(),
                        "stdout": stdout_buf.getvalue(),
                        "stderr": stderr_buf.getvalue(),
                    })
                    return

            got = str(out_text).strip()
            if got != expected_norm:
                conn.send({
                    "category": "test_failure",
                    "failing_test_index": i,
                    "input": input_str_norm,
                    "expected": expected_norm,
                    "got": got,
                    "stdout": stdout_buf.getvalue(),
                    "stderr": stderr_buf.getvalue() or "Output mismatch",
                })
                return

        conn.send({
            "category": "success",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        })

    except Exception:
        conn.send({
            "category": "execution_error",
            "error": traceback.format_exc(),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        })
    finally:
        conn.close()


def _worker_humaneval(generated_solution: str, test_code: str, entry_point: str, conn):
    """
    Child process for HumanEval-style tasks.

    - exec(generated_solution) to get the candidate function (entry_point).
    - exec(test_code) to define check(candidate).
    - call check(candidate).
    """
    import io, sys

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf

    try:
        globs = {"__name__": "__unit_test__"}

        # Load candidate
        exec(generated_solution, globs)
        candidate = globs.get(entry_point)
        if candidate is None or not callable(candidate):
            conn.send({
                "category": "execution_error",
                "error": f"No callable entry_point '{entry_point}' found after exec().",
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
            })
            return

        # Provide candidate into env, then load tests
        globs["candidate"] = candidate
        exec(test_code, globs)

        check = globs.get("check")
        if check is None or not callable(check):
            conn.send({
                "category": "execution_error",
                "error": "No callable 'check' function found in HumanEval test code.",
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
            })
            return

        try:
            check(candidate)
        except AssertionError as e:
            conn.send({
                "category": "test_failure",
                "error": f"AssertionError in check(candidate): {e}",
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
            })
            return
        except Exception as e:
            conn.send({
                "category": "execution_error",
                "error": f"Exception while running check(candidate): {repr(e)}",
                "traceback": traceback.format_exc(),
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
            })
            return

        conn.send({
            "category": "success",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        })

    except Exception:
        conn.send({
            "category": "execution_error",
            "error": traceback.format_exc(),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        })
    finally:
        conn.close()


def _run_standard(problem, timeout=10):
    generated_solution = problem.get("generated_solution", "")
    public_tests = problem.get("public_tests", [])
    problem_id = str(problem.get("problem_id"))
    solution_id = problem.get("solution_id", str(uuid.uuid4()))
    completion_index = problem.get("completion_index")

    solution_code = problem.get("solution_code")
    dataclass_code = problem.get("dataclass_code")
    complexity = problem.get("time_complexity_inferred") or problem.get("complexity")

    base_result = {
        "problem_id": problem_id,
        "solution_id": solution_id,
        "completion_index": completion_index,
        "generated_solution": generated_solution,
        "num_tests": len(public_tests),
        "public_tests": public_tests,
        "solution_code": solution_code,
        "dataclass_code": dataclass_code,
        "complexity": complexity,
        "raw_llm_output": problem.get("raw_llm_output", ""),
    }

    if not str(generated_solution).strip():
        base_result.update({
            "category": "empty_code_extracted",
            "error": "No code provided in 'generated_solution'."
        })
        return base_result

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker_standard, args=(generated_solution, public_tests, child_conn))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        base_result.update({
            "category": "timeout",
            "error": f"Execution timed out after {timeout} seconds"
        })
        return base_result

    if parent_conn.poll():
        base_result.update(parent_conn.recv())
    else:
        base_result.update({
            "category": "no_response",
            "error": "No response received from subprocess"
        })

    return base_result


def _run_humaneval(problem, timeout=10):
    generated_solution = problem.get("generated_solution", "")
    problem_id = str(problem.get("problem_id"))
    solution_id = problem.get("solution_id", str(uuid.uuid4()))
    completion_index = problem.get("completion_index")

    test_code = problem.get("test", "") or ""
    entry_point = problem.get("entry_point", "") or "candidate"

    # heuristic count of assertions
    num_tests = sum(1 for line in test_code.splitlines() if "assert " in line)

    base_result = {
        "problem_id": problem_id,
        "solution_id": solution_id,
        "completion_index": completion_index,
        "generated_solution": generated_solution,
        "num_tests": num_tests,
        "humaneval_test_code": test_code,
        "entry_point": entry_point,
        "raw_llm_output": problem.get("raw_llm_output", ""),
    }

    if not str(generated_solution).strip():
        base_result.update({
            "category": "empty_code_extracted",
            "error": "No code provided in 'generated_solution'."
        })
        return base_result

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker_humaneval, args=(generated_solution, test_code, entry_point, child_conn))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        base_result.update({
            "category": "timeout",
            "error": f"Execution timed out after {timeout} seconds"
        })
        return base_result

    if parent_conn.poll():
        base_result.update(parent_conn.recv())
    else:
        base_result.update({
            "category": "no_response",
            "error": "No response received from subprocess"
        })

    return base_result


def run_public_tests(problem, timeout=10, dataset_hint="auto"):
    """
    Dispatch to the appropriate runner, based on dataset_hint and per-row metadata.
    """
    row_ds = str(problem.get("dataset", "") or "").lower()
    dataset = (dataset_hint.lower() if dataset_hint != "auto" else row_ds) or "auto"

    if dataset == "humaneval" or (
        dataset == "auto"
        and "test" in problem
        and "entry_point" in problem
        and "prompt" in problem
        and "public_tests" not in problem
    ):
        return _run_humaneval(problem, timeout=timeout)
    else:
        return _run_standard(problem, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_to_file", required=True, help="Input generated JSONL.")
    ap.add_argument("--output_file", required=True, help="Output JSONL with test results.")
    ap.add_argument("--timeout", type=int, default=10, help="Per-solution timeout (sec).")
    ap.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "codecontests", "bigobench", "humaneval"],
        help="Dataset hint; 'auto' will infer from each row's 'dataset' field or structure."
    )
    args = ap.parse_args()

    problems = load_jsonl(args.path_to_file)
    print(f"Loaded {len(problems)} rows from {args.path_to_file}")
    print(f"Dataset mode: {args.dataset}")

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for problem in tqdm(problems, desc="🔍 Running unit tests"):
            result = run_public_tests(problem, timeout=args.timeout, dataset_hint=args.dataset)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()