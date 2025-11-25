#!/usr/bin/env python3
"""
Disposable worker for HumanEval dynamic tracing.

Reads a JSON payload from stdin:
  {
    "solution": { ... },
    "tests": { "problem_id": ..., "test": "<test code>", "entry_point": "<fn>" },
    "timeout_per_test": <seconds>
  }

Executes the candidate under the HumanEval `check(candidate)` harness,
counting Python opcodes for the candidate function only.

Emits a single JSON line to stdout with fields including:
  - problem_id
  - solution_id
  - original_category
  - status: "pass"/"fail"
  - reason
  - test_case_index: 0
  - opcodes: { OPCODE_NAME: count, ... }
"""

from __future__ import annotations

import dis
import json
import sys
import time
import traceback
from collections import Counter
from typing import Any, Dict


class TimeLimitExceeded(Exception):
    pass


def make_tracer(deadline: float, counter: Counter) -> Any:
    """
    Trace only opcodes executed in code objects whose filename is "<solution>".
    """
    def _tracer(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
        elif event == "opcode":
            # Only count opcodes from the candidate code, not from the test harness
            if frame.f_code.co_filename == "<solution>":
                opname = dis.opname[frame.f_code.co_code[frame.f_lasti]]
                counter[opname] += 1
            if time.perf_counter() > deadline:
                raise TimeLimitExceeded
        return _tracer

    return _tracer


def evaluate(solution: Dict[str, Any], tests: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    pid = solution.get("problem_id")
    sid = solution.get("solution_id")
    category = solution.get("category")

    src = solution.get("generated_solution") or ""
    test_code = tests.get("test") or ""
    entry_point = tests.get("entry_point") or solution.get("entry_point") or "candidate"

    base: Dict[str, Any] = {
        "problem_id": pid,
        "solution_id": sid,
        "original_category": category,
        "entry_point": entry_point,
        "test_case_index": 0,
        "test_type": "humaneval",
        "opcodes": {},
    }

    if not src.strip():
        base.update(
            {"status": "fail", "reason": "empty_code", "traceback": "No generated_solution code."}
        )
        return base

    if not test_code.strip():
        base.update(
            {"status": "fail", "reason": "missing_test_code", "traceback": "No HumanEval test code found."}
        )
        return base

    try:
        # Compile and exec candidate
        sol_code = compile(src, "<solution>", "exec")
    except Exception:
        base.update(
            {
                "status": "fail",
                "reason": "compile_error",
                "traceback": traceback.format_exc(),
            }
        )
        return base

    globs: Dict[str, Any] = {"__name__": "__humaneval_dynamic__"}
    try:
        exec(sol_code, globs)
    except Exception:
        base.update(
            {
                "status": "fail",
                "reason": "runtime_error_during_candidate_exec",
                "traceback": traceback.format_exc(),
            }
        )
        return base

    candidate = globs.get(entry_point)
    if candidate is None or not callable(candidate):
        base.update(
            {
                "status": "fail",
                "reason": "missing_entry_point",
                "traceback": f"Entry point '{entry_point}' not found or not callable.",
            }
        )
        return base

    # Compile test harness (defines METADATA, check(candidate))
    try:
        test_obj = compile(test_code, "<tests>", "exec")
    except Exception:
        base.update(
            {
                "status": "fail",
                "reason": "test_compile_error",
                "traceback": traceback.format_exc(),
            }
        )
        return base

    counter: Counter = Counter()
    deadline = time.perf_counter() + timeout
    tracer = make_tracer(deadline, counter)

    try:
        sys.settrace(tracer)
        try:
            # Inject candidate into globals for the test harness
            globs["candidate"] = candidate
            exec(test_obj, globs)
            check = globs.get("check")
            if check is None or not callable(check):
                base.update(
                    {
                        "status": "fail",
                        "reason": "missing_check_function",
                        "traceback": "No callable check(candidate) found in test code.",
                    }
                )
                return base

            # Run the HumanEval checker
            check(candidate)

        finally:
            sys.settrace(None)

    except TimeLimitExceeded:
        base.update(
            {
                "status": "fail",
                "reason": "timeout_cooperative",
                "traceback": f"Execution exceeded cooperative timeout of {timeout} seconds.",
            }
        )
        base["opcodes"] = dict(counter)
        return base
    except AssertionError as e:
        base.update(
            {
                "status": "fail",
                "reason": "test_failure",
                "traceback": f"AssertionError in check(candidate): {e}",
            }
        )
        base["opcodes"] = dict(counter)
        return base
    except Exception:
        base.update(
            {
                "status": "fail",
                "reason": "runtime_error_during_check",
                "traceback": traceback.format_exc(),
            }
        )
        base["opcodes"] = dict(counter)
        return base

    # Success
    base.update({"status": "pass", "reason": "all_tests_passed"})
    base["opcodes"] = dict(counter)
    return base


if __name__ == "__main__":
    try:
        payload = json.load(sys.stdin)
        sol_data = payload["solution"]
        test_data = payload["tests"]
        timeout = int(payload.get("timeout_per_test", 20))

        result = evaluate(sol_data, test_data, timeout)
        print(json.dumps(result))
    except Exception:
        # Fail loudly so the watchdog can record a worker_crash
        sys.exit(1)