#!/usr/bin/env python3
"""
Disposable Worker for Fail-safe Evaluator (Corrected Output Handling)
-------------------
This script is executed as a subprocess by the main `produce_dynamic_traces_cc.py` watchdog.
"""
from __future__ import annotations

import builtins
import ctypes
import dis
import io
import json
import sys
import time
import traceback
from collections import Counter
from inspect import Parameter, signature
from queue import Queue, Empty
from threading import Thread
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# I/O shim (captures stdout)
# --------------------------------------------------------------------------- #
class _StdioShim:
    def __init__(self, stdin_text: str) -> None:
        self.stdin_text = stdin_text
        self.orig_stdin, self.orig_stdout, self.orig_stderr = (
            sys.stdin,
            sys.stdout,
            sys.stderr,
        )
        self.orig_input = builtins.input

    def __enter__(self):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        sys.stdin, sys.stdout, sys.stderr = (
            io.StringIO(self.stdin_text),
            self.stdout,
            self.stderr,
        )
        builtins.input = lambda prompt=None: sys.stdin.readline().rstrip("\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdin, sys.stdout, sys.stderr = (
            self.orig_stdin,
            self.orig_stdout,
            self.orig_stderr,
        )
        builtins.input = self.orig_input


# --------------------------------------------------------------------------- #
# Tracing / timeout utilities
# --------------------------------------------------------------------------- #
class TimeLimitExceeded(Exception):
    pass


def make_tracer(deadline: float, counter: Counter) -> Any:
    def _tracer(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
        elif event == "opcode":
            counter[dis.opname[frame.f_code.co_code[frame.f_lasti]]] += 1
            if time.perf_counter() > deadline:
                raise TimeLimitExceeded
        return _tracer

    return _tracer


# --------------------------------------------------------------------------- #
# Adaptive solve() invoker
# --------------------------------------------------------------------------- #
def _call_solve_adaptively(solve, input_lines: List[str]):
    try:
        sig = signature(solve)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        ]
        mandatory = [p for p in params if p.default is Parameter.empty]
        if len(mandatory) == 0:
            return solve()
        if len(mandatory) == 1:
            return solve(input_lines)
        raise TypeError("solve() has unsupported signature")
    except (ValueError, TypeError):
        try:
            return solve()
        except TypeError:
            return solve(input_lines)


# --------------------------------------------------------------------------- #
# Forced thread stopper
# --------------------------------------------------------------------------- #
def _async_raise_thread(t: Thread, exctype=SystemExit) -> None:
    if not t.is_alive():
        return
    tid = t.ident
    if tid is None:
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid), ctypes.py_object(exctype)
    )
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)


# --------------------------------------------------------------------------- #
# Low‑level worker
# --------------------------------------------------------------------------- #
def _inner_test_worker(
    q: Queue, code_obj, test_input: str, expected_output: str, timeout_s: int
) -> None:
    counter: Counter = Counter()
    deadline = time.perf_counter() + timeout_s
    tracer = make_tracer(deadline, counter)
    try:
        globs: Dict[str, Any] = {}
        with _StdioShim(test_input) as stdio:
            sys.settrace(tracer)
            try:
                exec(code_obj, globs)
                solve = globs.get("solve")
                if not callable(solve):
                    q.put(
                        {
                            "status": "fail",
                            "reason": "missing_solve_function",
                            "traceback": "The 'solve' function was not found or is not callable.",
                            "opcodes": dict(counter),
                        }
                    )
                    return
                result = _call_solve_adaptively(solve, test_input.splitlines())
            finally:
                sys.settrace(None)

            actual_output = ""
            if result is not None:
                if isinstance(result, (list, tuple)):
                    actual_output = " ".join(map(str, result))
                else:
                    actual_output = str(result)
            else:
                actual_output = stdio.stdout.getvalue()

        if actual_output.strip() == expected_output.strip():
            q.put(
                {"status": "pass", "reason": "correct_output", "opcodes": dict(counter)}
            )
        else:
            q.put(
                {
                    "status": "fail",
                    "reason": "output_mismatch",
                    "details": f"Expected {expected_output!r}, got {actual_output!r}",
                    "opcodes": dict(counter),
                }
            )

    except TimeLimitExceeded:
        q.put(
            {
                "status": "fail",
                "reason": "timeout_cooperative",
                "traceback": f"Execution exceeded cooperative timeout of {timeout_s} seconds.",
                "opcodes": dict(counter),
            }
        )
    except BaseException:
        q.put(
            {
                "status": "fail",
                "reason": "runtime_error",
                "traceback": traceback.format_exc(),
                "opcodes": dict(counter),
            }
        )


# --------------------------------------------------------------------------- #
# Public “run one test” API
# --------------------------------------------------------------------------- #
def run_single_test(
    code_obj, test_input: str, expected_output: str, timeout_s: int
) -> Dict[str, Any]:
    q: Queue = Queue(maxsize=1)
    t = Thread(
        target=_inner_test_worker,
        args=(q, code_obj, test_input, expected_output, timeout_s),
        daemon=True,
    )
    t.start()
    try:
        return q.get(timeout=timeout_s + 1.0)
    except Empty:
        _async_raise_thread(t)
        return {
            "status": "fail",
            "reason": "timeout_unresponsive",
            "traceback": f"Worker thread did not produce a result within {timeout_s + 1.0}s. Presumed stuck and abandoned.",
            "opcodes": {},
        }


# --------------------------------------------------------------------------- #
# Main evaluation logic for the worker
# --------------------------------------------------------------------------- #
def evaluate_and_print(sol_data, test_data, timeout):
    pid = sol_data["problem_id"]
    sid = sol_data.get("solution_id")
    category = sol_data.get("category")
    time_complexity_inferred = test_data.get("time_complexity_inferred")

    def _write_result(res: Dict[str, Any]) -> None:
        record = {
            "problem_id": pid,
            "solution_id": sid,
            "original_category": category,
            "time_complexity_inferred": time_complexity_inferred,
            **res,
        }
        print(json.dumps(record))

    code = sol_data.get("generated_solution", "")
    if not code.strip():
        _write_result(
            {
                "test_type": "N/A",
                "test_case_index": -1,
                "status": "fail",
                "reason": "empty_code",
                "opcodes": {},
            }
        )
        return

    try:
        code_obj = compile(code, "<solution>", "exec")
    except (SyntaxError, ValueError, TypeError):
        _write_result(
            {
                "test_type": "N/A",
                "test_case_index": -1,
                "status": "fail",
                "reason": "compile_error",
                "traceback": traceback.format_exc(),
                "opcodes": {},
            }
        )
        return

    test_cases = [
        (t, "private") for t in test_data.get("tests", {}).get("private_tests", [])[:3]
    ]
    if not test_cases:
        _write_result(
            {
                "test_type": "N/A",
                "test_case_index": -1,
                "status": "fail",
                "reason": "no_tests_found",
                "opcodes": {},
            }
        )
        return

    for idx, (case, test_type) in enumerate(test_cases):
        res = run_single_test(
            code_obj, case.get("input", ""), case.get("output", ""), timeout
        )
        res.update({"test_type": test_type, "test_case_index": idx})
        _write_result(res)
        if "timeout" in res.get("reason", ""):
            _write_result(
                {
                    "test_type": "N/A",
                    "test_case_index": -1,
                    "status": "fail",
                    "reason": "aborted_due_to_previous_timeout",
                    "opcodes": {"ABORTED_DUE_TO_HANG": 1},
                }
            )
            return


if __name__ == "__main__":
    try:
        payload = json.load(sys.stdin)
        sol_data = payload["solution"]
        test_data = payload["tests"]
        timeout = payload["timeout_per_test"]
        evaluate_and_print(sol_data, test_data, timeout)
    except Exception:
        sys.exit(1)