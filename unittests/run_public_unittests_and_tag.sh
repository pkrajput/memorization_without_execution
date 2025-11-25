#!/bin/bash
set -euo pipefail

# ==============================================================================
# Run public unit tests on generated solutions and tag originals
#
# What it does:
#  1) For each generated JSONL:
#       python unit_test.py --path_to_file IN --output_file TEST_OUT [--timeout T] [--dataset auto|codecontests|bigobench|humaneval]
#  2) Merge TEST_OUT back into a COPY of IN:
#       adds:
#         - success: "pass" if category=="success" else "failure"
#         - status: "pass" if success=="pass" else "fail"   (kept for legacy)
#         - public_unittest_category: category from unit_test.py
#         - public_unittest_error / public_unittest_stdout (if present)
#       keying by (problem_id, solution_id),
#       fallback by (problem_id, completion_index) if needed.
#
# Notes:
#   - For CodeContests / BigO-Bench: uses .public_tests list.
#   - For HumanEval: runs the `test` code via its `check(candidate)` entry.
#
# Usage:
#   ./run_public_unittests_and_tag.sh <out_dir> <gen1.jsonl> [gen2.jsonl ...]
#
# Example:
#   ./run_public_unittests_and_tag.sh \
#     /abs/path/to/unittested_out \
#     /abs/path/to/withoutfuzzing.jsonl \
#     /abs/path/to/fuzz0.5.jsonl \
#     /abs/path/to/fuzz0.9.jsonl
#
# Optional env overrides:
#   UNITTEST_SCRIPT=unit_test.py
#   TIMEOUT=10
# ==============================================================================

if [[ $# -lt 2 ]]; then
  echo "Error: Missing args."
  echo "Usage: $0 <out_dir> <gen1.jsonl> [gen2.jsonl ...]"
  exit 1
fi

OUT_DIR="$(realpath "$1")"; shift
UNITTEST_SCRIPT="${UNITTEST_SCRIPT:-unit_test.py}"
TIMEOUT="${TIMEOUT:-10}"

if [[ ! -f "$UNITTEST_SCRIPT" ]]; then
  echo "Error: unit test script not found: $UNITTEST_SCRIPT"
  exit 1
fi

mkdir -p "$OUT_DIR"

for INFILE in "$@"; do
  if [[ ! -f "$INFILE" ]]; then
    echo "Error: input JSONL not found: $INFILE"
    exit 1
  fi

  base="$(basename "$INFILE")"
  TEST_OUT="$OUT_DIR/unittest_${base}"
  TAGGED_OUT="$OUT_DIR/unittested_${base}"

  echo "======================================================================="
  echo "Running public unit tests:"
  echo "  INFILE   = $INFILE"
  echo "  TEST_OUT = $TEST_OUT"
  echo "  TAGGED   = $TAGGED_OUT"
  echo "  TIMEOUT  = $TIMEOUT sec/solution"
  echo "======================================================================="

  # 1) Run unit tester (dataset auto-detected from each row unless overridden)
  python "$UNITTEST_SCRIPT" \
    --path_to_file "$INFILE" \
    --output_file "$TEST_OUT" \
    --timeout "$TIMEOUT"

  # 2) Merge test results back into a copy of originals
  python - <<'PY' "$INFILE" "$TEST_OUT" "$TAGGED_OUT"
import json, sys

orig_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Load test results into maps
by_pid_sid = {}
by_pid_ci  = {}

with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue

        pid = str(d.get("problem_id"))
        sid = d.get("solution_id")
        ci  = d.get("completion_index")

        if pid and sid:
            by_pid_sid[(pid, str(sid))] = d
        if pid and ci is not None:
            try:
                by_pid_ci[(pid, int(ci))] = d
            except Exception:
                pass

def success_from_category(cat: str):
    return "pass" if cat == "success" else "failure"

n_total = 0
n_tagged = 0

with open(orig_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_total += 1
        try:
            o = json.loads(line)
        except Exception:
            continue

        pid = str(o.get("problem_id"))
        sid = o.get("solution_id")
        ci  = o.get("completion_index")

        tr = None
        if pid and sid and (pid, str(sid)) in by_pid_sid:
            tr = by_pid_sid[(pid, str(sid))]
        elif pid and ci is not None and (pid, int(ci)) in by_pid_ci:
            tr = by_pid_ci[(pid, int(ci))]

        if tr is not None:
            cat = tr.get("category", "unknown")
            o["public_unittest_category"] = cat
            o["success"] = success_from_category(cat)
            o["status"] = "pass" if o["success"] == "pass" else "fail"
            if "error" in tr:
                o["public_unittest_error"] = tr.get("error")
            if "stdout" in tr:
                o["public_unittest_stdout"] = tr.get("stdout")
            if "stderr" in tr and tr.get("stderr"):
                o["public_unittest_stderr"] = tr.get("stderr")
            if "failing_test_index" in tr:
                o["public_unittest_failing_test_index"] = tr.get("failing_test_index")
            n_tagged += 1
        else:
            o["public_unittest_category"] = "missing_unittest_result"
            o["success"] = "failure"
            o["status"] = "fail"

        g.write(json.dumps(o, ensure_ascii=False) + "\n")

print(f"[merge] tagged {n_tagged}/{n_total} rows -> {out_path}")
PY

  echo "Done: $TAGGED_OUT"
  echo ""
done

echo "All files unit-tested + tagged. Outputs in: $OUT_DIR"