#!/bin/bash
set -euo pipefail

# ==============================================================================
# Unified fuzzing study driver (UNIT-TESTED INPUTS)
#
# Supported datasets:
#   - bigo         : BigOBench-style (tests separate, dynamic traces on public-pass)
#   - codecontests : CodeContests-style (tests embedded in solutions, dynamic traces on public-pass)
#   - humaneval    : HumanEval-style (tests extracted from solutions, dynamic traces on ALL solutions)
#
# What it does per condition:
#   1) Make SCTD-compatible copy:
#        - ensure "category" exists (use public_unittest_category if present)
#   2) SCTD on appropriate solution set (all solutions)
#   3) Dataset-specific dynamic pipeline:
#        - bigo:
#            3a) Filter to public-pass solutions only (status=="pass")
#            3b) Extract per-problem tests JSONL (for BigO watchdog tracer)
#            3c) Produce dynamic traces on public-pass solutions
#            3d) Filter traces to dynamic-pass only (status=="pass")
#            3e) DCTD on dynamic-pass traces
#        - codecontests:
#            3a) Filter to public-pass solutions only (status=="pass")
#            3b) Produce dynamic traces on public-pass solutions
#            3c) Filter traces to dynamic-pass only (status=="pass")
#            3d) DCTD on dynamic-pass traces
#        - humaneval:
#            3a) Extract per-problem test snippets (HumanEval `test` + `entry_point`)
#            3b) Produce dynamic traces on ALL solutions
#            3c) DCTD on ALL traces (pass + fail)
#   4) Merge study-wide CSVs
#
# Usage:
#   ./run_fuzz_metrics_unittested.sh \
#       <dataset> \
#       <unittested_no_fuzz.jsonl> \
#       <unittested_fuzzA.jsonl> \
#       <unittested_fuzzB.jsonl> \
#       <out_dir>
#
# Where:
#   dataset = bigo | codecontests | humaneval
#
# Fuzz levels by dataset:
#   - bigo / humaneval   : 0.0, 0.5, 0.9   (labels: fuzz0, fuzz0.5, fuzz0.9)
#   - codecontests       : 0.0, 0.3, 0.6   (labels: fuzz0, fuzz0.3, fuzz0.6)
#
# Optional env overrides:
#   TIMEOUT=20
#   WORKERS=4
#   SCTD_SCRIPT=calculate_sctd.py
#   DCTD_SCRIPT=calculate_dctd.py
#   TRACE_SCRIPT=produce_dynamic_traces.py    # unified watchdog driver (expects --dataset)
# ==============================================================================

if [[ $# -lt 5 ]]; then
  echo "Error: Missing args."
  echo "Usage: $0 <dataset> <unittested_no_fuzz.jsonl> <unittested_fuzzA.jsonl> <unittested_fuzzB.jsonl> <out_dir>"
  echo "  dataset: bigo | codecontests | humaneval"
  exit 1
fi

DATASET="$1"
shift

if [[ "$DATASET" != "bigo" && "$DATASET" != "codecontests" && "$DATASET" != "humaneval" ]]; then
  echo "Error: invalid dataset '$DATASET'. Expected: bigo | codecontests | humaneval"
  exit 1
fi

NO_FUZZ="$1"
FUZZ_A="$2"
FUZZ_B="$3"
OUT_ROOT="$(realpath "$4")"

for f in "$NO_FUZZ" "$FUZZ_A" "$FUZZ_B"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: input JSONL not found: $f"
    exit 1
  fi
done

SCTD_SCRIPT="${SCTD_SCRIPT:-calculate_sctd.py}"
DCTD_SCRIPT="${DCTD_SCRIPT:-calculate_dctd.py}"
TRACE_SCRIPT="${TRACE_SCRIPT:-produce_dynamic_traces.py}"  # unified Python watchdog

for s in "$SCTD_SCRIPT" "$DCTD_SCRIPT" "$TRACE_SCRIPT"; do
  if [[ ! -f "$s" ]]; then
    echo "Error: script not found: $s"
    exit 1
  fi
done

TIMEOUT="${TIMEOUT:-20}"
WORKERS="${WORKERS:-4}"

mkdir -p "$OUT_ROOT"

# ------------------------------------------------------------------------------
# Dataset-specific fuzz labels / probabilities
# ------------------------------------------------------------------------------
declare -a LABELS
declare -a FUZZPROBS

if [[ "$DATASET" == "codecontests" ]]; then
  LABELS=("fuzz0" "fuzz0.3" "fuzz0.6")
  FUZZPROBS=("0.0" "0.3" "0.6")
else
  # bigo + humaneval
  LABELS=("fuzz0" "fuzz0.5" "fuzz0.9")
  FUZZPROBS=("0.0" "0.5" "0.9")
fi

declare -a INPUTS=("$NO_FUZZ" "$FUZZ_A" "$FUZZ_B")

echo "DATASET=$DATASET"
echo "OUT_ROOT=$OUT_ROOT"
echo "TIMEOUT=$TIMEOUT  WORKERS=$WORKERS"
echo "SCTD_SCRIPT=$SCTD_SCRIPT"
echo "TRACE_SCRIPT=$TRACE_SCRIPT"
echo "DCTD_SCRIPT=$DCTD_SCRIPT"
echo ""

# ------------------------------------------------------------------------------
# Helper: ensure category field (SCTD needs pid+code+category)
# Use public_unittest_category if category missing.
# ------------------------------------------------------------------------------
prepare_sctd_input () {
  local infile="$1"
  local outfile="$2"
  python - <<'PY' "$infile" "$outfile"
import json, sys
inp, outp = sys.argv[1], sys.argv[2]
with open(inp, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            d=json.loads(line)
        except Exception:
            continue
        if d.get("category") is None:
            d["category"] = d.get("public_unittest_category", "N/A")
        g.write(json.dumps(d, ensure_ascii=False) + "\n")
PY
}

# ------------------------------------------------------------------------------
# Helper: filter to public-pass solutions only (status=="pass")
# ------------------------------------------------------------------------------
filter_public_pass_solutions () {
  local infile="$1"
  local outfile="$2"
  python - <<'PY' "$infile" "$outfile"
import json, sys
inp, outp = sys.argv[1], sys.argv[2]
kept = 0
total = 0
with open(inp, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
    for line in f:
        line=line.strip()
        if not line:
            continue
        total += 1
        try:
            d=json.loads(line)
        except Exception:
            continue
        if d.get("status") == "pass":
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
print(f"[public-pass] kept {kept}/{total} solutions -> {outp}")
PY
}

# ------------------------------------------------------------------------------
# Helper: keep only dynamic-pass traces (status=="pass")
# ------------------------------------------------------------------------------
filter_dynamic_pass_traces () {
  local infile="$1"
  local outfile="$2"
  python - <<'PY' "$infile" "$outfile"
import json, sys
inp, outp = sys.argv[1], sys.argv[2]
kept = 0
total = 0
with open(inp, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
    for line in f:
        line=line.strip()
        if not line:
            continue
        total += 1
        try:
            d=json.loads(line)
        except Exception:
            continue
        if d.get("status") == "pass":
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
print(f"[dynamic-pass] kept {kept}/{total} traces -> {outp}")
PY
}

# ------------------------------------------------------------------------------
# Helper (bigo): extract per-problem tests JSONL required by BigO watchdog tracer
# It expects rows like:
#   {"problem_id": "...", "tests": {"public_tests": [...], "private_tests": [...], "generated_tests":[...]}}
# We take the first seen tests per problem_id from the solutions file.
# ------------------------------------------------------------------------------
extract_tests_jsonl_bigo () {
  local solutions_file="$1"
  local tests_out="$2"
  python - <<'PY' "$solutions_file" "$tests_out"
import json, sys
sol_path, out_path = sys.argv[1], sys.argv[2]

tests_by_pid = {}
with open(sol_path, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            d=json.loads(line)
        except Exception:
            continue
        pid = str(d.get("problem_id"))
        if not pid or pid in tests_by_pid:
            continue

        pub = d.get("public_tests", [])
        priv = d.get("private_tests", [])
        gen = d.get("generated_tests", [])

        tests_by_pid[pid] = {
            "problem_id": pid,
            "tests": {
                "public_tests": pub,
                "private_tests": priv,
                "generated_tests": gen
            }
        }

with open(out_path, "w", encoding="utf-8") as g:
    for pid, row in tests_by_pid.items():
        g.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[tests-jsonl] wrote {len(tests_by_pid)} problems -> {out_path}")
PY
}

# ------------------------------------------------------------------------------
# Helper (humaneval): extract HumanEval tests JSONL
# Rows: {"problem_id": "...", "test": "<test code>", "entry_point": "<fn name>"}
# Take the first test/entry_point per problem_id from the solutions file.
# ------------------------------------------------------------------------------
extract_tests_jsonl_humaneval () {
  local solutions_file="$1"
  local tests_out="$2"
  python - <<'PY' "$solutions_file" "$tests_out"
import json, sys
sol_path, out_path = sys.argv[1], sys.argv[2]

tests_by_pid = {}
with open(sol_path, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            d=json.loads(line)
        except Exception:
            continue
        pid = str(d.get("problem_id"))
        if not pid or pid in tests_by_pid:
            continue

        test_code = d.get("test")
        entry_point = d.get("entry_point")
        if not test_code or not entry_point:
            continue

        tests_by_pid[pid] = {
            "problem_id": pid,
            "test": test_code,
            "entry_point": entry_point,
        }

with open(out_path, "w", encoding="utf-8") as g:
    for pid, row in tests_by_pid.items():
        g.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[tests-jsonl] wrote {len(tests_by_pid)} problems -> {out_path}")
PY
}

# ------------------------------------------------------------------------------
# Main loop over fuzzing conditions
# ------------------------------------------------------------------------------
for idx in 0 1 2; do
  LABEL="${LABELS[$idx]}"
  INFILE="${INPUTS[$idx]}"
  FP="${FUZZPROBS[$idx]}"

  OUT_DIR="$OUT_ROOT/$LABEL"
  mkdir -p "$OUT_DIR"

  echo "======================================================================="
  echo "Dataset: $DATASET   Condition: $LABEL   fuzz_prob=$FP"
  echo "Input: $INFILE"
  echo "Output dir: $OUT_DIR"
  echo "======================================================================="

  # 1) SCTD-compatible copy (adds category if missing)
  SOLS_SCTD="$OUT_DIR/solutions_sctd_ready.jsonl"
  echo "[1/?] Preparing SCTD-compatible solutions JSONL..."
  prepare_sctd_input "$INFILE" "$SOLS_SCTD"

  # 2) SCTD on all solutions
  SCTD_OUT="$OUT_DIR/sctd_${LABEL}.csv"
  echo "[2/?] Computing SCTD..."
  python "$SCTD_SCRIPT" "$SOLS_SCTD" "$SCTD_OUT"

  if [[ "$DATASET" == "bigo" ]]; then
    # ------------------------------ BigOBench path ---------------------------
    # 3) Public-pass-only solutions for dynamic tracing
    SOLS_PUBLIC_PASS="$OUT_DIR/solutions_public_pass.jsonl"
    echo "[3/?] (bigo) Filtering to public-pass solutions only..."
    filter_public_pass_solutions "$SOLS_SCTD" "$SOLS_PUBLIC_PASS"

    # 4) Extract tests JSONL for watchdog tracer
    TESTS_JSONL="$OUT_DIR/tests_by_problem.jsonl"
    echo "[4/?] (bigo) Extracting per-problem tests JSONL..."
    extract_tests_jsonl_bigo "$SOLS_PUBLIC_PASS" "$TESTS_JSONL"

    # 5) Dynamic traces on public-pass solutions
    TRACE_OUT="$OUT_DIR/dynamic_trace_${LABEL}_timeout${TIMEOUT}.jsonl"
    echo "[5/?] (bigo) Producing dynamic traces (public-pass solutions)..."
    python "$TRACE_SCRIPT" \
      --dataset bigo \
      --solutions "$SOLS_PUBLIC_PASS" \
      --tests "$TESTS_JSONL" \
      --output "$TRACE_OUT" \
      --timeout "$TIMEOUT" \
      --workers "$WORKERS"

    # 6) Filter dynamic traces to dynamic-pass only
    TRACE_PASS="$OUT_DIR/dynamic_trace_${LABEL}_passonly.jsonl"
    echo "[6/?] (bigo) Filtering dynamic-pass-only traces..."
    filter_dynamic_pass_traces "$TRACE_OUT" "$TRACE_PASS"

    # 7) DCTD
    DCTD_OUT="$OUT_DIR/dctd_${LABEL}.csv"
    echo "[7/?] (bigo) Computing DCTD..."
    python "$DCTD_SCRIPT" "$TRACE_PASS" "$DCTD_OUT"

  elif [[ "$DATASET" == "codecontests" ]]; then
    # ------------------------------ CodeContests path -----------------------
    # 3) Public-pass-only solutions for dynamic tracing
    SOLS_PUBLIC_PASS="$OUT_DIR/solutions_public_pass.jsonl"
    echo "[3/?] (codecontests) Filtering to public-pass solutions only..."
    filter_public_pass_solutions "$SOLS_SCTD" "$SOLS_PUBLIC_PASS"

    # 4) Dynamic traces on public-pass solutions
    TRACE_OUT="$OUT_DIR/dynamic_trace_${LABEL}_timeout${TIMEOUT}.jsonl"
    echo "[4/?] (codecontests) Producing dynamic traces (public-pass solutions)..."
    python "$TRACE_SCRIPT" \
      --dataset codecontests \
      --solutions "$SOLS_PUBLIC_PASS" \
      --output "$TRACE_OUT" \
      --timeout "$TIMEOUT" \
      --workers "$WORKERS"

    # 5) Filter dynamic traces to dynamic-pass only
    TRACE_PASS="$OUT_DIR/dynamic_trace_${LABEL}_passonly.jsonl"
    echo "[5/?] (codecontests) Filtering dynamic-pass-only traces..."
    filter_dynamic_pass_traces "$TRACE_OUT" "$TRACE_PASS"

    # 6) DCTD
    DCTD_OUT="$OUT_DIR/dctd_${LABEL}.csv"
    echo "[6/?] (codecontests) Computing DCTD..."
    python "$DCTD_SCRIPT" "$TRACE_PASS" "$DCTD_OUT"

  else
    # ------------------------------ HumanEval path --------------------------
    # 3) Extract HumanEval tests JSONL (one per problem)
    TESTS_JSONL="$OUT_DIR/tests_by_problem.jsonl"
    echo "[3/?] (humaneval) Extracting HumanEval tests JSONL..."
    extract_tests_jsonl_humaneval "$SOLS_SCTD" "$TESTS_JSONL"

    # 4) Dynamic traces on ALL solutions
    TRACE_OUT="$OUT_DIR/dynamic_trace_${LABEL}_timeout${TIMEOUT}.jsonl"
    echo "[4/?] (humaneval) Producing dynamic traces for all solutions..."
    python "$TRACE_SCRIPT" \
      --dataset humaneval \
      --solutions "$SOLS_SCTD" \
      --tests "$TESTS_JSONL" \
      --output "$TRACE_OUT" \
      --timeout "$TIMEOUT" \
      --workers "$WORKERS"

    # 5) DCTD on all traces (pass + fail)
    DCTD_OUT="$OUT_DIR/dctd_${LABEL}.csv"
    echo "[5/?] (humaneval) Computing DCTD on all traces..."
    python "$DCTD_SCRIPT" "$TRACE_OUT" "$DCTD_OUT"
  fi

  echo "Done for $LABEL"
  echo ""
done

# ------------------------------------------------------------------------------
# Merge per-condition CSVs into study-wide CSVs
# ------------------------------------------------------------------------------
echo "Merging study-wide CSVs..."
python - <<'PY' "$OUT_ROOT" "$DATASET"
import csv, os, sys
out_root = sys.argv[1]
dataset = sys.argv[2]

if dataset == "codecontests":
    conds = [("fuzz0","0.0"), ("fuzz0.3","0.3"), ("fuzz0.6","0.6")]
else:
    conds = [("fuzz0","0.0"), ("fuzz0.5","0.5"), ("fuzz0.9","0.9")]

def merge(kind, out_name):
    rows = []
    for label, fp in conds:
        path = os.path.join(out_root, label, f"{kind}_{label}.csv")
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row["fuzz_prob"] = fp
                row["condition"] = label
                rows.append(row)
    if not rows:
        print(f"No rows found for {kind}; skipping merge.")
        return
    headers = []
    for row in rows:
        for k in row.keys():
            if k not in headers:
                headers.append(k)
    out_path = os.path.join(out_root, out_name)
    with open(out_path, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")

merge("sctd", "all_sctd_fuzz_study.csv")
merge("dctd", "all_dctd_fuzz_study.csv")
PY

echo "======================================================================="
echo "All fuzzing conditions processed for dataset '$DATASET'. Study CSVs in: $OUT_ROOT"
echo " - all_sctd_fuzz_study.csv"
echo " - all_dctd_fuzz_study.csv"
echo "======================================================================="