# Logprobs + Dynamic Stability Fuzzing Pipeline

This repo contains a small pipeline for studying how **input fuzzing** (synonym substitution) affects:

- Token-level **log probability distributions**
- **Static** opcode stability (SCTD)
- **Dynamic** opcode stability (DCTD)
- **Pass rates** across code benchmarks

The pipeline supports **three datasets** via a common flag:

- `humaneval`
- `codecontests`
- `bigobench`

---

## High-Level Pipeline

For each dataset + fuzz setting, the workflow is:

1. **Generate model samples + logprobs** with `logprobs.py`
2. **Tag solutions as public-pass / fail** with `unittests/run_public_unittests_and_tag.sh`
3. **Compute SCTD/DCTD metrics** with `dynamic_stability/run_fuzz_metrics_unittested.sh`
4. **Plot SCTD/DCTD + logprob distributions** with `plot_sctd_dctd_and_logprobs_new.py`

All three datasets share this same structure; only the `--dataset` flag and fuzz levels differ.

---

## Repo Structure (key files only)

```text
.
├── logprobs.py                       # Main sample generation + logprobs script
├── calculateLogits.py                # API wrappers + logprob helpers
├── evaluate_quixbugs_instance.py     # Legacy evaluation helper
├── synonym_substitution_standalone.py# Synonym-substitution transformer

├── dynamic_stability/
│   ├── calculate_sctd.py             # Static Canonical Trace Divergence (SCTD)
│   ├── calculate_dctd.py             # Dynamic Canonical Trace Divergence (DCTD)
│   ├── produce_dynamic_traces.py     # Unified watchdog driver (bigo/cc/humaneval)
│   ├── run_fuzz_metrics_unittested.sh# Unified fuzz-study driver (bigo/cc/humaneval)
│   ├── worker_bigo.py                # BigOBench-specific trace worker
│   ├── worker_cc.py                  # CodeContests-style worker
│   └── worker_humaneval.py           # HumanEval-style worker

├── unittests/
│   ├── auto_unit_test.py
│   ├── unit_test.py
│   └── run_public_unittests_and_tag.sh   # Tags solutions as public pass/fail

├── plot_sctd_dctd_and_logprobs_new.py    # Plotting (SCTD/DCTD + logprob summaries)
├── requirements.txt
└── README.md
```

---

## Installation

From the repo root:

```bash
pip install -r requirements.txt
```

Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Download NLTK WordNet data:

```bash
python - <<'PY'
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
PY
```

Set up OpenAI API credentials via env (recommended):

```bash
export OPENAI_API_KEY=your_key_here
```

---

## 1. Generating Samples + Logprobs (`logprobs.py`)

`logprobs.py` is the generator. It:

- Reads a dataset (`--dataset` flag)
- Calls an OpenAI model with logprobs enabled
- Optionally applies synonym-substitution fuzzing to the prompt
- Writes one JSONL row per completion with:
  - problem metadata
  - model code
  - token logprobs
  - any auxiliary fields needed downstream (tests, categories, etc.)

### Dataset switch

```bash
--dataset {humaneval, codecontests, bigobench}
```

### Example: HumanEval (no fuzzing, 5 completions per problem)

```bash
python logprobs.py   --data /home/prateek/mem/memorization_without_execution/datasets/humaneval/human_eval.jsonl   --dataset humaneval   --model gpt-4o   --temperature 0.7   --max-problems 50   --output /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/withoutfuzzing/he_wf.jsonl   --completions-per-problem 5
```

You then repeat this command for your fuzzed variants (with the appropriate fuzzing flags inside `logprobs.py`), producing JSONLs such as:

- `generated_solutions/humaneval/withoutfuzzing/he_wf.jsonl`  (`fuzz0`)
- `generated_solutions/humaneval/withfuzzing/he_f_0.5.jsonl` (`fuzz0.5`)
- `generated_solutions/humaneval/withfuzzing/he_f_0.9.jsonl` (`fuzz0.9`)

Analogously for:

- `--dataset codecontests`  → CodeContests runs
- `--dataset bigobench`     → BigOBench runs

---

## 2. Tagging Public Pass/Fail (`unittests/`)

Once you have JSONL solution files, you tag each solution as **public-pass / public-fail** using the unittest harness.

From the repo root:

```bash
cd unittests
TIMEOUT=15 ./run_public_unittests_and_tag.sh   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/withoutfuzzing/he_wf.jsonl   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/withfuzzing/he_f_0.5.jsonl   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/withfuzzing/he_f_0.9.jsonl
```

This script:

- Runs public tests on each solution (with `TIMEOUT` seconds per test)
- Writes **unit-tested** JSONLs into the target directory, typically:

```text
generated_solutions/humaneval/unittested/
  ├── unittested_he_wf.jsonl
  ├── unittested_he_f_0.5.jsonl
  └── unittested_he_f_0.9.jsonl
```

Each row is now annotated with at least:

- `status`: `"pass"` / `"fail"` on public tests  
- `public_unittest_category`: category used for SCTD if `category` is missing

You follow the same pattern for `codecontests` and `bigobench`—just point to the corresponding dataset-specific solution files.

---

## 3. Dynamic Stability Metrics (`dynamic_stability/`)

Metrics are computed in `dynamic_stability` using a unified shell driver and Python scripts.

### 3.1. Scripts

- `calculate_sctd.py` — Static Canonical Trace Divergence (SCTD)
- `calculate_dctd.py` — Dynamic Canonical Trace Divergence (DCTD)
- `produce_dynamic_traces.py` — Unified watchdog that dispatches to:
  - `worker_bigo.py`        (BigOBench-style)
  - `worker_cc.py`          (CodeContests-style)
  - `worker_humaneval.py`   (HumanEval-style)
- `run_fuzz_metrics_unittested.sh` — High-level driver that:
  - runs SCTD
  - runs the dynamic tracer
  - optionally filters dynamic-pass traces
  - runs DCTD
  - merges per-fuzz CSVs into study-wide CSVs

From the repo root:

```bash
cd dynamic_stability
```

### 3.2. Unified metrics driver: `run_fuzz_metrics_unittested.sh`

Usage:

```bash
./run_fuzz_metrics_unittested.sh   <dataset>   <unittested_no_fuzz.jsonl>   <unittested_fuzzA.jsonl>   <unittested_fuzzB.jsonl>   <out_dir>
```

Where `<dataset>` ∈ `{bigo, codecontests, humaneval}`.

The **fuzz grid** depends on the dataset:

- `bigo` / `humaneval`:
  - labels: `fuzz0`, `fuzz0.5`, `fuzz0.9`
  - probs:  `0.0`, `0.5`, `0.9`
- `codecontests`:
  - labels: `fuzz0`, `fuzz0.3`, `fuzz0.6`
  - probs:  `0.0`, `0.3`, `0.6`

Optional env overrides:

```bash
TIMEOUT=20       # per-test timeout for dynamic tracing
WORKERS=4        # watchdog parallelism
SCTD_SCRIPT=calculate_sctd.py
DCTD_SCRIPT=calculate_dctd.py
TRACE_SCRIPT=produce_dynamic_traces.py
```

#### Example: HumanEval metrics

```bash
./run_fuzz_metrics_unittested.sh   humaneval   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested/unittested_he_wf.jsonl   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested/unittested_he_f_0.5.jsonl   /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested/unittested_he_f_0.9.jsonl   /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study
```

This will create a directory structure like:

```text
metrics/humaneval/fuzz_study/
  ├── fuzz0/
  │   ├── sctd_fuzz0.csv
  │   └── dctd_fuzz0.csv
  ├── fuzz0.5/
  │   ├── sctd_fuzz0.5.csv
  │   └── dctd_fuzz0.5.csv
  ├── fuzz0.9/
  │   ├── sctd_fuzz0.9.csv
  │   └── dctd_fuzz0.9.csv
  ├── all_sctd_fuzz_study.csv
  └── all_dctd_fuzz_study.csv
```

### 3.3. Dataset-specific behavior (inside `run_fuzz_metrics_unittested.sh`)

- **BigOBench (`bigo`)**
  - Filters to `status == "pass"` for **public-pass** solutions
  - Extracts tests per problem into a separate JSONL for the watchdog
  - Dynamic tracing only on public-pass solutions
  - Filters dynamic traces to `status == "pass"` before DCTD

- **CodeContests (`codecontests`)**
  - Tests are embedded in each solution (`private_tests`)
  - Filters to `status == "pass"` and traces only these
  - Filters traces to `status == "pass"` before DCTD

- **HumanEval (`humaneval`)**
  - Extracts `test` + `entry_point` per problem from solutions
  - Runs dynamic tracing on **all** solutions
  - Runs DCTD on **all** traces (pass + fail)

---

## 4. Plotting SCTD/DCTD + Logprobs

Once metrics are computed, use `plot_sctd_dctd_and_logprobs_new.py` from the repo root.

### Example: HumanEval fuzz study

```bash
python plot_sctd_dctd_and_logprobs_new.py   --dataset humaneval   --sctd-csvs     /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study/fuzz0/sctd_fuzz0.csv     /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study/fuzz0.5/sctd_fuzz0.5.csv     /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study/fuzz0.9/sctd_fuzz0.9.csv   --dctd-csvs     /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study/fuzz0/dctd_fuzz0.csv     /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study/fuzz0.5/dctd_fuzz0.5.csv     /home/prateek/mem/memorization_without_execution/metrics/humaneval/fuzz_study/fuzz0.9/dctd_fuzz0.9.csv   --jsonls     /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested/unittested_he_wf.jsonl     /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested/unittested_he_f_0.5.jsonl     /home/prateek/mem/memorization_without_execution/generated_solutions/humaneval/unittested/unittested_he_f_0.9.jsonl   --out-dir     /home/prateek/mem/memorization_without_execution/plots/humaneval_fuzz_study
```

Flags:

- `--dataset` — one of: `humaneval`, `codecontests`, `bigobench`
- `--sctd-csvs` — one CSV per fuzz level in **(fuzz0, fuzzA, fuzzB)** order
- `--dctd-csvs` — same order as SCTD (for datasets where DCTD is computed)
- `--jsonls` — unit-tested JSONLs in matching fuzz order
- `--out-dir` — directory where plots and auxiliary CSVs are written

The script produces, per dataset:

- SCTD / DCTD **box-with-points** plots (with per-problem markers)
- **Delta violin** plots for `fuzz0 − fuzzX`
- Logprob trajectories (mean, naive entropy, spread via IQR/std)
- Per-problem logprob spread CSVs and point-ID maps for linking back to `problem_id`

---

## Notes & Extensions

- All scripts expect per-solution JSONL with consistent `problem_id` keys.
- Fuzzed vs non-fuzzed runs are handled by filenames and argument order; there is no hidden state.
- To add a new dataset, you would:
  - extend `logprobs.py` to support `--dataset newdataset`
  - add a worker / branch in `produce_dynamic_traces.py`
  - add a small dataset-specific branch in `run_fuzz_metrics_unittested.sh`
  - optionally extend the plotting script for any dataset-specific quirks

---

## License / Citation

Please refer to the parent project’s license.

If you build on this work, please cite:

- HumanEval: Chen et al., 2021  
- Any papers where SCTD / DCTD are introduced and described.

---

## Datasets

All JSONL datasets used in this pipeline (HumanEval, CodeContests, BigOBench variants, etc.) are available here:

- Full dataset folder: https://drive.google.com/drive/folders/1kC-65rvDoXFNlagWZzPkl9RqEEhf-u7J?usp=sharing

