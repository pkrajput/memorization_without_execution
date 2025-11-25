#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Generator with optional prompt fuzzing
and chosen-token probabilities.

Supports multiple JSONL dataset formats (CodeContests, BigO-Bench, Humaneval).

What this script does:
  1) Loads a JSONL dataset (format chosen by --dataset or auto).
  2) For each problem, generates K independent solutions using an OpenAI chat model.
  3) Optionally perturbs (fuzzes) the prompt via synonym substitution before generation
  4) Writes one JSONL row per completion to --output.

Each output row includes:
  - problem_id
  - solution_id, completion_index
  - original description
  - fuzzed_description (if fuzzing changed it)
  - generated_solution (extracted code from a python fenced block if possible)
  - raw_llm_output (full assistant text)
  - token_probs: compact list of chosen tokens and their probabilities
  - dataset (as passed)
  - dataset-specific metadata when available

Important:
  - No installs/downloads happen inside this script.
  - If fuzzing is enabled, you must preinstall:
      * spaCy model (default: en_core_web_sm)
      * NLTK WordNet corpus
"""

import argparse
import json
import os
import re
import time
import uuid
from math import exp
from typing import Optional, List, Dict, Any, Tuple

from openai import OpenAI
from tqdm import tqdm

# Optional prompt fuzzing imports, guarded so script can run without them.
try:
    import spacy
    import nltk
    from synonym_substitution_standalone import synonym_substitution
except Exception:
    spacy = None
    nltk = None
    synonym_substitution = None


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------

def normalize_problem_id(problem: dict, line_num: int, dataset: str) -> str:
    """
    Robust problem_id derivation supporting multiple datasets.
    """
    pid = (
        problem.get("problem_id")
        or problem.get("id")
        or problem.get("task_id")
        or problem.get("name")
        or problem.get("problem_name")  # BigO-Bench
        or problem.get("title")
    )
    if not pid:
        pid = f"line_{line_num}"
        print(f"Warning: no id field on line {line_num}; using synthetic problem_id '{pid}'.")
    return str(pid)


def normalize_description(problem: dict, dataset: str) -> str:
    """
    Robust description extraction supporting multiple datasets.
    """
    # Most datasets already use description
    desc = problem.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()

    # Fallbacks (covers CodeContests/BigO-Bench and Humaneval's 'prompt')
    for k in ("statement", "prompt", "question", "problem_description", "text"):
        v = problem.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def load_dataset(path: str, dataset: str = "auto") -> List[dict]:
    """
    Load problems from a JSONL file.

    Ensures:
      - guaranteed 'problem_id'
      - guaranteed 'description' (best-effort)
      - dedup on problem_id

    Args:
        path: JSONL dataset path.
        dataset: 'codecontests', 'bigobench', 'humaneval', or 'auto'
    """
    problems: List[dict] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                problem = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed JSON at line {line_num}: {e}")
                continue

            if not isinstance(problem, dict):
                print(f"Warning: line {line_num} is not a JSON object; skipping.")
                continue

            pid = normalize_problem_id(problem, line_num, dataset)
            if pid in seen_ids:
                continue

            problem["problem_id"] = pid
            problem["description"] = normalize_description(problem, dataset)

            seen_ids.add(pid)
            problems.append(problem)

    return problems


def extract_code_block(content: str) -> str:
    """
    Extract a Python code block from model output.

    Preferred:
      ```python
      ...
      ```

    Fallback:
      take text from the first 'def solve' line onward.
    """
    match = re.search(
        r"```(?:python|py)?\s*\n(.*?)\n\s*```",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    lines = content.splitlines()
    code_lines: List[str] = []
    inside_code = False
    for line in lines:
        if line.strip().startswith("def solve"):
            inside_code = True
        if inside_code:
            code_lines.append(line)

    return "\n".join(code_lines).strip() if code_lines else ""


# ---------------------------------------------------------------------
# Logprob utilities
# ---------------------------------------------------------------------

def serialize_token_probs(lp_obj) -> Optional[List[Dict[str, Any]]]:
    """
    Convert OpenAI logprobs into compact chosen-token probabilities.

    Output format:
      [{"token": "<token_str>", "prob": <linear_prob>}, ...]
    """
    if lp_obj is None or getattr(lp_obj, "content", None) is None:
        return None

    out: List[Dict[str, Any]] = []
    for tok in lp_obj.content:
        token_str = getattr(tok, "token", None)
        logp = getattr(tok, "logprob", None)
        if token_str is None or logp is None:
            continue
        out.append({"token": token_str, "prob": float(exp(logp))})

    return out


def _extract_text_from_completion(completion) -> str:
    """
    Robustly extract assistant text from a chat completion object,
    handling slight SDK/format differences.
    """
    choice = completion.choices[0]

    msg = getattr(choice, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(p.get("text", ""))
            joined = "".join(parts).strip()
            if joined:
                return joined

    legacy_text = getattr(choice, "text", None)
    if isinstance(legacy_text, str) and legacy_text.strip():
        return legacy_text

    return ""


def create_completion_openai(
    client: OpenAI,
    model: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    retries: int = 5,
    retry_sleep: float = 1.0,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Call OpenAI Chat Completions with retries and chosen-token logprobs.

    We try max_completion_tokens first (newer models/SDKs),
    then fall back to max_tokens for compatibility.

    Returns:
        (assistant_text, token_probs)
    """
    last_err = None

    for attempt in range(retries):
        try:
            params: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                logprobs=True,
            )
            if seed is not None:
                params["seed"] = seed

            try:
                completion = client.chat.completions.create(
                    **params, max_completion_tokens=max_tokens
                )
            except TypeError:
                completion = client.chat.completions.create(
                    **params, max_tokens=max_tokens
                )

            text = _extract_text_from_completion(completion).strip()
            if not text:
                fr = getattr(completion.choices[0], "finish_reason", None)
                raise RuntimeError(f"Empty completion text (finish_reason={fr})")

            token_probs = serialize_token_probs(completion.choices[0].logprobs)
            return text, token_probs

        except Exception as e:
            last_err = e
            sleep_s = retry_sleep * (2 ** attempt)
            time.sleep(min(sleep_s, 8.0))

    raise last_err


# ---------------------------------------------------------------------
# Prompt fuzzing utilities
# ---------------------------------------------------------------------

def setup_fuzzing(prompt_fuzzing: Optional[float], spacy_model: str):
    """
    Initialize synonym-based prompt fuzzing resources.
    """
    if prompt_fuzzing is None:
        return None

    if synonym_substitution is None or spacy is None or nltk is None:
        raise ImportError(
            "Prompt fuzzing requested, but synonym_substitution_standalone "
            "or dependencies (spacy, nltk) are not available."
        )

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        raise LookupError(
            "NLTK WordNet corpus not found. Install externally, e.g.\n"
            "  python -c \"import nltk; nltk.download('wordnet')\""
        )

    try:
        nlp = spacy.load(spacy_model)
    except OSError as e:
        raise OSError(
            f"spaCy model '{spacy_model}' not found. Install externally, e.g.\n"
            f"  python -m spacy download {spacy_model}"
        ) from e

    return nlp


def maybe_fuzz_description(
    description: str,
    nlp,
    prompt_fuzzing: Optional[float],
    base_seed: int,
    problem_id: str,
    completion_index: int,
    temperature: float,
) -> str:
    """
    Fuzz a problem description with synonym substitution if enabled.
    """
    if nlp is None or prompt_fuzzing is None:
        return description

    local_seed = (
        base_seed
        ^ hash(problem_id)
        ^ completion_index
        ^ int(temperature * 1000)
    ) & 0xFFFFFFFF

    variants = synonym_substitution(
        text=description,
        spacy_pipeline=nlp,
        seed=local_seed,
        prob=prompt_fuzzing,
        max_outputs=1,
    )
    return variants[0] if variants else description


# ---------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------

def resolve_output_file(output_path: str, out_prefix: str, model: str, temperature: float) -> str:
    """
    Resolve final output file location.
    """
    safe_model = model.replace("/", "_")
    output_path = os.path.abspath(output_path)

    is_dir = output_path.endswith(os.sep) or (
        os.path.exists(output_path) and os.path.isdir(output_path)
    )
    if is_dir:
        os.makedirs(output_path, exist_ok=True)
        fname = f"{out_prefix}_{safe_model}_temp{temperature}.jsonl"
        return os.path.join(output_path, fname)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


# ---------------------------------------------------------------------
# Dataset-specific metadata packers
# ---------------------------------------------------------------------

def pack_codecontests_fields(problem: dict) -> Dict[str, Any]:
    return {
        "difficulty": problem.get("difficulty"),
        "solutions": problem.get("solutions"),
        "generated_tests": problem.get("generated_tests"),
        "public_tests": (problem.get("tests") or {}).get("public_tests", []),
        "private_tests": (problem.get("tests") or {}).get("private_tests", []),
    }


def pack_bigobench_fields(problem: dict) -> Dict[str, Any]:
    tests = problem.get("tests") or {}
    return {
        "problem_name": problem.get("problem_name"),
        "reference_solution_id": problem.get("solution_id"),
        "reference_solution_code": problem.get("solution_code"),
        "reference_dataclass_code": problem.get("dataclass_code"),
        "inputs_example": problem.get("inputs_example"),
        "time_complexity_inferred": problem.get("time_complexity_inferred"),
        "time_curve_coefficient": problem.get("time_curve_coefficient"),
        "problem_time_curve_coefficient_list": problem.get("problem_time_curve_coefficient_list"),
        "public_tests": tests.get("public_tests", []),
        "private_tests": tests.get("private_tests", []),
        "generated_tests": tests.get("generated_tests", []),
    }


def pack_humaneval_fields(problem: dict) -> Dict[str, Any]:
    """
    Pack relevant HumanEval fields.

    Example structure:
      {
        "task_id": "test/0",
        "prompt": "def return1():\n",
        "canonical_solution": "    return 1",
        "test": "def check(candidate):\n    assert candidate() == 1",
        "entry_point": "return1"
      }
    """
    return {
        "task_id": problem.get("task_id"),
        "prompt": problem.get("prompt"),
        "canonical_solution": problem.get("canonical_solution"),
        "test": problem.get("test"),
        "entry_point": problem.get("entry_point"),
    }


def pack_dataset_fields(problem: dict, dataset: str) -> Dict[str, Any]:
    if dataset == "codecontests":
        return pack_codecontests_fields(problem)
    if dataset == "bigobench":
        return pack_bigobench_fields(problem)
    if dataset == "humaneval":
        return pack_humaneval_fields(problem)

    # auto: include whatever exists
    out: Dict[str, Any] = {}
    out.update(pack_codecontests_fields(problem))
    out.update(pack_bigobench_fields(problem))
    out.update(pack_humaneval_fields(problem))
    return out


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate completions sequentially with OpenAI, optional prompt fuzzing, and token probabilities."
    )
    parser.add_argument("--data", required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "codecontests", "bigobench", "humaneval"],
        help="Dataset schema hint. Use 'auto' to infer by fields."
    )
    parser.add_argument("--model", required=True, help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, required=True, help="Single temperature value.")
    parser.add_argument("--output", required=True, help="Output file path or directory.")
    parser.add_argument(
        "--out-prefix",
        default="generated_completions",
        help="Used only if --output is a directory."
    )
    parser.add_argument("--completions-per-problem", type=int, default=5)
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Only process first N problems."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for fuzzing (and optionally OpenAI seed)."
    )
    parser.add_argument(
        "--use-openai-seed",
        action="store_true",
        help="If set, pass --seed into OpenAI calls."
    )
    parser.add_argument(
        "--prompt-fuzzing",
        type=float,
        default=None,
        help="Synonym substitution probability in [0,1]. Omit for no fuzzing."
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model to use if fuzzing enabled."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional custom base_url. Leave unset for official OpenAI endpoint."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max output token budget."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries per request on transient failures."
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Base sleep seconds between retries."
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Optional sleep seconds between successful calls."
    )

    args = parser.parse_args()

    # Humaneval uses code-only 'prompt' fields; fuzzing is not meaningful there.
    if args.dataset == "humaneval" and args.prompt_fuzzing is not None:
        print("Warning: --prompt-fuzzing is ignored for 'humaneval' (code-only prompts).")
        args.prompt_fuzzing = None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    problems = load_dataset(args.data, dataset=args.dataset)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    output_file = resolve_output_file(args.output, args.out_prefix, args.model, args.temperature)

    completed_problem_keys = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_problem_keys.add((data["problem_id"], data.get("completion_index", 0)))
                except Exception:
                    continue

    nlp = setup_fuzzing(args.prompt_fuzzing, args.spacy_model)

    print("Run configuration")
    print(f"  dataset: {args.dataset}")
    print(f"  problems: {len(problems)}")
    print(f"  completions_per_problem: {args.completions_per_problem}")
    print(f"  model: {args.model}")
    print(f"  temperature: {args.temperature}")
    print(f"  prompt_fuzzing: {args.prompt_fuzzing}")
    print(f"  output_file: {output_file}")
    if completed_problem_keys:
        print(f"  resume_from_existing: {len(completed_problem_keys)} items already present")

    total_jobs = len(problems) * args.completions_per_problem
    pbar = tqdm(total=total_jobs, desc="Generating completions", dynamic_ncols=True)

    with open(output_file, "a", encoding="utf-8") as out_f:
        for problem in problems:
            pid = problem["problem_id"]
            original_desc = str(problem.get("description") or "").strip()

            for i in range(args.completions_per_problem):
                key = (pid, i)
                if key in completed_problem_keys:
                    pbar.update(1)
                    continue
                else:
                    fuzzed_desc = maybe_fuzz_description(
                        description=original_desc,
                        nlp=nlp,
                        prompt_fuzzing=args.prompt_fuzzing,
                        base_seed=args.seed,
                        problem_id=str(pid),
                        completion_index=i,
                        temperature=args.temperature,
                    )

                # Dataset-specific prompting
                if args.dataset == "humaneval":
                    prompt_text = problem.get("prompt") or fuzzed_desc
                    entry_point = problem.get("entry_point") or "solve"
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a highly skilled Python programmer.\n"
                                "You will be given the signature (and possibly docstring) of a function.\n"
                                "Your task is to implement this function correctly.\n\n"
                                "Requirements:\n"
                                "- Do NOT change the function name or its arguments.\n"
                                "- Do NOT write any test code, main function, or print statements for debugging.\n"
                                "- You may add imports and helper functions if needed.\n"
                                "- Return ONLY valid Python code, enclosed in a single code block:\n"
                                "```python\n<your_code_here>\n```"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Complete the following Python function so that it satisfies its specification "
                                "and passes a hidden test suite:\n\n"
                                f"{prompt_text}\n\n"
                                "Write the full implementation of this function."
                            ),
                        },
                    ]
                else:
                    # CodeContests / BigO-Bench / others: standard 'solve(input_lines)' template
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a highly skilled competitive programmer. "
                                "Your task is to write ONLY the complete Python code for a function named `solve`.\n\n"
                                "### Function Requirements:\n"
                                "- The function signature must be: `def solve(input_lines: list[str]) -> str`\n"
                                "- It will receive input as a list of strings (`input_lines`).\n"
                                "- Return the final output as a string (not using print).\n"
                                "- Match the exact expected format.\n\n"
                                "### Code Block Format:\n"
                                "- Return your solution enclosed in triple backticks like:\n"
                                "```python\n<your_code_here>\n```"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Here is the problem statement:\n\n"
                                f"{fuzzed_desc}\n\n"
                                "Write the correct implementation of `solve` based on this description."
                            ),
                        },
                    ]

                try:
                    raw_llm_output, token_probs = create_completion_openai(
                        client=client,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        seed=args.seed if args.use_openai_seed else None,
                        retries=args.retries,
                        retry_sleep=args.retry_sleep,
                    )

                    generated_solution_code = extract_code_block(raw_llm_output)

                    info = {
                        "dataset": args.dataset,
                        "problem_id": pid,
                        "solution_id": str(uuid.uuid4()),
                        "completion_index": i,
                        "description": original_desc,
                        "fuzzed_description": (
                            fuzzed_desc if fuzzed_desc != original_desc else None
                        ),
                        "prompt_fuzzing_prob": args.prompt_fuzzing,
                        "generated_solution": generated_solution_code,
                        "raw_llm_output": raw_llm_output,
                        "token_probs": token_probs,
                        "model": args.model,
                        "temperature": args.temperature,
                    }
                    info.update(pack_dataset_fields(problem, args.dataset))

                except Exception as e:
                    info = {
                        "dataset": args.dataset,
                        "problem_id": pid,
                        "solution_id": str(uuid.uuid4()),
                        "completion_index": i,
                        "description": original_desc,
                        "fuzzed_description": (
                            fuzzed_desc if fuzzed_desc != original_desc else None
                        ),
                        "prompt_fuzzing_prob": args.prompt_fuzzing,
                        "error_category": "api_call_failure",
                        "error_message": str(e),
                        "generated_solution": "",
                        "raw_llm_output": "",
                        "token_probs": None,
                        "model": args.model,
                        "temperature": args.temperature,
                    }
                    info.update(pack_dataset_fields(problem, args.dataset))

                out_f.write(json.dumps(info, ensure_ascii=False) + "\n")
                out_f.flush()

                completed_problem_keys.add(key)
                pbar.update(1)

                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)

    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()