"""
Log Probabilities Analysis for Code Generation

This script analyzes log probabilities and perplexity metrics for code generation
on the HumanEval dataset. It compares original prompts with synonym-substituted
prompts to evaluate the impact on code generation quality.

The script performs two main experiments:
1. Original Input: Generate code from original prompts
2. Modified Input: Generate code from synonym-substituted prompts

Results are saved to JSON files in Result_Original/ and Result_Mutation/ directories.
"""

from datasets import load_dataset
from synonym_substitution_standalone import SynonymSubstitution
from calculateLogits import get_completion, extract_logprobs
from tqdm import tqdm
import numpy as np
import pandas as pd
from evaluate_quixbugs_instance import evaluate_quixbugs_instance
import random
import concurrent.futures
import os


def get_metrics(response, my_model):
    """
    Extract metrics from API response including mean log probability,
    perplexity, and top log probabilities.
    
    Args:
        response: API response object
        my_model: Model name string
        
    Returns:
        tuple: (mean_logprob, perplexity, top_logs)
    """
    logs = []
    logprob_values = extract_logprobs(response, my_model)
    if logprob_values:
        mean_logprob = float(np.mean(logprob_values))
        perplexity = float(np.exp(-mean_logprob))
    else:
        mean_logprob = float('-inf')
        perplexity = float('inf')

    # Safely extract top logprobs if available
    if hasattr(response, 'choices') and len(response.choices) > 0:
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            if hasattr(response.choices[0].logprobs, 'content') and response.choices[0].logprobs.content:
                for token in response.choices[0].logprobs.content:
                    if hasattr(token, 'top_logprobs') and token.top_logprobs:
                        token_log = [{"token": t.token, "logprob": t.logprob} for t in token.top_logprobs]
                        logs.append(token_log)
        
    return mean_logprob, perplexity, logs


def process_row(idx, row, my_model, TOP_LOGPROB, TEMPERATURE):
    """
    Process a single row from the dataset by generating code and evaluating it.
    
    Args:
        idx: Row index
        row: DataFrame row containing prompt and test information
        my_model: Model name to use
        TOP_LOGPROB: Number of top logprobs to retrieve
        TEMPERATURE: Sampling temperature
        
    Returns:
        tuple: (idx, mean_log, perplexity, top_logs, passed)
    """
    try:
        instruction = row["prompt"]
        messages = [
            {"role": "system", "content": "Carefully consider edge cases in the code. Write idiomatic code that respects the idioms of the programming language. Output only source code, without any commentary surrounding it."},
            {"role": "user", "content": f"{instruction}"}
        ]
        
        response = get_completion(messages=messages, model=my_model, logprobs=True, top_logprobs=TOP_LOGPROB, temperature=TEMPERATURE)
        mean_log, perplexity, top_logs = get_metrics(response, my_model)

        tests = row["test"].split("def check(candidate):", 1)[1].replace("    ", "").split("\n")
        while "" in tests:
            tests.remove("")
        
        # Safely extract response content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and response.choices[0].message:
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    passed = evaluate_quixbugs_instance(response.choices[0].message.content, str(tests)).passed
                else:
                    passed = False
            else:
                passed = False
        else:
            passed = False
        
        return idx, mean_log, perplexity, top_logs, passed
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, None, None, None, False


def run_original_experiment(df_tmp, my_model, MAX_RUN, TOP_LOGPROB, TEMPERATURE, output_dir="Result_Original"):
    """
    Run experiment on original (unmodified) prompts.
    
    Args:
        df_tmp: DataFrame with dataset
        my_model: Model name to use
        MAX_RUN: Number of runs to perform
        TOP_LOGPROB: Number of top logprobs to retrieve
        TEMPERATURE: Sampling temperature
        output_dir: Directory to save results
    """
    print("## Generation: Original Input")
    print(f"Running generation process against {len(df_tmp)} samples of the original HumanEval dataset.")
    print(f"Model: {my_model}, Runs: {MAX_RUN}, Top Logprobs: {TOP_LOGPROB}, Temperature: {TEMPERATURE}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for run in range(MAX_RUN):
        print(f"Run {run + 1}/{MAX_RUN}")
        df = df_tmp.copy()
        df["top_logs"], df["mean_log"], df["perplexity"], df["passed"] = None, None, None, None
        
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, idx, row, my_model, TOP_LOGPROB, TEMPERATURE): idx 
                      for idx, row in df.iterrows()}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing run {run + 1}"):
                try:
                    idx, mean_log, perplexity, top_logs, passed = future.result()
                    df.at[idx, "mean_log"] = mean_log
                    df.at[idx, "perplexity"] = perplexity
                    df.at[idx, "top_logs"] = top_logs
                    df.at[idx, "passed"] = passed
                except Exception as e:
                    print(f"Error processing future: {e}")
        
        output_file = f"./{output_dir}/HUMANEVAL_FULL_Sample_Code_Synthesis_LogProbs_Results_GPT_3.5_run_{run}.json"
        df.to_json(output_file, orient="records", indent=2)
        print(f"✓ Results saved to {output_file}\n")


def run_mutation_experiment(df_tmp, my_model, MAX_RUN, TOP_LOGPROB, TEMPERATURE, output_dir="Result_Mutation"):
    """
    Run experiment on synonym-substituted (modified) prompts.
    
    Args:
        df_tmp: DataFrame with dataset
        my_model: Model name to use
        MAX_RUN: Number of runs to perform
        TOP_LOGPROB: Number of top logprobs to retrieve
        TEMPERATURE: Sampling temperature
        output_dir: Directory to save results
    """
    print("\n## Generation: Modified Input (Synonym Substitution)")
    print(f"Running generation process against {len(df_tmp)} samples with synonym-substituted prompts.")
    print(f"Model: {my_model}, Runs: {MAX_RUN}, Top Logprobs: {TOP_LOGPROB}, Temperature: {TEMPERATURE}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for run in range(MAX_RUN):
        print(f"Run {run + 1}/{MAX_RUN}")
        df = df_tmp.copy()
        df["original_input"] = df["prompt"]
        df["top_logs"], df["mean_log"], df["perplexity"], df["passed"] = None, None, None, None

        print("ALTERNATIVE INPUT GENERATION")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying synonym substitution"):
            transformer = SynonymSubstitution(seed=random.randint(0, 100), prob=0.5, max_outputs=1)
            df.at[idx, "prompt"] = transformer.generate(row["original_input"])[0]

        print("PROCESS DONE ✅\n")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, idx, row, my_model, TOP_LOGPROB, TEMPERATURE): idx 
                      for idx, row in df.iterrows()}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing run {run + 1}"):
                try:
                    idx, mean_log, perplexity, top_logs, passed = future.result()
                    df.at[idx, "mean_log"] = mean_log
                    df.at[idx, "perplexity"] = perplexity
                    df.at[idx, "top_logs"] = top_logs
                    df.at[idx, "passed"] = passed
                except Exception as e:
                    print(f"Error processing future: {e}")
        
        output_file = f"./{output_dir}/HUMAN_EVAL_Sample_Code_Synthesis_LogProbs_Results_GPT_3.5_run_{run}.json"
        df.to_json(output_file, orient="records", indent=2)
        print(f"✓ Results saved to {output_file}\n")


def analyze_results(original_dir="./Result_Original/HumanEval", mutation_dir="./Result_Mutation/HumanEval"):
    """
    Analyze and compare results from original and mutation experiments.
    
    Args:
        original_dir: Directory containing original experiment results
        mutation_dir: Directory containing mutation experiment results
    """
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Analyze original results
    if os.path.exists(original_dir):
        files = [os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.endswith(".json")]
        if files:
            df_original = pd.DataFrame()
            run = 0
            for f in sorted(files):
                df_tmp = pd.read_json(f, orient="records")
                df_tmp["run"] = run
                run += 1
                df_original = pd.concat([df_original, df_tmp], ignore_index=True)

            print("\nResult: Original Input")
            print(df_original.groupby("run")["passed"].value_counts())
    
    # Analyze mutation results
    if os.path.exists(mutation_dir):
        files = [os.path.join(mutation_dir, f) for f in os.listdir(mutation_dir) if f.endswith(".json")]
        if files:
            df_mutation = pd.DataFrame()
            run = 0
            for f in sorted(files):
                df_tmp = pd.read_json(f, orient="records")
                df_tmp["run"] = run
                run += 1
                df_mutation = pd.concat([df_mutation, df_tmp], ignore_index=True)

            print("\nResult: Modified Input - Synonym Substitution")
            print(df_mutation.groupby("run")["passed"].value_counts())
    
    print("\n" + "="*70)


def main():
    """
    Main function to run the log probabilities analysis experiments.
    """
    # Configuration
    my_model = "gpt-3.5-turbo-0125"
    MAX_RUN = 10
    TOP_LOGPROB = 20
    TEMPERATURE = 0.3
    
    # Load dataset
    print("Loading HumanEval dataset...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    df = ds.to_pandas()
    print(f"Loaded {len(df)} samples from HumanEval dataset.\n")
    
    # Prepare data
    df_tmp = df.copy()
    print(f"Prepared {len(df_tmp)} samples for processing.\n")
    
    # Run experiments
    print("="*70)
    print("STARTING EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: Original Input
    run_original_experiment(df_tmp, my_model, MAX_RUN, TOP_LOGPROB, TEMPERATURE)
    
    # Experiment 2: Modified Input (Synonym Substitution)
    run_mutation_experiment(df_tmp, my_model, MAX_RUN, TOP_LOGPROB, TEMPERATURE)
    
    # Analyze results
    analyze_results()
    
    print("\n✓ All experiments completed!")


if __name__ == "__main__":
    main()

