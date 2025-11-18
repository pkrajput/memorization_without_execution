# Output Log Distribution Analysis

This directory contains tools and scripts for analyzing log probabilities and perplexity metrics in code generation tasks. The main focus is on comparing code generation quality between original prompts and synonym-substituted prompts using the HumanEval dataset.

## Overview

This project evaluates the impact of input perturbation (synonym substitution) on code generation metrics, specifically:
- **Mean Log Probability**: Average log probability of generated tokens
- **Perplexity**: Exponential of negative mean log probability
- **Test Pass Rate**: Whether generated code passes the test cases

## Project Structure

```
Output_Log_Distribution/
├── logprobs.py                           # Main analysis script (converted from notebook)
├── logprobs.ipynb                        # Original Jupyter notebook
├── calculateLogits.py                    # Utility functions for API interactions and logprob extraction
├── evaluate_quixbugs_instance.py         # Code evaluation and testing utilities
├── synonym_substitution_standalone.py    # Synonym substitution transformation
├── visualize_ast_example.py              # AST visualization example
├── Result_Original/                      # Results from original prompt experiments
│   ├── HumanEval/                       # HumanEval dataset results
│   └── OpencodeInstruct/                # OpenCodeInstruct dataset results
├── Result_Mutation/                      # Results from synonym-substituted prompt experiments
    ├── HumanEval/                       # HumanEval dataset results
    └── OpencodeInstruct/                # OpenCodeInstruct dataset results

```

## Features

### Main Script: `logprobs.py`

The main script performs two experiments:

1. **Original Input Experiment**: Generates code from original prompts and evaluates:
   - Log probabilities and perplexity
   - Test pass rate
   - Top log probabilities for each token

2. **Modified Input Experiment**: Applies synonym substitution to prompts, then generates code:
   - Same metrics as original experiment
   - Comparison of pass rates between original and modified inputs

### Supporting Modules

- **`calculateLogits.py`**: 
  - OpenAI API wrapper with logprob support
  - Log probability extraction utilities
  - AST-based metrics computation
  - Visualization functions

- **`evaluate_quixbugs_instance.py`**: 
  - Code evaluation against test cases
  - Test execution and result reporting
  - Code cleaning utilities

- **`synonym_substitution_standalone.py`**: 
  - Synonym substitution using spaCy and WordNet
  - Text augmentation for prompt perturbation

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

3. Download NLTK data (WordNet):
```python
python -c "import nltk; nltk.download('wordnet')"
```

4. Set up OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - Or set it in `calculateLogits.py` directly (not recommended for production)

## Usage

### Running the Main Analysis

```bash
python logprobs.py
```

This will:
1. Load the HumanEval dataset
2. Run 10 iterations of the original input experiment
3. Run 10 iterations of the synonym-substituted input experiment
4. Save results to JSON files in `Result_Original/` and `Result_Mutation/`
5. Display summary statistics comparing both experiments

### Configuration

Edit the configuration variables in `main()` function:

```python
my_model = "gpt-3.5-turbo-0125"  # Model to use
MAX_RUN = 10                      # Number of experimental runs
TOP_LOGPROB = 20                  # Number of top logprobs to retrieve
TEMPERATURE = 0.3                 # Sampling temperature
```

### Running Individual Components

#### Evaluate a single code instance:
```python
from evaluate_quixbugs_instance import evaluate_quixbugs_instance

code = "def add(a, b): return a + b"
tests = "assert add(2, 3) == 5"
result = evaluate_quixbugs_instance(code, tests)
print(f"Passed: {result.passed}")
```

#### Apply synonym substitution:
```python
from synonym_substitution_standalone import SynonymSubstitution

transformer = SynonymSubstitution(seed=42, prob=0.5, max_outputs=1)
result = transformer.generate("Write a function to calculate the sum")
print(result[0])
```

#### Extract log probabilities:
```python
from calculateLogits import get_completion, extract_logprobs

messages = [{"role": "user", "content": "Write a Python function"}]
response = get_completion(messages, model="gpt-3.5-turbo-0125", logprobs=True, top_logprobs=10)
logprobs = extract_logprobs(response, "gpt-3.5-turbo-0125")
print(f"Mean logprob: {np.mean(logprobs)}")
```

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "task_id": "...",
  "prompt": "...",
  "test": "...",
  "mean_log": -2.5,
  "perplexity": 12.18,
  "top_logs": [[{"token": "...", "logprob": -1.2}, ...]],
  "passed": true
}
```

For mutation experiments, an additional field:
```json
{
  "original_input": "...",
  ...
}
```

## Dependencies

See `requirements.txt` for the complete list. Key dependencies include:
- `openai`: OpenAI API client
- `datasets`: HuggingFace datasets library
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `spacy`: NLP processing
- `nltk`: Natural language toolkit
- `codebleu`: Code BLEU metric calculation

## Results Analysis

The script automatically analyzes results and displays:
- Pass rate statistics for each run
- Comparison between original and modified inputs
- Run-by-run breakdown of test results

Example output:
```
Result: Original Input
run  passed
0    True      149
     False      15
...

Result: Modified Input - Synonym Substitution
run  passed
0    True      152
     False      12
...
```

## Notes

- The script uses parallel processing via `ThreadPoolExecutor` for faster execution
- Results are saved incrementally (one file per run)
- The synonym substitution uses a 50% substitution probability by default
- Make sure you have sufficient API credits for large-scale experiments
- Results directories are created automatically if they don't exist

## License

Please refer to the parent project's license.

## Citation

If you use this code, please cite the relevant papers:
- HumanEval dataset: [Chen et al., 2021]
- ReAPR dataset: ReAPR: Automatic Program Repair via Retrieval-Augmented Large Language Models

