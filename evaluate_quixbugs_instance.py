"""
Standalone QuixBugs instance evaluation module.

This module provides functionality to evaluate code responses by running them against
test cases. It combines the evaluation logic from evaluation/metrics/pass_test.py
and utility functions from evaluation/metrics/code_utils.py into a standalone,
self-contained module.

Based on the code from:
- evaluation/metrics/pass_test.py (evaluate_quixbugs_instance)
- evaluation/metrics/code_utils.py (clean_code, remove_duplicate)
"""

import sys
import os
import tempfile
import subprocess
import time
import re
from dataclasses import dataclass


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestReport:
    """Report indicating test execution results."""
    passed: bool
    functional_error: bool
    runtime_error: bool
    message: str


# ============================================================================
# Code Utility Functions
# ============================================================================

def remove_comments(response: str) -> str:
    """
    Remove comments and docstrings from the code response.
    
    Args:
        response (str): The code response to clean.
        
    Returns:
        str: The code response without comments and docstrings.
    """
    lines = response.splitlines()
    cleaned_lines = []
    for line in lines:
        if not line.strip().startswith("#"):
            cleaned_lines.append(line)
    # Remove docstrings
    cleaned_lines = [re.sub(r'"""(.*?)"""', '', line, flags=re.DOTALL) for line in cleaned_lines]
    
    return "\n".join(cleaned_lines).strip()


def clean_code(response: str) -> str:
    """
    Clean the code response by removing code fences, comments, and whitespace.
    
    Args:
        response (str): The code response to clean.
        
    Returns:
        str: The cleaned code response.
    """
    if "```python" in response:
        response = response.split("```python")[1]
    if "```" in response:
        response = response.split("```")[0]
    response = remove_comments(response)
    return response.strip()


def remove_duplicate(res: str, ref_output: str) -> str:
    """
    Clean the response by capturing the relevant lines from the reference output.
    Used to handle cases where LLM responses contain duplicate code or blabbering.

    Args:
        res (str): The response output from the LLM.
        ref_output (str): The reference output to compare against.
        
    Returns:
        str: The cleaned response containing only the relevant lines.
    """
    ref_lines = ref_output.splitlines()
    ref_out_lines = res.splitlines()

    # In case of multiple answers, we take the first one
    try:
        if ref_out_lines.count(ref_lines[0]) > 1:
            res_ = ref_out_lines[0] + "\n"
            for idx in range(len(ref_out_lines) + 1):
                if ref_out_lines[idx] != ref_out_lines[0] and idx != 0:
                    res_ += ref_out_lines[idx] + "\n"
                elif idx != 0:
                    break
            return res_.replace("User: ", "").replace("Assistant: ", "").strip()
    except IndexError:
        pass

    # In case of blabbering, we identify the code based on the first line of the reference output
    try:
        f_index = ref_out_lines.index(ref_lines[0])
    except ValueError:
        f_index = -1
        for idx in range(len(ref_lines)):
            if "import " in ref_out_lines[idx] or ("def" in ref_out_lines[idx] and ":" in ref_out_lines[idx]):
                f_index = idx
    if f_index != -1:
        res = "\n".join(ref_out_lines[f_index:f_index+len(ref_lines)+1])
        if len(res.splitlines()[-1].split()) > 5:
            return "\n".join(ref_out_lines[f_index:f_index+len(ref_lines)]).replace("User: ", "").replace("Assistant: ", "").strip()
    return res.replace("User: ", "").replace("Assistant: ", "").strip()


# ============================================================================
# Test Name Update Function
# ============================================================================

def update_test_name(response: str, test: str) -> str:
    """
    Update the test name in the response code to match the function name.
    
    Args:
        response (str): The code response from the LLM.
        test (str): The test to be updated.
        
    Returns:
        str: The updated code with the new test name.
    """
    function_name = response.rsplit('def ', 1)[-1].split('(', 1)[0].strip()
    test_name = test.split("assert", 1)[1].split("(", 1)[0].strip()
    if function_name not in test:
        test = test.replace(test_name, function_name)
    return test


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_quixbugs_instance(
    response: str, 
    tests: str, 
    timeout: int = 10, 
    programming_language: str = "python", 
    ref_output: str = "", 
    hard_clean: bool = False
) -> TestReport:
    """
    Evaluate a QuixBugs instance by running the provided code against tests.

    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds. Defaults to 10.
        programming_language (str): Programming language of the code. Defaults to "python".
        ref_output (str): Reference output for hard cleaning (optional). Defaults to "".
        hard_clean (bool): If True, use remove_duplicate for aggressive cleaning. Defaults to False.

    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    response = clean_code(response)
    if hard_clean:
        response = remove_duplicate(response, ref_output)

    if programming_language == "python":
        tests = update_test_name(response, tests)
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
            combined_code = f"{response}\n\n# Tests\n{tests}"
            temp_file.write(combined_code)
            temp_file_path = temp_file.name
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return TestReport(
                    passed=True,
                    functional_error=False,
                    runtime_error=False,
                    message=f"All tests passed successfully.\nOutput: {result.stdout}"
                )
            else:
                if "AssertionError" in result.stderr or "assert" in result.stderr:
                    return TestReport(
                        passed=False,
                        functional_error=True,
                        runtime_error=False,
                        message=f"Test assertion failed: {result.stderr}"
                    )
                else:
                    return TestReport(
                        passed=False,
                        functional_error=False,
                        runtime_error=True,
                        message=f"Code execution error: {result.stderr}"
                    )
                    
        except subprocess.TimeoutExpired:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message="Test timed out"
            )
        except Exception as e:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message=f"Runtime error: {str(e)}"
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    else:
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message=f"Unsupported programming language: {programming_language}"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example Python code with test
    python_code = """
def calculate_sum(a, b):
    return a + b
"""

    test_code = """
assert calculate_sum(2, 3) == 5
assert calculate_sum(0, 0) == 0
assert calculate_sum(-1, 1) == 0
print("All tests passed!")
"""

    print("Testing evaluate_quixbugs_instance...")
    print("\nOriginal code:")
    print(python_code)
    print("\nTest code:")
    print(test_code)
    print("\n" + "="*60 + "\n")

    # Test 1: Valid code with passing tests
    report1 = evaluate_quixbugs_instance(
        response=python_code,
        tests=test_code,
        timeout=10,
        programming_language="python"
    )
    
    print("Test 1 - Valid code with passing tests:")
    print(f"  Passed: {report1.passed}")
    print(f"  Functional Error: {report1.functional_error}")
    print(f"  Runtime Error: {report1.runtime_error}")
    print(f"  Message: {report1.message[:100]}..." if len(report1.message) > 100 else f"  Message: {report1.message}")
    print("\n" + "="*60 + "\n")

    # Test 2: Code with failing test
    failing_code = """
def calculate_sum(a, b):
    return a + b + 1  # Bug: adds 1 extra
"""

    failing_test = """
assert calculate_sum(2, 3) == 5
print("All tests passed!")
"""

    report2 = evaluate_quixbugs_instance(
        response=failing_code,
        tests=failing_test,
        timeout=10,
        programming_language="python"
    )
    
    print("Test 2 - Code with failing test:")
    print(f"  Passed: {report2.passed}")
    print(f"  Functional Error: {report2.functional_error}")
    print(f"  Runtime Error: {report2.runtime_error}")
    print(f"  Message: {report2.message[:100]}..." if len(report2.message) > 100 else f"  Message: {report2.message}")
    print("\n" + "="*60 + "\n")

    # Test 3: Code with syntax error
    syntax_error_code = """
def calculate_sum(a, b):
    return a + b  # Missing closing parenthesis
"""

    report3 = evaluate_quixbugs_instance(
        response=syntax_error_code,
        tests=test_code,
        timeout=10,
        programming_language="python"
    )
    
    print("Test 3 - Code execution (should handle gracefully):")
    print(f"  Passed: {report3.passed}")
    print(f"  Functional Error: {report3.functional_error}")
    print(f"  Runtime Error: {report3.runtime_error}")
    print(f"  Message: {report3.message[:100]}..." if len(report3.message) > 100 else f"  Message: {report3.message}")

