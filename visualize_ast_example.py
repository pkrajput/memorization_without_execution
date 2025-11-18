"""
Example script demonstrating AST visualization with perplexity and mean_log metrics.
"""

from calculateLogits import get_completion, compute_ast_node_metrics, visualize_ast_graph, visualize_ast_graph_simple

# Example: Get a response and visualize AST
my_model = "gpt-3.5-turbo-0125"

example_buggy_code = """
def gcd(a, b):
    if a == 0:
        return b
    return gcd(b % a, a)
"""

messages = [
    {"role": "system", "content": "You are a code repair assistant. Only output code. Do not include any explanations, comments, or natural language text in your response. Only generate the code itself. Do not print code comment markers or quotes at the beginning or end of the code block."},
    {"role": "user", "content": f"Repair this buggy code:\n\n{example_buggy_code}"}
]

print("Getting response from API...")
response = get_completion(messages=messages, model=my_model, logprobs=True, top_logprobs=10, temperature=0.3)

print("Computing AST node metrics...")
ast_metrics = compute_ast_node_metrics(response, my_model)

print("\nExtracted Code:")
print(ast_metrics["code"])
print(f"\nTotal AST nodes: {len(ast_metrics['ast_nodes'])}")
print(f"Nodes with metrics: {len([n for n in ast_metrics['ast_nodes'] if n['perplexity'] is not None])}")

# Visualize as graph
print("\nGenerating graph visualization (perplexity)...")
try:
    visualize_ast_graph(ast_metrics, output_file="ast_graph_perplexity.png", metric_type="perplexity")
    print("✓ Graph saved to ast_graph_perplexity.png")
except Exception as e:
    print(f"Error creating graph visualization: {e}")
    print("Trying simpler visualization...")
    visualize_ast_graph_simple(ast_metrics, output_file="ast_graph_simple_perplexity.png", metric_type="perplexity")
    print("✓ Simple graph saved to ast_graph_simple_perplexity.png")

# Visualize with mean_log
print("\nGenerating graph visualization (mean_log)...")
try:
    visualize_ast_graph(ast_metrics, output_file="ast_graph_mean_log.png", metric_type="mean_log")
    print("✓ Graph saved to ast_graph_mean_log.png")
except Exception as e:
    print(f"Error creating graph visualization: {e}")
    print("Trying simpler visualization...")
    visualize_ast_graph_simple(ast_metrics, output_file="ast_graph_simple_mean_log.png", metric_type="mean_log")
    print("✓ Simple graph saved to ast_graph_simple_mean_log.png")

print("\nDone!")

