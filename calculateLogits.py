from openai import OpenAI
import numpy as np
import os
import dotenv
from typing import List, Dict, Tuple
import ast

dotenv.load_dotenv()

try:
    DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "120"))
except ValueError:
    DEFAULT_REQUEST_TIMEOUT = 120.0

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=3,
    timeout=DEFAULT_REQUEST_TIMEOUT,
)


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4-mini",
    max_tokens=2000,
    temperature=0.7,
    stop=None,
    tools=None,
    logprobs=None,
    top_logprobs=None,
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
):
    """Get completion from OpenAI API - handles both chat/completions and responses endpoints"""

    uses_responses_endpoint = "gpt-4-mini" in model
    api_client = client if request_timeout is None else client.with_options(timeout=request_timeout)

    if uses_responses_endpoint:
        input_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            input_messages.append({
                "role": role,
                "content": content,
            })

        params = {
            "model": model,
            "input": input_messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        if logprobs:
            params["logprobs"] = True
            if top_logprobs is not None:
                params["top_logprobs"] = top_logprobs

        completion = api_client.responses.create(**params)

    else:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stop": stop,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

        if model.startswith("o1") or ("gpt-4-mini" in model):
            params["max_completion_tokens"] = max_tokens
        elif ("codex" not in model):
            params["max_tokens"] = max_tokens
        else:
            params["gpt-4-mini"] = True

        if tools:
            params["tools"] = tools

        completion = api_client.chat.completions.create(**params)

    return completion


def extract_response_content(response, model: str) -> str:
    """Extract text content from response, handling both endpoints"""
    uses_responses_endpoint = "gpt-4-mini" in model
    
    if uses_responses_endpoint:
        # Responses endpoint format
        return getattr(response, "output_text", "")
    else:
        # Chat completions endpoint format
        return response.choices[0].message.content


def extract_logprobs(response, model: str):
    """Extract token logprobs as floats from response regardless of endpoint"""
    uses_responses_endpoint = "gpt-4-mini" in model

    logprob_values: List[float] = []

    if uses_responses_endpoint:
        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) != "output_text":
                    continue
                logprobs_info = getattr(content, "logprobs", None)
                if logprobs_info is None:
                    continue

                token_logprobs = getattr(logprobs_info, "token_logprobs", None)
                if token_logprobs is not None:
                    logprob_values.extend(token_logprobs)
                    continue

                tokens = getattr(logprobs_info, "tokens", [])
                for token in tokens:
                    logprob = getattr(token, "logprob", None)
                    if logprob is not None:
                        logprob_values.append(logprob)
    else:
        logprob_content = getattr(response.choices[0].logprobs, "content", None)
        if logprob_content:
            for token in logprob_content:
                logprob = getattr(token, "logprob", None)
                if logprob is not None:
                    logprob_values.append(logprob)

    return logprob_values


def extract_tokens_with_logprobs(response, model: str) -> List[Dict]:
    """
    Extract tokens with their text and logprobs from response.
    
    Args:
        response: API response object
        model: Model name
        
    Returns:
        List of dicts with 'token', 'logprob', and 'text' keys
    """
    uses_responses_endpoint = "gpt-4-mini" in model
    tokens = []

    if uses_responses_endpoint:
        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) != "output_text":
                    continue
                logprobs_info = getattr(content, "logprobs", None)
                if logprobs_info is None:
                    continue

                token_list = getattr(logprobs_info, "tokens", [])
                for token_obj in token_list:
                    token_text = getattr(token_obj, "token", "")
                    logprob = getattr(token_obj, "logprob", None)
                    tokens.append({
                        "token": token_text,
                        "logprob": logprob,
                        "text": token_text
                    })
    else:
        logprob_content = getattr(response.choices[0].logprobs, "content", None)
        if logprob_content:
            for token_obj in logprob_content:
                token_text = getattr(token_obj, "token", "")
                logprob = getattr(token_obj, "logprob", None)
                tokens.append({
                    "token": token_text,
                    "logprob": logprob,
                    "text": token_text
                })

    return tokens


def compute_ast_node_metrics(response, model: str) -> Dict:
    """
    Extract code from response, compute AST tree, and compute perplexity and mean_log
    for each AST node.
    
    Args:
        response: API response object
        model: Model name
        
    Returns:
        Dictionary with:
            - 'code': extracted code string
            - 'ast_nodes': list of dicts with node info, perplexity, and mean_log
            - 'ast_tree': string representation of AST
    """
    from evaluate_quixbugs_instance import clean_code
    
    # Extract code from response
    code = extract_response_content(response, model)
    code = clean_code(code)
    
    # Extract tokens with logprobs
    tokens = extract_tokens_with_logprobs(response, model)
    
    # Reconstruct the full text from tokens
    token_texts = [t["text"] for t in tokens]
    reconstructed_text = "".join(token_texts)
    
    # Map tokens to character positions in reconstructed text
    token_positions = []
    char_pos = 0
    for token in tokens:
        token_text = token["text"]
        start_pos = char_pos
        char_pos += len(token_text)
        end_pos = char_pos
        token_positions.append({
            "token": token,
            "start_char": start_pos,
            "end_char": end_pos
        })
    
    # Try to align reconstructed text with code
    # Find the best alignment by matching code in reconstructed text
    code_start_in_reconstructed = reconstructed_text.find(code[:min(50, len(code))])
    if code_start_in_reconstructed == -1:
        # Try reverse: find code that matches part of reconstructed
        code_clean = code.replace(' ', '').replace('\n', '')
        reconstructed_clean = reconstructed_text.replace(' ', '').replace('\n', '')
        code_start_in_reconstructed = reconstructed_clean.find(code_clean[:min(50, len(code_clean))])
        if code_start_in_reconstructed == -1:
            code_start_in_reconstructed = 0
    
    # Parse code to AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "code": code,
            "error": f"Syntax error in code: {str(e)}",
            "ast_nodes": [],
            "ast_tree": None
        }
    
    # Build character position to line/column mapping for code
    lines = code.split('\n')
    char_to_line_col = {}
    char_idx = 0
    for line_num, line in enumerate(lines):
        for col_num in range(len(line) + 1):  # +1 to include newline
            char_to_line_col[char_idx] = (line_num + 1, col_num)  # 1-indexed line
            char_idx += 1
        # Newline character
        char_to_line_col[char_idx] = (line_num + 1, len(line))
        char_idx += 1
    
    # Visitor class to collect AST nodes with their positions
    class ASTNodeVisitor(ast.NodeVisitor):
        def __init__(self, source_code, tokens_with_positions, code_start_offset):
            self.nodes = []
            self.source_code = source_code
            self.tokens_with_positions = tokens_with_positions
            self.code_start_offset = code_start_offset
            self.lines = source_code.split('\n')
            
        def _get_node_char_range(self, node):
            """Get character range for a node in the source code."""
            if not (hasattr(node, 'lineno') and hasattr(node, 'col_offset')):
                return None, None
            
            start_line = node.lineno - 1  # 0-indexed
            end_line = getattr(node, 'end_lineno', node.lineno) - 1
            start_col = node.col_offset
            end_col = getattr(node, 'end_col_offset', 
                            len(self.lines[end_line]) if end_line < len(self.lines) else 0)
            
            # Calculate character positions
            start_char = sum(len(self.lines[i]) + 1 for i in range(start_line)) + start_col
            end_char = sum(len(self.lines[i]) + 1 for i in range(end_line)) + end_col
            
            return start_char, end_char
            
        def visit(self, node):
            # Get node's character range in source code
            node_start_char, node_end_char = self._get_node_char_range(node)
            
            if node_start_char is not None and node_end_char is not None:
                # Map to positions in reconstructed text
                node_start_in_reconstructed = self.code_start_offset + node_start_char
                node_end_in_reconstructed = self.code_start_offset + node_end_char
                
                # Find tokens that overlap with this node's range
                node_logprobs = []
                node_tokens = []
                
                for token_pos in self.tokens_with_positions:
                    token_start = token_pos["start_char"]
                    token_end = token_pos["end_char"]
                    token = token_pos["token"]
                    
                    # Check if token overlaps with node range
                    if (token_start < node_end_in_reconstructed and 
                        token_end > node_start_in_reconstructed and
                        token["logprob"] is not None):
                        node_tokens.append(token)
                        node_logprobs.append(token["logprob"])
                
                # If no tokens found, try proportional allocation
                if not node_logprobs and self.tokens_with_positions:
                    # Estimate based on node size relative to code
                    total_code_chars = len(self.source_code)
                    node_chars = node_end_char - node_start_char
                    if total_code_chars > 0:
                        token_ratio = node_chars / total_code_chars
                        num_tokens = max(1, int(len(self.tokens_with_positions) * token_ratio))
                        # Estimate start token index based on line position
                        if hasattr(node, 'lineno'):
                            line_ratio = (node.lineno - 1) / max(1, len(self.lines))
                            start_token_idx = int(len(self.tokens_with_positions) * line_ratio)
                            end_token_idx = min(len(self.tokens_with_positions), start_token_idx + num_tokens)
                            node_logprobs = [tp["token"]["logprob"] 
                                           for tp in self.tokens_with_positions[start_token_idx:end_token_idx]
                                           if tp["token"]["logprob"] is not None]
                
                # Extract source code for this node
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', node.lineno) - 1
                start_col = node.col_offset
                end_col = getattr(node, 'end_col_offset', 
                                len(self.lines[end_line]) if end_line < len(self.lines) else 0)
                
                if start_line == end_line:
                    node_text = self.lines[start_line][start_col:end_col]
                else:
                    node_text = self.lines[start_line][start_col:] + '\n'
                    for line_idx in range(start_line + 1, end_line):
                        node_text += self.lines[line_idx] + '\n'
                    if end_line < len(self.lines):
                        node_text += self.lines[end_line][:end_col]
                
                # Compute metrics for this node
                if node_logprobs:
                    mean_logprob = float(np.mean(node_logprobs))
                    perplexity = float(np.exp(-mean_logprob))
                else:
                    mean_logprob = None
                    perplexity = None
                
                node_info = {
                    "type": type(node).__name__,
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "end_lineno": getattr(node, 'end_lineno', None),
                    "end_col_offset": getattr(node, 'end_col_offset', None),
                    "source": node_text[:200],  # Truncate for readability
                    "num_tokens": len(node_logprobs),
                    "mean_log": mean_logprob,
                    "perplexity": perplexity
                }
                
                self.nodes.append(node_info)
            
            self.generic_visit(node)
    
    visitor = ASTNodeVisitor(code, token_positions, code_start_in_reconstructed)
    visitor.visit(tree)
    
    # Build node ID to metrics mapping for visualization
    node_id_to_metrics = {}
    for i, node_info in enumerate(visitor.nodes):
        node_id = f"{node_info['type']}_{node_info['lineno']}_{node_info['col_offset']}"
        node_id_to_metrics[node_id] = {
            "index": i,
            "metrics": node_info
        }
    
    return {
        "code": code,
        "ast_nodes": visitor.nodes,
        "ast_tree": ast.dump(tree),
        "total_tokens": len(tokens),
        "tokens_with_logprobs": len([t for t in tokens if t["logprob"] is not None]),
        "tree": tree,  # Keep the AST tree for visualization
        "node_id_to_metrics": node_id_to_metrics
    }


def visualize_ast_graph(ast_metrics: Dict, output_file: str = None, metric_type: str = "perplexity", 
                       layout: str = "hierarchical", figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Visualize the AST tree as a graph with nodes colored/sized by perplexity or mean_log.
    
    Args:
        ast_metrics: Output from compute_ast_node_metrics
        output_file: Optional path to save the figure (e.g., "ast_graph.png")
        metric_type: "perplexity" or "mean_log" to use for coloring
        layout: "hierarchical" (tree layout) or "spring" (force-directed)
        figsize: Figure size as (width, height)
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        raise ImportError("networkx and matplotlib are required for visualization. Install with: pip install networkx matplotlib")
    
    if "tree" not in ast_metrics or ast_metrics["tree"] is None:
        print("Error: AST tree not available in metrics. Cannot visualize.")
        return
    
    tree = ast_metrics["tree"]
    ast_nodes = ast_metrics["ast_nodes"]
    
    # Build a mapping from AST node objects to their metrics
    node_to_metrics = {}
    for node_info in ast_nodes:
        # Create a key based on position
        key = (node_info['lineno'], node_info['col_offset'], node_info['type'])
        node_to_metrics[key] = node_info
    
    # Build graph structure by traversing AST
    G = nx.DiGraph()
    node_labels = {}
    node_colors = []
    node_sizes = []
    node_metrics_data = {}
    
    def get_node_id(node):
        """Generate a unique ID for an AST node."""
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            return f"{type(node).__name__}_{node.lineno}_{node.col_offset}"
        return f"{type(node).__name__}_{id(node)}"
    
    def add_node_to_graph(node, parent_id=None):
        """Recursively add AST nodes to the graph."""
        node_id = get_node_id(node)
        
        # Get metrics for this node
        metrics = None
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            key = (node.lineno, node.col_offset, type(node).__name__)
            if key in node_to_metrics:
                metrics = node_to_metrics[key]
        
        # Create label
        node_type = type(node).__name__
        if metrics:
            source_preview = metrics['source'][:30].replace('\n', ' ')
            label = f"{node_type}\n{source_preview}..."
        else:
            label = node_type
        
        # Get metric value for coloring
        if metrics and metrics[metric_type] is not None:
            metric_value = metrics[metric_type]
        else:
            metric_value = None
        
        # Add node to graph
        if node_id not in G:
            G.add_node(node_id, label=label, node_type=node_type, metrics=metrics)
            node_labels[node_id] = label
            node_metrics_data[node_id] = metric_value
            
            # Set color based on metric
            if metric_value is not None:
                if metric_type == "perplexity":
                    # Normalize perplexity (assuming range 1.0 to 2.0, adjust as needed)
                    normalized = min(1.0, max(0.0, (metric_value - 1.0) / 1.0))
                    node_colors.append(plt.cm.RdYlGn_r(normalized))  # Red = high perplexity, Green = low
                else:  # mean_log
                    # Normalize mean_log (assuming range -5 to 0, adjust as needed)
                    normalized = min(1.0, max(0.0, (metric_value + 5) / 5))
                    node_colors.append(plt.cm.RdYlGn_r(normalized))  # Red = low logprob, Green = high
                
                # Size based on metric (inverse for perplexity, direct for mean_log)
                if metric_type == "perplexity":
                    size = 300 + (metric_value - 1.0) * 500  # Scale based on perplexity
                else:
                    size = 300 + abs(metric_value) * 100  # Scale based on absolute mean_log
                node_sizes.append(max(100, min(2000, size)))
            else:
                node_colors.append('lightgray')
                node_sizes.append(300)
        
        # Add edge from parent
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Recursively add children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        add_node_to_graph(item, node_id)
            elif isinstance(value, ast.AST):
                add_node_to_graph(value, node_id)
    
    # Build graph starting from root
    add_node_to_graph(tree)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "hierarchical":
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=20, 
                          edge_color='gray', ax=ax)
    
    # Draw labels (only for important nodes to avoid clutter)
    important_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] in 
                      ['FunctionDef', 'ClassDef', 'If', 'For', 'While', 'Return', 'Call']]
    labels_to_show = {n: node_labels[n] for n in important_nodes if n in node_labels}
    nx.draw_networkx_labels(G, pos, labels_to_show, font_size=8, ax=ax)
    
    # Create colorbar
    if metric_type == "perplexity":
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                                   norm=plt.Normalize(vmin=1.0, vmax=2.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Perplexity (Red=High, Green=Low)', rotation=270, labelpad=20)
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                                   norm=plt.Normalize(vmin=-5, vmax=0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Mean Log Probability (Red=Low, Green=High)', rotation=270, labelpad=20)
    
    ax.set_title(f'AST Tree Visualization - {metric_type.capitalize()}', fontsize=16, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_file}")
    else:
        plt.show()
    
    return fig


def visualize_ast_graph_simple(ast_metrics: Dict, output_file: str = None, 
                               metric_type: str = "perplexity") -> None:
    """
    A simpler visualization using a tree-like structure.
    
    Args:
        ast_metrics: Output from compute_ast_node_metrics
        output_file: Optional path to save the figure
        metric_type: "perplexity" or "mean_log" to use for coloring
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    ast_nodes = ast_metrics["ast_nodes"]
    
    if not ast_nodes:
        print("No AST nodes to visualize.")
        return
    
    # Filter nodes with valid metrics
    nodes_with_metrics = [n for n in ast_nodes if n[metric_type] is not None]
    
    if not nodes_with_metrics:
        print(f"No nodes with valid {metric_type} metrics.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(nodes_with_metrics) * 0.5)))
    
    # Prepare data
    node_types = [n['type'] for n in nodes_with_metrics]
    metric_values = [n[metric_type] for n in nodes_with_metrics]
    line_numbers = [n['lineno'] for n in nodes_with_metrics]
    
    # Normalize colors
    if metric_type == "perplexity":
        norm_values = [(v - min(metric_values)) / (max(metric_values) - min(metric_values)) 
                      if max(metric_values) > min(metric_values) else 0.5 
                      for v in metric_values]
        colors = [plt.cm.RdYlGn_r(n) for n in norm_values]
        ylabel = "Perplexity"
    else:
        norm_values = [(v - min(metric_values)) / (max(metric_values) - min(metric_values)) 
                      if max(metric_values) > min(metric_values) else 0.5 
                      for v in metric_values]
        colors = [plt.cm.RdYlGn_r(n) for n in norm_values]
        ylabel = "Mean Log Probability"
    
    # Create horizontal bar chart
    y_pos = range(len(nodes_with_metrics))
    bars = ax.barh(y_pos, metric_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add labels
    labels = [f"{nt}\nLine {ln}" for nt, ln in zip(node_types, line_numbers)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_title(f'AST Node {ylabel} Visualization', fontsize=14, pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', ha='left' if width > 0 else 'right', 
               va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_file}")
    else:
        plt.show()
    
    return fig


