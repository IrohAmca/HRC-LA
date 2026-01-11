import json
import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def load_latest_results(results_dir: str) -> Optional[Dict]:
    """Load the most recent benchmark results."""
    
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return None
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    
    if not json_files:
        logger.warning("No benchmark results found.")
        return None
    
    latest_file = sorted(json_files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_markdown_table(results: List[Dict]) -> str:
    """Generate a markdown table from benchmark results."""
    
    lines = [
        "| Model | Best Accuracy (%) | Val Loss | Train Time (s) | Memory (MB) | Parameters |",
        "|-------|-------------------|----------|----------------|-------------|------------|",
    ]
    
    for r in results:
        memory = f"{r['peak_memory_mb']:.1f}" if r.get('peak_memory_mb') else "N/A"
        lines.append(
            f"| {r['model_name']} | "
            f"{r['best_val_accuracy']*100:.2f} | "
            f"{r['final_val_loss']:.4f} | "
            f"{r['total_training_time_seconds']:.2f} | "
            f"{memory} | "
            f"{r['total_parameters']:,} |"
        )
    
    return "\n".join(lines)

def generate_latex_table(results: List[Dict]) -> str:
    """Generate a LaTeX table from benchmark results."""
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Results}",
        r"\label{tab:benchmark_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Best Acc. (\%) & Val Loss & Time (s) & Memory (MB) & Params \\",
        r"\midrule",
    ]
    
    for r in results:
        memory = f"{r['peak_memory_mb']:.1f}" if r.get('peak_memory_mb') else "N/A"
        model_name = r['model_name'].replace("Ω", r"$\Omega$")
        lines.append(
            f"{model_name} & "
            f"{r['best_val_accuracy']*100:.2f} & "
            f"{r['final_val_loss']:.4f} & "
            f"{r['total_training_time_seconds']:.2f} & "
            f"{memory} & "
            f"{r['total_parameters']:,} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)

def plot_training_curves(results: List[Dict], output_dir: str):
    """Generate and save training curve plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    for r in results:
        if r.get('val_losses'):
            plt.plot(r['val_losses'], label=f"{r['model_name']} (Val)")
    
    plt.title("Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for r in results:
        if r.get('val_accuracies'):
            plt.plot(r['val_accuracies'], label=f"{r['model_name']} (Val)")
            
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_accuracy.png"))
    plt.close()
