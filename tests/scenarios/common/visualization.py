"""
Visualization Module for HRC-LA Benchmark Scenarios.

This module provides standardized plotting utilities for training metrics,
performance analysis, and result visualization across all benchmark scenarios.

Reference:
    Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
    Child et al. (2019). Generating Long Sequences with Sparse Transformers.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Academic color palette (colorblind-friendly)
COLORS = {
    "primary": "#0072B2",      # Blue
    "secondary": "#D55E00",    # Vermillion
    "tertiary": "#009E73",     # Bluish green
    "quaternary": "#CC79A7",   # Reddish purple
    "train": "#0072B2",
    "val": "#D55E00",
    "loss": "#E69F00",
    "accuracy": "#009E73",
    "memory": "#CC79A7",
    "time": "#56B4E9",
}

# Matplotlib style configuration for academic papers
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


@dataclass
class TrainingMetrics:
    """Container for training metrics from a single experiment."""
    
    experiment_name: str
    epochs: List[int]
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]
    epoch_times: List[float]
    memory_usage: List[float]  # GPU memory in MB per epoch
    best_val_accuracy: float
    total_time: float
    peak_memory_mb: float
    config: Dict[str, Any]
    
    @classmethod
    def from_history(
        cls,
        experiment_name: str,
        history: List[Dict],
        best_val_accuracy: float,
        total_time: float,
        config: Dict[str, Any],
    ) -> "TrainingMetrics":
        """Create TrainingMetrics from a training history list."""
        memory_usage = [h.get("memory_mb", 0.0) for h in history]
        peak_memory = max(memory_usage) if memory_usage else 0.0
        
        return cls(
            experiment_name=experiment_name,
            epochs=[h["epoch"] for h in history],
            train_losses=[h["train_loss"] for h in history],
            val_losses=[h["val_loss"] for h in history],
            train_accuracies=[h["train_acc"] for h in history],
            val_accuracies=[h["val_acc"] for h in history],
            epoch_times=[h["epoch_time"] for h in history],
            memory_usage=memory_usage,
            best_val_accuracy=best_val_accuracy,
            total_time=total_time,
            peak_memory_mb=peak_memory,
            config=config,
        )


class BenchmarkVisualizer:
    """
    Standardized visualization class for HRC-LA benchmark experiments.
    
    Provides consistent plotting across all benchmark scenarios with
    publication-ready figure formatting.
    
    Attributes:
        results_dir: Directory to save generated figures.
        experiment_name: Name of the current experiment.
    """
    
    def __init__(self, results_dir: str, experiment_name: str = "benchmark"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Path to save output figures.
            experiment_name: Identifier for the experiment.
        """
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        os.makedirs(results_dir, exist_ok=True)
        
        plt.rcParams.update(PLOT_STYLE)
    
    def plot_training_curves(
        self,
        metrics: TrainingMetrics,
        save: bool = True,
        show: bool = False,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Generate comprehensive training curve visualization.
        
        Creates a 2x3 subplot figure showing:
        - Loss curves (train/val)
        - Accuracy curves (train/val)
        - Memory usage per epoch
        - Epoch timing
        - Learning progress summary
        
        Args:
            metrics: TrainingMetrics object containing experiment data.
            save: Whether to save the figure to disk.
            show: Whether to display the figure.
            
        Returns:
            Tuple of (figure, axes array).
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"{metrics.experiment_name} Training Analysis",
            fontsize=14,
            fontweight="bold",
        )
        
        epochs = metrics.epochs
        
        # Loss curves (top-left)
        ax = axes[0, 0]
        ax.plot(epochs, metrics.train_losses, 
                color=COLORS["train"], label="Train Loss", linewidth=2)
        ax.plot(epochs, metrics.val_losses, 
                color=COLORS["val"], label="Val Loss", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Loss Curves")
        ax.legend(loc="upper right")
        ax.set_xlim(epochs[0], epochs[-1])
        
        # Accuracy curves (top-right)
        ax = axes[0, 1]
        ax.plot(epochs, metrics.train_accuracies, 
                color=COLORS["train"], label="Train Acc", linewidth=2)
        ax.plot(epochs, metrics.val_accuracies, 
                color=COLORS["val"], label="Val Acc", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Curves")
        ax.legend(loc="lower right")
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(0, 1.05)
        
        # Memory usage (top-right-most)
        ax = axes[0, 2]
        if metrics.memory_usage and any(m > 0 for m in metrics.memory_usage):
            ax.plot(epochs, metrics.memory_usage, 
                    color=COLORS["memory"], linewidth=2, marker='o', markersize=4)
            ax.axhline(y=metrics.peak_memory_mb, 
                       color=COLORS["secondary"], linestyle="--", 
                       label=f"Peak: {metrics.peak_memory_mb:.1f} MB")
            ax.fill_between(epochs, metrics.memory_usage, alpha=0.3, color=COLORS["memory"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("GPU Memory (MB)")
            ax.set_title("Memory Usage")
            ax.legend(loc="upper right")
            ax.set_xlim(epochs[0], epochs[-1])
        else:
            ax.text(0.5, 0.5, "Memory tracking\nnot available\n(CPU mode)", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_title("Memory Usage")
        
        # Learning rate / Loss decrease rate (bottom-left)
        ax = axes[1, 0]
        # Plot loss convergence rate
        loss_decrease = [0] + [metrics.train_losses[i-1] - metrics.train_losses[i] 
                               for i in range(1, len(metrics.train_losses))]
        colors_bar = [COLORS["tertiary"] if d >= 0 else COLORS["secondary"] for d in loss_decrease]
        ax.bar(epochs, loss_decrease, color=colors_bar, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Decrease")
        ax.set_title("Learning Progress (Loss Δ)")
        
        # Epoch timing (bottom-middle)
        ax = axes[1, 1]
        ax.bar(epochs, metrics.epoch_times, color=COLORS["time"], alpha=0.7)
        ax.axhline(y=np.mean(metrics.epoch_times), 
                   color=COLORS["secondary"], linestyle="--", 
                   label=f"Mean: {np.mean(metrics.epoch_times):.2f}s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Epoch Duration")
        ax.legend()
        
        # Summary statistics (bottom-right)
        ax = axes[1, 2]
        ax.axis("off")
        
        # Memory info string
        memory_info = ""
        if metrics.memory_usage and any(m > 0 for m in metrics.memory_usage):
            memory_info = (
                f"\nMemory Usage:\n"
                f"  Peak Memory: {metrics.peak_memory_mb:.1f} MB\n"
                f"  Avg Memory: {np.mean(metrics.memory_usage):.1f} MB\n"
            )
        
        summary_text = (
            f"Experiment Summary\n"
            f"{'─' * 35}\n\n"
            f"Best Val Accuracy: {metrics.best_val_accuracy:.4f}\n"
            f"Final Train Loss: {metrics.train_losses[-1]:.4f}\n"
            f"Final Val Loss: {metrics.val_losses[-1]:.4f}\n\n"
            f"Total Time: {metrics.total_time:.2f}s\n"
            f"Avg Epoch Time: {np.mean(metrics.epoch_times):.2f}s\n"
            f"{memory_info}\n"
            f"Configuration:\n"
            f"  seq_len: {metrics.config.get('seq_len', 'N/A')}\n"
            f"  d_model: {metrics.config.get('d_model', 'N/A')}\n"
            f"  num_heads: {metrics.config.get('num_heads', 'N/A')}\n"
            f"  m_features: {metrics.config.get('m_features', 'N/A')}\n"
        )
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.results_dir, 
                f"{self.experiment_name}_training_{timestamp}.png"
            )
            plt.savefig(filepath)
            logger.info(f"Training curves saved to: {filepath}")
        
        if show:
            plt.show()
        
        return fig, axes
    
    def plot_performance_comparison(
        self,
        metrics_list: List[TrainingMetrics],
        save: bool = True,
        show: bool = False,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Compare performance across multiple experiments.
        
        Args:
            metrics_list: List of TrainingMetrics from different experiments.
            save: Whether to save the figure.
            show: Whether to display the figure.
            
        Returns:
            Tuple of (figure, axes array).
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Performance Comparison", fontsize=14, fontweight="bold")
        
        names = [m.experiment_name for m in metrics_list]
        accuracies = [m.best_val_accuracy for m in metrics_list]
        times = [m.total_time for m in metrics_list]
        final_losses = [m.val_losses[-1] for m in metrics_list]
        
        # Accuracy comparison
        ax = axes[0]
        bars = ax.bar(names, accuracies, color=COLORS["accuracy"], alpha=0.8)
        ax.set_ylabel("Best Validation Accuracy")
        ax.set_title("Accuracy Comparison")
        ax.set_ylim(0, 1.05)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{acc:.4f}", ha="center", fontsize=9)
        
        # Time comparison
        ax = axes[1]
        bars = ax.bar(names, times, color=COLORS["time"], alpha=0.8)
        ax.set_ylabel("Total Training Time (s)")
        ax.set_title("Training Time Comparison")
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{t:.1f}s", ha="center", fontsize=9)
        
        # Loss comparison
        ax = axes[2]
        bars = ax.bar(names, final_losses, color=COLORS["loss"], alpha=0.8)
        ax.set_ylabel("Final Validation Loss")
        ax.set_title("Loss Comparison")
        for bar, loss in zip(bars, final_losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{loss:.4f}", ha="center", fontsize=9)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.results_dir, 
                f"performance_comparison_{timestamp}.png"
            )
            plt.savefig(filepath)
            logger.info(f"Comparison plot saved to: {filepath}")
        
        if show:
            plt.show()
        
        return fig, axes
    
    def plot_depth_heatmap(
        self,
        depth_results: Dict[float, Dict[str, float]],
        seq_len: int,
        save: bool = True,
        show: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create heatmap visualization for NIAH depth analysis.
        
        Args:
            depth_results: Dictionary mapping depth -> {accuracy, num_samples}.
            seq_len: Sequence length for the experiment.
            save: Whether to save the figure.
            show: Whether to display the figure.
            
        Returns:
            Tuple of (figure, axes).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        depths = sorted(depth_results.keys())
        accuracies = [depth_results[d]["accuracy"] for d in depths]
        
        # Bar plot with color gradient
        colors = plt.cm.RdYlGn(accuracies)
        bars = ax.bar(range(len(depths)), accuracies, color=colors)
        
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d:.1f}" for d in depths])
        ax.set_xlabel("Needle Depth (0=Start, 1=End)")
        ax.set_ylabel("Retrieval Accuracy")
        ax.set_title(f"NIAH: Accuracy by Depth (seq_len={seq_len})")
        ax.set_ylim(0, 1.05)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Accuracy")
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{acc:.2f}", ha="center", fontsize=8)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.results_dir,
                f"niah_depth_analysis_{seq_len}_{timestamp}.png"
            )
            plt.savefig(filepath)
            logger.info(f"Depth heatmap saved to: {filepath}")
        
        if show:
            plt.show()
        
        return fig, ax


def save_experiment_results(
    results_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
    history: List[Dict],
    best_val_acc: float,
    total_time: float,
    extra_data: Optional[Dict] = None,
) -> str:
    """
    Save experiment results in standardized JSON format.
    
    Args:
        results_dir: Directory to save results.
        experiment_name: Name of the experiment.
        config: Configuration dictionary.
        history: Training history list.
        best_val_acc: Best validation accuracy achieved.
        total_time: Total training time in seconds.
        extra_data: Additional data to include (e.g., depth analysis).
        
    Returns:
        Path to the saved file.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq_len = config.get("seq_len", "unknown")
    filename = f"{experiment_name}_seq{seq_len}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    results = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "config": config,
        "results": {
            "best_val_accuracy": best_val_acc,
            "total_training_time_seconds": total_time,
            "history": history,
        },
    }
    
    if extra_data:
        results["results"].update(extra_data)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {filepath}")
    return filepath


def load_experiment_results(filepath: str) -> Optional[Dict]:
    """
    Load experiment results from JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        Dictionary containing experiment results, or None if loading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {filepath}: {e}")
        return None


def print_experiment_summary(
    experiment_name: str,
    config: Dict[str, Any],
    best_val_acc: float,
    total_time: float,
    results_dir: str,
) -> None:
    """
    Print formatted experiment summary to console.
    
    Args:
        experiment_name: Name of the experiment.
        config: Configuration dictionary.
        best_val_acc: Best validation accuracy.
        total_time: Total training time.
        results_dir: Directory where results are saved.
    """
    summary = f"""
{'═' * 60}
 EXPERIMENT COMPLETE: {experiment_name.upper()}
{'═' * 60}
 
 Results:
   Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
   Total Training Time      : {total_time:.2f}s
 
 Configuration:
   Sequence Length  : {config.get('seq_len', 'N/A')}
   Model Dimension  : {config.get('d_model', 'N/A')}
   Attention Heads  : {config.get('num_heads', 'N/A')}
   HRC-LA Features  : {config.get('m_features', 'N/A')}
   Batch Size       : {config.get('batch_size', 'N/A')}
   Learning Rate    : {config.get('learning_rate', 'N/A')}
 
 Output:
   Results saved to: {results_dir}
{'═' * 60}
"""
    print(summary)
