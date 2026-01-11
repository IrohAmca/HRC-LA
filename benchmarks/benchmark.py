"""
Core Benchmark Script for HRC-LA.

Compares three attention mechanisms:
1. Standard Softmax Attention (Baseline)
2. HRC-LA (Fixed Omega)
3. HRC-LA (Learnable Omega)

Evaluation Modes:
- "mse": Compare HRC outputs against Standard Softmax (MSE Error)
- "loss": Independent evaluation using reconstruction loss (Cross-Entropy)

Metrics: Inference Time, Peak Memory, and Error/Loss.
"""

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hrc_la import HRCMultiheadAttention
from hrc_la.utils import StandardAdapter, measure_performance, sync_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EvalMode(Enum):
    """Evaluation mode for benchmarking."""

    MSE = "mse"  # Compare against Standard Softmax output
    LOSS = "loss"  # Independent reconstruction loss evaluation


@dataclass
class BenchmarkConfig:
    d_model: int = 16
    num_heads: int = 4
    m_features: int = 16
    learning_rate: float = 5e-4
    seq_lengths: List[int] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_mode: EvalMode = EvalMode.MSE  # Default to MSE comparison
    vocab_size: int = 64  # For loss-based evaluation

    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [1024, 2048, 4096, 8192, 16384]


class AttentionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.models = {}
        self.results = []
        self._setup_models()

        # Output projection for loss-based evaluation
        if self.config.eval_mode == EvalMode.LOSS:
            self.output_proj = nn.Linear(
                self.config.d_model, self.config.vocab_size
            ).to(self.config.device)

    def _setup_models(self):
        """Initialize and synchronize models for fair comparison."""
        logger.info(f"Setting up models on {self.config.device}...")

        std_raw = nn.MultiheadAttention(
            self.config.d_model, self.config.num_heads, batch_first=True
        ).to(self.config.device)
        self.models["Standard"] = StandardAdapter(std_raw)

        hrc_fixed = HRCMultiheadAttention(
            self.config.d_model,
            self.config.num_heads,
            m_features=self.config.m_features,
            batch_first=True,
            learnable_omega=False,
        ).to(self.config.device)

        hrc_learnable = HRCMultiheadAttention(
            self.config.d_model,
            self.config.num_heads,
            m_features=self.config.m_features,
            batch_first=True,
            learnable_omega=True,
        ).to(self.config.device)

        sync_weights(std_raw, hrc_fixed)
        sync_weights(std_raw, hrc_learnable)

        self.models["HRC-LA (Fixed)"] = hrc_fixed
        self.models["HRC-LA (Learnable)"] = hrc_learnable

        for model in self.models.values():
            model.eval()

    def _compute_reconstruction_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> float:
        """
        Compute reconstruction loss (Cross-Entropy) for independent evaluation.

        Args:
            output: Model output [batch, seq_len, d_model]
            target: Target token indices [batch, seq_len]

        Returns:
            Cross-entropy loss value
        """
        # Project to vocabulary
        logits = self.output_proj(output)  # [batch, seq_len, vocab_size]

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), target.view(-1))
        return loss.item()

    def run(self):
        """Execute the benchmark loop across sequence lengths."""
        eval_mode_str = self.config.eval_mode.value
        metric_name = "MSE" if self.config.eval_mode == EvalMode.MSE else "Loss"

        logger.info(
            f"Running Benchmark: D={self.config.d_model}, H={self.config.num_heads}, M={self.config.m_features}"
        )
        logger.info(f"Evaluation Mode: {eval_mode_str.upper()}")
        logger.info(
            f"{'N':<8} | {'Model':<20} | {'Time (s)':<10} | {'Mem (MB)':<10} | {metric_name:<12}"
        )
        logger.info("-" * 70)

        for N in self.config.seq_lengths:
            x = torch.randn(1, N, self.config.d_model).to(self.config.device)

            # Generate random target for loss-based evaluation
            target = None
            if self.config.eval_mode == EvalMode.LOSS:
                target = torch.randint(
                    0, self.config.vocab_size, (1, N), device=self.config.device
                )

            # Get standard output for MSE comparison
            out_std = None
            if self.config.eval_mode == EvalMode.MSE:
                _, _, out_std = measure_performance(
                    self.models["Standard"], x, self.config.device
                )

            for name, model in self.models.items():
                time_taken, memory, output = measure_performance(
                    model, x, self.config.device
                )

                metric_value = 0.0
                if time_taken is not None and output is not None:
                    if self.config.eval_mode == EvalMode.MSE:
                        # MSE comparison against Standard
                        if name != "Standard" and out_std is not None:
                            metric_value = torch.mean((out_std - output) ** 2).item()
                    else:
                        # Independent loss evaluation
                        metric_value = self._compute_reconstruction_loss(output, target)

                if time_taken is not None:
                    self.results.append(
                        {
                            "Sequence Length": N,
                            "Model": name,
                            "Time (s)": time_taken,
                            "Memory (MB)": memory,
                            metric_name: metric_value,
                        }
                    )
                    logger.info(
                        f"{N:<8} | {name:<20} | {time_taken:<10.4f} | {memory:<10.2f} | {metric_value:<12.6f}"
                    )
                else:
                    logger.info(
                        f"{N:<8} | {name:<20} | {'OOM/Fail':<10} | {'-':<10} | {'-':<12}"
                    )

    def plot_results(self, save_path="benchmark_results.png"):
        """Visualize benchmark results."""
        if not self.results:
            logger.warning("No results to plot.")
            return

        df = pd.DataFrame(self.results)
        metric_name = "MSE" if self.config.eval_mode == EvalMode.MSE else "Loss"

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        sns.lineplot(
            data=df,
            x="Sequence Length",
            y="Time (s)",
            hue="Model",
            style="Model",
            marker="o",
            ax=axes[0],
        )
        axes[0].set_title("Inference Time (Lower is Better)")
        axes[0].set_yscale("log")

        sns.lineplot(
            data=df,
            x="Sequence Length",
            y="Memory (MB)",
            hue="Model",
            style="Model",
            marker="o",
            ax=axes[1],
        )
        axes[1].set_title("Peak Memory Usage (Lower is Better)")

        if self.config.eval_mode == EvalMode.MSE:
            # For MSE, exclude Standard (it's the reference)
            df_metric = df[df["Model"] != "Standard"]
            metric_title = "Approximation Error vs Standard (MSE)"
        else:
            # For Loss, include all models
            df_metric = df
            metric_title = "Reconstruction Loss (Cross-Entropy)"

        if not df_metric.empty:
            sns.lineplot(
                data=df_metric,
                x="Sequence Length",
                y=metric_name,
                hue="Model",
                style="Model",
                marker="o",
                ax=axes[2],
            )
            axes[2].set_title(metric_title)
            axes[2].set_yscale("log")

        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Plots saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark HRC-LA attention mechanisms"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mse", "loss"],
        default="mse",
        help="Evaluation mode: 'mse' (compare vs softmax) or 'loss' (independent reconstruction loss)",
    )
    parser.add_argument("--d_model", type=int, default=16, help="Model dimension")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--m_features", type=int, default=16, help="HRC-LA feature dimension"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=64, help="Vocabulary size (for loss mode)"
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.png", help="Output plot path"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        m_features=args.m_features,
        vocab_size=args.vocab_size,
        seq_lengths=args.seq_lengths,
        eval_mode=EvalMode.LOSS if args.mode == "loss" else EvalMode.MSE,
    )

    benchmark = AttentionBenchmark(config)
    benchmark.run()
    benchmark.plot_results(args.output)
