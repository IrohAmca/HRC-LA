"""
Core Benchmark Script for HRC-LA.

Compares three attention mechanisms:
1. Standard Softmax Attention (Baseline)
2. HRC-LA (Fixed Omega)
3. HRC-LA (Learnable Omega)

Metrics: Inference Time, Peak Memory, and MSE Approximation Error.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hrc_la import HRCMultiheadAttention
from hrc_la.utils import StandardAdapter, measure_performance, sync_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    d_model: int = 64
    num_heads: int = 4
    m_features: int = 256
    seq_lengths: List[int] = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [1024, 2048, 4096, 8192, 16384]


class AttentionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.models = {}
        self.results = []
        self._setup_models()

    def _setup_models(self):
        """Initialize and synchronize models for fair comparison."""
        logger.info(f"Setting up models on {self.config.device}...")
        
        std_raw = nn.MultiheadAttention(
            self.config.d_model, 
            self.config.num_heads, 
            batch_first=True
        ).to(self.config.device)
        self.models['Standard'] = StandardAdapter(std_raw)

        hrc_fixed = HRCMultiheadAttention(
            self.config.d_model, 
            self.config.num_heads, 
            m_features=self.config.m_features,
            batch_first=True, 
            learnable_omega=False
        ).to(self.config.device)
        
        hrc_learnable = HRCMultiheadAttention(
            self.config.d_model, 
            self.config.num_heads, 
            m_features=self.config.m_features,
            batch_first=True, 
            learnable_omega=True
        ).to(self.config.device)

        sync_weights(std_raw, hrc_fixed)
        sync_weights(std_raw, hrc_learnable)

        self.models['HRC-LA (Fixed)'] = hrc_fixed
        self.models['HRC-LA (Learnable)'] = hrc_learnable

        for model in self.models.values():
            model.eval()

    def run(self):
        """Execute the benchmark loop across sequence lengths."""
        logger.info(f"Running Benchmark: D={self.config.d_model}, H={self.config.num_heads}, M={self.config.m_features}")
        logger.info(f"{'N':<8} | {'Model':<20} | {'Time (s)':<10} | {'Mem (MB)':<10} | {'MSE':<12}")
        logger.info("-" * 70)

        for N in self.config.seq_lengths:
            x = torch.randn(1, N, self.config.d_model).to(self.config.device)
            
            # Baseline output for MSE
            _, _, out_std = measure_performance(self.models['Standard'], x, self.config.device)
            
            for name, model in self.models.items():
                time_taken, memory, output = measure_performance(model, x, self.config.device)
                
                mse = 0.0
                if name != 'Standard' and out_std is not None and output is not None:
                    mse = torch.mean((out_std - output) ** 2).item()
                
                if time_taken is not None:
                    self.results.append({
                        "Sequence Length": N,
                        "Model": name,
                        "Time (s)": time_taken,
                        "Memory (MB)": memory,
                        "MSE": mse
                    })
                    logger.info(f"{N:<8} | {name:<20} | {time_taken:<10.4f} | {memory:<10.2f} | {mse:<12.6f}")
                else:
                    logger.info(f"{N:<8} | {name:<20} | {'OOM/Fail':<10} | {'-':<10} | {'-':<12}")

    def plot_results(self, save_path="benchmark_results.png"):
        """Visualize benchmark results."""
        if not self.results:
            logger.warning("No results to plot.")
            return

        df = pd.DataFrame(self.results)
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        sns.lineplot(data=df, x="Sequence Length", y="Time (s)", hue="Model", style="Model", marker="o", ax=axes[0])
        axes[0].set_title("Inference Time (Lower is Better)")
        axes[0].set_yscale("log")
        
        sns.lineplot(data=df, x="Sequence Length", y="Memory (MB)", hue="Model", style="Model", marker="o", ax=axes[1])
        axes[1].set_title("Peak Memory Usage (Lower is Better)")
        
        df_mse = df[df["Model"] != "Standard"]
        if not df_mse.empty:
            sns.lineplot(data=df_mse, x="Sequence Length", y="MSE", hue="Model", style="Model", marker="o", ax=axes[2])
            axes[2].set_title("Approximation Error vs Standard")
            axes[2].set_yscale("log")

        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Plots saved to {save_path}")


if __name__ == "__main__":
    config = BenchmarkConfig()
    benchmark = AttentionBenchmark(config)
    benchmark.run()
    benchmark.plot_results()
