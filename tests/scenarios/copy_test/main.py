"""
Copy Task Benchmark for HRC-LA Transformer.

This module implements the sequence copying task, a fundamental benchmark
for evaluating sequence-to-sequence learning capabilities. The model must
learn to reproduce input sequences exactly, testing its ability to maintain
information across the entire sequence length.

Task Definition:
    Given an input sequence [x_1, x_2, ..., x_n], the model must output
    the identical sequence [x_1, x_2, ..., x_n].

References:
    Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines.
    Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from tests.models import HRCTransformer
from tests.scenarios.common.config import BenchmarkConfig
from tests.scenarios.common.datasets import CopyTaskDataset
from tests.scenarios.common.trainer import (
    count_parameters,
    get_lr_scheduler,
    train_epoch,
    validate,
)
from tests.scenarios.common.visualization import (
    BenchmarkVisualizer,
    TrainingMetrics,
    print_experiment_summary,
    save_experiment_results,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def print_usage() -> None:
    """Display usage information and command-line examples."""
    usage_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         COPY TASK BENCHMARK                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Tests the model's ability to copy input sequences - a fundamental           ║
║  sequence-to-sequence learning task.                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE EXAMPLES:                                                              ║
║                                                                               ║
║  Basic run:                                                                   ║
║    python tests/scenarios/copy_test/main.py                                   ║
║                                                                               ║
║  Custom sequence length:                                                      ║
║    python tests/scenarios/copy_test/main.py --seq_len 256 --epochs 30        ║
║                                                                               ║
║  Full configuration:                                                          ║
║    python tests/scenarios/copy_test/main.py \\                                ║
║        --seq_len 512 --epochs 50 --batch_size 64 \\                           ║
║        --d_model 128 --num_heads 8 --m_features 32 \\                         ║
║        --learning_rate 5e-4 --use_wandb                                       ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARGUMENTS:                                                                   ║
║    --seq_len       : Sequence length (default: 128)                           ║
║    --epochs        : Number of training epochs (default: 20)                  ║
║    --batch_size    : Batch size (default: 16)                                 ║
║    --num_samples   : Number of training samples (default: 10000)              ║
║    --vocab_size    : Vocabulary size (default: 64)                            ║
║    --d_model       : Model dimension (default: 16)                            ║
║    --num_heads     : Number of attention heads (default: 4)                   ║
║    --num_layers    : Number of transformer layers (default: 1)                ║
║    --m_features    : HRC-LA feature dimension (default: 8)                    ║
║    --learning_rate : Learning rate (default: 7e-4)                            ║
║    --use_wandb     : Enable Weights & Biases logging                          ║
║    --no_plot       : Disable plot generation                                  ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT:                                                                      ║
║    Results saved to: tests/scenarios/copy_test/results/                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(usage_text)


def train_copy_task(
    config: Optional[BenchmarkConfig] = None,
    use_wandb: bool = False,
    generate_plots: bool = True,
) -> float:
    """
    Train and evaluate the HRC-LA Transformer on the copy task.
    
    The copy task requires the model to output an exact replica of the input
    sequence, testing its ability to maintain and retrieve information across
    varying sequence lengths.
    
    Args:
        config: Experiment configuration. Uses defaults if None.
        use_wandb: Enable Weights & Biases logging.
        generate_plots: Generate training visualization plots.
        
    Returns:
        Best validation accuracy achieved during training.
    """
    if config is None:
        config = BenchmarkConfig()

    device = config.get_device()
    
    logger.info(f"Device: {device}")
    logger.info(f"Sequence Length: {config.seq_len}")
    logger.info(
        f"Model: d_model={config.d_model}, heads={config.num_heads}, "
        f"layers={config.num_layers}, m_features={config.m_features}"
    )

    if WANDB_AVAILABLE and use_wandb:
        wandb.init(project="hrc-copy-task", config=config.__dict__)

    # Dataset preparation
    train_dataset = CopyTaskDataset(
        config.vocab_size, config.seq_len, config.num_samples
    )
    val_dataset = CopyTaskDataset(
        config.vocab_size, config.seq_len, config.num_samples // 10, seed=43
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Model initialization
    model = HRCTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        m_features=config.m_features,
        dropout=config.dropout,
        learnable_omega=True,
        max_len=config.seq_len + 100,
        max_seq_len=config.seq_len + 100,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

    # Optimizer with AdamW and learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    # Training state
    train_history = []
    best_val_acc = 0.0
    total_start_time = time.time()

    # Main training loop
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Reset peak memory tracking for this epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            learnable_omega_penalty=config.learnable_omega_penalty,
        )

        val_loss, val_acc = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        best_val_acc = max(best_val_acc, val_acc)
        
        # Track GPU memory usage
        memory_mb = 0.0
        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
            "memory_mb": memory_mb,
        })

        logger.info(
            f"Epoch {epoch + 1:3d}/{config.num_epochs} │ "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} │ "
            f"Time: {epoch_time:.2f}s │ Mem: {memory_mb:.1f}MB"
        )

        if WANDB_AVAILABLE and use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "epoch_time": epoch_time,
                "memory_mb": memory_mb,
                "epoch": epoch,
            })

    total_time = time.time() - total_start_time

    # Configuration dictionary for saving
    config_dict = {
        "seq_len": config.seq_len,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "m_features": config.m_features,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "num_samples": config.num_samples,
    }

    # Save results
    save_experiment_results(
        results_dir=RESULTS_DIR,
        experiment_name="copy_task",
        config=config_dict,
        history=train_history,
        best_val_acc=best_val_acc,
        total_time=total_time,
    )

    # Generate visualization plots
    if generate_plots:
        visualizer = BenchmarkVisualizer(
            results_dir=RESULTS_DIR,
            experiment_name="copy_task",
        )
        
        metrics = TrainingMetrics.from_history(
            experiment_name=f"Copy Task (seq_len={config.seq_len})",
            history=train_history,
            best_val_accuracy=best_val_acc,
            total_time=total_time,
            config=config_dict,
        )
        
        visualizer.plot_training_curves(metrics, save=True, show=False)

    # Print summary
    print_experiment_summary(
        experiment_name="Copy Task",
        config=config_dict,
        best_val_acc=best_val_acc,
        total_time=total_time,
        results_dir=RESULTS_DIR,
    )

    if WANDB_AVAILABLE and use_wandb:
        wandb.finish()

    return best_val_acc


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Copy Task Benchmark for HRC-LA Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model architecture
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Input sequence length")
    parser.add_argument("--vocab_size", type=int, default=64,
                        help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=16,
                        help="Model embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of transformer layers")
    parser.add_argument("--m_features", type=int, default=8,
                        help="HRC-LA random feature dimension")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--learning_rate", type=float, default=7e-4,
                        help="Initial learning rate")
    
    # Logging and output
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plot generation")
    parser.add_argument("--show_usage", action="store_true",
                        help="Show usage examples and exit")
    
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_arguments()

    if args.show_usage:
        print_usage()
        sys.exit(0)

    print_usage()

    config = BenchmarkConfig(
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        m_features=args.m_features,
        learning_rate=args.learning_rate,
    )

    train_copy_task(
        config=config,
        use_wandb=args.use_wandb,
        generate_plots=not args.no_plot,
    )
