"""
Needle In A Haystack (NIAH) Test for HRC-LA Transformer.

This module implements the NIAH evaluation, a critical benchmark for assessing
long-context retrieval capabilities. The model must locate and retrieve specific
key-value pairs ("needles") hidden at various depths within long sequences of
random tokens ("haystack").

Task Definition:
    Given a sequence containing random tokens with a key-value pair inserted
    at depth d ∈ [0, 1], the model must correctly predict the value token
    when presented with the key.

Evaluation Methodology:
    - Single Depth: Test retrieval at a specific position
    - Multi-Depth: Sweep across all depths to create retrieval heatmaps

References:
    Liu, N., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
    Kamradt, G. (2023). Needle In A Haystack - Pressure Testing LLMs.
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project root configuration
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from tests.models import HRCTransformer
from tests.scenarios.common.config import BenchmarkConfig
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
from tests.scenarios.needle_test.datasets import (
    MultiDepthNIAHDataset,
    NeedleInHaystackDataset,
)

logger = logging.getLogger(__name__)

# Results directory for this scenario
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# WandB availability check
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class NIAHConfig(BenchmarkConfig):
    """
    Configuration for NIAH experiments.
    
    Extends BenchmarkConfig with NIAH-specific parameters for controlling
    needle placement, multi-depth evaluation, and retrieval settings.
    
    Attributes:
        needle_depth: Position of needle in sequence (0.0=start, 1.0=end).
        needle_length: Number of tokens in each needle key-value pair.
        num_needles: Number of needle pairs to insert per sequence.
        test_multiple_depths: Enable multi-depth sweep evaluation.
        num_depth_levels: Number of depth levels for multi-depth test.
        samples_per_depth: Samples to generate per depth level.
        use_wandb: Enable WandB logging.
    """
    needle_depth: float = 0.5
    needle_length: int = 2
    num_needles: int = 1
    test_multiple_depths: bool = False
    num_depth_levels: int = 10
    samples_per_depth: int = 100
    use_wandb: bool = False


def print_usage() -> None:
    """Display usage information and command-line examples."""
    usage_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEEDLE IN A HAYSTACK (NIAH) TEST                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Evaluates the model's ability to retrieve specific key-value pairs           ║
║  hidden at various depths within long sequences.                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE EXAMPLES:                                                              ║
║                                                                               ║
║  Basic run (single depth):                                                    ║
║    python tests/scenarios/needle_test/main.py                                 ║
║                                                                               ║
║  Custom depth position (0.0=start, 1.0=end):                                  ║
║    python tests/scenarios/needle_test/main.py --needle_depth 0.3              ║
║                                                                               ║
║  Multi-depth sweep (recommended for full evaluation):                         ║
║    python tests/scenarios/needle_test/main.py --multi_depth --num_depths 10   ║
║                                                                               ║
║  Long sequence test:                                                          ║
║    python tests/scenarios/needle_test/main.py \\                              ║
║        --seq_len 2048 --d_model 128 --num_heads 8                             ║
║                                                                               ║
║  Full configuration with WandB:                                               ║
║    python tests/scenarios/needle_test/main.py \\                              ║
║        --seq_len 1024 --epochs 30 --batch_size 64 \\                          ║
║        --d_model 128 --num_heads 8 --m_features 32 \\                         ║
║        --multi_depth --num_depths 10 --use_wandb                              ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARGUMENTS:                                                                   ║
║    --seq_len       : Sequence length (default: 512)                           ║
║    --epochs        : Number of training epochs (default: 20)                  ║
║    --batch_size    : Batch size (default: 32)                                 ║
║    --num_samples   : Number of training samples (default: 10000)              ║
║    --vocab_size    : Vocabulary size (default: 64)                            ║
║    --d_model       : Model dimension (default: 64)                            ║
║    --num_heads     : Number of attention heads (default: 4)                   ║
║    --num_layers    : Number of transformer layers (default: 1)                ║
║    --m_features    : HRC-LA feature dimension (default: 16)                   ║
║    --learning_rate : Learning rate (default: 7e-4)                            ║
║                                                                               ║
║  NIAH-SPECIFIC:                                                               ║
║    --needle_depth  : Needle position 0.0-1.0 (default: 0.5)                   ║
║    --num_needles   : Number of needle pairs (default: 1)                      ║
║    --multi_depth   : Enable multi-depth evaluation                            ║
║    --num_depths    : Number of depth levels (default: 10)                     ║
║                                                                               ║
║  LOGGING:                                                                     ║
║    --use_wandb     : Enable Weights & Biases logging                          ║
║    --no_plot       : Disable plot generation                                  ║
║                                                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT:                                                                      ║
║    Results saved to: tests/scenarios/needle_test/results/                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(usage_text)


def evaluate_needle_retrieval(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    needle_positions: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """
    Evaluate needle retrieval accuracy.
    
    Computes both overall sequence accuracy and needle-specific retrieval
    accuracy when needle positions are provided.
    
    Args:
        model: The model to evaluate.
        dataloader: DataLoader containing NIAH samples.
        device: Computation device.
        needle_positions: Optional list of needle positions per sample.
        
    Returns:
        Dictionary containing loss, accuracy, and needle_accuracy metrics.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    needle_correct = 0
    needle_total = 0
    batch_idx = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), 
                target.view(-1)
            )
            total_loss += loss.item()
            
            pred = output.argmax(dim=-1)
            total_correct += (pred == target).sum().item()
            total_tokens += target.numel()
            
            # Needle-specific accuracy computation
            if needle_positions is not None:
                batch_size = data.size(0)
                for i in range(batch_size):
                    sample_idx = batch_idx * dataloader.batch_size + i
                    if sample_idx < len(needle_positions):
                        for pos in needle_positions[sample_idx]:
                            if pos + 1 < data.size(1):
                                needle_total += 1
                                if pred[i, pos] == target[i, pos]:
                                    needle_correct += 1
            
            batch_idx += 1
    
    results = {
        "loss": total_loss / len(dataloader),
        "accuracy": total_correct / total_tokens,
    }
    
    if needle_total > 0:
        results["needle_accuracy"] = needle_correct / needle_total
    
    return results


def evaluate_by_depth(
    model: torch.nn.Module,
    dataset: MultiDepthNIAHDataset,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model performance at each depth level.
    
    Creates a depth-wise performance profile by evaluating retrieval
    accuracy at each needle depth position.
    
    Args:
        model: The model to evaluate.
        dataset: MultiDepthNIAHDataset with samples at various depths.
        device: Computation device.
        batch_size: Evaluation batch size.
        
    Returns:
        Dictionary mapping depth -> {accuracy, num_samples}.
    """
    model.eval()
    results_by_depth = {}
    
    for depth in dataset.depth_levels:
        indices = dataset.get_samples_by_depth(depth)
        
        if not indices:
            continue
        
        subset_data = dataset.data[indices]
        subset_targets = dataset.targets[indices]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_data = subset_data[i:i+batch_size].to(device)
                batch_targets = subset_targets[i:i+batch_size].to(device)
                
                output = model(batch_data)
                pred = output.argmax(dim=-1)
                
                correct += (pred == batch_targets).sum().item()
                total += batch_targets.numel()
        
        results_by_depth[depth] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "num_samples": len(indices),
        }
    
    return results_by_depth


def train_niah(
    config: Optional[NIAHConfig] = None,
    generate_plots: bool = True,
) -> float:
    """
    Train and evaluate on the Needle In A Haystack task.
    
    Implements the complete NIAH evaluation pipeline including training,
    validation, and optional multi-depth analysis with visualization.
    
    Args:
        config: NIAH experiment configuration. Uses defaults if None.
        generate_plots: Generate training and depth analysis plots.
        
    Returns:
        Best validation accuracy achieved during training.
    """
    if config is None:
        config = NIAHConfig()

    device = config.get_device()
    
    logger.info(f"Device: {device}")
    logger.info(f"Sequence Length: {config.seq_len}")
    logger.info(f"Needle Depth: {config.needle_depth}")
    logger.info(f"Multi-Depth Mode: {config.test_multiple_depths}")
    logger.info(
        f"Model: d_model={config.d_model}, heads={config.num_heads}, "
        f"layers={config.num_layers}, m_features={config.m_features}"
    )

    if WANDB_AVAILABLE and config.use_wandb:
        wandb.init(project="hrc-niah-test", config=config.__dict__)

    # Dataset preparation based on evaluation mode
    if config.test_multiple_depths:
        train_dataset = MultiDepthNIAHDataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            samples_per_depth=config.num_samples // config.num_depth_levels,
            num_depths=config.num_depth_levels,
            needle_length=config.needle_length,
        )
        val_dataset = MultiDepthNIAHDataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            samples_per_depth=config.num_samples // (config.num_depth_levels * 10),
            num_depths=config.num_depth_levels,
            needle_length=config.needle_length,
            seed=43,
        )
    else:
        train_dataset = NeedleInHaystackDataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            num_samples=config.num_samples,
            needle_depth=config.needle_depth,
            needle_length=config.needle_length,
            num_needles=config.num_needles,
        )
        val_dataset = NeedleInHaystackDataset(
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            num_samples=config.num_samples // 10,
            needle_depth=config.needle_depth,
            needle_length=config.needle_length,
            num_needles=config.num_needles,
            seed=43,
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

    # Optimizer configuration
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

        if WANDB_AVAILABLE and config.use_wandb:
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

    # Depth-wise evaluation for multi-depth mode
    depth_results = None
    if config.test_multiple_depths and isinstance(val_dataset, MultiDepthNIAHDataset):
        logger.info("\n" + "═" * 60)
        logger.info("DEPTH-WISE EVALUATION")
        logger.info("═" * 60)
        
        depth_results = evaluate_by_depth(
            model, val_dataset, device, config.batch_size
        )
        
        for depth, metrics in sorted(depth_results.items()):
            logger.info(f"  Depth {depth:.2f}: Accuracy = {metrics['accuracy']:.4f}")
            
            if WANDB_AVAILABLE and config.use_wandb:
                wandb.log({f"depth_{depth:.2f}_accuracy": metrics["accuracy"]})

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
        "needle_depth": config.needle_depth,
        "num_needles": config.num_needles,
        "test_multiple_depths": config.test_multiple_depths,
        "num_depth_levels": config.num_depth_levels,
    }

    # Prepare extra data for saving
    extra_data = None
    if depth_results is not None:
        extra_data = {
            "depth_analysis": {str(k): v for k, v in depth_results.items()}
        }

    # Determine experiment name
    mode_str = "multi_depth" if config.test_multiple_depths else f"depth{config.needle_depth}"
    experiment_name = f"niah_{mode_str}"

    # Save results
    save_experiment_results(
        results_dir=RESULTS_DIR,
        experiment_name=experiment_name,
        config=config_dict,
        history=train_history,
        best_val_acc=best_val_acc,
        total_time=total_time,
        extra_data=extra_data,
    )

    # Generate visualization plots
    if generate_plots:
        visualizer = BenchmarkVisualizer(
            results_dir=RESULTS_DIR,
            experiment_name="niah",
        )
        
        metrics = TrainingMetrics.from_history(
            experiment_name=f"NIAH (seq_len={config.seq_len})",
            history=train_history,
            best_val_accuracy=best_val_acc,
            total_time=total_time,
            config=config_dict,
        )
        
        visualizer.plot_training_curves(metrics, save=True, show=False)
        
        # Generate depth heatmap for multi-depth evaluation
        if depth_results is not None:
            visualizer.plot_depth_heatmap(
                depth_results=depth_results,
                seq_len=config.seq_len,
                save=True,
                show=False,
            )

    # Print summary
    print_experiment_summary(
        experiment_name="Needle In A Haystack",
        config=config_dict,
        best_val_acc=best_val_acc,
        total_time=total_time,
        results_dir=RESULTS_DIR,
    )

    if WANDB_AVAILABLE and config.use_wandb:
        wandb.finish()

    return best_val_acc


def run_depth_sweep(
    seq_lengths: List[int],
    num_depths: int = 10,
    config: Optional[NIAHConfig] = None,
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """
    Run a full depth sweep across multiple sequence lengths.
    
    Creates comprehensive data for NIAH heatmap visualization by
    evaluating retrieval accuracy at all depth levels for each
    specified sequence length.
    
    Args:
        seq_lengths: List of sequence lengths to evaluate.
        num_depths: Number of depth levels per sequence length.
        config: Base configuration. Uses defaults if None.
        
    Returns:
        Nested dictionary: seq_len -> depth -> {accuracy, num_samples}.
    """
    if config is None:
        config = NIAHConfig()
    
    results = {}
    
    for seq_len in seq_lengths:
        logger.info(f"\n{'═' * 60}")
        logger.info(f"Testing Sequence Length: {seq_len}")
        logger.info(f"{'═' * 60}")
        
        config.seq_len = seq_len
        config.test_multiple_depths = True
        config.num_depth_levels = num_depths
        
        device = config.get_device()
        
        model = HRCTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            m_features=config.m_features,
            dropout=config.dropout,
            learnable_omega=True,
            max_len=seq_len + 100,
            max_seq_len=seq_len + 100,
        ).to(device)
        
        dataset = MultiDepthNIAHDataset(
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            samples_per_depth=100,
            num_depths=num_depths,
        )
        
        depth_results = evaluate_by_depth(model, dataset, device, config.batch_size)
        results[seq_len] = depth_results
    
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Needle In A Haystack (NIAH) Test for HRC-LA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model architecture
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Input sequence length")
    parser.add_argument("--vocab_size", type=int, default=64,
                        help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Model embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of transformer layers")
    parser.add_argument("--m_features", type=int, default=16,
                        help="HRC-LA random feature dimension")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--learning_rate", type=float, default=7e-4,
                        help="Initial learning rate")
    
    # NIAH-specific parameters
    parser.add_argument("--needle_depth", type=float, default=0.5,
                        help="Needle position (0.0=start, 1.0=end)")
    parser.add_argument("--num_needles", type=int, default=1,
                        help="Number of needle pairs per sequence")
    parser.add_argument("--multi_depth", action="store_true",
                        help="Enable multi-depth evaluation sweep")
    parser.add_argument("--num_depths", type=int, default=10,
                        help="Number of depth levels for multi-depth test")
    
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

    config = NIAHConfig(
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        num_samples=args.num_samples,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        m_features=args.m_features,
        needle_depth=args.needle_depth,
        num_needles=args.num_needles,
        test_multiple_depths=args.multi_depth,
        num_depth_levels=args.num_depths,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb,
    )

    train_niah(config=config, generate_plots=not args.no_plot)
