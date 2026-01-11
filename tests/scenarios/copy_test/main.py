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

logger = logging.getLogger(__name__)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")


def train_copy_task(config: Optional[BenchmarkConfig] = None):
    """
    Trainer for the Copy Task using HRC Transformer.
    """
    if config is None:
        config = BenchmarkConfig()

    device = config.get_device()
    logger.info(f"Device: {device}")

    if WANDB_AVAILABLE:
        wandb.init(project="hrc-copy-task-single", config=config.__dict__)

    train_dataset = CopyTaskDataset(
        config.vocab_size, config.seq_len, config.num_samples
    )
    val_dataset = CopyTaskDataset(
        config.vocab_size, config.seq_len, config.num_samples // 10
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = HRCTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        m_features=config.m_features,
        dropout=config.dropout,
        learnable_omega=True,
        max_len=config.seq_len + 100,  # Buffer for positional encoding
        max_seq_len=config.seq_len + 100,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    for epoch in range(config.num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            learnable_omega_penalty=config.learnable_omega_penalty,
        )

        val_loss, val_acc = validate(model, val_loader, device)

        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time,
                    "epoch": epoch,
                }
            )

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    config = BenchmarkConfig(seq_len=args.seq_len, num_epochs=args.epochs)

    train_copy_task(config)
