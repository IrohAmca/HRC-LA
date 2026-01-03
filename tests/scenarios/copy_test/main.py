import argparse
import logging
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from tests.models import HRCTransformer
from tests.scenarios.copy_test.config import CopyTaskConfig

logger = logging.getLogger(__name__)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")


class CopyTaskDataset(Dataset):
    """
    Copy Task Dataset.

    The dataset generates sequences where the first half is random and the second half is a copy of the first half.
    Each sequence is of length `seq_len` and consists of integers in the range [0, vocab_size).
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        half_len = seq_len // 2
        random_part = torch.randint(0, vocab_size, (num_samples, half_len))
        self.data = torch.cat([random_part, random_part], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return sequence, sequence  # input == target (copy task)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Warmup + Cosine decay learning rate scheduler."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_copy_task(config: Optional[CopyTaskConfig] = None):
    """
    Trainer for the Copy Task using HRC Transformer.
    """

    if config is None:
        config = CopyTaskConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "vocab_size": config.vocab_size,
                "seq_len": config.seq_len,
                "d_model": config.d_model,
                "num_heads": config.num_heads,
                "num_layers": config.num_layers,
                "d_ff": config.d_ff,
                "m_features": config.m_features,
                "dropout": config.dropout,
                "learnable_omega": config.learnable_omega,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
            },
        )

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
        max_len=config.seq_len + 10,
        dropout=config.dropout,
        learnable_omega=config.learnable_omega,
        learnable_omega_penalty=config.learnable_omega_penalty,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    omega_params = []
    standart_params = []
    for name, param in model.named_parameters():
        if "omega" in name:
            omega_params.append(param)
        else:
            standart_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": standart_params, "lr": config.learning_rate},
            {"params": omega_params, "lr": config.learning_rate * 0.33},
        ],
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

    global_step = 0
    best_val_acc = 0.0

    logger.info("\n Training started...\n")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

            if config.learnable_omega:
                loss = loss + model.ortho_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            train_correct += (predictions == targets).sum().item()
            train_total += targets.numel()

            global_step += 1

            if global_step % 10 == 0 and config.use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/accuracy": (predictions == targets)
                        .float()
                        .mean()
                        .item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    }
                )

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        logger.info("===== Validation =====")
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size), targets.view(-1)
                )

                val_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {val_acc:.4f}")

        logger.info(
            f"Epoch {epoch + 1:2d}/{config.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "train/epoch_accuracy": train_acc,
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_acc,
                    "val/best_accuracy": best_val_acc,
                }
            )

    logger.info(f"\n Training completed! Best Val Accuracy: {best_val_acc:.4f}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return model, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRC Transformer Copy Task Training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--learnable-omega", action="store_true", help="Use learnable omega"
    )
    parser.add_argument(
        "--learnable-omega-penalty",
        type=float,
        default=0.0001,
        help="Learnable omega penalty",
    )
    args = parser.parse_args()

    config = CopyTaskConfig(
        use_wandb=not args.no_wandb,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        learnable_omega=args.learnable_omega,
        learnable_omega_penalty=args.learnable_omega_penalty,
    )
    train_copy_task(config)
