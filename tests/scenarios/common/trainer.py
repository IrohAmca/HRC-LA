import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


def count_parameters(model) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    learnable_omega_penalty: float = 0.0,
) -> Tuple[float, float]:
    """Run a single training epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))

        if hasattr(model, "layers"):
            for layer in model.layers:
                if hasattr(layer.self_attn, "omega") and isinstance(
                    layer.self_attn.omega, nn.Parameter
                ):
                    omega = layer.self_attn.omega
                    for h in range(omega.shape[0]):
                        w = omega[h]
                        wtw = torch.matmul(w.T, w)
                        identity = torch.eye(w.shape[1], device=device)
                        ortho_loss = torch.norm(wtw - identity) ** 2
                        loss += learnable_omega_penalty * ortho_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        pred = output.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += target.numel()

    return total_loss / len(dataloader), correct / total


def validate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Run validation."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()

            pred = output.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.numel()

    return total_loss / len(dataloader), correct / total
