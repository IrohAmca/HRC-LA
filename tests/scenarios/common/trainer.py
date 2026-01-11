"""
Training Utilities for HRC-LA Benchmark Scenarios.

This module provides standardized training and validation functions
used across all benchmark scenarios, ensuring consistent training
procedures and fair comparisons.

The training pipeline includes:
    - Warmup + Cosine decay learning rate scheduling
    - Gradient clipping for training stability
    - Optional orthogonality regularization for learnable omega

References:
    Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent
        with Warm Restarts. ICLR.
    Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
"""

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    The schedule follows:
        1. Linear warmup from 0 to base_lr over warmup_steps
        2. Cosine decay from base_lr to 0.1 * base_lr over remaining steps
    
    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        
    Returns:
        LambdaLR scheduler instance.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: The PyTorch model to analyze.
        
    Returns:
        Tuple of (total_parameters, trainable_parameters).
    """
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
    """
    Execute a single training epoch.
    
    Performs forward pass, loss computation, backpropagation, and
    parameter updates for all batches in the dataloader.
    
    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimizer for parameter updates.
        scheduler: Learning rate scheduler.
        device: Computation device.
        learnable_omega_penalty: Regularization weight for omega orthogonality.
        
    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Cross-entropy loss for next-token prediction
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))

        # Orthogonality regularization for learnable omega matrices
        if learnable_omega_penalty > 0 and hasattr(model, "layers"):
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
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: The model to evaluate.
        dataloader: Validation data loader.
        device: Computation device.
        
    Returns:
        Tuple of (average_loss, accuracy) on validation set.
    """
    model.eval()
    total_loss = 0.0
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
