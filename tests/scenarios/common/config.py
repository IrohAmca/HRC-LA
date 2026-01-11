"""
Configuration Classes for HRC-LA Benchmark Scenarios.

This module provides standardized configuration dataclasses for
benchmark experiments, ensuring consistent hyperparameter management
across different evaluation scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class BenchmarkConfig:
    """
    Base configuration for benchmark experiments.
    
    Defines all hyperparameters for model architecture, training procedure,
    and experiment settings. Subclasses may extend this for scenario-specific
    parameters.
    
    Attributes:
        vocab_size: Size of the token vocabulary.
        seq_len: Input sequence length.
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        d_ff: Feed-forward network hidden dimension.
        m_features: HRC-LA random feature dimension.
        dropout: Dropout probability.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        num_epochs: Number of training epochs.
        warmup_steps: Learning rate warmup steps.
        num_samples: Number of training samples.
        learnable_omega_penalty: Orthogonality regularization weight.
        random_seed: Random seed for reproducibility.
        device: Computation device specification.
    """
    # Model configuration
    vocab_size: int = 64
    seq_len: int = 16000
    d_model: int = 16
    num_heads: int = 4
    num_layers: int = 1
    d_ff: int = 16
    m_features: int = 8
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 7e-4
    num_epochs: int = 20
    warmup_steps: int = 50
    num_samples: int = 10000
    
    # Learnable omega settings
    learnable_omega_penalty: float = 0.0001
    
    # Experiment settings
    random_seed: int = 42
    device: str = "auto"
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    attention_type: str
    seq_len: int = 0 
    
    best_val_accuracy: float = 0.0
    final_val_accuracy: float = 0.0
    final_train_accuracy: float = 0.0
    
    final_val_loss: float = 0.0
    final_train_loss: float = 0.0
    
    total_training_time_seconds: float = 0.0
    avg_epoch_time_seconds: float = 0.0
    total_parameters: int = 0
    trainable_parameters: int = 0
    
    peak_memory_mb: Optional[float] = None
    
    train_losses: Optional[List[float]] = None
    val_losses: Optional[List[float]] = None
    train_accuracies: Optional[List[float]] = None
    val_accuracies: Optional[List[float]] = None
    
    config: Optional[Dict] = None
