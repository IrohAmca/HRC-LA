"""
Common Dataset Implementations for HRC-LA Benchmarks.

This module provides standardized dataset classes used across multiple
benchmark scenarios, ensuring consistent data generation and evaluation.

References:
    Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines.
"""

import torch
from torch.utils.data import Dataset


class CopyTaskDataset(Dataset):
    """
    Copy Task Dataset for sequence memorization evaluation.
    
    Generates sequences where the model must learn to reproduce the input
    exactly. This tests the fundamental ability to store and retrieve
    sequential information through the attention mechanism.
    
    Sequence Structure:
        Input:  [x_1, x_2, ..., x_n, x_1, x_2, ..., x_n]
        Target: [x_1, x_2, ..., x_n, x_1, x_2, ..., x_n]
        
    The first half contains random tokens, and the second half is an
    exact copy. The model must learn the copying pattern.
    
    Attributes:
        vocab_size: Number of unique tokens in vocabulary.
        seq_len: Total sequence length (must be even).
        num_samples: Number of samples in the dataset.
    """

    def __init__(
        self, 
        vocab_size: int, 
        seq_len: int, 
        num_samples: int, 
        seed: int = 42
    ):
        """
        Initialize the Copy Task Dataset.
        
        Args:
            vocab_size: Size of the token vocabulary.
            seq_len: Total sequence length (will use half for pattern).
            num_samples: Number of sequences to generate.
            seed: Random seed for reproducibility.
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        generator = torch.Generator().manual_seed(seed)
        
        half_len = seq_len // 2
        random_part = torch.randint(
            0, vocab_size, (num_samples, half_len), generator=generator
        )
        self.data = torch.cat([random_part, random_part], dim=1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        sequence = self.data[idx]
        return sequence, sequence
