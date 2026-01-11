"""
Needle In A Haystack (NIAH) Dataset Implementation.

This module provides dataset classes for the NIAH evaluation benchmark,
which tests a model's ability to retrieve specific information hidden
at various positions within long sequences.

The NIAH task is critical for evaluating:
    1. Long-context retrieval capabilities
    2. Position-invariant information access
    3. Attention mechanism effectiveness across sequence lengths

References:
    Liu, N., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
    Kamradt, G. (2023). Needle In A Haystack - Pressure Testing LLMs.
"""

import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class NeedleInHaystackDataset(Dataset):
    """
    Single-depth Needle In A Haystack Dataset.

    Generates sequences containing random "haystack" tokens with key-value
    "needle" pairs inserted at a specified depth. The model must learn to
    associate keys with their corresponding values regardless of position.

    Sequence Structure:
        [haystack_tokens][KEY_i][VALUE_i][haystack_tokens]

    Training Objective:
        Given the sequence, predict the next token at each position.
        The critical evaluation metric is accuracy on VALUE prediction
        when KEY is presented.

    Attributes:
        vocab_size: Total vocabulary size.
        seq_len: Length of generated sequences.
        num_samples: Number of samples in the dataset.
        needle_depth: Fractional position of needle (0.0=start, 1.0=end).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        needle_depth: float = 0.5,
        needle_length: int = 2, 
        num_needles: int = 1,  
        depth_range: Optional[Tuple[float, float]] = None,  
        seed: int = 42,
    ):
        """
        Initialize the NIAH dataset.

        Args:
            vocab_size: Size of the vocabulary
            seq_len: Total sequence length
            num_samples: Number of samples to generate
            needle_depth: Where to place the needle (0.0-1.0, fraction of seq_len)
            needle_length: Length of each needle (key + value tokens)
            num_needles: Number of needle pairs to insert
            depth_range: If provided, randomly sample depth from this range
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.needle_depth = needle_depth
        self.needle_length = needle_length
        self.num_needles = num_needles
        self.depth_range = depth_range

        self.key_start = 0
        self.value_start = num_needles
        self.haystack_start = 2 * num_needles

        random.seed(seed)
        torch.manual_seed(seed)

        self.data = []
        self.targets = []
        self.needle_positions = []

        self._generate_samples()

    def _generate_samples(self):
        """Generate all samples."""
        for _ in range(self.num_samples):
            sequence, target, positions = self._generate_single_sample()
            self.data.append(sequence)
            self.targets.append(target)
            self.needle_positions.append(positions)

        self.data = torch.stack(self.data)
        self.targets = torch.stack(self.targets)

    def _generate_single_sample(self) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Generate a single NIAH sample."""
        sequence = torch.randint(self.haystack_start, self.vocab_size, (self.seq_len,))

        if self.depth_range is not None:
            depth = random.uniform(*self.depth_range)
        else:
            depth = self.needle_depth

        max_needle_pos = self.seq_len - self.needle_length * self.num_needles - 1
        needle_start = int(depth * max_needle_pos)

        positions = []
        target_values = []

        for i in range(self.num_needles):
            key_token = self.key_start + i
            value_token = self.value_start + i

            pos = needle_start + i * self.needle_length
            sequence[pos] = key_token
            sequence[pos + 1] = value_token

            positions.append(pos)
            target_values.append(value_token)

        target = sequence.clone()

        return sequence, target, positions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_needle_positions(self, idx):
        """Get needle positions for a specific sample."""
        return self.needle_positions[idx]


class MultiDepthNIAHDataset(Dataset):
    """
    Multi-depth Needle In A Haystack Dataset.

    Generates samples with needles placed at various depth levels to
    evaluate retrieval capability across the full context window.
    This enables creation of NIAH heatmaps for comprehensive analysis.

    The depth levels are evenly distributed from 0.1 to 0.9, avoiding
    the extreme edges where positional effects may dominate.

    Attributes:
        depth_levels: List of evaluated depth positions.
        samples_per_depth: Number of samples generated per depth.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        samples_per_depth: int = 100,
        num_depths: int = 10, 
        needle_length: int = 2,
        seed: int = 42,
    ):
        """
        Initialize multi-depth NIAH dataset.

        Args:
            vocab_size: Size of the vocabulary
            seq_len: Total sequence length
            samples_per_depth: Number of samples per depth level
            num_depths: Number of evenly spaced depth levels
            needle_length: Length of each needle
            seed: Random seed
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.samples_per_depth = samples_per_depth
        self.num_depths = num_depths
        self.needle_length = needle_length

        # Calculate depth levels (evenly spaced from 0.1 to 0.9)
        self.depth_levels = [
            0.1 + 0.8 * i / (num_depths - 1) for i in range(num_depths)
        ]

        self.data = []
        self.targets = []
        self.depths = []

        torch.manual_seed(seed)
        random.seed(seed)

        self._generate_samples()

    def _generate_samples(self):
        """Generate samples for all depth levels."""
        for depth in self.depth_levels:
            dataset = NeedleInHaystackDataset(
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
                num_samples=self.samples_per_depth,
                needle_depth=depth,
                needle_length=self.needle_length,
                seed=random.randint(0, 100000),
            )

            for i in range(len(dataset)):
                seq, target = dataset[i]
                self.data.append(seq)
                self.targets.append(target)
                self.depths.append(depth)

        self.data = torch.stack(self.data)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_depth(self, idx):
        """Get the needle depth for a specific sample."""
        return self.depths[idx]

    def get_samples_by_depth(self, depth: float) -> list:
        """Get all sample indices for a specific depth."""
        return [i for i, d in enumerate(self.depths) if abs(d - depth) < 0.01]
