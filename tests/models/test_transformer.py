"""
Transformer implementation using HRC Attention.
This file contains a simple transformer for training and testing purposes.
"""

import math

import torch
import torch.nn as nn

from hrc_la import HRCMultiheadAttention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Transformer Feed Forward Network."""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class HRCTransformerEncoderLayer(nn.Module):
    """Single Encoder Layer of the HRC Transformer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        m_features: int = 256,
        dropout: float = 0.1,
        learnable_omega: bool = False,
        learnable_omega_penalty: float = 0.001,
    ):
        super().__init__()

        self.self_attn = HRCMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            m_features=m_features,
            batch_first=True,
            learnable_omega=learnable_omega,
            learnable_omega_penalty=learnable_omega_penalty,
        )

        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(self.norm1(x))
        x = x + self.dropout1(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x

    @property
    def ortho_loss(self) -> torch.Tensor:
        return getattr(self.self_attn, "ortho_loss", 0)


class HRCTransformer(nn.Module):
    """Transformer model using HRC Attention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        m_features: int = 256,
        max_len: int = 512,
        dropout: float = 0.1,
        learnable_omega: bool = True,
        learnable_omega_penalty: float = 0.0001,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [
                HRCTransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    m_features=m_features,
                    dropout=dropout,
                    learnable_omega=learnable_omega,
                    learnable_omega_penalty=learnable_omega_penalty,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits

    @property
    def ortho_loss(self) -> torch.Tensor:
        total_loss = 0
        for layer in self.layers:
            total_loss = total_loss + layer.ortho_loss
        return total_loss
