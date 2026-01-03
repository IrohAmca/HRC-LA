from dataclasses import dataclass
from typing import Optional

@dataclass
class CopyTaskConfig:
    vocab_size: int = 256
    seq_len: int = 64
    num_samples: int = 10000
    
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    m_features: int = 128
    dropout: float = 0.1
    learnable_omega: bool = False
    learnable_omega_penalty: float = 0.0001
    
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    warmup_steps: int = 100
    
    use_wandb: bool = True
    wandb_project: str = "hrc-transformer-copy-task"
    wandb_run_name: Optional[str] = None
