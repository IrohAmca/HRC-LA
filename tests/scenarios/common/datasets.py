import torch
from torch.utils.data import Dataset


class CopyTaskDataset(Dataset):
    """
    Copy Task Dataset.
    
    The dataset generates sequences where the first half is random 
    and the second half is a copy of the first half.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        generator = torch.Generator().manual_seed(seed)
        
        half_len = seq_len // 2
        random_part = torch.randint(
            0, vocab_size, (num_samples, half_len), generator=generator
        )
        self.data = torch.cat([random_part, random_part], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return sequence, sequence
