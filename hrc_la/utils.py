import time

import torch
from torch import nn


def sync_weights(std_model, hrc_model):
    """
    Synchronizes weights from a standard MultiheadAttention model to the HRC model.
    """
    d_model = std_model.embed_dim

    # Standard weights: [3 * d_model, d_model]
    qkv_weight = std_model.in_proj_weight
    qkv_bias = std_model.in_proj_bias

    # Split weights
    hrc_model.q_proj.weight.data = qkv_weight[:d_model, :].clone()
    hrc_model.k_proj.weight.data = qkv_weight[d_model : 2 * d_model, :].clone()
    hrc_model.v_proj.weight.data = qkv_weight[2 * d_model :, :].clone()

    hrc_model.q_proj.bias.data = qkv_bias[:d_model].clone()
    hrc_model.k_proj.bias.data = qkv_bias[d_model : 2 * d_model].clone()
    hrc_model.v_proj.bias.data = qkv_bias[2 * d_model :].clone()

    # Output weights
    hrc_model.out_proj.weight.data = std_model.out_proj.weight.data.clone()
    hrc_model.out_proj.bias.data = std_model.out_proj.bias.data.clone()


class StandardAdapter(nn.Module):
    def __init__(self, std_model):
        super().__init__()
        self.std_model = std_model

    def forward(self, x):
        # Standard model returns (output, weights) tuple, we only need output [0]
        return self.std_model(x, x, x)[0]


def measure_performance(model, x, device):
    # Warm-up
    try:
        with torch.no_grad():
            _ = model(x)
    except RuntimeError:
        return None, None, None

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_t = time.time()

    try:
        with torch.no_grad():
            output = model(x)
    except RuntimeError:
        return None, None, None

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start_t

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    return elapsed, peak_mem, output


# --- 3. ANA KARŞILAŞTIRMA TESTİ ---
