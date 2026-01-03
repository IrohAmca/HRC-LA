import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hrc_la import HRCMultiheadAttention
from hrc_la.utils import StandardAdapter, sync_weights


def test_output_shape():
    """Test if the output shape matches the input shape for various sequence lengths."""
    D_MODEL = 64
    NUM_HEADS = 4
    SEQ_LENGTHS = [512, 1024, 2048]
    
    model = HRCMultiheadAttention(D_MODEL, NUM_HEADS)
    
    for N in SEQ_LENGTHS:
        x = torch.randn(1, N, D_MODEL)
        output = model(x)
        assert output.shape == (1, N, D_MODEL)

def test_batch_first_false():
    """Test if the model handles batch_first=False correctly."""
    batch_size = 2
    seq_len = 16
    d_model = 64
    num_heads = 4
    
    model = HRCMultiheadAttention(d_model, num_heads, batch_first=False)
    x = torch.randn(seq_len, batch_size, d_model)
    output = model(x)
    
    assert output.shape == (seq_len, batch_size, d_model)

def test_gradient_flow():
    """Test if gradients can flow through the model."""
    d_model = 32
    num_heads = 4
    model = HRCMultiheadAttention(d_model, num_heads)
    x = torch.randn(1, 8, d_model, requires_grad=True)
    
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient not found for {name}"

def test_divisibility_error():
    """Test if the model raises an error when d_model is not divisible by num_heads."""
    with pytest.raises(AssertionError):
        HRCMultiheadAttention(d_model=64, num_heads=5)

@pytest.mark.parametrize("learnable", [False, True])
def test_omega_type(learnable):
    """Test if omega is registered correctly as buffer or parameter."""
    model = HRCMultiheadAttention(d_model=64, num_heads=4, learnable_omega=learnable)
    
    if learnable:
        assert isinstance(model.omega, nn.Parameter)
        assert model.omega.requires_grad
    else:
        assert isinstance(model.omega, torch.Tensor)
        assert not isinstance(model.omega, nn.Parameter)
        # Buffers are not in model.parameters()
        assert "omega" in dict(model.named_buffers())

@pytest.mark.parametrize("N", [512, 1024, 2048])
def test_approximation_accuracy(N):
    """Test if the approximation error is within a reasonable range compared to standard MHA."""
    D_MODEL = 64
    NUM_HEADS = 4
    M_FEATURES = 128
    
    std_model_raw = nn.MultiheadAttention(D_MODEL, NUM_HEADS, batch_first=True)
    hrc_model = HRCMultiheadAttention(D_MODEL, NUM_HEADS, m_features=M_FEATURES, batch_first=True, learnable_omega=False)
    
    sync_weights(std_model_raw, hrc_model)
    std_model = StandardAdapter(std_model_raw)
    
    x = torch.randn(1, N, D_MODEL)
    
    with torch.no_grad():
        out_std = std_model(x)
        out_hrc = hrc_model(x)
    
    mse = torch.mean((out_std - out_hrc)**2).item()
    # Threshold is slightly higher for random features but should be low
    assert mse < 0.05

if __name__ == "__main__":
    pytest.main([__file__])
