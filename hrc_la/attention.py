import torch
import torch.nn as nn


class HRCMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, m_features=256, batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.m = m_features
        self.head_dim = d_model // num_heads
        self.batch_first = batch_first
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
        
        omega = torch.randn(num_heads, m_features, self.head_dim)
        self.register_buffer("omega", omega)
        
    def _reset_parameters(self):
        # Xavier Init (Scaled initialization to prevent variance explosion)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def complex_feature_map(self, u, theta):
        """
        u: (Batch, Heads, Seq, Dim)
        theta: (Batch, Heads, Seq, M)
        """
        norm_sq = torch.sum(u ** 2, dim=-1, keepdim=True) # (B, H, N, 1)
        
        # Kernel Scaling
        scale = torch.exp(norm_sq / 2.0) / (self.m ** 0.5)
        
        # Euler's Formula
        z = torch.complex(torch.cos(theta), torch.sin(theta))
        
        return scale * z # (B, H, N, M)

    def forward(self, x):

        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        
        :return: Description
        Implements the HRC-LA attention mechanism as described in the paper:
        HRC-LA(X) = Re(φ_ℂ(Q) · (φ_ℂ(K)ᴴ V)) / Re(φ_ℂ(Q) · (φ_ℂ(K)ᴴ 1_N))
        
        where φ_ℂ(·) is the complex feature map.
        1_N is a vector of ones with length N (sequence length).
        ᴴ denotes the Hermitian transpose (conjugate transpose).
        The method processes the input tensor `x`, computes the query, key, and value
        projections, applies the complex feature map, and performs linear attention
        using the HRC-LA approach.
        """

        if not self.batch_first:
            x = x.transpose(0, 1)
            
        B, N, C = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scale_factor = self.head_dim ** 0.25
        q = q / scale_factor
        k = k / scale_factor
        
        # Phase Calculation (Theta)
        theta_q = torch.matmul(q, self.omega.transpose(-1, -2))
        theta_k = torch.matmul(k, self.omega.transpose(-1, -2))
        
        # Complex Feature Map
        phi_q = self.complex_feature_map(q, theta_q)
        phi_k = self.complex_feature_map(k, theta_k)
        
        # Linear Aggregation
        v_complex = v.to(torch.complex64)
        
        # S = K^H * V
        kv = torch.matmul(phi_k.mH, v_complex)
        
        # Numerator
        numerator = torch.matmul(phi_q, kv).real
        
        # Denominator
        k_sum = phi_k.sum(dim=-2) 
        denominator = torch.matmul(phi_q, k_sum.unsqueeze(-1)).real
        
        # Division with epsilon to prevent division by zero
        epsilon = 1e-6
        output = numerator / (denominator + epsilon)
        
        # Concatenate Heads
        output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)
        
        # Final Projection
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output
