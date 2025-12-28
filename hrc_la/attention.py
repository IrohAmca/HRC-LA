import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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

        omega = self._create_orthogonal_matrix(self.num_heads, self.m, self.head_dim)
        self.register_buffer("omega", omega)

    def _create_orthogonal_matrix(self, num_heads, m_features, head_dim):
        """
        Creates an orthogonal matrix for the HRC-LA attention mechanism.
        
        Args:
            num_heads: Number of attention heads.
            m_features: Number of features for the complex feature map.
            head_dim: Dimension of each head.
        
        Returns:
            omega: Orthogonal matrix for the HRC-LA attention mechanism.
        """
        if m_features % head_dim != 0:
            logger.warning(
                "m_features is not a multiple of head_dim. To most effectively use orthogonal features, m_features ceiling should be a multiple of head_dim."
            )

        omega_stack_count = math.ceil(m_features / head_dim)

        omegas = torch.zeros(num_heads, m_features, head_dim)
        for i in range(num_heads):
            w = torch.zeros(m_features, head_dim)
            for j in range(omega_stack_count):
                if j * head_dim >= m_features:
                    break
                q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
                w[j * head_dim : min((j + 1) * head_dim, m_features), :] = q[: min(head_dim, m_features - j * head_dim), :]
            scale = head_dim**0.5
            omegas[i] = w * scale
        return omegas

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
        norm_sq = torch.sum(u**2, dim=-1, keepdim=True)  # (B, H, N, 1)

        # Kernel Scaling
        scale = torch.exp(norm_sq / 2.0) / (self.m**0.5)

        # Euler's Formula
        z = torch.complex(torch.cos(theta), torch.sin(theta))

        return scale * z  # (B, H, N, M)

    def forward(self, x):
        """
        Performs the forward pass of the Hybrid Real-Complex Linear Attention (HRC-LA).
        
        This method implements the HRC-LA mechanism with Orthogonal Random Features.
        It projects the input into Query, Key, and Value spaces, applies a complex feature map
        using orthogonal random matrices to approximate the softmax kernel, and computes
        linear attention.

        Theory:
            HRC-LA(X) = Re(φ_ℂ(Q) · (φ_ℂ(K)ᴴ V)) / Re(φ_ℂ(Q) · (φ_ℂ(K)ᴴ 1_N))
            
            where φ_ℂ(·) is the complex feature map defined as:
            φ_ℂ(u) = (exp(‖u‖² / 2) / √m) * exp(i * u * Ωᵀ)
            
            - Ω is the orthogonal random matrix.
            - 1_N is a vector of ones with length N.
            - ᴴ denotes the Hermitian transpose.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, D_Model) or (Seq_Len, Batch, D_Model)
                              depending on `batch_first`.

        Returns:
            torch.Tensor: Output tensor of the same shape as input `x`.
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

        scale_factor = self.head_dim**0.25
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
