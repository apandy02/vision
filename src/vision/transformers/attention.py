"""
This module contains the building blocks for a 2D attention mechanism.
Author: Aryaman Pandya
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from torch.nn import functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    d_k: int,
    is_masked: bool,
) -> torch.Tensor:
    """
    Scaled dot product attention.

    [Usage deprecated]

    Args:
        q: query tensor
        k: key tensor
        d_k: dimension of the key
        mask: whether to use a mask

    Returns:
        attention: attention tensor
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if is_masked:
        mask = torch.tril(torch.ones(scores.shape)).to(q.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    return nn.Softmax(-1)(scores)


class Attention(nn.Module):
    """
    Multihead attention.
    """
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_channels: int,
        num_groups: int = 8,
        d_k: Optional[int] = None,
        is_masked: bool = False
    ):
        """
        Args:
            d_k: dimension of the key
            dropout: dropout rate
            num_heads: number of heads
            num_channels: number of channels
            num_groups: number of groups for group normalization
            mask: whether to use a mask
        """
        super(Attention, self).__init__()
        self.d_k = d_k if d_k is not None else num_channels
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.mask = is_masked
        self.num_channels = num_channels

        self.query_projection = nn.Linear(num_channels, num_heads * self.d_k)
        self.key_projection = nn.Linear(num_channels, num_heads * self.d_k)
        self.value_projection = nn.Linear(num_channels, num_heads * self.d_k)

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=num_channels
        )
        self.output_layer = nn.Linear(num_heads * self.d_k, num_channels)
        

    def attention_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes attention values.
        """
        batch_size = x.shape[0]
        residual = x

        if y is not None:
            k, q, v = y, x, y
        else:
            k, q, v = x, x, x

        k_len, q_len, v_len = k.size(1), q.size(1), v.size(1)

        k = self.key_projection(k).view(batch_size, k_len, self.num_heads, self.d_k)
        q = self.query_projection(q).view(batch_size, q_len, self.num_heads, self.d_k)
        v = self.value_projection(v).view(batch_size, v_len, self.num_heads, self.d_k)

        attn_mask = None
        if self.mask:
            attn_mask = torch.tril(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool))
        
        output = F.scaled_dot_product_attention(
            query=q,
            key=k, 
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.mask and attn_mask is None
        )
        output = self.output_layer(output.transpose(1, 2).contiguous().view(batch_size, q_len, -1))

        return self.dropout(output) + residual

    def forward(self, x, y=None):
        """
        forward pass for the attention mechanism.

        Args:
            x: input tensor
            y: optional tensor for cross-attention
        """
        return self.attention_values(x, y)


class Attention2D(Attention):
    """
    Multihead attention.
    """
    def forward(self, x, y=None):
        """
        forward pass for the attention mechanism.

        Args:
            x: input tensor
            y: optional tensor for cross-attention
        """
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, height * width).permute(0, 2, 1)
        x = self.group_norm(x)

        if y is not None:
            return self.attention_values(x, y).view(batch_size, n_channels, height, width)

        return self.attention_values(x, None).permute(0, 2, 1).view(batch_size, n_channels, height, width)
