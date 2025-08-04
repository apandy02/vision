import torch
import torch.nn as nn

from .attention import Attention

class MLP(nn.Module): 
    """
    Class implementation of a position wise MLP
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float, num_layers: int = 2) -> None:
        super(MLP, self).__init__()
        
        layers = [nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout)]
        
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(d_ff, d_ff, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(d_ff, d_model, bias=True))
        self.mlp_layers = nn.Sequential(*layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(self.layer_norm(x))


class EncoderLayer(nn.Module):
    """
    Encoder layer block for ViT
    """
    def __init__(
        self, 
        num_heads: int,
        num_channels: int,
        d_linear: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        is_masked: bool = False
    ):
        super(EncoderLayer, self).__init__()
        self.norm1, self.norm2 = nn.LayerNorm(num_channels),  nn.LayerNorm(num_channels)
        self.mha = Attention(dropout, num_heads, num_channels, num_groups)
        self.mlp = MLP(num_channels, d_linear, dropout, num_linear_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mha(self.norm1(x)) + x
        return self.mlp(self.norm2(h)) + h


class Encoder(nn.Module):
    def __init__(
        self, 
        num_heads: int,
        num_channels: int,
        num_layers: int,
        d_linear: int,
        num_linear_layers: int = 2,
        num_groups: int = 8,
        dropout: float = 0.1,
        is_masked: bool = False
    ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                num_heads, num_channels, d_linear, num_linear_layers, num_groups, dropout, is_masked
            ) for _ in range(num_layers)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x 