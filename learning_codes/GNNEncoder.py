# GNNEncoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv1(x, edge_index))