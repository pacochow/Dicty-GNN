import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, order):
        super(GCN, self).__init__()

        self.order = order
        # Initialise layers
        self.conv1 = GCNConv(self.order, hidden_channels)
        self.fc1 = Linear(hidden_channels, 500)
        self.fc2 = Linear(500, 500)
        self.fc3 = Linear(500, 500)
        self.out = Linear(500, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        x = self.out(x)
        return x
    
