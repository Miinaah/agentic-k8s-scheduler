"""
Graph Neural Network Encoder for Cluster State.

Encodes cluster topology and per-node resource metrics into 64-dimensional
node embeddings equipped with global/neighborhood cluster context.
Supports SAGEConv and GATConv, variable layer counts (2-4), residual connections,
and PyG Batch collation for efficient mini-batch training.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, SAGEConv


class ClusterGNNEncoder(nn.Module):
    """GNN Encoder mapping (N, 12) cluster states to (N, 64) contextual node embeddings."""

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 64,
        gnn_layers: int = 2,
        gnn_type: str = "sage",  # "sage" or "gat"
        gat_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = False,
        use_gnn: bool = True,
    ):
        """Initialize ClusterGNNEncoder.

        Args:
            node_feat_dim: Dimension of raw node features (default 12).
            hidden_dim: Embedding dimension (default 64).
            gnn_layers: Number of message passing layers (range 2-4, default 2).
            gnn_type: Type of GNN conv layer ('sage' or 'gat').
            gat_heads: Attention heads for GATConv (default 4, averaged back to hidden_dim).
            dropout: Dropout probability (default 0.1).
            residual: Whether to use residual skip connections (especially for layers > 2).
            use_gnn: Flag for ablation studies; if False, skips message passing.
        """
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.gnn_layers = max(2, min(4, gnn_layers))
        self.gnn_type = gnn_type.lower()
        self.gat_heads = gat_heads
        self.dropout_p = dropout
        self.residual = residual
        self.use_gnn = use_gnn

        # Input projection: Linear(12 -> 64) -> ReLU
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if self.use_gnn:
            for _ in range(self.gnn_layers):
                if self.gnn_type == "gat":
                    conv = GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=gat_heads,
                        concat=False,  # Averaged back to hidden_dim
                        dropout=dropout,
                    )
                else:  # Default SAGEConv
                    conv = SAGEConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                    )
                self.convs.append(conv)
                self.dropouts.append(nn.Dropout(p=dropout))

    def forward(
        self,
        data_or_x: Union[Data, Batch, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Accepts either:
        1. A PyG Data or Batch object containing .x and .edge_index
        2. Tensors x of shape (N, 12) and edge_index of shape (2, E)

        Returns:
            torch.Tensor of shape (N, 64) containing node embeddings.
        """
        if isinstance(data_or_x, (Data, Batch)) or hasattr(data_or_x, "x"):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
        else:
            x = data_or_x
            if edge_index is None and self.use_gnn:
                raise ValueError("edge_index must be provided when use_gnn is True")

        # 1. Project input features: (N, 12) -> (N, 64)
        h = self.input_proj(x)

        # If ablation mode (no GNN), return projected features without message passing
        if not self.use_gnn:
            return h

        # 2. Message-passing layers
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):
            h_in = h
            h = conv(h, edge_index)
            h = F.relu(h)
            h = dropout(h)
            # Residual connection for layers > 2 or when configured
            if self.residual and i > 0:
                h = h + h_in

        return h
