"""Small signed weighted GCN for graph-level ASD classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SignedGraphConvolution(nn.Module):
    """Aggregate positive and negative graph neighbors with separate weights."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.positive_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.negative_projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(
        self,
        node_features: torch.Tensor,
        positive_adjacency: torch.Tensor,
        negative_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        positive_messages = torch.bmm(positive_adjacency, node_features)
        negative_messages = torch.bmm(negative_adjacency, node_features)
        return (
            self.self_projection(node_features)
            + self.positive_projection(positive_messages)
            + self.negative_projection(negative_messages)
        )


class ConnectomeGCN(nn.Module):
    """Two graph-convolution layers followed by graph pooling and a classifier."""

    def __init__(
        self,
        node_feature_dim: int = 5,
        covariate_dim: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.graph_conv1 = SignedGraphConvolution(node_feature_dim, hidden_dim)
        self.graph_conv2 = SignedGraphConvolution(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        positive_adjacency: torch.Tensor,
        negative_adjacency: torch.Tensor,
        covariates: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.graph_conv1(
            node_features, positive_adjacency, negative_adjacency
        )
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.graph_conv2(
            hidden, positive_adjacency, negative_adjacency
        )
        hidden = F.relu(hidden)

        pooled_mean = hidden.mean(dim=1)
        pooled_max = hidden.max(dim=1).values
        graph_features = torch.cat(
            [pooled_mean, pooled_max, covariates], dim=1
        )
        return self.classifier(graph_features)
