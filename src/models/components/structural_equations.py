import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .hyper_nets import HyperLocallyConnected


class LinearStructuralEquationModel(nn.Module):
    """Creates a linear structural equation. (x, G) --> dx Given a graph and input computes a
    masked linear layer (masked by G)

    so that each output dx is computed as ((G @ W)^t @ x) + b
    """

    def __init__(self, n_dim: int, bias=True):
        super().__init__()
        self.n_dim = n_dim
        self.weight = nn.Parameter(torch.Tensor(n_dim, n_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_dim))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.n_dim
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, G, x):
        # G: [n_graphs, n_dim, n_dim]
        # x: [batch, t, n_dim]
        # Note that this means self.weight contains transposed weights
        gw = (G.to(self.weight) * self.weight).transpose(-2, -1).unsqueeze(-3)
        # gw [n_graphs, 1, n_dim, n_dim]
        out = torch.matmul(x, gw)

        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out


class HyperStructuralEquationModel(nn.Module):
    """Hyper Structural equation which builds a separate MLP for each node whose weights depend on
    an input graph."""

    def __init__(self, dims: List[int], hyper=None, hyper_args=None):
        super().__init__()
        if hyper_args is None:
            hyper_args = {"bias": True, "n_ens": 1, "hyper_hidden_dims": None}
        self.hyper = hyper

        layers = []
        if hyper == "per_node_mlp":
            for i in range(len(dims) - 1):
                layers.append(
                    HyperLocallyConnected(
                        dims[0],  # num_linear
                        dims[i],  # input_features
                        dims[i + 1],  # output_features
                        hyper=hyper,
                        **hyper_args,
                    )
                )
        else:
            for i in range(len(dims) - 1):
                layers.append(
                    HyperLocallyConnected(
                        dims[0],  # num_linear
                        dims[i],  # input_features
                        dims[i + 1],  # output_features
                        hyper=hyper,
                        **hyper_args,
                    )
                )
        self.layers = nn.ModuleList(layers)

    def forward(self, G, x, per_node=False):  # ([d, d] x [n, 1, d]) -> [n, 1, d]
        if not per_node:
            Gt = G.transpose(-2, -1).unsqueeze(1)
            x = Gt.to(x) * x
        x = x.unsqueeze(dim=3)  # [n_ens, batch, d, t, d]
        # print(x.shape)
        # print(asdas)
        # print(x.shape)
        x = self.layers[0](x, G)
        for layer in self.layers[1:]:
            # x = layer(F.elu(x), G)  # [n_ens, batch, d, t, mi]
            x = layer(x, G)  # [n_ens, batch, d, t, mi]
        # print(x.shape, G.shape)
        x = x.transpose(-3, -1).squeeze(-2)
        return x  # x.shape [n_ens, batch, t, d]
