r"""
Torch modules that represent drift functions for predicitng dx.

Running forward on these modules should return a predcition for dx given inputs x and sampled graphs G \sim Q(G | D)
"""
import torch
import torch.nn.functional
from torch import nn

from .base import Intervenable
from .graph_samplers import GraphLayer, GraphLayerSVGD, GraphLayerVI
from .hyper_nets import AnalyiticLinearLocallyConnected, HyperLocallyConnected
from .kernel import RBF


class BayesianDrift(Intervenable):
    """Define a Bayesian Drift function."""

    def __init__(
        self,
        dims,
        n_ens=25,
        k_hidden=1,
        alpha=0.1,
        gamma=0,
        w_init_std=1e-2,
        deepens=None,
        hyper=None,
        bias=True,
        time_invariant=True,
    ):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.n_ens = n_ens
        self.time_invariant = time_invariant
        self.alpha = alpha
        self.gamma = gamma
        self.deepens = deepens
        self.current_epoch = 0
        if not time_invariant:
            dims[0] += 1

        if ~self.deepens and self.gamma == 0:
            self.graphs = GraphLayerVI(n_ens, dims[0], k_hidden, alpha)
        elif ~self.deepens and self.gamma != 0:
            self.graphs = GraphLayerSVGD(n_ens, dims[0], k_hidden, alpha, gamma, w_init_std)
        else:
            self.graphs = GraphLayer(n_ens, dims[0], k_hidden, alpha)

        if hyper != "linear":
            layers = []
            for i in range(len(dims) - 1):
                layers.append(
                    HyperLocallyConnected(
                        dims[0],  # num_linear
                        dims[i],  # input_features
                        dims[i + 1],  # output_features
                        n_ens=n_ens,
                        hyper=hyper,
                        bias=bias,
                    )
                )
            self.fc2 = nn.ModuleList(layers)

    def phi(self):
        """Implements an SVGD penalty on the ensembled set of graph parameter tuples.

        Updates the grad function with an RBF kernel across parameters.
        """
        kernel = RBF(sigma=self.gamma)
        Z = self.graphs.Z()
        Z_vec = Z.reshape(Z.shape[0], -1)
        kz = kernel(Z_vec, Z_vec.detach())
        grad_Ks = torch.autograd.grad(kz.sum(), self.graphs.parameters())
        for i, param in enumerate(self.graphs.parameters()):
            weighted_negative_score = torch.mean(
                kz.detach().reshape((*kz.shape, *([1] * (Z.dim() - 1)))) * param.grad,
                dim=0,
            )
            negative_phi = weighted_negative_score + grad_Ks[i]
            param.grad = negative_phi

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        G = self.graphs()
        Gt = G.transpose(-2, -1).unsqueeze(1)
        x = Gt * x
        x = x.unsqueeze(dim=3)  # [n_ens, batch, d, t, d]
        for fc in self.fc2:
            x = fc(x, G)  # [n_ens, batch, d, t, mi]
        x = x.transpose(-3, -1).squeeze(-2)
        return x  # x.shape [n_ens, batch, t, d]

    def l2_reg(self):
        """L2 regularization on input layer parameters."""
        return torch.sum(self.graphs() ** 2)

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        G = self.graphs()
        return torch.sum(torch.abs(G)) / G.shape[-1]

    def svgd_reg(self):
        """SVGD sparse prior for scale-free graphs."""
        l1 = torch.sum(torch.abs(self.graphs().transpose(-2, -1)), dim=-1)
        prod = torch.prod((torch.ones_like(l1) + l1) ** (-3), dim=-1)
        return torch.sum(prod)

    def kl_reg(self):
        """KL divergence regularization on input layer parameters for Baysian SVI."""
        return self.graphs._get_kl()

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        weights = torch.sum(self.graphs() ** 2, dim=0).pow(gamma).data  # [i, j]
        return weights

    def get_structure(self, eval_n_graphs=None, test_mode=None):
        """Score each edge based on the the weight sum."""
        return self.graphs(eval_n_graphs, test_mode)


class LinearBayesianDrift(BayesianDrift):
    def __init__(
        self,
        dims,
        n_ens=25,
        k_hidden=1,
        alpha=0.1,
        gamma=0,
        w_init_std=1e-2,
        deepens=None,
        hyper=None,
        bias=True,
        time_invariant=True,
    ):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__(
            dims=dims,
            n_ens=n_ens,
            k_hidden=k_hidden,
            alpha=alpha,
            gamma=gamma,
            w_init_std=1e-2,
            deepens=deepens,
            hyper=hyper,
            bias=bias,
            time_invariant=time_invariant,
        )
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.n_ens = n_ens
        self.time_invariant = time_invariant
        self.alpha = alpha
        self.current_epoch = 0
        if not time_invariant:
            dims[0] += 1

        self.fc2 = AnalyiticLinearLocallyConnected(
            dims[0],  # num_linear
            dims[0],  # input_features
            n_ens=n_ens,
            hyper=hyper,
            bias=bias,
        )

    def phi(self):
        """Implements an SVGD penalty on the ensembled set of graph parameter tuples.

        Updates the grad function with an RBF kernel across parameters.
        """
        kernel = RBF(sigma=self.gamma)
        Z = self.graphs.Z()
        Z_vec = Z.reshape(Z.shape[0], -1)
        kz = kernel(Z_vec, Z_vec.detach())
        grad_Ks = torch.autograd.grad(kz.sum(), self.parameters())
        for i, param in enumerate(self.parameters()):
            weighted_negative_score = torch.mean(
                kz.detach().reshape((*kz.shape, *([1] * (Z.dim() - 1)))) * param.grad,
                dim=0,
            )
            negative_phi = weighted_negative_score + grad_Ks[i]
            param.grad = negative_phi

    def forward(self, t, x, dx):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        G = self.graphs()
        x = x.squeeze(1).unsqueeze(0).unsqueeze(-1)  # [1, batch, d, 1]
        x = self.fc2(x, dx, G)  # [n_ens, batch, d, t, mi]
        x = x.squeeze(-3).squeeze(-2).squeeze(-1)
        return x  # x.shape [n_ens, batch, t, d]
