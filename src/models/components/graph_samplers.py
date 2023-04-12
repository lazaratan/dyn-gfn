r"""
Torch modules that represent distributions over graphs.

Running forward on these modules should return a set of graphs. Represented as a dense float32 tensor of shape G=[n_graphs, n_var, n_var] where G_ij ~= 1 iff v_i \in Pa(v_j).
"""
import math

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class GraphLayer(Module):
    def __init__(
        self,
        n_graphs: int,
        n_var: int,
        n_embed: int,
        alpha: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_graphs = n_graphs
        self.n_var = n_var
        self.n_embed = n_embed
        self.alpha = alpha
        self.t = 1
        # define network weights
        self.w = Parameter(
            torch.empty((self.n_graphs, self.n_var, self.n_embed), **factory_kwargs)
        )
        self.v = Parameter(
            torch.empty((self.n_graphs, self.n_var, self.n_embed), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def forward(self, eval_n_graphs=None):
        Z = torch.matmul(self.w, self.v.transpose(-2, -1))
        self.alpha_t = self.t * self.alpha
        self.t += 1
        G = torch.sigmoid(self.alpha_t * Z)
        return G


class GraphLayerVI(Module):
    def __init__(
        self,
        n_graphs: int,
        n_var: int,
        n_embed: int,
        alpha: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_graphs = n_graphs
        self.n_var = n_var
        self.n_embed = n_embed
        self.alpha = alpha
        self.t = 1
        # define network weights
        self.w_mu = Parameter(torch.empty((self.n_var, self.n_embed), **factory_kwargs))
        self.w_isp_std = Parameter(
            torch.empty((self.n_var, self.n_embed), **factory_kwargs)
        )
        self.v_mu = Parameter(torch.empty((self.n_var, self.n_embed), **factory_kwargs))
        self.v_isp_std = Parameter(
            torch.empty((self.n_var, self.n_embed), **factory_kwargs)
        )
        self.eps = 1e-6
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.w_mu)
        torch.nn.init.zeros_(self.v_mu)
        self.w_isp_std.data.fill_(-4)
        self.v_isp_std.data.fill_(-4)

    def sample(self, n_graphs=500):
        w_sigma, v_sigma = (
            F.softplus(self.w_isp_std) + self.eps,
            F.softplus(self.v_isp_std) + self.eps,
        )
        w = (
            self.w_mu
            + torch.randn(size=(n_graphs, self.n_var, self.n_embed)).to(self.w_mu)
            * w_sigma
        )
        v = (
            self.v_mu
            + torch.randn(size=(n_graphs, self.n_var, self.n_embed)).to(self.v_mu)
            * v_sigma
        )
        return w, v

    def _get_kl(self, prior_log_sigma=-2):
        kl = torch.sum(
            prior_log_sigma
            - self.w_isp_std
            + 0.5 * (F.softplus(self.w_isp_std) ** 2) / (math.exp(prior_log_sigma * 2))
        )
        kl = torch.sum(
            prior_log_sigma
            - self.w_isp_std
            + 0.5 * (F.softplus(self.w_isp_std) ** 2) / (math.exp(prior_log_sigma * 2))
        )
        kl += 0.5 * torch.sum(self.w_mu**2) / math.exp(prior_log_sigma * 2)
        kl += 0.5 * torch.sum(self.v_mu**2) / math.exp(prior_log_sigma * 2)
        return kl

    def forward(self, eval_n_graphs=None, test_mode=None):
        if eval_n_graphs is None:
            self.w, self.v = self.sample(self.n_graphs)
        else:
            self.w, self.v = self.sample(eval_n_graphs)
        Z = torch.matmul(self.w, self.v.transpose(-2, -1))
        if test_mode:
            return Z
        # self.alpha_t = self.t * self.alpha
        self.alpha_t = self.t ** (0.5) * self.alpha
        self.t += 1
        G = torch.sigmoid(self.alpha_t * Z)
        return G


class GraphLayerSVGD(Module):
    def __init__(
        self,
        n_graphs: int,
        n_var: int,
        n_embed: int,
        alpha: float = 1.0,
        gamma: float = 1.0,
        w_init_std: float = 1e-2,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_graphs = n_graphs
        self.n_var = n_var
        self.n_embed = n_embed
        self.alpha = alpha
        self.gamma = gamma
        self.w_init_std = w_init_std
        self.t = 1
        self.alpha_t = 1

        # define network weights
        self.w = Parameter(
            torch.empty((self.n_graphs, self.n_var, self.n_embed), **factory_kwargs)
        )
        self.v = Parameter(
            torch.empty((self.n_graphs, self.n_var, self.n_embed), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.w, mean=0, std=self.w_init_std
        )  # std=1e-2 lin-sys lin-sol, lin-sys 1e-3 hyper-sol
        torch.nn.init.normal_(self.v, mean=0, std=self.w_init_std)  # std=1e-2 for linear sovler

    def Z(self, eval_n_graphs=None):
        if eval_n_graphs is None:
            self.alpha_t = self.t ** (0.5) * self.alpha
            self.t += 1
        return torch.matmul(self.w, self.v.transpose(-2, -1))

    def forward(self, eval_n_graphs=None, test_mode=None):
        Z = self.Z(eval_n_graphs)
        if test_mode:
            return Z
        if eval_n_graphs is None:
            return Z
        G = torch.sigmoid(self.alpha_t * Z)
        return G
