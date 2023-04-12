r"""velocity_module.py.

Here we define the following modules:

    *  VelocityLitModule: Base Velocity Module, implements data loading,
    evaluation calculations, and general loss function.
    Has the following subclasses:

    1. LinearLitModule: Uses linear SCM to parameterize dx = f(x).
        SCM parameters are solved using anlytic linear solver
    2. HyperLitModule: Uses hyper-network to learn parameters for
        dx = f(x). SCM parameters are parameterized more expressively.
"""

import itertools
from typing import Any, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torchmetrics import MeanSquaredError

from .components.bayesian_drift import BayesianDrift, LinearBayesianDrift
from .components.evaluation import (
    compare_graphs,
    compare_graphs_bayesian_cover,
    compare_graphs_bayesian_shd,
    compute_graphs_bayesian_diversity,
    compute_graphs_sparsity,
    kl_distance_true_sigmoid,
)
from .node_module import NODELitModule


class VelocityLitModule(NODELitModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.01,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        kl_reg: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        gamma: float = 1.0,
        deepens: bool = False,
        svgd: bool = False,
        svgd_gamma: float = 0.0,
        **kwargs,
    ):
        """Initializes a Bayesian Velocity Module.

        Args:
            net: defines bayesian_drift network model that parameterize graphs
            lr: learning rate
            l1_reg: controls degree of sparsity of graphs
            l2_reg: controls degree of L2 regularization
            kl_reg: controls degree of KL term in variational inference
            temperature: scaling for energy likelihood (1/temperature**2)
            weight_decay: option to add weight_decay through optimizer
            optimizer: selects optimizer
            gamma: parameter for learning rate schedule
            deepens (bool): if true use Deep Ensemble parameterization
                and learning of graphs
            svgd (bool): if true use DiBS parameterization
                and learning of graphs
            svgd_gamma: controls particle separation in SVGD for DiBS
                method
        """
        super(NODELitModule, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.save_hyperparameters(logger=False)
        self.net = net
        self.criterion = torch.nn.MSELoss()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor):
        # Set t = None
        return self.net(None, x)

    def step(self, batch: Any, train_mode: bool = False):
        x, y, gc = batch
        pred = self.forward(x)[:, 0, :]
        loss = self.criterion(pred, y)
        return loss, pred, y

    def compute_grn_modes(self, graph):
        """Computes ground truth admissible graphs for RNA-velocity dataset."""
        n = graph.shape[0]
        mask_list = []
        for var in [1]:
            line = graph[var]
            mask = line.clone().to(bool)
            m = mask.sum()
            mask_list.append(mask)
            sub_graphs = torch.zeros((3**m, 3, n))
            for i, line in enumerate(list(itertools.product(range(3), repeat=4))):
                for j, index in zip(line, np.arange(n)[mask]):
                    sub_graphs[i, j, index] = 1
        masks = torch.stack(mask_list)
        prod = sub_graphs
        certain_graph0 = graph[:1].repeat(prod.shape[0], 1, 1)
        certain_graph_mid = graph[2:-2].repeat(prod.shape[0], 1, 1)
        floppier_graph = torch.cat(
            [certain_graph0, prod[:, :1], certain_graph_mid, prod[:, 1:]], dim=1
        )
        return floppier_graph

    def compute_true_modes(self, graph):
        """Computes ground truth admissible graphs for syntehtic dataset."""
        graph = torch.tensor(graph)
        n = graph.shape[0]
        graph_list = []
        mask_list = []
        for var in [0, 1, 2]:
            line = graph[var]
            mask = line.clone().to(bool)
            graphs = torch.zeros((2 ** mask.sum(), n))
            m = mask.sum()
            mask_list.append(mask)
            graphs[:, mask] = torch.tensor(
                list(itertools.product(range(2), repeat=m))
            ).to(torch.float32)
            graph_list.append(graphs)
        masks = torch.stack(mask_list)
        prod = torch.stack([torch.stack(s) for s in itertools.product(*graph_list)])
        other = masks.to(torch.float32).to(graphs) - prod
        certain_graph = graph[3:-3]
        certain_graph = certain_graph.repeat(prod.shape[0], 1, 1).to(graphs)
        floppier_graph = torch.cat([prod, certain_graph, other], dim=1)
        return floppier_graph

    def eval_step(self, batch: Any, batch_idx: int, prefix: str, mse_fn):
        loss, preds, targets = self.step(batch)
        if len(preds.shape) == len(targets.shape) + 1:
            targets = targets.expand(self.hparams.n_ens, *([-1] * len(targets.shape)))
        mse_fn(preds, targets)
        if self.hparams.n_ens > 200:
            graph = self.net.get_structure(eval_n_graphs=self.hparams.eval_batch_size)
        else:
            graph = self.net.get_structure()
        if isinstance(graph, Tensor):
            graph = graph.cpu().detach().numpy()

        if prefix == "test":
            if batch_idx == 0:
                gc = batch[2][0].cpu().detach().numpy()
                if gc.shape[-1] == 5:
                    true_graphs = self.compute_grn_modes(torch.tensor(gc))
                else:
                    true_graphs = self.compute_true_modes(gc)
                Z = self.net.get_structure(
                    eval_n_graphs=self.hparams.eval_batch_size, test_mode=True
                )
                #print("\b CHECK \n")
                #print(Z)
                print("\n alpha_t:", self.net.graphs.alpha_t, "\n")
                #kl_div = kl_distance_true_sigmoid(Z, true_graphs, self.hparams.alpha)
                #kl_div = kl_distance_true_sigmoid(Z, true_graphs, self.net.graphs.alpha_t)
                kl_div = kl_distance_true_sigmoid(Z, true_graphs, 1e8)
                self.log(f"{prefix}/kl_div", kl_div, on_step=False, on_epoch=True)
                self.log(
                    f"{prefix}/alpha", self.hparams.alpha, on_step=False, on_epoch=True
                )

        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        to_return = {"loss": loss, "preds": preds, "targets": targets}
        if batch_idx == 0:
            gc = batch[2][0].cpu().detach().numpy()
            if len(graph.shape) > 2:
                # Assume it is a group of graphs, take mean over single_graph metrics
                metrics = [compare_graphs(gc, g) for g in graph]
                metrics = {
                    k: np.mean([d[k] for d in metrics]) for k in metrics[0].keys()
                }
            else:
                metrics = compare_graphs(gc, graph)
            to_return.update(metrics)
            for k, v in metrics.items():
                self.log(f"{prefix}/{k}", v, on_step=False, on_epoch=True)
        return to_return

    def eval_epoch_end(self, outputs: List[Any], prefix: str, mse_fn):
        mse_fn.reset()


class LinearLitModule(VelocityLitModule):
    """LinearLitModule implements analytic linear parameter hyper network specific training and
    evaluation code."""

    def __init__(
        self,
        dm_conf,
        dims: list = [],
        bias: bool = True,
        time_invariant: bool = True,
        n_ens: int = 25,
        k_hidden: int = 1,
        alpha: float = 0.1,
        svgd_gamma: float = 0.0,
        deepens: bool = False,
        hyper: Optional[str] = None,
        **kwargs,
    ) -> None:
        r"""Initializes a HyperLitModule Module.

        Args:
            dm_conf: datamodule configuration
            dims: layer dimensions of structural equations functions
            bias (bool): if true uses bias parameters
            time_invariant (bool): if true uses time invariance
            n_ens: number of graph samples / graphs per model
            k_hidden: defines dimension of graph representation matrices
                G = simoid(w^t*v), where w,v \in R^{d x k}.
            alpha: initial scaling parameter in sigmoid function -
                G = simoid(alpha_t * w^t*v)
            svgd_gamma: controls particle separation in SVGD for DiBS
                method
            deepens (bool): if true use Deep Ensemble parameterization
                and learning of graphs
            hyper: selects hyper-network parameterization
        """
        if hyper == "None":
            hyper = None
        self.hyper = hyper

        self.save_hyperparameters(logger=False)

        print(
            "\n Analytic Linear-SCM using DynDiBS, DynBCD, DynDeepEns \n Using",
            self.hparams.n_ens,
            "Models \n",
        )
        net = LinearBayesianDrift(
            dims=[dm_conf.p, *dims],
            n_ens=n_ens,
            k_hidden=k_hidden,
            alpha=alpha,
            gamma=svgd_gamma,
            w_init_std=1e-2,
            deepens=deepens,
            hyper=hyper,
            bias=bias,
            time_invariant=time_invariant,
        )

        super().__init__(
            net=net,
            **kwargs,
        )

    def eval_step(self, batch: Any, batch_idx: int, prefix: str, mse_fn):
        G = self.net.get_structure(eval_n_graphs=self.hparams.eval_batch_size)
        G = G.cpu().detach().numpy()
        G = (G > 0.5).astype(
            float
        )  # additional threshold to not cause issues in bayes_cover calc
        gc = batch[2][0].cpu().detach().numpy()
        np.set_printoptions(precision=2, suppress=True)
        if self.hparams.hyper == "linear":
            print("A_est")
            print(self.net.fc2.weights.mean(dim=0))
        print(G.mean(axis=0))
        print(G[0])
        print(gc)
        print("Number of edges: ", np.sum(gc > 0))
        print(G.shape, gc.shape)
        if np.any(gc < 0):
            # in the uncertain setting if the ground truth graph has any values < 0
            bayes_shd, bayes_tshd = compare_graphs_bayesian_shd(gc, G)
            self.log(f"{prefix}/bayes_shd", bayes_shd, on_step=False, on_epoch=True)
            self.log(f"{prefix}/bayes_tshd", bayes_tshd, on_step=False, on_epoch=True)
            bayes_cover = compare_graphs_bayesian_cover(gc, G)
            self.log(f"{prefix}/bayes_cover", bayes_cover, on_step=False, on_epoch=True)
        bayes_diversity = compute_graphs_bayesian_diversity(G)
        sparsity = 0.0
        for m in range(G.shape[0]):
            sparsity += compute_graphs_sparsity(G[m])
        avg_sparsity = sparsity / G.shape[0]
        self.log(
            f"{prefix}/bayes_diversity",
            bayes_diversity,
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{prefix}/avg_sparsity", avg_sparsity, on_step=False, on_epoch=True)
        return super().eval_step(batch, batch_idx, prefix, mse_fn)

    def step(self, batch: Any, train_mode: bool = False):
        x, y, gc = batch
        if self.hyper == "linear":
            pred = self.net(None, x[:, None, :], y[:, None, :]).squeeze(
                dim=-2
            )  # [n_ens, n, d]
        else:
            pred = self.forward(x[:, None, :]).squeeze(dim=-2)  # [n_ens, n, d]
        loss = self.criterion(
            pred, y.expand(self.hparams.n_ens, *([-1] * len(y.shape)))
        )
        return loss, pred, y


class HyperLitModule(VelocityLitModule):
    """HyperLitModule implements parameter hyper network specific training and evaluation code."""

    def __init__(
        self,
        dm_conf,
        dims: list = [],
        bias: bool = True,
        time_invariant: bool = True,
        n_ens: int = 25,
        k_hidden: int = 1,
        alpha: float = 0.1,
        svgd_gamma: float = 0.0,
        deepens: bool = False,
        hyper: Optional[str] = None,
        **kwargs,
    ) -> None:
        r"""Initializes a HyperLitModule Module.

        Args:
            dm_conf: datamodule configuration
            dims: layer dimensions of structural equations functions
            bias (bool): if true uses bias parameters
            time_invariant (bool): if true uses time invariance
            n_ens: number of graph samples / graphs per model
            k_hidden: defines dimension of graph representation matrices
                G = simoid(w^t*v), where w,v \in R^{d x k}.
            alpha: initial scaling parameter in sigmoid function -
                G = simoid(alpha_t * w^t*v)
            svgd_gamma: controls particle separation in SVGD for DiBS
                method
            deepens (bool): if true use Deep Ensemble parameterization
                and learning of graphs
            hyper: selects hyper-network parameterization
        """
        if hyper == "None":
            hyper = None
        self.hyper = hyper

        self.save_hyperparameters(logger=False)

        print(
            "\n Hyper-network using DynDiBS, DynBCD, DynDeepEns \n Using",
            self.hparams.n_ens,
            "Models \n",
        )
        net = BayesianDrift(
            dims=[dm_conf.p, *dims],
            n_ens=n_ens,
            k_hidden=k_hidden,
            alpha=alpha,
            gamma=svgd_gamma,
            w_init_std=1e-3,
            deepens=deepens,
            hyper=hyper,
            bias=bias,
            time_invariant=time_invariant,
        )

        super().__init__(
            net=net,
            **kwargs,
        )

    def eval_step(self, batch: Any, batch_idx: int, prefix: str, mse_fn):
        G = self.net.get_structure(eval_n_graphs=self.hparams.eval_batch_size)
        G = G.cpu().detach().numpy()
        G = (G > 0.5).astype(
            float
        )  # additional threshold to not cause issues in bayes_cover calc
        gc = batch[2][0].cpu().detach().numpy()
        np.set_printoptions(precision=2, suppress=True)
        if self.hparams.hyper == "linear":
            print("A_est")
            print(self.net.fc2.weights.mean(dim=0))
        print(G.mean(axis=0))
        print(G[0])
        print(gc)
        print("Number of edges: ", np.sum(gc > 0))
        print(G.shape, gc.shape)
        if np.any(gc < 0):
            # in the uncertain setting if the ground truth graph has any values < 0
            bayes_shd, bayes_tshd = compare_graphs_bayesian_shd(gc, G)
            self.log(f"{prefix}/bayes_shd", bayes_shd, on_step=False, on_epoch=True)
            self.log(f"{prefix}/bayes_tshd", bayes_tshd, on_step=False, on_epoch=True)
            bayes_cover = compare_graphs_bayesian_cover(gc, G)
            self.log(f"{prefix}/bayes_cover", bayes_cover, on_step=False, on_epoch=True)
        bayes_diversity = compute_graphs_bayesian_diversity(G)
        sparsity = 0.0
        for m in range(G.shape[0]):
            sparsity += compute_graphs_sparsity(G[m])
        avg_sparsity = sparsity / G.shape[0]
        self.log(
            f"{prefix}/bayes_diversity",
            bayes_diversity,
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{prefix}/avg_sparsity", avg_sparsity, on_step=False, on_epoch=True)
        return super().eval_step(batch, batch_idx, prefix, mse_fn)

    def step(self, batch: Any, train_mode: bool = False):
        x, y, gc = batch
        if self.hyper == "linear":
            pred = self.net(None, x[:, None, :], y[:, None, :]).squeeze(
                dim=-2
            )  # [n_ens, n, d]
        else:
            pred = self.forward(x[:, None, :]).squeeze(dim=-2)  # [n_ens, n, d]
        loss = self.criterion(
            pred, y.expand(self.hparams.n_ens, *([-1] * len(y.shape)))
        )
        return loss, pred, y
