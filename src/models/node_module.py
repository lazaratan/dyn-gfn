r"""node_module.py.

Here we define the following modules:

    *  NODELitModule: Base NODE Module, implements data loading,
    evaluation calculations, and general loss function.
    This module is more general than velocity_module.py and allows
    for time-series extension for NeuralODE solver.
"""
from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchdyn.core import NeuralODE
from torchmetrics import MeanSquaredError

from .components.evaluation import compare_graphs


class NODELitModule(LightningModule):
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
        solver: str = "tsit5",
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
            solver: if using NeuralODE parameterization (instead of velocity
                D ={x, dx}), this defines which solver to use.
                NOTE: not used in this version.
        """
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.save_hyperparameters(logger=False)
        self.net = net
        self.node = NeuralODE(self.net, solver=solver)
        self.criterion = torch.nn.MSELoss()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_mse doesn't store accuracy from these checks
        self.val_mse.reset()

    def random_time_subset(self, batch: Any, horizon=5) -> Any:
        """Takes a random subset of the batch of length horizon."""
        obs, ts, gc = batch
        n, t, d = obs.shape
        start_idx = np.random.randint(t - horizon)
        obs = obs[:, start_idx : start_idx + horizon]
        ts = ts[:, start_idx : start_idx + horizon]
        return (obs, ts, gc)

    def step(self, batch: Any, train_mode: bool = False):
        if train_mode:
            batch = self.random_time_subset(batch)
        obs, ts, _ = batch
        obs0 = obs[:, :1, :]
        # Warning: Assumes all samples in a batch have the same times
        _, preds = self.node(obs0, ts[0])
        # Preds returns shape [Times, samples, 1, dim] change to [S, T, D] format
        preds = preds[:, :, 0, :]
        preds = torch.transpose(preds, 0, 1)
        loss = self.criterion(preds, obs.detach())
        return loss, preds, obs

    def compute_loss(self, batch: Any, batch_idx: int):
        mse, preds, targets = self.step(batch, train_mode=True)
        reg = torch.tensor([0.0], device=self.device)
        if self.hparams.l2_reg != 0:
            reg += self.hparams.l2_reg * self.net.l2_reg()
        if self.hparams.l1_reg != 0:
            reg += self.hparams.l1_reg * self.net.l1_reg()
        if self.hparams.kl_reg != 0:
            reg += self.hparams.kl_reg * self.net.kl_reg()
        if hasattr(self.hparams, "de_reg") and self.hparams.de_reg != 0:
            reg += (
                self.hparams.de_reg
                * self.net.DeepEns_prior(self.net, self.prior_var)
                / len(batch[0].detach())
            )
        loss = mse / (self.hparams.temperature**2) + reg
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/reg_total", reg, on_step=False, on_epoch=True, prog_bar=True)
        return loss, mse, reg

    def training_step(self, batch: Any, batch_idx: int):
        opt = self.optimizers()
        opt.zero_grad()
        loss, mse, reg = self.compute_loss(batch, batch_idx)
        # loss interpreted as negative log probs
        self.manual_backward(loss)
        if self.hparams.svgd:
            # Updates .grads with svgd
            self.net.phi()
        opt.step()
        return {"loss": loss, "mse": mse.detach(), "reg_total": reg.detach()}

    def eval_step(self, batch: Any, batch_idx: int, prefix: str, mse_fn):
        loss, preds, targets = self.step(batch)
        mse_fn(preds, targets)
        if self.hparams.n_ens > 200:
            graph = self.net.get_structure(eval_n_graphs=self.hparams.eval_batch_size)
        else:
            graph = self.net.get_structure()
        if isinstance(graph, Tensor):
            graph = graph.cpu().detach().numpy()
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

    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val", self.val_mse)

    def eval_epoch_end(self, outputs: List[Any], prefix: str, mse_fn):
        mse = mse_fn.compute()
        v = {k: torch.cat([d[k] for d in outputs]) for k in ["preds", "targets"]}
        preds, targets = v["preds"], v["targets"]
        z_p, mse_np = preds.cpu().detach().numpy(), mse.item()
        graph = self.net.get_structure()[0]
        if isinstance(graph, Tensor):
            graph = graph.cpu().detach().numpy()
        mse_fn.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "val", self.val_mse)

    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test", self.test_mse)

    def test_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "test", self.test_mse)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.hparams.optimizer == "sgld":
            from .components.SGLD import SGLD

            optimizer = SGLD
        elif self.hparams.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop
        else:
            raise ValueError(f"Unrecognized optimizer {self.hparams.optimizer}")
        opt = optimizer(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.gamma == 1.0:
            # If gamma is one, then don't bother to initialize the scheduler
            return opt
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=self.hparams.gamma
            ),
            "name": "exponential_scheduler",
        }
        return [opt], [lr_scheduler]
