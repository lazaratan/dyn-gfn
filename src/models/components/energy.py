import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torchdyn.core import NeuralODE


class VelocityEnergy(nn.Module):
    def __init__(self, structure_model, prior_lambda, temperature):
        super().__init__()
        self.structure_model = structure_model
        self.best_mse = 1
        self.pretraining = True
        self.alpha = 0.95
        self.temperature = temperature
        self.prior_lambda = prior_lambda
        self.t = 1
        self.temp = 1
        self.mse_mean_data = None

    def likelihood(self, G, batch):
        x, dx, gc = batch

        # Reshape x, dx like they have an additional time dimension
        # [batch, time, dim]
        x = x.unsqueeze(1)
        dx = dx.unsqueeze(1)
        pred = self.structure_model(G, x)
        mses = nn.functional.mse_loss(pred, dx.expand(*pred.shape), reduction="none")
        mses = mses.mean((1, 2, 3))
        return mses / (self.temperature**2)

    def forward(
        self,
        G,
        batch,
        return_mse=False,
    ):
        likelihood = self.likelihood(G, batch)
        if return_mse:
            return (likelihood * (self.temperature**2)).mean()
        prior = G.to(likelihood).sum((1, 2)) / G.shape[-1]
        return likelihood + self.prior_lambda * prior

    def step(self, epoch=None):
        pass


class PerNodeVelocityEnergy(nn.Module):
    def __init__(self, structure_model, prior_lambda, temperature):
        super().__init__()
        self.structure_model = structure_model
        self.best_mse = 1
        self.pretraining = True
        self.alpha = 0.95
        self.temperature = temperature
        self.prior_lambda = prior_lambda
        self.t = 1
        self.temp = 1
        self.mse_mean_data = None

    def likelihood(self, G, batch, node_idx=None):
        x, dx, gc = batch

        # Reshape x, dx like they have an additional time dimension
        # [batch, time, dim]
        x = x.unsqueeze(1)
        dx = dx.unsqueeze(1)
        pred_p = []

        if G.shape[1] == 1:
            x_masked = G.unsqueeze(1) * x
            pred = self.structure_model[node_idx](
                G.transpose(-2, -1), x_masked, per_node=True
            )
            mses = nn.functional.mse_loss(
                pred,
                dx[:, :, node_idx].unsqueeze(-1).expand(*pred.shape),
                reduction="none",
            )
            mses = mses.mean((1, 2, 3))
            return mses

        Gt = G.transpose(-2, -1).to(x)
        x_masked = Gt.unsqueeze(1) * x.unsqueeze(0)  # [n_ens, n, d_out, d_in]
        for p in range(G.shape[-1]):
            pred_tmp = self.structure_model[p](
                G[:, p, :].unsqueeze(1),
                x_masked[:, :, p, :].unsqueeze(2),
                per_node=True,
            )
            pred_p.append(pred_tmp)
        pred = torch.stack(pred_p, dim=-1).to(x).squeeze(-2)

        mses = nn.functional.mse_loss(pred, dx.expand(*pred.shape), reduction="none")
        mses = mses.mean((1, 2, 3))
        return mses

    def forward(
        self,
        G,
        batch,
        return_mse=False,
        node_idx=None,
    ):
        if return_mse:
            return self.likelihood(G, batch)

        likelihood = self.likelihood(G, batch, node_idx=node_idx) / (
            self.temperature**2
        )
        prior = G.to(likelihood).sum((1, 2)) / G.shape[-1]
        return likelihood + self.prior_lambda * prior

    def step(self, epoch=None):
        pass


class SimpleAnalyticBayesVelocityEnergy(nn.Module):
    def __init__(self, n_dim, beta, prior_lambda, temperature):
        super().__init__()
        self.temperature = temperature
        self.prior_lambda = prior_lambda
        self.beta = beta
        self.n_dim = n_dim

    def likelihood(self, G, batch):
        x, dx, gc = batch
        # Reshape x, dx like they have an additional time dimension
        # [batch, time, dim]
        Gt = torch.transpose(G.to(x), -2, -1).unsqueeze(1)
        x = x.unsqueeze(1).unsqueeze(0)
        dx = dx.unsqueeze(1)
        x_masked = Gt * x
        A_est = []
        for p in range(self.n_dim):
            w_est = torch.linalg.solve(
                (torch.transpose(x_masked[:, :, p, :], -2, -1) @ x_masked[:, :, p, :])
                + self.beta * torch.eye(self.n_dim).unsqueeze(0).type_as(x_masked),
                torch.transpose(x_masked[:, :, p, :], -2, -1) @ dx[:, :, p],
            )
            A_est.append(w_est)
        A_est = torch.cat(A_est, dim=2)
        A_est = A_est.unsqueeze(1)
        pred = A_est @ torch.transpose(x, -2, -1)
        pred = torch.transpose(pred, -2, -1)
        mses = nn.functional.mse_loss(pred, dx.expand(*pred.shape), reduction="none")
        mses = mses.mean((1, 2, 3))
        return mses

    def forward(
        self,
        G,
        batch,
        return_mse=False,
        current_epoch=None,
    ):
        if current_epoch is None:
            T = self.temperature
        elif current_epoch % 2 == 0:
            T = 1.0
        else:
            T = self.temperature

        if return_mse:
            return self.likelihood(G, batch)
        else:
            likelihood = self.likelihood(G, batch) / (T**2)

            prior = G.to(likelihood).sum((1, 2)) / G.shape[-1]
            return likelihood + self.prior_lambda * prior


class PerNodeSimpleAnalyticBayesVelocityEnergy(nn.Module):
    def __init__(self, n_dim, beta, prior_lambda, temperature):
        super().__init__()
        self.temperature = temperature
        self.prior_lambda = prior_lambda
        self.beta = beta
        self.n_dim = n_dim

    def likelihood(self, G, batch, node_idx=None):
        x, dx, gc = batch
        # Reshape x, dx like they have an additional time dimension
        # [batch, time, dim]
        Gt = torch.transpose(G.to(x), -2, -1).unsqueeze(1)
        x = x.unsqueeze(1).unsqueeze(0)
        dx = dx.unsqueeze(1)
        if G.shape[1] == 1:
            x_masked = G.unsqueeze(1) * x[:, :, :, node_idx].unsqueeze(-1)
            A_est = []
            w_est = torch.linalg.solve(
                (torch.transpose(x_masked[:, :, 0, :], -2, -1) @ x_masked[:, :, 0, :])
                + self.beta * torch.eye(self.n_dim).unsqueeze(0).type_as(x_masked),
                torch.transpose(x_masked[:, :, 0, :], -2, -1) @ dx[:, 0, node_idx],
            )
            A_est.append(w_est)
            A_est = torch.stack(A_est, dim=2)
            A_est = A_est.unsqueeze(1)
            A_est = torch.transpose(A_est, -1, -2)
            pred = A_est @ torch.transpose(x, -2, -1)
            pred = torch.transpose(pred, -2, -1)
            mses = nn.functional.mse_loss(
                pred,
                dx[:, :, node_idx].unsqueeze(-1).expand(*pred.shape),
                reduction="none",
            )
            mses = mses.mean((1, 2, 3))
        else:
            x_masked = Gt * x
            A_est = []
            for p in range(self.n_dim):
                w_est = torch.linalg.solve(
                    (
                        torch.transpose(x_masked[:, :, p, :], -2, -1)
                        @ x_masked[:, :, p, :]
                    )
                    + self.beta * torch.eye(self.n_dim).unsqueeze(0).type_as(x_masked),
                    torch.transpose(x_masked[:, :, p, :], -2, -1) @ dx[:, :, p],
                )
                A_est.append(w_est)
            A_est = torch.cat(A_est, dim=2)
            A_est = A_est.unsqueeze(1)
            pred = A_est @ torch.transpose(x, -2, -1)
            pred = torch.transpose(pred, -2, -1)
            mses = nn.functional.mse_loss(
                pred, dx.expand(*pred.shape), reduction="none"
            )
            mses = mses.mean((1, 2, 3))
        return mses

    def forward(
        self,
        G,
        batch,
        return_mse=False,
        test_mode=False,
        current_epoch=None,
        node_idx=None,
    ):
        if current_epoch is None:
            T = self.temperature
        elif current_epoch % 2 == 0:
            T = 1.0
        else:
            T = self.temperature

        if test_mode:
            return self.likelihood(G, batch, node_idx=node_idx)

        if return_mse:
            return self.likelihood(G, batch)

        likelihood = self.likelihood(G, batch, node_idx=node_idx) / (T**2)
        prior = G.to(likelihood).sum((1, 2)) / G.shape[-1]
        return likelihood + self.prior_lambda * prior


def shd(a, b):
    return torch.sum(torch.abs(a - b), dim=(-2, -1))


class HammingEnergy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def likelihood(self, G, batch):
        return self.forward(G, batch)

    def base_likelihood(self, G, batch):
        return self.forward(G, batch)

    def forward(self, G, batch, return_mse=False):
        # hamming distance
        target = batch[2][0]

        # single graph
        if torch.all(target >= 0):
            hamming = shd(G, target.to(G))
            return 2 * hamming * math.log(target.shape[-1])
            # sqrt seems very important here... minimum not so much.  without sqrt
            # works well for target = eye or target = ones, but not other graphs
            # with sqrt seems to work more generally.
            # return -((2) ** (torch.maximum(6 - torch.sqrt(hamming), torch.tensor(-10))))
        # Handle Bayesian case

        target = target.squeeze().type(torch.int64)
        var_maps = torch.minimum(torch.tensor(0), target)[:, 0]
        var_mask = var_maps < 0
        vars_to_deidentify = -(var_maps[var_mask] + 1)
        summed_estimated_graph = G[:, ~var_mask]
        # Distance to the nearest admissible graph.
        for i, v in enumerate(vars_to_deidentify):
            summed_estimated_graph[:, v] += G[:, var_mask][:, i]
        hamming = shd(target[~var_mask], summed_estimated_graph.to(target))
        # return -((2) ** (torch.maximum(6 - torch.sqrt(hamming), torch.tensor(-10))))
        return 2 * hamming * math.log(target.shape[-1])
