r""" energy_gfn_module.py

Implements Energy-based GflowNet training. Primarily, this separates out the
GFlowNet, Environment, and Energy models. The GFlowNet and Energy Models are
trainable and represent distributions over the space defined by the
environment. The Environment defines the allowable forward and backwards
transitions over the domain.

Here we define the following modules:

    *  EnergyGFNModule: Base Energy Gflownet module, implements general loss
    functions and data loading. Has the following subclasses:

        1.  HyperGridGFlowNetModule: Implements an EnergyGFNModule over the
        standard hypergrid environment. Ignores batch data.
        2.  FixedGraphGFlowNetModule: Implements an EnergyGFNModule over
        directed graphs where an action corresponds to adding an edge. Ignores
        batch data.
        3.  TrainableCausalGraphGFlowNetModule: Implements training of a
        GFlowNet with a reward based on structural equations. Here we train the
        energy model to minimize the loss E_{G \sim Q(G | D)} \| f(x, theta, G)
        - dx \|_2^2, i.e. fit the energy model on the posterior graphs.
"""
import math
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchdyn.core import NeuralODE

from .components.energy import (
    HammingEnergy,
    SimpleAnalyticBayesVelocityEnergy,
    VelocityEnergy,
)
from .components.environments import GFlowNetEnv, GraphEnv
from .components.evaluation import (
    compare_graph_distribution,
    compare_graphs,
    compare_graphs_bayesian_cover,
    compare_graphs_bayesian_shd,
    compute_graphs_bayesian_diversity,
    compute_graphs_sparsity,
)
from .components.gfn_models import MLPFlow
from .components.structural_equations import (
    HyperStructuralEquationModel,
    LinearStructuralEquationModel,
)


class EnergyGFNModule(LightningModule):
    """Energy GFN Module.

    Implements an energy-based GFN training procedure. The idea here is to
    implement two models, a sampler (GFN) and an energy model that can
    learn from each other. The GFN learns to sample according to the energy
    function, and the energy function learns the data distribution.

    Example:

    The GFN is sampling causal dynamics structures i.e. G contains an edge i,j
    iff d f_i / d x_j is non-zero. The energy function is how well a model can
    represent the data + an L0 penalty on the number of edges.
    """

    _debug = True

    def __init__(
        self,
        gfn_model,
        energy_model,
        env: GFlowNetEnv,
        loss_fn: str = "trajectory_balance",
        alpha: float = 0.0,  # Exploration
        n_steps: int = 0,
        env_batch_size: int = 64,
        eval_batch_size: int = 1000,
        lr: float = 1e-3,
        full_posterior_eval: bool = False,
    ) -> None:
        """Initializes a Energy GFN.

        Args:
            gfn_model: torch module mapping state --> log action probabilities
            energy_model: torch module from (state, batch) --> energy
            env: gflownet environment representing state and actions
                determining valid states and actions for each state.
            loss_fn: gflownet loss function type
            alpha: parameter controlling GFN exploration
            n_steps: number of edges to sample in GFN env (default 0 turns off this feature)
            env_batch_size: number of environments to step in parallel during training
            eval_batch_size: number of environments to use in evaluation
            lr: learning rate
            full_posterior_eval (bool): if true attempts to evaluate the energy
                on all possible states. This should only be performed on small
                state spaces.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["gfn_model", "energy_model", "env"])
        assert 0 <= alpha <= 1
        self.gfn_model = gfn_model
        self.energy_model = energy_model
        self.env_constructor = env
        self.Z = nn.Parameter(torch.ones((1,)) * 0.02)
        self.full_posterior_eval = full_posterior_eval
        self.state_probs = None
        self.actions_count = None

    def reset_envs(self):
        for e in self.envs:
            e.reset()

    def forward(self, envs):
        states_and_masks = self.get_states_and_masks(envs)
        return self.gfn_model(*states_and_masks)

    def get_states_and_masks(self, envs):
        """Get states and forward / backward masks from a list of partially completed DAGs."""
        input_states = torch.stack([e.get_state_input() for e in envs])
        forward_masks = torch.stack([e.forward_mask for e in envs])
        backward_masks = torch.stack(
            [e.backward_mask for e in envs if e.backward_mask is not None]
        )
        return input_states, forward_masks, backward_masks

    def trajectory_balance(self, batch: Any, batch_idx: int):
        r"""
        L_TB =  [log {(Z \prod_t P_F(s_t | s_{t-1})) / (R(x) \prod_t P_B(s_{t-1} | s_t))} ]**2

        Parameterized functions:
            - P_F
            - P_B (default set to uniform)
            - Z
            - R(x) (if using trainable energy)
        """
        self.reset_envs()
        envs = self.envs
        log_likelihood_diff = torch.zeros(self.hparams.env_batch_size).type_as(self.Z)
        log_likelihood_diff += 100 * self.Z
        actions = None
        loss = torch.tensor(0.0, requires_grad=True)
        TERM = self.envs[0].terminal_index
        if self.actions_count is None:
            self.t = torch.zeros(1)
            self.actions_count = torch.zeros((self.hparams.env_batch_size, TERM + 1))

        # Forward sample some complete trajectories accumulating the
        # Trajectory Balance loss along the way
        while not np.alltrue([e.done for e in envs]):
            active_indices = [i for i, e in enumerate(envs) if not e.done]
            active_envs = [e for e in envs if not e.done]
            forward_logprob, back_logprob = self.forward(active_envs)

            # collect backward log likelihood as a function of previous action.
            if actions is not None:
                log_likelihood_diff[active_indices] -= back_logprob.gather(
                    1, actions[actions != TERM, None]
                ).squeeze(1)

            # sample new action
            actions = self.sample_actions(forward_logprob, active_indices)

            # collect forward log likelihood
            log_likelihood_diff[active_indices] += forward_logprob.gather(
                1, actions[:, None]
            ).squeeze(1)

            for a, env in zip(actions, active_envs):
                env.step(a)

        # Maximize log likelihood
        states = torch.stack([e.state for e in envs])
        log_rewards = -self.energy_model(states, batch)
        log_likelihood_diff -= log_rewards
        loss = (log_likelihood_diff**2).mean()
        self.log(
            "train/log_reward",
            log_rewards.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss, log_rewards

    def detailed_balance(self, batch: Any, batch_idx: int):
        """
        L_DB = [log { (R(s_i) P_B(s_{i-1} | s_i) P_F(s_n | s_{i-1})) / ((R(s_{i-1}) P_F(s_i | s_{i-1}) P_F(s_n | s_i))) }]**2

        Parameterized functions:
            - P_F
            - P_B (default set to uniform)
            - R(s) if using trainable energy
        """
        self.reset_envs()
        envs = self.envs
        actions = None
        prev_forward_logprob = None
        prev_log_rewards = None
        prev_active_indices = None
        loss = torch.tensor(0.0, requires_grad=True)
        TERM = self.envs[0].terminal_index
        if self.actions_count is None:
            self.t = torch.zeros(1)
            self.actions_count = torch.zeros((self.hparams.env_batch_size, TERM + 1))

        while not np.alltrue([e.done for e in envs]):
            # Setup
            active_indices = [i for i, e in enumerate(envs) if not e.done]
            active_envs = [e for e in envs if not e.done]
            forward_logprob, back_logprob = self.forward(active_envs)

            # R(s)
            states = torch.stack([e.state for e in active_envs])
            log_rewards = -self.energy_model(states, batch)

            if actions is not None:  # change to 'if actions is not None:'
                prev_sub_indices = [
                    i
                    for i, e_n in enumerate([envs[j] for j in prev_active_indices])
                    if not e_n.done
                ]
                # next_rwd_active_indices = [prev_active_indices[i] for i in next_fwd_active_indices]
                # calculate loss using current and previous state
                #   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
                # R(s') - R(s)
                error = log_rewards - prev_log_rewards[prev_sub_indices]
                error += back_logprob.gather(1, actions[actions != TERM, None]).squeeze(
                    1
                )  # P_B(s|s')
                error += prev_forward_logprob[prev_sub_indices, -1]  # P(s_f|s)
                error -= forward_logprob[:, -1].detach()  # P(s_f|s')
                error -= (
                    prev_forward_logprob[prev_sub_indices]
                    .gather(1, actions[actions != TERM, None])
                    .squeeze(1)
                )  # P_theta(s'|s)
                #   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

                # accumulate loss
                loss = loss + F.huber_loss(
                    error,
                    torch.zeros_like(error),
                    delta=1.0,
                )

            # sample new action
            actions = self.sample_actions(forward_logprob, active_indices)

            # step
            for a, env in zip(actions, active_envs):
                env.step(a)

            # save previous log-probs and log-rewards
            prev_forward_logprob = forward_logprob
            prev_log_rewards = log_rewards
            prev_active_indices = active_indices

        self.t += 1
        # Maximize log likelihood
        return loss, log_rewards

    def sample_actions(self, forward_logprob, active, TERM):
        # Upper confidence bound exploration
        if self.hparams.confidence > 0 and self.hparams.alpha == 0:
            if self.t > 1000:
                # upper confidence bound (for exploration)
                # DOESN't REALLY WORK
                c = self.hparams.confidence
                U = torch.sqrt(torch.log(self.t) / (self.actions_count + 1e-6)).to(
                    forward_logprob
                )
                Q = torch.log(torch.exp(forward_logprob) + c * U[active])
                actions = Q.softmax(1).multinomial(1).squeeze(1)
            else:
                # normal sampling
                actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
            self.actions_count[active.cpu(), actions.cpu()] += 1
            # print(self.actions_count[0])
            self.t += 1

        # greedy-epsilon decay exploration
        elif self.hparams.alpha > 0 and self.hparams.confidence == 0:
            actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
            epsilon = self.hparams.alpha * (0.5**self.current_epoch)
            choice = torch.bernoulli(
                torch.ones_like(actions).to(actions) * epsilon
            ).bool()
            actions = torch.where(
                choice,
                torch.randint(high=TERM, size=actions.size()).to(actions),
                actions,
            )

        # softmax tempering
        elif self.hparams.temper_period != 0:
            c_temper = (
                0.475
                * math.cos(
                    2 * 3.14159 / self.hparams.temper_period * self.current_epoch
                    + 3.14159 / 2
                )
                + 0.525
            )
            if c_temper < 1e-4:
                c_temper = 0.001
            actions = (forward_logprob * c_temper).softmax(1).multinomial(1).squeeze(1)

        # no exploration
        else:
            actions = (forward_logprob).softmax(1).multinomial(1).squeeze(1)

        return actions

    def gflownet_step(self, batch: Any, batch_idx: int):
        if self.hparams.loss_fn == "detailed_balance":
            loss, log_rewards = self.detailed_balance(batch, batch_idx)
        elif self.hparams.loss_fn == "trajectory_balance":
            loss, log_rewards = self.trajectory_balance(batch, batch_idx)
            self.log("train/Z", self.Z, on_step=False, on_epoch=True, prog_bar=False)

        else:
            raise ValueError("\nNo loss_fn selected!\n")
        self.log(
            "train/reward",
            log_rewards.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def energy_step(
        self,
        graphs,
        batch: Any,
        batch_idx: int,
        pretraining: bool = False,
        train_mode: bool = False,
    ):
        return self.energy_model(
            graphs,
            batch,
            return_mse=True,
        )

    def on_fit_start(self):
        self.envs = [
            self.env_constructor(self.device)
            for _ in range(self.hparams.env_batch_size)
        ]

    def training_step(self, batch: Any, batch_idx: int):
        return self.gflownet_step(batch, batch_idx)

    def configure_optimizers(self):
        gfn_opt = torch.optim.Adam(
            [
                {"params": self.gfn_model.parameters()},
                {"params": self.Z, "lr": self.hparams.lr},
            ],
            lr=self.hparams.lr,
        )
        return gfn_opt

    def sample_forward(self, envs=None):
        """Sample states from the GFlowNet."""
        if envs is None:
            # by default use stored environments
            envs = self.envs
        elif isinstance(envs, int):
            # Constructing new envs to sample from
            envs = [self.env_constructor(self.device) for _ in range(envs)]
        while not np.alltrue([e.done for e in envs]):
            active_envs = [e for e in envs if not e.done]
            forward_logprob = self.forward(active_envs)[0]
            actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
            for a, e in zip(actions, active_envs):
                e.step(a)
        return envs

    def sample_backwards(self, envs):
        raise NotImplementedError

    def on_eval_start(self):
        self.eval_envs = self.sample_forward(
            envs=self.hparams.eval_batch_size, steps=self.hparams.n_steps
        )

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        if self.hparams.full_posterior_eval and self.state_probs is None:
            self.list_of_states = self.eval_envs.compute_test_set(
                self.hparams.n_steps, batch[2][0]
            )
            self.state_probs = self.calculate_px(batch)

    def on_validation_start(self):
        return self.on_eval_start()

    def on_test_start(self):
        return self.on_eval_start()

    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test")

    def calculate_px(self, batch):
        energies = []
        for idx in range(0, self.list_of_states.shape[0], 10000):
            if idx % 100000 == 0:
                print(idx)
            energies.append(
                self.energy_model(self.list_of_states[idx : idx + 10000], batch)
            )
        energies = torch.cat(energies)
        mses = energies
        px = torch.exp(-energies)
        tot_energy = px.sum().detach().cpu().numpy()
        state_probs = {
            tuple(s.flatten()): (r / tot_energy, e, m)
            for s, r, e, m in zip(
                self.list_of_states.cpu().numpy(),
                px.detach().cpu().numpy(),
                energies.detach().cpu().numpy(),
                mses.detach().cpu().numpy(),
            )
        }
        return state_probs

    def compute_gc_modes(self, gc_0, p):
        if p == 3:
            gc_0[0, -1, :] = torch.zeros(p)
            gc_0[1, 0, :] = torch.zeros(p)
            gc_0[2, 0, 0], gc_0[2, -1, 1], gc_0[2, 0, 2] = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            gc_0[3, -1, 0], gc_0[3, -1, 1], gc_0[3, 0, 2] = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            gc_0[4, -1, 0], gc_0[4, 0, 1], gc_0[4, 0, 2] = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            gc_0[5, 0, 0], gc_0[5, 0, 1], gc_0[5, -1, 2] = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            gc_0[6, 0, 0], gc_0[6, -1, 1], gc_0[6, -1, 2] = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            gc_0[7, -1, 0], gc_0[7, 0, 1], gc_0[7, -1, 2] = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            return gc_0
        else:
            gc_0[0, -1, :] = torch.zeros(p)
            return gc_0[0]

    def compute_loss_true_graphs(self, graphs, batch, return_mse=False):
        # calculate val losses
        with torch.no_grad():
            p = graphs.shape[-1]
            loss = self.energy_model(graphs, batch, return_mse=return_mse).mean()
            if p == 3:
                gc_ones = torch.ones((2**p, p, p))
                loss_gc_0 = torch.zeros(2**p)
                gc_0 = self.compute_gc_modes(gc_ones, p)
                for i in range(2**p):
                    loss_gc_0[i] = self.energy_model(
                        gc_0[i].expand(*graphs.shape), batch, return_mse=return_mse
                    ).mean()
            else:
                gc_ones = torch.ones((2**5, p, p))
                gc_0 = self.compute_gc_modes(gc_ones, p)
                loss_gc_0 = self.energy_model(
                    gc_0.expand(*graphs.shape), batch, return_mse=return_mse
                ).mean()
                loss_gc_0 = loss_gc_0.unsqueeze(0)

            gc_1 = torch.ones((p, p))
            gc_1[-1, :] = torch.zeros(p)
            gc_1[0, 0] = torch.zeros(1)
            gc_1 = gc_1.expand(*graphs.shape)
            gc_ones = gc_ones[0].expand(*graphs.shape)

            loss_gc_1 = self.energy_model(gc_1, batch, return_mse=return_mse).mean()
            loss_gc_ones = self.energy_model(
                gc_ones, batch, return_mse=return_mse
            ).mean()

        return loss, loss_gc_0, loss_gc_1, loss_gc_ones

    def eval_graph_metrics(self, batch: Any, batch_idx: int, prefix: str, graphs):
        # calculate val losses
        loss, _, _, _ = self.compute_loss_true_graphs(graphs, batch, return_mse=False)
        mse, _, _, _ = self.compute_loss_true_graphs(graphs, batch, return_mse=True)
        # loss = torch.zeros(1)
        graphs = graphs.detach().cpu().numpy()
        # log losses
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        to_return = {"loss": loss.item()}
        if batch_idx == 0:
            gc = batch[2][0].cpu().detach().numpy()
            print("GC", gc)
            if np.any(gc < 0):
                # in the uncertain setting if the ground truth graph has any values < 0
                bayes_shd, bayes_tshd = compare_graphs_bayesian_shd(gc, graphs)
                self.log(f"{prefix}/bayes_shd", bayes_shd, on_step=False, on_epoch=True)
                self.log(
                    f"{prefix}/bayes_tshd", bayes_tshd, on_step=False, on_epoch=True
                )
                bayes_cover = compare_graphs_bayesian_cover(gc, graphs)
                self.log(
                    f"{prefix}/bayes_cover", bayes_cover, on_step=False, on_epoch=True
                )
                if prefix == "test":
                    self.log(
                        f"{prefix}/kl_div", self.kl_div, on_step=False, on_epoch=True
                    )
                (
                    bayes_dist_unif,
                    _,
                    bayes_dist_prop,
                ) = compare_graph_distribution(gc, graphs)
                self.log(
                    f"{prefix}/bayes_dist_unif",
                    bayes_dist_unif,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"{prefix}/bayes_dist_prop",
                    bayes_dist_prop,
                    on_step=False,
                    on_epoch=True,
                )
            bayes_diversity = compute_graphs_bayesian_diversity(graphs)

            self.log(
                f"{prefix}/bayes_diversity",
                bayes_diversity,
                on_step=False,
                on_epoch=True,
            )
            sparsity = 0.0
            for m in range(graphs.shape[0]):
                sparsity += compute_graphs_sparsity(graphs[m])
            avg_sparsity = sparsity / graphs.shape[0]
            self.log(
                f"{prefix}/avg_sparsity", avg_sparsity, on_step=False, on_epoch=True
            )
            # Assume it is a group of graphs, take mean over single_graph metrics
            metrics = [compare_graphs(gc, g) for g in graphs]
            metrics = {k: np.mean([d[k] for d in metrics]) for k in metrics[0].keys()}
            to_return.update(metrics)
            for k, v in metrics.items():
                self.log(f"{prefix}/{k}", v, on_step=False, on_epoch=True)
            if self.hparams.full_posterior_eval:
                self.state_probs = self.calculate_px(batch)
        return to_return

    def eval_epoch_end(self, outputs, prefix: str) -> None:
        if self.hparams.full_posterior_eval:
            # Recalculate stateprobs
            # get total visits to all state visits
            # If we have access to ground truth probabilities then check the total
            # variation distance between predicted and ground truth.
            # estimate p(x) from samples
            total_counts = {}
            total_sum = 0
            graphs = self.eval_envs.get_state().int().cpu().numpy()
            for state in graphs:
                state_id = tuple(state.flatten())
                total_counts[state_id] = total_counts.get(state_id, 0) + 1
                total_sum += 1
            l1_loss = 0
            for state, prob in self.state_probs.items():
                l1_loss += abs(prob[0] - total_counts.get(state, 0) / total_sum)

            if self._debug:

                def print_dict(d):
                    np.set_printoptions(suppress=True, precision=3)
                    top = sorted(d.items(), key=lambda x: x[1][1], reverse=True)[:16]
                    print("--- top energy ---")
                    for k, v in top:
                        print(f"{np.array(k).astype(int)}: {np.array(v)}")
                    top = sorted(d.items(), key=lambda x: x[1][3], reverse=False)[:16]
                    print("--- min mse ---")
                    for k, v in top:
                        print(f"{np.array(k).astype(int)}: {np.array(v)}")
                    print("--- top predicted ---")
                    top = sorted(d.items(), key=lambda x: x[1][0], reverse=True)[:16]
                    for k, v in top:
                        print(f"{np.array(k).astype(int)}: {np.array(v)}")

                print("Predicted vs actual vs energy vs mse:")
                pred = {k: v / total_sum for k, v in total_counts.items()}
                pred_vs_actual = {
                    k: (pred.get(k, 0.0), *v) for k, v in self.state_probs.items()
                }
                print_dict(pred_vs_actual)
                # for p in self.energy_model.structure_model.parameters():
                #    print("structure_params", p.name, p.data)
            l1_loss /= 2  # Convert to total variation loss
            self.log(f"{prefix}/loss", l1_loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "test")


def construct_gfn_graph_mlp(n_dim, uniform_backwards, hidden_dim):
    """Minimimal MLPFlow Constructor."""
    return MLPFlow(
        in_dim=n_dim**2,
        out_dim=n_dim**2,
        misc_out_dim=0,
        uniform_backwards=uniform_backwards,
        hidden_dim=hidden_dim,
    )


def construct_per_node_gfn_graph_mlp(n_dim, uniform_backwards, hidden_dim):
    """Per-node Minimimal MLPFlow Constructor."""
    pMLPFlow = nn.ModuleList()
    for _ in range(n_dim):
        pMLPFlow.append(
            MLPFlow(
                in_dim=n_dim,
                out_dim=n_dim,
                misc_out_dim=0,
                uniform_backwards=uniform_backwards,
                hidden_dim=hidden_dim,
            )
        )
    return pMLPFlow


class FixedGraphGFlowNetModule(EnergyGFNModule):
    """Implements a single-mode fixed target GFN trainer.

    This can be used to validate that the GFlowNet training is working as expected for n_dim < 5
    where we can enumerate the true posterior over all possible graphs.
    """

    def __init__(
        self,
        n_dim: int = 2,
        env_batch_size: int = 64,
        eval_batch_size: int = 1000,
        uniform_backwards: float = False,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        full_posterior_eval: bool = True,
        **kwargs,
    ) -> None:

        gfn_model = construct_gfn_graph_mlp(n_dim, uniform_backwards, hidden_dim)

        def energy_model(x, batch):
            del batch
            """Measure distance to identity matrix"""
            hamming = torch.abs(x - torch.eye(x.shape[-1]).type_as(x)).sum(dim=(-2, -1))
            ret = torch.zeros_like(hamming)
            ret += 1e-8
            ret[hamming <= 3] = 10
            ret[hamming <= 1] = 100
            ret[hamming == 0] = 1000
            ret = -torch.log(ret)
            return ret

        def env(device):
            return GraphEnv(n_dim, device)

        super().__init__(
            gfn_model=gfn_model,
            energy_model=energy_model,
            env=env,
            env_batch_size=env_batch_size,
            eval_batch_size=eval_batch_size,
            lr=lr,
            full_posterior_eval=full_posterior_eval,
        )
        self.save_hyperparameters(logger=False)


class TrainableCausalGraphGFlowNetModule(EnergyGFNModule):
    _debug_generate_full_graphs = False

    def __init__(
        self,
        n_dim: int,
        structural_eq_model,
        env_batch_size: int = 64,
        eval_batch_size: int = 1000,
        uniform_backwards: float = False,
        hidden_dim: int = 64,
        embed_dim: int = 128,
        lr: float = 1e-3,
        alpha: float = 0.0,
        temperature: float = 1.0,
        temper_period: int = 1.0,
        w_mse: float = 1.0,
        w_sparse: float = 0.01,
        prior_lambda: float = 1.0,
        beta: float = 1e-4,
        confidence: float = 0.0,
        energy_freq: int = 10,
        full_posterior_eval: bool = False,
        debug_use_shd_energy: bool = False,
        analytic_use_simple_mse_energy: bool = False,
        solver: str = None,
        **kwargs,
    ) -> None:
        assert 0 <= alpha <= 1

        gfn_model = construct_gfn_graph_mlp(n_dim, uniform_backwards, hidden_dim)

        if solver is None:
            energy_model = VelocityEnergy(
                structural_eq_model, w_mse, w_sparse, temperature
            )

        elif debug_use_shd_energy:
            print("Warning: using cheater hamming distance energy")
            energy_model = HammingEnergy()

        elif analytic_use_simple_mse_energy:
            print("Using simple analytic reward MAP energy")
            energy_model = SimpleAnalyticBayesVelocityEnergy(
                n_dim=n_dim,
                beta=beta,
                prior_lambda=prior_lambda,
                temperature=temperature,
            )
        else:
            raise Exception("Need to select type of energy reward from boolean list")

        def env(device):
            return GraphEnv(n_dim, device)

        super().__init__(
            gfn_model=gfn_model,
            energy_model=energy_model,
            env=env,
            env_batch_size=env_batch_size,
            eval_batch_size=eval_batch_size,
            lr=lr,
            full_posterior_eval=full_posterior_eval,
        )
        self.save_hyperparameters(ignore=["structural_eq_model"], logger=False)

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int = 0):
        if optimizer_idx == 0:
            graphs = torch.stack([e.state for e in self.envs])
            if batch_idx % 50 == 0:
                torch.set_printoptions(precision=1)
                print(graphs.mean(0))
                print(self.energy_model.structure_model.weight)
            return self.gflownet_step(batch, batch_idx)
        self.log(
            "avg_shd",
            torch.sum(torch.abs(batch[2][0] - graphs.mean(0))),
            on_step=True,
            prog_bar=True,
        )
        loss = self.energy_step(graphs, batch, batch_idx).mean()
        self.log(
            "train/energy_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )
        return {"loss": loss}

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        super().eval_step(batch, batch_idx, prefix)
        graphs = torch.stack([e.state for e in self.eval_envs])
        graph_metrics = self.eval_graph_metrics(batch, batch_idx, prefix, graphs)
        return graph_metrics

    def on_train_epoch_start(self):
        # Update energy temperature
        self.energy_model.step(self.current_epoch)

    def configure_optimizers(self):
        gfn_opt = torch.optim.Adam(
            [
                {"params": self.gfn_model.parameters()},
                {"params": self.Z},
            ],
            lr=self.hparams.lr,
        )
        if self.hparams.debug_use_shd_energy or self.hparams.analytic_use_map_energy:
            # don't train energy
            return ({"optimizer": gfn_opt},)
        energy_opt = torch.optim.Adam(
            self.energy_model.parameters(), lr=self.hparams.lr
        )
        return (
            {"optimizer": gfn_opt, "frequency": 1},
            {"optimizer": energy_opt, "frequency": self.hparams.energy_freq},
        )


class LinearTrainableCausalGraphGFlowNetModule(TrainableCausalGraphGFlowNetModule):
    def __init__(
        self,
        dm_conf,
        bias: bool = True,
        **kwargs,
    ) -> None:
        n_dim = dm_conf.p
        structural_eq_model = LinearStructuralEquationModel(dm_conf.p, bias=bias)
        super().__init__(
            n_dim=n_dim,
            structural_eq_model=structural_eq_model,
            **kwargs,
        )
        self.save_hyperparameters(
            ignore=["dm_conf", "structural_eq_model"], logger=False
        )


class HyperTrainableCausalGraphGFlowNetModule(TrainableCausalGraphGFlowNetModule):
    def __init__(
        self,
        dm_conf,
        hyper,
        dims=None,
        hyper_args=None,
        **kwargs,
    ) -> None:
        n_dim = dm_conf.p
        if dims is None:
            dims = [1]
        structural_eq_model = HyperStructuralEquationModel(
            [n_dim, *dims], hyper=hyper, hyper_args=hyper_args
        )
        super().__init__(
            n_dim=n_dim,
            structural_eq_model=structural_eq_model,
            **kwargs,
        )
        self.save_hyperparameters(
            ignore=["dm_conf", "structural_eq_model"], logger=False
        )
