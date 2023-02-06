r""" parallel_energy_gfn_module.py

Paralellized implementation of energy_gfn_module.py.

Here we define the following primary modules:

    *  ParallelEnergyGFNModule: Base Parallel Energy Gflownet module, implements general loss
    functions and data loading. Has the following subclasses:

        1. ParallelTrainableCausalGraphGFlowNetModule: Implements training of a
        GFlowNet with a reward based on structural equations. Here we can either use
        analytic solver for linear SCM parameters to minimize the loss or train the energy model.
        to minimize the loss E_{G \sim Q(G | D)} \| f(x, theta, G)
        - dx \|_2^2, i.e. fit the energy model on the posterior graphs.
        2. PerNodeParallelTrainableCausalGraphGFlowNetModule: Identical to (1) except
        uses per-node graph factorization
"""
import itertools
import math
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F

from .components.energy import (
    HammingEnergy,
    PerNodeSimpleAnalyticBayesVelocityEnergy,
    PerNodeVelocityEnergy,
    SimpleAnalyticBayesVelocityEnergy,
    VelocityEnergy,
)
from .components.environments import GraphEnvs, PerNodeGraphEnvs
from .components.structural_equations import (
    HyperStructuralEquationModel,
    LinearStructuralEquationModel,
)
from .energy_gfn_module import (
    EnergyGFNModule,
    construct_gfn_graph_mlp,
    construct_per_node_gfn_graph_mlp,
)


class ParallelEnergyGFNModule(EnergyGFNModule):
    """Parallelized GFlowNet module with energy-based reward energy."""

    def reset_envs(self):
        self.envs.reset()

    def get_states_and_masks(self, envs):
        """Get states and forward / backward masks from a list of partially completed DAGs."""
        done_mask = envs.done
        return (
            envs.get_state()[~done_mask],
            envs.forward_mask[~done_mask],
            envs.backward_mask[~done_mask],
        )

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
        TERM = envs.terminal_index
        actions = None
        if self.actions_count is None:
            self.t = torch.zeros(1)
            self.actions_count = torch.zeros((self.hparams.env_batch_size, TERM + 1))

        # Forward sample some complete trajectories accumulating the
        # Trajectory Balance loss along the way
        while not torch.all(envs.done):
            active = ~envs.done
            forward_logprob, back_logprob = self.forward(envs)

            # collect backward log likelihood as a function of previous action.
            if actions is not None:
                log_likelihood_diff[active] -= back_logprob.gather(
                    1, actions[actions != self.envs.terminal_index, None]
                ).squeeze(1)

            # sample new action
            actions = self.sample_actions(forward_logprob, active, TERM)

            # collect forward log likelihood
            log_likelihood_diff[active] += forward_logprob.gather(
                1, actions[:, None]
            ).squeeze(1)

            envs.step(actions)

        # Maximize log likelihood
        log_rewards = -self.energy_model(envs.get_state(), batch)
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
        env = self.envs
        actions = None
        prev_log_rewards = torch.empty(env.done.shape[0]).type_as(env.state)
        prev_forward_logprob = None
        loss = torch.tensor(0.0, requires_grad=True)
        TERM = env.terminal_index
        if self.actions_count is None:
            self.t = torch.zeros(1).type_as(prev_log_rewards).int()
            self.actions_count = torch.zeros(
                (self.hparams.env_batch_size, TERM + 1)
            ).type_as(self.t)

        while not torch.all(env.done):
            # Setup
            active = ~env.done
            forward_logprob, back_logprob = self.forward(env)

            # init temp vars
            # log_rewards = torch.zeros(self.hparams.env_batch_size)

            # R(s)
            # log_rewards = -self.energy_model(env.get_state()[active], batch, current_epoch=self.current_epoch)
            log_rewards = -self.energy_model(env.get_state()[active], batch)

            if actions is not None:
                # calculate loss using current and previous state
                #   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
                # R(s') - R(s)
                error = log_rewards - prev_log_rewards[active]
                error += back_logprob.gather(1, actions[actions != TERM, None]).squeeze(
                    1
                )  # P_B(s|s')
                error += prev_forward_logprob[active, -1]  # P(s_f|s)
                error -= forward_logprob[:, -1].detach()  # P(s_f|s')
                error -= (
                    prev_forward_logprob[active]
                    .gather(1, actions[actions != TERM, None])
                    .squeeze(1)
                )  # P_theta(s'|s)
                #   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

                # accumulate loss
                loss = loss + F.huber_loss(
                    error,
                    torch.zeros_like(error),
                    delta=1.0,
                    reduction="none",
                )
                # loss = loss * (1.0 + torch.exp(log_rewards)) # scale GFN-loss by R(s)
                # loss = loss.mean(0)
                loss = loss * log_rewards.softmax(0)
                loss = loss.mean(0)

            # get actions
            actions = self.sample_actions(forward_logprob, active, TERM)
            env.step(actions)

            # save previous log-probs and log-rewards
            if prev_forward_logprob is None:
                prev_forward_logprob = torch.empty_like(forward_logprob)
            prev_forward_logprob[active] = forward_logprob
            prev_log_rewards[active] = log_rewards

        # Maximize log likelihood
        return loss, log_rewards

    def sample_actions(self, forward_logprob, active, TERM):
        # Upper confidence bound exploration
        if self.hparams.confidence > 0 and self.hparams.alpha == 0:
            if self.t > 1000:
                # upper confidence bound (for exploration)
                c = self.hparams.confidence
                U = torch.sqrt(torch.log(self.t) / (self.actions_count + 1e-6))
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
                torch.ones_like(actions).type_as(actions) * epsilon
            ).bool()
            actions = torch.where(
                choice,
                torch.randint(high=TERM, size=actions.size()).to(actions),
                actions,
            )

        # softmax tempering
        elif self.hparams.temper_period != 0 and self.hparams.alpha == 0:
            self.c_temper = (
                0.475
                * math.cos(
                    2 * 3.14159 / self.hparams.temper_period * self.current_epoch
                    + 3.14159 / 2
                )
                + 0.525
            )
            if self.c_temper < 1e-4:
                self.c_temper = 0.001
            actions = (
                (forward_logprob * self.c_temper).softmax(1).multinomial(1).squeeze(1)
            )
            # Temperature regularize on non-terminating actions
            actions[actions != TERM] = (
                (forward_logprob[actions != TERM, :-1] * self.c_temper)
                .softmax(1)
                .multinomial(1)
                .squeeze()
            )

        # softmax tempering + greedy-epsilon
        elif self.hparams.temper_period != 0 and self.hparams.alpha > 0:
            self.c_temper = (
                0.475
                * math.cos(
                    2 * 3.14159 / self.hparams.temper_period * self.current_epoch
                    + 3.14159 / 2
                )
                + 0.525
            )
            if self.c_temper < 1e-3:
                self.c_temper = 0.001
            actions = (
                (forward_logprob * self.c_temper).softmax(1).multinomial(1).squeeze(1)
            )
            # Temperature regularize on non-terminating actions
            actions[actions != TERM] = (
                (forward_logprob[actions != TERM, :-1] * self.c_temper)
                .softmax(1)
                .multinomial(1)
                .squeeze()
            )
            # greedy-eps
            epsilon = self.hparams.alpha * (0.5**self.current_epoch)
            choice = torch.bernoulli(
                torch.ones_like(actions).type_as(actions) * epsilon
            ).bool()
            actions = torch.where(
                choice,
                torch.randint(high=TERM, size=actions.size()).to(actions),
                actions,
            )

        # no exploration
        else:
            actions = (forward_logprob).softmax(1).multinomial(1).squeeze(1)

        return actions

    def on_fit_start(self):
        self.envs = self.env_constructor(self.hparams.env_batch_size, self.device)

    def sample_forward(
        self, envs: Optional[Union[int, Any]] = None, steps: Optional[int] = None
    ):
        """Sample states from the GFlowNet.

        Args:
            envs: defaults to self.envs, if specified as an integer creates
            that many new environments.

            steps: if specified, envs are rolled out for exactly this many
            steps, ignoring the exit probabilities.
        """
        if envs is None:
            # by default use stored environments
            envs = self.envs
        elif isinstance(envs, int):
            # Constructing new envs to sample from
            envs = self.env_constructor(envs, self.device)
        if self.hparams.n_steps > 0:
            steps = self.hparams.n_steps
        if steps:
            assert steps >= 0
            TERM = envs.terminal_index
            for i in range(steps):
                forward_logprob = self.forward(envs)[0]
                forward_logprob = forward_logprob[:, :TERM]
                actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
                envs.step(actions)
            return envs

        while not torch.all(envs.done):
            forward_logprob = self.forward(envs)[0]
            actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
            envs.step(actions)
        return envs


class ParallelTrainableCausalGraphGFlowNetModule(ParallelEnergyGFNModule):
    _debug_generate_full_graphs = False

    def __init__(
        self,
        n_dim: int,
        structural_eq_model,
        env_batch_size: int = 64,
        eval_batch_size: int = 1000,
        uniform_backwards: float = False,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        alpha: float = 0.0,
        temperature: float = 1.0,
        temper_period: int = 1.0,
        prior_lambda: float = 1.0,
        beta: float = 1e-4,
        full_posterior_eval: bool = False,
        debug_use_shd_energy: bool = False,
        analytic_use_simple_mse_energy: bool = False,
        solver: str = None,
        **kwargs,
    ) -> None:
        r"""Initializes a Parallel Energy GFN.

        Args:
            n_dim: number of variables in dynamic system (n_dim**2 = number of nodes in graph)
            structural_eq_model: structural equation defining reward energy
            env_batch_size: number of environments to step in parallel during training
            eval_batch_size: number of environments to use in evaluation
            uniform_backwards (bool): if true uses uniform distribution for backward transition probabilities
            hidden_dim: number of hidden units for GFN transition prob parameterization
            lr: learning rate
            alpha: hparam for uniform mixing probability to encourage GFN policy exploration
            temperature: scaling for energy likelihood (1/temperature**2)
            temper_period: period of cosine schedule for softmax tempering of GFN policy
            prior_lambda: controls degree of sparsity enforcement on graphs (\lambda * ||G||_0)
            beta: regression parameters for analytic linear solver
            full_posterior_eval (bool): if true attempts to evaluate the energy
                on all possible states. This should only be performed on small
                state spaces.
            debug_use_shd_energy (bool): if true uses known hamming distance energy.
                This is uses ground truth graphs. Used for designing shape of energy
                reward.
            analytic_use_simple_mse_energy (bool): if true uses analytic linear solver
                for linear SCM parameters.
        """
        assert 0 <= alpha <= 1
        if temper_period != 0:
            self.c_temper = 1.0

        gfn_model = construct_gfn_graph_mlp(n_dim, uniform_backwards, hidden_dim)

        if debug_use_shd_energy:
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
            print("\n Using Velocity hyper-net reward eneryg \n")
            energy_model = VelocityEnergy(
                structural_eq_model, prior_lambda, temperature
            )

        def env(n_graphs, device):
            return GraphEnvs(n_graphs, n_dim, device)

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
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int):
        graphs = self.envs.get_state()
        if (
            self.hparams.debug_use_shd_energy
            or self.hparams.analytic_use_simple_mse_energy
        ):
            gfn_opt = self.optimizers()
        else:
            gfn_opt, energy_opt = self.optimizers()

        # Train GFN
        for i in range(self.hparams.gfn_freq):
            gfn_loss = self.gflownet_step(batch, batch_idx)["loss"]
            gfn_opt.zero_grad()
            self.manual_backward(gfn_loss)
            gfn_opt.step()
        self.log_dict(
            {
                "train/gfn_loss": gfn_loss,
                "train/avg_shd": torch.sum(torch.abs(batch[2][0] - graphs.mean(0))),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Train energy if needed
        if (
            self.hparams.debug_use_shd_energy
            or self.hparams.analytic_use_simple_mse_energy
        ):
            return
        if self.current_epoch < self.hparams.pretraining_epochs:
            graphs = torch.ones_like(graphs)
        for i in range(self.hparams.energy_freq):
            energy_loss = self.energy_step(
                graphs,
                batch,
                batch_idx,
                pretraining=self.current_epoch < self.hparams.pretraining_epochs,
            )
            energy_opt.zero_grad()
            self.manual_backward(energy_loss)
            energy_opt.step()
        self.log_dict(
            {"train/energy_loss": energy_loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_eval_start(self):
        super().on_eval_start()
        graphs = self.eval_envs.get_state()
        print("Val Graphs")
        print("----------")
        print(graphs.mean(0))
        print("----------")
        if (
            not self.hparams.debug_use_shd_energy
            and not self.hparams.analytic_use_simple_mse_energy
        ):
            print("best_mse", self.energy_model.best_mse)
            print("mse_mean_data", self.energy_model.mse_mean_data)

            weights, biases = self.energy_model.structure_model.layers[0].hyper_layer(
                graphs
            )
            print(weights.mean(0).squeeze(), biases.mean(0).squeeze())

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        super().eval_step(batch, batch_idx, prefix)
        graphs = self.eval_envs.get_state()
        if batch_idx == 0:
            optimal_edges = (
                torch.sum(torch.clip(batch[2][0], min=0)).cpu().detach().item()
            )
            self.optimal_eval_envs = self.sample_forward(
                envs=self.hparams.eval_batch_size, steps=self.hparams.n_steps
            )
            optimal_graphs = self.optimal_eval_envs.get_state()
            print("Optimal Edges Val Graphs")
            print("----------")
            print(optimal_graphs.mean(0))
            print("----------")
            print(self.energy_model(graphs, batch)[:10])
            if not self.hparams.debug_use_shd_energy:
                print(self.energy_model.likelihood(graphs, batch)[:10])
            print("----------")
            print("Single sampled graph e.g.")
            print(graphs[0])
        graph_metrics = self.eval_graph_metrics(batch, batch_idx, prefix, graphs)
        return graph_metrics

    def configure_optimizers(self):
        if self.hparams.analytic_use_simple_mse_energy:
            gfn_opt = torch.optim.Adam(
                [
                    {"params": self.gfn_model.parameters()},
                    {"params": self.Z},
                ],
                lr=self.hparams.lr / 1,
            )
        else:
            gfn_opt = torch.optim.Adam(
                [
                    {"params": self.gfn_model.parameters()},
                    {"params": self.Z},
                ],
                lr=self.hparams.lr / 0.1,
            )
        if (
            self.hparams.debug_use_shd_energy
            or self.hparams.analytic_use_simple_mse_energy
        ):
            return gfn_opt
        energy_opt = torch.optim.Adam(
            self.energy_model.parameters(), lr=self.hparams.lr
        )
        return gfn_opt, energy_opt


class ParallelLinearTrainableCausalGraphGFlowNetModule(
    ParallelTrainableCausalGraphGFlowNetModule
):
    def __init__(
        self,
        dm_conf,
        bias: bool = True,
        debug_use_true_weights: bool = False,
        **kwargs,
    ) -> None:
        n_dim = dm_conf.p
        if debug_use_true_weights:
            bias = False
            structural_eq_model = LinearStructuralEquationModel(dm_conf.p, bias=bias)
        else:
            structural_eq_model = LinearStructuralEquationModel(dm_conf.p, bias=bias)
        super().__init__(
            n_dim=n_dim,
            structural_eq_model=structural_eq_model,
            **kwargs,
        )
        self.save_hyperparameters(
            ignore=["dm_conf", "structural_eq_model"], logger=False
        )


class PerNodeParallelTrainableCausalGraphGFlowNetModule(ParallelEnergyGFNModule):
    r"""
    Implements Per-node GFN factorized model:

    Q(G | D) \ \prod_{i \in 1,...,d} Q_i(G_i | D).

    This module uses d GFNs, 1 GFN for each per-variable graph.
    """

    _debug_generate_full_graphs = False

    def __init__(
        self,
        n_dim: int,
        structural_eq_model,
        env_batch_size: int = 64,
        eval_batch_size: int = 1000,
        uniform_backwards: float = False,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        alpha: float = 0.0,
        temperature: float = 1.0,
        temper_period: int = 1.0,
        prior_lambda: float = 1.0,
        beta: float = 1e-4,
        gfn_freq: int = 10,
        energy_freq: int = 10,
        load_pretrain: bool = False,
        pretraining_epochs: int = 15,
        full_posterior_eval: bool = False,
        debug_use_shd_energy: bool = False,
        analytic_use_simple_mse_energy: bool = False,
        **kwargs,
    ) -> None:
        r"""Initializes a Per-node Parallel Energy GFN.

        Args:
            n_dim: number of variables in dynamic system (n_dim**2 = number of nodes in graph)
            structural_eq_model: structural equation defining reward energy
            env_batch_size: number of environments to step in parallel during training
            eval_batch_size: number of environments to use in evaluation
            uniform_backwards (bool): if true uses uniform distribution for backward transition probabilities
            hidden_dim: number of hidden units for GFN transition prob parameterization
            lr: learning rate
            alpha: hparam for uniform mixing probability to encourage GFN policy exploration
            temperature: scaling for energy likelihood (1/temperature**2)
            temper_period: period of cosine schedule for softmax tempering of GFN policy
            prior_lambda: controls degree of sparsity enforcement on graphs (\lambda * ||G||_0)
            beta: regression parameters for analytic linear solver
            gfn_freq: number of GFN optimizer steps per epoch
            energy_freq: if using trainable energy, number of energy model optimizer steps per epoch
            load_pretrain (bool): if true, use pre-trained GFN flow model
            pretraining_epochs: number of epochs to pre-train GFN model with fixed graphs
            full_posterior_eval (bool): if true attempts to evaluate the energy
                on all possible states. This should only be performed on small
                state spaces.
            debug_use_shd_energy (bool): if true uses known hamming distance energy.
                This is uses ground truth graphs. Used for designing shape of energy
                reward.
            analytic_use_simple_mse_energy (bool): if true uses analytic linear solver
                for linear SCM parameters.
        """
        assert 0 <= alpha <= 1
        if temper_period != 0:
            self.c_temper = 1.0

        self.n_dim = n_dim

        gfn_model = construct_per_node_gfn_graph_mlp(
            n_dim, uniform_backwards, hidden_dim
        )

        if debug_use_shd_energy:
            print("Warning: using cheater hamming distance energy")
            energy_model = HammingEnergy()

        elif analytic_use_simple_mse_energy:
            print("Using simple analytic reward MAP energy")
            energy_model = PerNodeSimpleAnalyticBayesVelocityEnergy(
                n_dim=n_dim,
                beta=beta,
                prior_lambda=prior_lambda,
                temperature=temperature,
            )
        else:
            print("\n Using Per-node Velocity Hyper-network reward energy\n")
            energy_model = PerNodeVelocityEnergy(
                structural_eq_model, prior_lambda, temperature
            )

        def env(n_graphs, device):
            envs = []
            for _ in range(n_dim):
                envs.append(PerNodeGraphEnvs(n_graphs, n_dim, device))
            return envs

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
        self.automatic_optimization = False

    def reset_envs(self):
        self.envs[self.p].reset()

    def detailed_balance(self, batch: Any, batch_idx: int):
        """Per-node detailed balance. Each state s_i represents a complete trajectory G_i.

        L_DB = [log { (R(s_i) P_B(s_{i-1} | s_i) P_F(s_n | s_{i-1})) / ((R(s_{i-1}) P_F(s_i | s_{i-1}) P_F(s_n | s_i))) }]**2

        Parameterized functions:
            - P_F
            - P_B (default set to uniform)
            - R(s) if using trainable energy
        """
        self.reset_envs()
        env = self.envs[self.p]
        actions = None
        prev_log_rewards = torch.empty(env.done.shape[0]).type_as(env.state)
        prev_forward_logprob = None
        loss = torch.tensor(0.0, requires_grad=True)
        TERM = env.terminal_index
        if self.actions_count is None:
            self.t = torch.zeros(1).type_as(prev_log_rewards).int()
            self.actions_count = torch.zeros(
                (self.hparams.env_batch_size, TERM + 1)
            ).type_as(self.t)

        while not torch.all(env.done):
            # Setup
            active = ~env.done
            forward_logprob, back_logprob = self.forward(env)

            log_rewards = -self.energy_model(
                env.get_state()[active], batch, node_idx=self.p
            )

            if actions is not None:
                # calculate loss using current and previous state
                #   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
                # R(s') - R(s)
                error = log_rewards - prev_log_rewards[active]
                error += back_logprob.gather(1, actions[actions != TERM, None]).squeeze(
                    1
                )  # P_B(s|s')
                error += prev_forward_logprob[active, -1]  # P(s_f|s)
                error -= forward_logprob[:, -1].detach()  # P(s_f|s')
                error -= (
                    prev_forward_logprob[active]
                    .gather(1, actions[actions != TERM, None])
                    .squeeze(1)
                )  # P_theta(s'|s)
                #   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

                # accumulate loss
                loss = loss + F.huber_loss(
                    error,
                    torch.zeros_like(error),
                    delta=1.0,
                    reduction="none",
                )
                loss = loss * log_rewards.softmax(0)
                loss = loss.mean(0)

            # get actions
            actions = self.sample_actions(forward_logprob, active, TERM)
            env.step(actions)

            # save previous log-probs and log-rewards
            if prev_forward_logprob is None:
                prev_forward_logprob = torch.empty_like(forward_logprob)
            prev_forward_logprob[active] = forward_logprob
            prev_log_rewards[active] = log_rewards

        # Maximize log likelihood
        return loss, log_rewards

    def forward(self, envs, envs_idx=None):
        if envs_idx is None:
            gfn_model = self.gfn_model[self.p]
            states_and_masks = self.get_states_and_masks(envs)
            return gfn_model(*states_and_masks)
        else:
            gfn_model = self.gfn_model[envs_idx]
            states_and_masks = self.get_states_and_masks(envs)
            return gfn_model(*states_and_masks)

    def sample_forward(
        self, envs: Optional[Union[int, Any]] = None, steps: Optional[int] = None
    ):
        """Sample states from the GFlowNet.

        Args:
            envs: defaults to self.envs, if specified as an integer creates
            that many new environments.

            steps: if specified, envs are rolled out for exactly this many
            steps, ignoring the exit probabilities.
        """
        if envs is None:
            # by default use stored environments
            envs = self.envs
        elif isinstance(envs, int):
            # Constructing new envs to sample from
            envs = self.env_constructor(envs, self.device)
        if self.hparams.n_steps > 0:
            steps = self.hparams.n_steps
        if steps:
            assert steps >= 0
            TERM = envs.terminal_index
            for i in range(steps):
                forward_logprob = self.forward(envs)[0]
                forward_logprob = forward_logprob[:, :TERM]
                actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
                envs.step(actions)
            return envs

        p = 0
        for es in envs:
            while not torch.all(es.done):
                forward_logprob = self.forward(es, p)[0]
                actions = forward_logprob.softmax(1).multinomial(1).squeeze(1)
                es.step(actions)
            envs[p] = es
            p += 1
        return envs

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
                for j, index in zip(line, np.arange(n)[mask.cpu()]):
                    sub_graphs[i, j, index] = 1
        masks = torch.stack(mask_list)
        prod = sub_graphs.to(graph)
        certain_graph0 = graph[:1].repeat(prod.shape[0], 1, 1).to(graph)
        certain_graph_mid = graph[2:-2].repeat(prod.shape[0], 1, 1)
        floppier_graph = torch.cat(
            [certain_graph0, prod[:, :1], certain_graph_mid, prod[:, 1:]], dim=1
        )
        return floppier_graph

    def compute_true_modes(self, graph):
        """Computes ground truth admissible graphs for syntehtic dataset."""
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

    def compute_traj_likelihood(self, per_node_graph, gfn_model, nll_envs):
        """Computes likelihood Q(G) of a complete trajectory (construction of G)."""
        G_prob_like = 0
        action_trajectories = nll_envs.action_trajectoreis(per_node_graph)
        log_sum_args = []
        for action_traj in action_trajectories:
            if len(action_traj.shape) > 0:
                nll_envs.reset()
                log_like_traj = 0
                for a in action_traj:
                    states_and_masks = self.get_states_and_masks(nll_envs)
                    forward_logprob = gfn_model(*states_and_masks)[0]
                    log_like_traj += forward_logprob[0, a]
                    nll_envs.step(a)
                log_like_traj += forward_logprob[
                    0, per_node_graph.shape[0]
                ]  # add termination log_prob
                log_sum_args.append(torch.tensor(log_like_traj))
            else:
                nll_envs.reset()
                states_and_masks = self.get_states_and_masks(nll_envs)
                forward_logprob = gfn_model(*states_and_masks)[0]
                log_like_traj = forward_logprob[0, action_traj]
                log_sum_args.append(torch.tensor(log_like_traj))

        g = torch.tensor(log_sum_args)
        G_prob_like += torch.logsumexp(g, dim=0)

        return G_prob_like

    def kl_distance_true(self, true_graphs):
        """KL metric between learned posterior Q(G | D) and ground truth posterior distribution
        P(G*) over admissible structures G*.

        KL(P || Q) = P(G*) (log P(G*) - log Q(G | D))
        """
        P = true_graphs.shape[-1]
        num_graphs = true_graphs.shape[0]

        # compute log Q(G) for per-node GFN.
        log_Q_G = []
        for i in range(num_graphs):
            log_Q_Gp = 0
            g = true_graphs[i]
            if i % 5 == 00:
                print(f"{i}/{num_graphs} graph probs computed")
            for p in range(P):
                nll_envs = self.eval_envs[p]
                gfn_model = self.gfn_model[p]
                log_Q_Gp += self.compute_traj_likelihood(g[p], gfn_model, nll_envs)

            if torch.isinf(log_Q_Gp):
                print("ERROR in KL compute: Infinity!!")
            log_Q_G.append(log_Q_Gp)

        logprobs = torch.stack(log_Q_G)
        logprobs = logprobs.cpu().detach().numpy()
        np.save("logprobs.npy", logprobs)

        # compute discrete KL(P || Q)
        P_G = torch.tensor(1 / num_graphs)
        log_P_G = torch.log(P_G)
        kl = 0
        for i in range(len(log_Q_G)):
            kl += P_G * (log_P_G - log_Q_G[i])

        return kl

    def training_step(self, batch: Any, batch_idx: int):
        full_gfn_loss, full_graphs = [], []
        for p in range(self.hparams.n_dim):
            self.p = p
            graphs = self.envs[p].get_state()
            full_graphs.append(graphs)
            if (
                self.hparams.debug_use_shd_energy
                or self.hparams.analytic_use_simple_mse_energy
            ):
                gfn_opt = self.optimizers()
            else:
                gfn_opt, energy_opt = self.optimizers()

            # Train GFN
            if not self.current_epoch < self.hparams.pretraining_epochs:
                for i in range(self.hparams.gfn_freq):
                    p_gfn_loss = self.gflownet_step(batch, batch_idx)["loss"]
                full_gfn_loss.append(p_gfn_loss.to(batch[0]))

        if not self.current_epoch < self.hparams.pretraining_epochs:
            full_gfn_loss = torch.stack(full_gfn_loss).to(batch[0])  # weird error?
            gfn_loss = full_gfn_loss.sum()

            gfn_opt.zero_grad()
            self.manual_backward(gfn_loss)
            gfn_opt.step()

            self.log_dict(
                {
                    "train/gfn_loss": gfn_loss,
                    "train/avg_shd": torch.sum(torch.abs(batch[2][0] - graphs.mean(0))),
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        graphs = torch.stack(full_graphs).squeeze(2)
        graphs = torch.transpose(graphs, 0, 1)

        # Train energy if needed
        if (
            self.hparams.debug_use_shd_energy
            or self.hparams.analytic_use_simple_mse_energy
        ):
            return
        if not self.hparams.load_pretrain:
            if self.current_epoch < self.hparams.pretraining_epochs:
                graphs = torch.ones_like(graphs)
        for i in range(self.hparams.energy_freq):
            energy_loss = self.energy_step(
                graphs,
                batch,
                batch_idx,
                pretraining=self.current_epoch < self.hparams.pretraining_epochs,
            )
            energy_loss = energy_loss.mean()
            energy_opt.zero_grad()
            self.manual_backward(energy_loss)
            energy_opt.step()
        self.log_dict(
            {"train/energy_loss": energy_loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_eval_start(self):
        super().on_eval_start()
        full_graphs_mean = []
        for p in range(self.hparams.n_dim):
            graphs = self.eval_envs[p].get_state()
            full_graphs_mean.append(graphs.mean(0))
            if (
                not self.hparams.debug_use_shd_energy
                and not self.hparams.analytic_use_simple_mse_energy
            ):
                print("best_mse", self.energy_model.best_mse)
                print("mse_mean_data", self.energy_model.mse_mean_data)

                if self.hparams.analytic_use_simple_mse_energy:
                    weights, biases = self.energy_model.structure_model.layers[
                        0
                    ].hyper_layer(graphs)
                else:
                    weights, biases = [], []
                    for p in range(self.n_dim):
                        w, b = (
                            self.energy_model.structure_model[p]
                            .layers[0]
                            .hyper_layer(graphs.squeeze(1))
                        )
                        weights.append(w)
                        biases.append(b)
                    weights = torch.stack(weights)
                    biases = torch.stack(biases)
                print(weights.mean(0).squeeze(), biases.mean(0).squeeze())
        graphs_mean = torch.stack(full_graphs_mean)
        print("Val Graphs")
        print("----------")
        print(graphs_mean)
        print("----------")

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        super().eval_step(batch, batch_idx, prefix)
        full_graphs, full_optimal_graphs = [], []
        with torch.no_grad():
            self.optimal_eval_envs = self.sample_forward(
                envs=self.hparams.eval_batch_size, steps=self.hparams.n_steps
            )

        for p in range(self.hparams.n_dim):
            per_node_graphs = self.eval_envs[p].get_state()

            full_graphs.append(per_node_graphs)
            if batch_idx == 0:
                per_node_optimal_graphs = self.optimal_eval_envs[p].get_state()
                full_optimal_graphs.append(per_node_optimal_graphs)

        graphs = torch.stack(full_graphs).squeeze(2)
        graphs = torch.transpose(graphs, 0, 1)

        # want to only compute once
        # compute KL
        if batch_idx == 0:
            if prefix == "test":
                print(prefix)
                GC = batch[-1]
                # for RNA-velocity dataset
                if GC.shape[-1] == 5:
                    true_graphs = self.compute_grn_modes(GC[0])
                # for synthetic dataset
                else:
                    true_graphs = self.compute_true_modes(GC[0])
                self.kl_div = self.kl_distance_true(true_graphs)

        if batch_idx == 0:
            optimal_graphs = torch.stack(full_optimal_graphs).squeeze(2)
            optimal_graphs = torch.transpose(optimal_graphs, 0, 1)
            print("Optimal Edges Val Graphs")
            print("----------")
            print(optimal_graphs.mean(0))
            print("----------")
            print(self.energy_model(graphs, batch)[:10])
        if not self.hparams.debug_use_shd_energy:
            with torch.no_grad():
                print(self.energy_model.likelihood(graphs, batch)[:10])
        print("----------")
        print("Single sampled graph e.g.")
        print(graphs[0])
        graph_metrics = self.eval_graph_metrics(batch, batch_idx, prefix, graphs)
        return graph_metrics

    def configure_optimizers(self):
        if self.hparams.analytic_use_simple_mse_energy:
            gfn_opt = torch.optim.Adam(
                [
                    {"params": self.gfn_model.parameters()},
                    {"params": self.Z},
                ],
                lr=self.hparams.lr / 1,
            )
        else:
            gfn_opt = torch.optim.Adam(
                [
                    {"params": self.gfn_model.parameters()},
                    {"params": self.Z},
                ],
                lr=self.hparams.lr / 0.1,
            )
        if (
            self.hparams.debug_use_shd_energy
            or self.hparams.analytic_use_simple_mse_energy
        ):
            return gfn_opt
        energy_opt = torch.optim.Adam(
            self.energy_model.parameters(), lr=self.hparams.lr
        )
        return gfn_opt, energy_opt


class PerNodeParallelLinearTrainableCausalGraphGFlowNetModule(
    PerNodeParallelTrainableCausalGraphGFlowNetModule
):
    def __init__(
        self,
        dm_conf,
        bias: bool = True,
        debug_use_true_weights: bool = False,
        **kwargs,
    ) -> None:
        n_dim = dm_conf.p
        if debug_use_true_weights:
            bias = False
            structural_eq_model = LinearStructuralEquationModel(dm_conf.p, bias=bias)
        else:
            structural_eq_model = LinearStructuralEquationModel(dm_conf.p, bias=bias)
        super().__init__(
            n_dim=n_dim,
            structural_eq_model=structural_eq_model,
            **kwargs,
        )
        self.save_hyperparameters(
            ignore=["dm_conf", "structural_eq_model"], logger=False
        )


class PerNodeParallelHyperTrainableCausalGraphGFlowNetModule(
    PerNodeParallelTrainableCausalGraphGFlowNetModule
):
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
        structural_eq_model = torch.nn.ModuleList()
        for _ in range(n_dim):
            structural_eq_model.append(
                HyperStructuralEquationModel(
                    [n_dim, *dims], hyper=hyper, hyper_args=hyper_args
                )
            )
        super().__init__(
            n_dim=n_dim,
            structural_eq_model=structural_eq_model,
            **kwargs,
        )
        self.save_hyperparameters(
            ignore=["dm_conf", "structural_eq_model"], logger=False
        )


class ParallelHyperTrainableCausalGraphGFlowNetModule(
    ParallelTrainableCausalGraphGFlowNetModule
):
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
