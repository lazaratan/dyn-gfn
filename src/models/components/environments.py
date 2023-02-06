"""GFlowNet environments."""
from abc import abstractmethod
from itertools import chain, combinations, permutations, product
from typing import Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")


class GFlowNetEnv:
    """Implements a GFlowNet environment.

    A GFlowNet environment is similar to any interactive reinforcement learning environment in that
    it must provide reset, step, reward, and done functions.
    """

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def forward_mask(self):
        raise NotImplementedError

    def backward_mask(self):
        raise NotImplementedError

    def done(self):
        raise NotImplementedError

    def reward(self, state: StateType):
        raise NotImplementedError

    @property
    def terminal_index(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActType) -> Tuple[StateType, bool]:
        """A step performs an action on the environment, it returns a tuple containing a state and
        a done flag."""
        raise NotImplementedError


def get_true_probs(num_vars, num_edge_types, reward_fn):
    """Iterate over all possible graphs of size num_vars x num_vars with all possible edge types to
    compute the reward for each possible state. This function is extremely expensive so should only
    be call for very small graphs. The number of states scales as:

    (num_edge_types+1) ** (num_vars**2)

    E.g. with 2 edge types and 4 variables, we need to evaluate 43046721 states.
    """

    def powerset(iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def to_tuple(x):
        return tuple(np.array(x.flatten(), dtype=int))

    all_states = {}
    total_reward = 0
    all_possible_edges = [
        [divmod(j, num_vars) for j in edges] for edges in powerset(range(num_vars**2))
    ]
    for edge_list in range(all_possible_edges):
        for edge_labels in product({0, 1}, repeat=len(edge_list)):
            state = np.zeros((num_vars, num_vars, num_edge_types), dtype=int)
            for k, (i, j) in zip(edge_labels, edge_list):
                state[i, j, k] = 1
            all_states[to_tuple(state)] = np.exp(reward_fn(state))
            total_reward += all_states[to_tuple(state)]
    for k, v in all_states.items():
        all_states[k] = v / total_reward
    return all_states


class DenseMultiGraph(GFlowNetEnv):
    """Simple graph sampling environment.

    No restriction on the graph, so self loops, cycles etc are allowed.
    """

    def __init__(
        self,
        num_vars=10,
        num_edge_types=2,
        reward=None,
        compute_true_probs: bool = False,
    ):
        self.num_vars = num_vars
        self.num_edge_types = num_edge_types
        self.reward = reward
        self.true_state_probabilities = None
        self.reset()
        if compute_true_probs:
            if self.reward is None:
                raise ValueError("Need a reward function to compute true probabilities")
            else:
                if num_vars > 4:
                    raise ValueError("Too many edges")
                else:
                    self.true_state_probabilities = get_true_probs(
                        self.num_vars, self.num_edge_types, self.reward
                    )

    @staticmethod
    def to_trinary(x):
        """State is represented as a (num_var, num_var, num_edge) tensor.

        This simple helper function converts the state to a weighted adjacency matrix with edge
        weights either -1 or 1.
        """
        return (x * (np.array([-1, 1])[None, None, :])).sum(axis=-1)

    def reset(self):
        self.state = np.zeros((self.num_vars, self.num_vars, self.num_edge_types))
        self.action_space_mask = np.ones(
            self.num_vars**2 * self.num_edge_types + 1, dtype=int
        )
        self.done = False
        self._action_lookup = np.arange(self.state.flatten().shape[0]).reshape(
            self.state.shape
        )

    @property
    def forward_mask(self):
        forward_mask = np.ones_like(self.state, dtype=int)
        # get all cells that don't have an edge
        flat_mask = self.state.reshape((-1, self.num_edge_types)).max(axis=1) == 0
        mask = flat_mask.reshape((self.num_vars, self.num_vars, 1))
        forward_mask *= mask
        forward_mask = np.concatenate(
            [forward_mask.flatten(), np.array([1 - int(self.done)])]
        )
        if self.done:
            return None
        else:
            return forward_mask

    @property
    def backward_mask(self):
        backward_mask = np.array(self.state.copy().flatten(), dtype=int)
        return backward_mask

    def get_state_input(self):
        return self.state

    def state_probabilities(self):
        return self.true_state_probabilities

    @property
    def terminal_index(self):
        return self.num_vars**2 * self.num_edge_types

    def step(self, action):
        if self.done:
            # if already done, do nothing
            pass
        elif action == self.terminal_index:
            # if terminal action is played, set done and return nothing
            self.done = True
        else:
            # the action lookup is a little confusing when you have
            # multiple edge types. The simplest solution that I could
            # think of is have a simple index array that is exactly the
            # same shape as the state which we can use for np.where().
            # Using this approach, each state element is uniquely associated
            # with a single action.
            self.state[np.where(self._action_lookup == int(action))] = 1
        return self.state, self.done


class StructuralODEEnv(DenseMultiGraph):
    def __init__(
        self,
        num_vars=10,
        num_edge_types=2,
        compute_true_probs: bool = False,
    ):
        super().__init__(
            num_vars,
            num_edge_types,
            reward=self.reward,
            compute_true_probs=compute_true_probs,
        )

    def reward(self, G):
        from .reward import ODEReward

        A = torch.eye(self.num_vars)
        A[1, 0] = 1
        A[0, 1] = -1
        # A = torch.eye(self.num_vars)
        R = ODEReward(A, n_cells=1, sigma=0.5)
        with torch.no_grad():
            G = DenseMultiGraph.to_trinary(G)
            return float(R.forward(torch.tensor(G).float()))


class GraphEnvs(GFlowNetEnv):
    """GraphEnvs is similar to GraphEnv except that it paramterizes a set of graphs.

    The state is initialized with zero edges. Taking a step in this environment
    adds an edge to the graph. This means that the environment has a maximum of
    n^2 steps each with n^2+1 possible actions (including terminate).

    Clearly many paths lead to the same graph as there are only 2^(n^2)
    possible graphs.
    """

    def __init__(self, n_graphs, num_vars, device):
        self.n_graphs = n_graphs
        self.num_vars = num_vars
        self.device = device
        self.reset()

    def reset(self):
        self.state = torch.zeros(
            (self.n_graphs, self.num_vars, self.num_vars), device=self.device
        )
        self.action_space_mask = torch.ones(
            self.n_graphs, self.num_vars**2 + 1, dtype=int, device=self.device
        )
        self.done = torch.zeros(self.n_graphs, dtype=bool, device=self.device)

    @property
    def forward_mask(self):
        return self.action_space_mask

    @property
    def backward_mask(self):
        backward_mask = 1 - self.action_space_mask[:, :-1]
        return backward_mask

    def get_state_input(self):
        return self.get_state()

    def get_state(self):
        return self.state

    @property
    def terminal_index(self):
        return self.num_vars**2

    def step(self, action):
        full_actions = np.ones_like(self.done.cpu(), dtype=int) * self.terminal_index
        full_actions[~self.done.cpu()] = action.cpu()
        done_actions = full_actions == self.terminal_index
        self.done[done_actions] = True
        self.action_space_mask[np.arange(self.n_graphs), full_actions] = 0
        valid_actions = np.arange(self.n_graphs)[~done_actions]
        i, j = np.divmod(full_actions[~done_actions], self.num_vars)
        self.state[valid_actions, i, j] = 1
        return self.state, self.done

    def list_states(self, n_steps: Optional[int] = None):
        """List all possible states.

        Args:
            n_steps: if supplied, then only return states with that number of actions.
        """

        def int_to_bits(x, bits=None, dtype=torch.uint8):
            assert not (
                x.is_floating_point() or x.is_complex()
            ), "x isn't an integer type"
            if bits is None:
                bits = x.element_size() * 16
            mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
            states = x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=dtype)
            if n_steps is None or n_steps < 0:
                return states
            return states[states.sum(dim=1) == n_steps]

        ints = torch.arange(2 ** (self.num_vars**2), dtype=torch.int32)
        bits = int_to_bits(ints, bits=self.num_vars**2, dtype=torch.float)
        return bits.reshape(-1, self.num_vars, self.num_vars)

    def compute_test_set(
        self, n_steps: Optional[int] = None, GC: Optional[torch.Tensor] = None
    ):
        if GC is None or self.num_vars <= 4:
            return self.list_states(n_steps)

        # Get a list of states including GC with more and fewer edges, lets start with just removing and adding all of the edges.

        # First add all the possible edges
        GCnp = GC.cpu().detach().numpy().flatten().astype(bool)
        list_of_states = []
        tmp = GCnp
        for nz in np.nonzero(GCnp)[0]:
            tmp = tmp.copy()
            tmp[nz] = False
            list_of_states.append(tmp)
        tmp = GCnp
        for zero in np.nonzero(~GCnp)[0]:
            tmp = tmp.copy()
            tmp[zero] = True
            list_of_states.append(tmp)
        print(
            " eval shape",
            np.array(list_of_states, dtype=float)
            .reshape(-1, self.num_vars, self.num_vars)
            .shape,
        )
        return (
            torch.tensor(list_of_states, dtype=float)
            .reshape(-1, self.num_vars, self.num_vars)
            .type_as(GC)
        )


class GraphFwdBckEnvs(GraphEnvs):
    """GraphFwdBckEnvs is similar to GraphEnvs except allows for taking backwards action sapce
    steps.

    In the context of GFN training, this environment allows for building a graph sequentially by
    sampling actions from the forward log-probability. But also allows removal of edges (i.e.
    backwards steps) by sampling actions from the backward log-probability.
    """

    def __init__(self, n_graphs, num_vars, device):
        self.n_graphs = n_graphs
        self.num_vars = num_vars
        self.device = device
        self.reset()

    def reset(self):
        self.state = torch.zeros(
            (self.n_graphs, self.num_vars, self.num_vars), device=self.device
        )
        self.action_space_mask = torch.ones(
            self.n_graphs, self.num_vars**2 + 1, dtype=int, device=self.device
        )
        self.done = torch.zeros(self.n_graphs, dtype=bool, device=self.device)

    def back_step(self, action):
        full_actions = np.ones_like(self.done.cpu(), dtype=int) * self.terminal_index
        full_actions[~self.done.cpu()] = action.cpu()
        done_actions = full_actions == self.terminal_index
        self.done[done_actions] = True
        self.action_space_mask[np.arange(self.n_graphs), full_actions] = 0
        valid_actions = np.arange(self.n_graphs)[~done_actions]
        i, j = np.divmod(full_actions[~done_actions], self.num_vars)
        self.state[valid_actions, i, j] = 0
        return self.state, self.done


class PerNodeGraphEnvs(GFlowNetEnv):
    """PerNodeGraphEnvs is similar to GraphEnvs except it construct an evniroment to build graphs
    sequentially for 1 node a time.

    Graphs are built for a repsective node by adding directed edges between the node and its
    children.
    """

    def __init__(self, n_graphs, num_vars, device):
        self.n_graphs = n_graphs
        self.num_vars = num_vars
        self.device = device
        self.reset()

    def reset(self):
        self.state = torch.zeros((self.n_graphs, 1, self.num_vars), device=self.device)
        self.action_space_mask = torch.ones(
            self.n_graphs, self.num_vars + 1, dtype=int, device=self.device
        )
        self.done = torch.zeros(self.n_graphs, dtype=bool, device=self.device)

    @property
    def forward_mask(self):
        return self.action_space_mask

    @property
    def backward_mask(self):
        backward_mask = 1 - self.action_space_mask[:, :-1]
        return backward_mask

    def get_state_input(self):
        return self.get_state()

    def get_state(self):
        return self.state

    @property
    def terminal_index(self):
        return self.num_vars

    def step(self, action):
        full_actions = np.ones_like(self.done.cpu(), dtype=int) * self.terminal_index
        full_actions[~self.done.cpu()] = action.cpu()
        done_actions = full_actions == self.terminal_index
        self.done[done_actions] = True
        self.action_space_mask[np.arange(self.n_graphs), full_actions] = 0
        valid_actions = np.arange(self.n_graphs)[~done_actions]
        i, j = np.divmod(full_actions[~done_actions], self.num_vars)
        self.state[valid_actions, i, j] = 1
        return self.state, self.done

    def list_states(self, n_steps: Optional[int] = None):
        """List all possible states.

        Args:
            n_steps: if supplied, then only return states with that number of actions.
        """

        def int_to_bits(x, bits=None, dtype=torch.uint8):
            assert not (
                x.is_floating_point() or x.is_complex()
            ), "x isn't an integer type"
            if bits is None:
                bits = x.element_size() * 16
            mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
            states = x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=dtype)
            if n_steps is None or n_steps < 0:
                return states
            return states[states.sum(dim=1) == n_steps]

        ints = torch.arange(2 ** (self.num_vars), dtype=torch.int32)
        bits = int_to_bits(ints, bits=self.num_vars, dtype=torch.float)
        return bits.reshape(-1, self.num_vars, 1)

    def compute_test_set(
        self, n_steps: Optional[int] = None, GC: Optional[torch.Tensor] = None
    ):
        if GC is None or self.num_vars <= 4:
            return self.list_states(n_steps)

        # Get a list of states including GC with more and fewer edges, lets start with just removing and adding all of the edges.

        # First add all the possible edges
        GCnp = GC.cpu().detach().numpy().flatten().astype(bool)
        list_of_states = []
        tmp = GCnp
        for nz in np.nonzero(GCnp)[0]:
            tmp = tmp.copy()
            tmp[nz] = False
            list_of_states.append(tmp)
        tmp = GCnp
        for zero in np.nonzero(~GCnp)[0]:
            tmp = tmp.copy()
            tmp[zero] = True
            list_of_states.append(tmp)
        print(
            " eval shape",
            np.array(list_of_states, dtype=float)
            .reshape(-1, self.num_vars, self.num_vars)
            .shape,
        )
        return (
            torch.tensor(list_of_states, dtype=float)
            .reshape(-1, self.num_vars, self.num_vars)
            .type_as(GC)
        )

    def action_trajectoreis(self, g):
        action_ids = (g == 1).squeeze().nonzero().squeeze()
        if torch.sum(g) > 1:
            trajectories = list(set(permutations(action_ids.tolist())))
            trajectories = torch.tensor(trajectories)
        else:
            trajectories = torch.tensor([g.shape[0]])
        return trajectories


class PerNodeNegLogLikeGraphEnvs(GFlowNetEnv):
    """PerNodeGraphEnvs used to compute NLL.

    This class computes possible trajectories used to construct a G.
    """

    def __init__(self, graphs, device):
        self.graphs = graphs
        self.n_graphs = graphs.shape[0]
        self.num_vars = graphs.shape[2]
        self.device = device
        self.reset()

    def reset(self):
        self.state = torch.zeros((self.n_graphs, 1, self.num_vars), device=self.device)
        self.action_space_mask = torch.ones(
            self.n_graphs, self.num_vars + 1, dtype=int, device=self.device
        )
        self.done = torch.zeros(self.n_graphs, dtype=bool, device=self.device)

    @property
    def forward_mask(self):
        return self.action_space_mask

    @property
    def backward_mask(self):
        backward_mask = 1 - self.action_space_mask[:, :-1]
        return backward_mask

    def get_state_input(self):
        return self.get_state()

    def get_state(self):
        return self.state

    @property
    def terminal_index(self):
        return self.num_vars

    def old_action_trajectoreis(self):
        self.trajectories = []
        for i in range(self.n_graphs):
            g = self.graphs[i]
            action_ids = (g == 1).squeeze().nonzero().squeeze()
            if torch.sum(g) > 1:
                trajectories = set(permutations(action_ids.tolist()))
                trajectories = torch.tensor(list(trajectories))
            else:
                trajectories = torch.tensor([])
            self.trajectories.append(trajectories)

    def action_trajectoreis(self, g):
        action_ids = (g == 1).squeeze().nonzero().squeeze()
        if torch.sum(g) > 1:
            trajectories = set(permutations(action_ids.tolist()))
            trajectories = torch.tensor(list(trajectories))
        else:
            trajectories = torch.tensor([])
        return trajectories

    def step(self, action):
        full_actions = np.ones_like(self.done.cpu(), dtype=int) * self.terminal_index
        full_actions[~self.done.cpu()] = action.cpu()
        done_actions = full_actions == self.terminal_index
        self.done[done_actions] = True
        self.action_space_mask[np.arange(self.n_graphs), full_actions] = 0
        valid_actions = np.arange(self.n_graphs)[~done_actions]
        i, j = np.divmod(full_actions[~done_actions], self.num_vars)
        self.state[valid_actions, i, j] = 1
        return self.state, self.done


class PerNodeFwdBckGraphEnvs(PerNodeGraphEnvs):
    def __init__(self, n_graphs, num_vars, device):
        self.n_graphs = n_graphs
        self.num_vars = num_vars
        self.device = device
        self.reset()

    def reset(self):
        self.state = torch.zeros((self.n_graphs, 1, self.num_vars), device=self.device)
        self.action_space_mask = torch.ones(
            self.n_graphs, self.num_vars + 1, dtype=int, device=self.device
        )
        self.done = torch.zeros(self.n_graphs, dtype=bool, device=self.device)

    def back_step(self, action):
        full_actions = np.zeros_like(self.done.cpu(), dtype=int) * self.terminal_index
        full_actions[~self.done.cpu()] = action.cpu()
        done_actions = full_actions == self.terminal_index
        self.done[done_actions] = True
        self.action_space_mask[np.arange(self.n_graphs), full_actions] = 0
        valid_actions = np.arange(self.n_graphs)[~done_actions]
        i, j = np.divmod(full_actions[~done_actions], self.num_vars)
        self.state[valid_actions, i, j] = 0
        return self.state, self.done


class GraphEnvsTerminal(GraphEnvs):
    def reset(self):
        super().reset()
        self.n_actions = (
            torch.ones(self.n_graphs, dtype=int, device=self.device)
            * self.num_vars**2
        )
        self.state = torch.zeros(
            (self.n_graphs, self.num_vars, self.num_vars), device=self.device
        )
        self.action_space_mask = torch.ones(
            self.n_graphs, self.num_vars**2, dtype=int, device=self.device
        )
        self.done = torch.zeros(self.n_graphs, dtype=bool, device=self.device)

    @property
    def backward_mask(self):
        backward_mask = 1 - self.action_space_mask
        return backward_mask

    def step(self, action):
        full_actions = np.ones_like(self.done.cpu(), dtype=int) * self.terminal_index
        full_actions[~self.done.cpu()] = action.cpu()
        self.done[self.n_actions == 0] = True

        i, j = np.divmod(full_actions[~self.done], self.num_vars)
        self.state[np.arange(self.n_graphs), i, j] = 1

        # Mask out actions that have been taken
        self.action_space_mask[np.arange(self.n_graphs), full_actions] = 0
        self.n_actions -= 1
        return self.state, self.done


class GraphEnv(GFlowNetEnv):
    """GraphEnv creates a graph environment where the state is an adjacency matrix.

    The state is initialized with zero edges. Taking a step in this environment
    adds an edge to the graph. This means that the environment has a maximum of
    n^2 steps each with n^2+1 possible actions (including terminate).

    Clearly many paths lead to the same graph as there are only 2^(n^2)
    possible graphs.
    """

    def __init__(self, num_vars, device):
        self.num_vars = num_vars
        self.device = device
        self.reset()

    def reset(self):
        self.state = torch.zeros((self.num_vars, self.num_vars), device=self.device)
        self.action_space_mask = torch.ones(
            self.num_vars**2 + 1, dtype=int, device=self.device
        )
        self.done = False

    @property
    def forward_mask(self):
        if self.done:
            return None
        return self.action_space_mask

    @property
    def backward_mask(self):
        backward_mask = 1 - self.action_space_mask[:-1]
        return backward_mask

    def get_state_input(self):
        return self.state

    @property
    def terminal_index(self):
        return self.num_vars**2

    def step(self, action):
        if self.done:
            # if already done, return nothing
            pass
        elif action == self.terminal_index:
            # if terminal action is played, set done and return nothing
            self.done = True
        else:
            # mask the selected action
            self.action_space_mask[action] = 0
            # we know action is < num_vars ** 2 so we can map the
            # action index -> i,j in the adjacency matrix
            i, j = divmod(int(action), self.num_vars)
            self.state[i, j] = 1
        return self.state, self.done

    def list_states(self):
        def int_to_bits(x, bits=None, dtype=torch.uint8):
            assert not (
                x.is_floating_point() or x.is_complex()
            ), "x isn't an integer type"
            if bits is None:
                bits = x.element_size() * 16
            mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
            return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=dtype)

        ints = torch.arange(2 ** (self.num_vars**2), dtype=torch.int32)
        bits = int_to_bits(ints, bits=self.num_vars**2, dtype=torch.float)
        return bits.reshape(-1, self.num_vars, self.num_vars)


class DenseGraph(GFlowNetEnv):
    """Simple graph sampling environment.

    No restriction on the graph, so self loops, cycles etc are allowed.
    """

    def __init__(self, num_vars=10):
        self.num_vars = num_vars
        self.reset()

    def reset(self):
        self.state = np.zeros((self.num_vars, self.num_vars))
        self.action_space_mask = np.ones(self.num_vars**2 + 1, dtype=int)
        self.done = False

    @property
    def forward_mask(self):
        forward_mask = self.action_space_mask.copy()
        if self.done:
            return None
        else:
            return forward_mask

    @property
    def backward_mask(self):
        backward_mask = 1 - self.action_space_mask[:-1]
        return backward_mask

    def get_state_input(self):
        return self.state

    @property
    def terminal_index(self):
        return self.num_vars**2

    def step(self, action):
        if self.done:
            # if already done, return nothing
            pass
        elif action == self.terminal_index:
            # if terminal action is played, set done and return nothing
            self.done = True
        else:
            # mask the selected action
            self.action_space_mask[action] = 0
            # we know action is < num_vars ** 2 so we can map the
            # action index -> i,j in the adjacency matrix
            i, j = divmod(int(action), self.num_vars)
            self.state[i, j] = 1
        return self.state, self.done


class HyperGrid(GFlowNetEnv):
    r"""HyperGrid environment from the original GFlowNet paper.

    This environment is a simple environment where the possible actions at each
    step are an increment in one of d dimensions. The reward is designed so
    that there are 2^d equally weighted modes at each "corner" of the
    hypergrid. This is designed to test the multi-modal sampling quality of
    sampling algorithms.

    The reward function is:
    $ R(x) = R_0 + R_1 \prod_i \mathbb{I}(0.25 < |x_i / H - 0.5|) + R_2 \prod_i \mathbb{I}(0.3 < |x_i / H -0.5| < 0.4)$
    """

    def __init__(
        self,
        horizon: int = 8,
        ndim: int = 2,
        R0: float = 1e-3,
        R1: float = 0.5,
        R2: float = 2.0,
    ) -> None:
        """
        Args:
            horizon (int): edge length of the hypercube.
            ndim (int): dimensionality of the hypercube.
            R0, R1, R2 (float): parameters controlling the base reward.
        """
        self.horizon = horizon
        self.ndim = ndim
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reset()

        j = np.zeros((horizon,) * ndim + (ndim,))
        for i in range(ndim):
            jj = np.linspace(0, horizon - 1, horizon)
            for _ in range(i):
                jj = jj[:, None]
                j[..., i] = jj

        self.truelr = self.reward(j)
        self.true_dist = self.truelr.flatten()
        self.true_dist /= self.true_dist.sum()

    def reset(self):
        self.done = False
        self.state = np.zeros(self.ndim, dtype=int)
        self._forward_mask_array = np.ones(self.ndim + 1, dtype=int)
        self._backward_mask_array = np.zeros(self.ndim, dtype=int)

    def state_probabilities(self):
        list_of_states = (
            np.array(np.meshgrid(*[np.arange(self.horizon)] * self.ndim))
            .reshape(self.ndim, -1)
            .T
        )
        rewards = self.reward(list_of_states)
        tot_reward = rewards.sum()
        state_dict = {tuple(s): r / tot_reward for s, r in zip(list_of_states, rewards)}
        return state_dict

    def list_states(self):
        n_states = self.horizon**self.ndim
        if n_states > 10000:
            print(f"Warning: computing {n_states} states may take a long time")
        list_of_states = (
            np.array(np.meshgrid(*[np.arange(self.horizon)] * self.ndim))
            .reshape(self.ndim, -1)
            .T
        )
        return list_of_states

    def get_state_input(self):
        return (
            F.one_hot(torch.tensor(self.state).long()[:, None], self.horizon)
            .flatten()
            .numpy()
        )

    @property
    def forward_mask(self):
        forward_mask = self._forward_mask_array.copy()
        if self.done:
            return None
        else:
            return forward_mask

    @property
    def backward_mask(self):
        return self._backward_mask_array.copy()

    def _update_forward_mask(self):
        self._forward_mask_array[:-1] = 1 * (self.state + 1 < self.horizon)

    def reward(self, x):
        ax = abs(x / (self.horizon - 1) * 2 - 1)
        return (
            (ax > 0.5).prod(-1) * self.R1
            + ((ax < 0.8) * (ax > 0.6)).prod(-1) * self.R2
            + self.R0
        )

    @property
    def terminal_index(self):
        return self.state.shape[0]

    def step(self, action):
        if self.done:
            # if already done, do nothing
            pass
        elif action == self.terminal_index:
            # if terminal action is played, set done and return nothing
            self.done = True
        else:
            # update backwards mask with the selected action
            self._backward_mask_array[action] = 1
            # update state
            self.state[action] += 1
            self.state = np.minimum(self.state, self.horizon)
            self._update_forward_mask()
        return self.state, self.done


class TorchEnvWrapper(GFlowNetEnv):
    """TorchEnvWrapper wraps an environment that returns numpy values with torch values."""

    def __init__(self, env, device):
        self.env = env
        self.device = device

    def reset(self):
        self.env.reset()

    def cast(self, arr, dtype=int):
        if arr is None:
            return arr
        return torch.tensor(arr, device=self.device, dtype=dtype)

    def get_state_input(self):
        return self.cast(self.env.get_state_input())

    @property
    def forward_mask(self):
        return self.cast(self.env.forward_mask)

    @property
    def backward_mask(self):
        return self.cast(self.env.backward_mask)

    def step(self, action):
        np_state, done = self.env.step(action)
        return self.cast(np_state, dtype=float), done

    @property
    def terminal_index(self):
        return self.env.terminal_index

    @property
    def state(self):
        return self.cast(self.env.state, dtype=float)

    def list_states(self):
        return self.cast(self.env.list_states(), dtype=float)


class EnvConstructor:
    def __init__(self, env, **kwargs):
        if isinstance(env, str):
            if env == "ODE":
                env = StructuralODEEnv
            elif env == "hyper":
                env = HyperGrid
            else:
                raise ValueError(f"Unknown environment {env}")
        self.env = env
        self.kwargs = kwargs

    def __call__(self):
        self.env(**self.kwargs)


class TorchHyperGrid(GFlowNetEnv):
    r"""HyperGrid environment from the original GFlowNet paper.

    This environment is a simple environment where the possible actions at each
    step are an increment in one of d dimensions. The reward is designed so
    that there are 2^d equally weighted modes at each "corner" of the
    hypergrid. This is designed to test the multi-modal sampling quality of
    sampling algorithms.

    The reward function is:
    $ R(x) = R_0 + R_1 \prod_i \mathbb{I}(0.25 < |x_i / H - 0.5|) + R_2 \prod_i \mathbb{I}(0.3 < |x_i / H -0.5| < 0.4)$
    """

    def __init__(
        self,
        horizon: int = 8,
        ndim: int = 2,
        R0: float = 1e-3,
        R1: float = 0.5,
        R2: float = 2.0,
        device="cpu",
    ) -> None:
        """
        Args:
            horizon (int): edge length of the hypercube.
            ndim (int): dimensionality of the hypercube.
            R0, R1, R2 (float): parameters controlling the base reward.
        """
        self.horizon = horizon
        self.ndim = ndim
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.device = device
        self.reset()

        j = np.zeros((horizon,) * ndim + (ndim,))
        for i in range(ndim):
            jj = np.linspace(0, horizon - 1, horizon)
            for _ in range(i):
                jj = jj[:, None]
                j[..., i] = jj

        self.truelr = self.reward(j)
        self.true_dist = self.truelr.flatten()
        self.true_dist /= self.true_dist.sum()
        self.device = device

    def reset(self):
        self.done = False
        self.state = torch.zeros(self.ndim, dtype=int, device=self.device)
        self._forward_mask_array = torch.ones(
            self.ndim + 1, dtype=int, device=self.device
        )
        self._backward_mask_array = torch.zeros(
            self.ndim, dtype=int, device=self.device
        )

    def list_states(self):
        n_states = self.horizon**self.ndim
        if n_states > 10000:
            print(f"Warning: computing {n_states} states may take a long time")
        list_of_states = (
            np.array(np.meshgrid(*[np.arange(self.horizon)] * self.ndim))
            .reshape(self.ndim, -1)
            .T
        )
        return self.cast(list_of_states, dtype=int)

    def get_state_input(self):
        state = F.one_hot(self.state[:, None], self.horizon).flatten().float()
        return state

    @property
    def forward_mask(self):
        forward_mask = self._forward_mask_array
        if self.done:
            return None
        else:
            return forward_mask

    @property
    def backward_mask(self):
        return self._backward_mask_array

    def _update_forward_mask(self):
        self._forward_mask_array[:-1] = 1 * (self.state + 1 < self.horizon)

    def reward(self, x):
        ax = abs(x / (self.horizon - 1) * 2 - 1)
        return (
            (ax > 0.5).prod(-1) * self.R1
            + ((ax < 0.8) * (ax > 0.6)).prod(-1) * self.R2
            + self.R0
        )

    @property
    def terminal_index(self):
        return self.state.shape[0]

    def step(self, action):
        if self.done:
            # if already done, do nothing
            pass
        elif action == self.terminal_index:
            # if terminal action is played, set done and return nothing
            self.done = True
        else:
            # update backwards mask with the selected action
            self._backward_mask_array[action] = 1
            # update state
            self.state[action] += 1
            self.state = torch.minimum(
                self.state, torch.tensor(self.horizon, device=self.device)
            )
            self._update_forward_mask()
        return self.state, self.done

    def cast(self, arr, dtype=int):
        if arr is None:
            return arr
        return torch.tensor(arr, device=self.device, dtype=dtype)
