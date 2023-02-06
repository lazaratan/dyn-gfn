import torch
from torch import nn


class MLPFlow(nn.Module):
    """Simple MLP architecture for modelling the probabilities in a GFlowNet.

    A Flow function has between 1 and 3 heads: forward_log_prob, backward_log_pro, misc. Forward
    parametrizes a distribution over possible forward actions. Backward parametrizes a distribution
    over possible backwards actions. this can be replaced with a uniform backwards strategy with
    the `uniform_backwards` flag. Misc is used by the detailed balance loss to parameterize the
    flow. The MLP first transforms the state to a common representation then applies up to three
    small heads to parameterize the flow.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        misc_out_dim: int = 0,
        hidden_dim: int = 64,
        uniform_backwards: bool = False,
    ):
        super().__init__()
        self.rep = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.forward_prob = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        self.stop = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1)
        )
        self.out_dim = out_dim
        self.uniform_backwards = uniform_backwards
        self.has_misc = misc_out_dim > 0
        if not self.uniform_backwards:
            self.backward_prob = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        if self.has_misc:
            self.misc = nn.Sequential(nn.Linear(hidden_dim, misc_out_dim))

    def _masked_softmax(self, x, mask, dim=1):
        return (x - (1 - mask) * 1e8).log_softmax(dim)

    def forward(self, x, forward_mask, backward_mask):
        x = x.reshape((x.shape[0], -1))

        # get state representation
        representation = self.rep(x)

        # compute backward probabilities
        if backward_mask.shape[0] > 0:
            if self.uniform_backwards:
                flattened = torch.zeros(x.shape[0], self.out_dim).type_as(x)
            else:
                flattened = self.backward_prob(representation).reshape(x.shape[0], -1)
            backward_log_prob = self._masked_softmax(flattened, backward_mask)
        else:
            # start state has no backward mask
            backward_log_prob = None

        # compute forward probabilities
        action_log_prob = self.forward_prob(representation).reshape(x.shape[0], -1)

        # set pooling by averaging across row / col representations (TODO:
        # attention / Transformer would be better)
        stop_log_prob = self.stop(representation)

        # concat because we have to choose between taking an action and stopping
        forward_log_prob = self._masked_softmax(
            torch.cat([action_log_prob, stop_log_prob], dim=1), forward_mask
        )
        if not self.has_misc:
            return forward_log_prob, backward_log_prob
        else:
            return forward_log_prob, backward_log_prob, self.misc(representation)


class LinearTransformerFlow(nn.Module):
    """Linear Transformer architecture for modelling the probabilities in a GFlowNet.

    A Flow function has between 1 and 3 heads: forward_log_prob, backward_log_pro, misc. Forward
    parametrizes a distribution over possible forward actions. Backward parametrizes a distribution
    over possible backwards actions. this can be replaced with a uniform backwards strategy with
    the `uniform_backwards` flag. Misc is used by the detailed balance loss to parameterize the
    flow. The Linear Transformer first transforms the state to a common representation then applies
    up to three small heads to parameterize the flow. The Transformer architecture is invariant to
    the order of the inputs, i.e. since G is represented as a set of edges.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        misc_out_dim: int = 0,
        hidden_dim: int = 64,
        embed_dim: int = 128,
        uniform_backwards: bool = False,
    ):
        from fast_transformers.builders import TransformerEncoderBuilder

        super().__init__()
        self.num_var = int(in_dim ** (1 / 2))
        self.embed = nn.Embedding(num_embeddings=in_dim * 2, embedding_dim=embed_dim)

        # Create the builder for our transformers
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=4,
            query_dimensions=int(embed_dim / 2),
            value_dimensions=int(embed_dim / 2),
            feed_forward_dimensions=embed_dim * 2,
            attention_type="linear",
        )

        self.rep = builder.get()

        self.forward_prob = builder.get()
        self.forward_linear = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # self.stop = nn.Sequential(
        #    nn.Linear(embed_dim*2, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1)
        # )
        # self.stop_linear = nn.Sequential(nn.Linear(out_dim, 1))
        self.stop = builder.get()
        self.stop_linear = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.out_dim = out_dim
        self.uniform_backwards = uniform_backwards
        self.has_misc = misc_out_dim > 0

        if not self.uniform_backwards:
            self.backward_prob = builder.get()
            self.backward_linear = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
            )

        if self.has_misc:
            self.misc = builder.get()
            self.misc_linear = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
            )

        # create edges as pair of indices (source, target)
        indices = torch.arange(self.num_var**2)
        sources, targets = torch.tensor(indices // self.num_var), torch.remainder(
            indices, self.num_var
        )
        self.edges = torch.stack((sources, targets + self.num_var), axis=1)

    def _masked_softmax(self, x, mask, dim=1):
        return (x - (1 - mask) * 1e8).log_softmax(dim)

    def forward(self, x, forward_mask, backward_mask):
        adjacency = x
        x = x.reshape((x.shape[0], -1))

        adjacency = adjacency.reshape((self.num_var**2, -1))

        # embed states
        embeddings = self.embed(self.edges.to(x).int())
        embeddings = embeddings.reshape(self.num_var**2, -1)

        # mask out edges
        src_embed = torch.einsum("ij,jb->ijb", embeddings.T, adjacency).T

        # get state representation
        representation = self.rep(src_embed)

        # compute backward probabilities
        if backward_mask.shape[0] > 0:
            if self.uniform_backwards:
                flattened = torch.zeros(x.shape[0], self.out_dim).type_as(x)
            else:
                flattened = self.backward_prob(representation)
                flattened = self.backward_linear(flattened).reshape(x.shape[0], -1)
            backward_log_prob = self._masked_softmax(flattened, backward_mask)
        else:
            # start state has no backward mask
            backward_log_prob = None

        # compute forward probabilities
        action_log_prob = self.forward_prob(representation)
        action_log_prob = self.forward_linear(action_log_prob).reshape(x.shape[0], -1)

        # set pooling by averaging across row / col representations (TODO:
        # attention / Transformer would be better)
        stop_log_prob = self.stop(representation)
        stop_log_prob = torch.mean(stop_log_prob, dim=1)
        stop_log_prob = self.stop_linear(stop_log_prob)

        # concat because we have to choose between taking an action and stopping
        forward_log_prob = self._masked_softmax(
            torch.cat([action_log_prob, stop_log_prob], dim=1), forward_mask
        )
        if not self.has_misc:
            return forward_log_prob, backward_log_prob
        else:
            return (
                forward_log_prob,
                backward_log_prob,
                self.misc_linear(self.misc(representation)),
            )
