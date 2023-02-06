import os

import numpy as np
import pytest

from src.models.components.evaluation import (
    compare_graphs_bayesian_dist,
    compare_graphs_bayesian_shd,
    compare_single_graph_bayesian_shd,
)

PATH = os.path.dirname(os.path.realpath(__file__))


def test_compare_graphs_all_positive():
    """Make sure it still is a standard SHD when there is only a single graph supplied without
    ambiguity."""
    # p = 10, var = [0, 1, 0]
    gc = np.load(os.path.join(PATH, "sample_graph.npy"))
    example = np.maximum(0, gc)
    example[0, -1, -1] = 1
    example[0, -2, -1] = 1
    compare_single_graph_bayesian_shd(np.maximum(0, gc), example)


def test_compare_graphs_bayesian_shd():
    # p = 10, var = [0, 1, 0]
    gc = np.load(os.path.join(PATH, "sample_graph.npy"))
    example = np.maximum(0, gc)
    example[0, -1, -1] = 1
    example[0, -2, -1] = 1
    compare_single_graph_bayesian_shd(gc, example)


def test_compare_graphs_bayesian_dist():
    # p = 10, var = [0, 1, 0]
    gc = np.load(os.path.join(PATH, "sample_graph.npy"))
    example = np.maximum(0, gc)
    example[0, -1, -1] = 1
    example[0, 0, -1] = 0
    example[0, -2, -1] = 1
    example[0, 1, -1] = 0
    example_2 = np.maximum(0, gc)
    example_2[0, -1, -1] = 1
    example_3 = np.maximum(0, gc)
    example_3[0, -1, -2] = 1
    example_3[0, 0, -2] = 0
    example_3[0, -2, -3] = 1
    example_3[0, 1, -3] = 0
    example_4 = np.maximum(0, gc)
    example = np.concatenate([example, example_2, example_3, example_4], axis=0)
    assert example.shape[0] == 4
    (
        seen_admissible,
        total_admissible,
        unique_admissible,
    ) = compare_graphs_bayesian_dist(gc, example)
    assert total_admissible == 279936
    assert seen_admissible == 3
    assert unique_admissible == 3
