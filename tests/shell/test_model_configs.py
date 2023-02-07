"""Tests for model configurations."""
import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@pytest.mark.slow
@pytest.mark.parametrize("experiment", ["per_node_linear_tcg", "per_node_sigmoid_tcg"])
def test_gfn_experiments_linear(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "trainer=cpu",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment",
    [
        "hyper_per_node_linear_tcg",
        "hyper_per_node_sigmoid_tcg",
    ],
)
def test_gfn_graph_experiments_hyper(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "model.load_pretrain=False",
        "trainer=cpu",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment",
    [
        "linear_bayes",
        "linear_svgd",
        "linear_sigmoid_bayes",
        "linear_sigmoid_svgd",
    ],
)
def test_bayes_drift_experiments_linear(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "trainer=cpu",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment",
    [
        "hyper_bayes",
        "hyper_svgd",
        "hyper_sigmoid_bayes",
        "hyper_sigmoid_svgd",
    ],
)
def test_bayes_drift_experiments_hyper(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "trainer=cpu",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
