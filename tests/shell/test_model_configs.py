"""Tests for model configurations."""
import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@pytest.mark.slow
def test_bayes_full():
    command = [
        "train.py",
        "logger=csv",
        "model=bayesian_velocity",
        "model.gamma=0.999",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_gfn():
    command = [
        "train.py",
        "logger=csv",
        "model=per_node_linear_tcg",
        "model.gamma=0.999",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        "model.proximal_lambda=0.001",
        "+model.proximal_eta=2",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
@pytest.mark.parametrize("model", ["bayesian_linear_velocity", "per_node_linear_tcg"])
def test_model_basic_gpu(model):
    """Test Basic models on GPU."""
    command = [
        "train.py",
        "logger=csv",
        "trainer=gpu",
        f"model={model}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["bayesian_linear_velocity", "per_node_linear_tcg"])
def test_model_basic_cpu(model):
    """Test Basic models on CPU."""
    command = [
        "train.py",
        "logger=csv",
        f"model={model}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment", ["per_node_linear_tcg", "per_node_sigmoid_tcg", "per_node_rna_tcg"]
)
def test_gfn_experiments_linear(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment",
    [
        "hyper_per_node_linear_tcg",
        "hyper_per_node_sigmoid_tcg",
        "hyper_per_node_rna_tcg",
    ],
)
def test_gfn_graph_experiments_hyper(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
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
        "linear_rna_bayes",
        "linear_rna_svgd",
    ],
)
def test_bayes_drift_experiments_linear(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
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
        "hyper_rna_bayes",
        "hyper_rna_svgd",
    ],
)
def test_bayes_drift_experiments_hyper(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
