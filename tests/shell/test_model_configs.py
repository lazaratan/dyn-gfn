"""Tests for model configurations.

mlp
ngm
"""
import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@pytest.mark.slow
def test_mlp_full():
    command = [
        "train.py",
        "logger=csv",
        "model=mlp",
        "model.gamma=0.999",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_ngm():
    command = [
        "train.py",
        "logger=csv",
        "model=ngm",
        "model.gamma=0.999",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        "model.proximal_lambda=0.001",
        "+model.proximal_eta=2",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_sgld():
    """Test SGLD."""
    command = [
        "train.py",
        "logger=csv",
        "model=mlp",
        "model.optimizer=sgld",
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
def test_sgld_gpu():
    """Test SGLD on GPU."""
    command = [
        "train.py",
        "logger=csv",
        "trainer.gpus=1",
        "model=mlp",
        "model.optimizer=sgld",
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
@pytest.mark.parametrize("model", ["mlp", "ngm"])
def test_model_basic_gpu(model):
    """Test Basic models on GPU."""
    command = [
        "train.py",
        "logger=csv",
        "trainer.gpus=1",
        f"model={model}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["mlp", "ngm"])
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
@pytest.mark.parametrize("experiment", ["hyper_gfn", "tcg"])
def test_gfn_experiments(experiment):
    command = [
        "train.py",
        "logger=csv",
        f"experiment={experiment}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_gfn_graph_experiments_hyper():
    command = [
        "train.py",
        "logger=csv",
        "experiment=hyper_gfn",
        "model=graph_gfn",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_gfn_graph_experiments_linear():
    command = [
        "train.py",
        "logger=csv",
        "experiment=linear_tcg",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
def test_gfn_graph_experiments_shd():
    command = [
        "train.py",
        "logger=csv",
        "experiment=linear_tcg",
        "model.debug_use_shd_energy=True",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
