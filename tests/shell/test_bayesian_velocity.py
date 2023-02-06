"""Test the velocity portion of the code."""

import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@pytest.mark.slow
def test_bayesian_velocity_basic():
    command = [
        "train.py",
        "experiment=linear_bayes",
        "++trainer.fast_dev_run=true",
        "logger=csv",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        "trainer=cpu",
    ]
    run_command(command)


@pytest.mark.slow
def test_bayesian_linear_velocity():
    command = [
        "train.py",
        "experiment=linear_bayes",
        "logger=csv",
        "trainer=cpu",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        "model.proximal_lambda=0.0001",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize("hyper", ["mlp", "per_graph", "invariant"])
def test_beaysian_velocity_hyper_cpu(hyper):
    command = [
        "train.py",
        "experiment=linear_bayes",
        "logger=csv",
        "trainer=cpu",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        f"model.hyper={hyper}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
@pytest.mark.parametrize("hyper", ["mlp", "per_graph", "invariant"])
def test_bayesian_velocity_hyper(hyper):
    command = [
        "train.py",
        "experiment=linear_bayes",
        "logger=csv",
        "trainer=gpu",
        "model.l1_reg=0.001",
        "model.l2_reg=0.001",
        f"model.hyper={hyper}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
