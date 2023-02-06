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
        "model.kl_reg=0.001",
        "trainer=cpu",
    ]
    run_command(command)


@pytest.mark.slow
def test_bayesian_velocity_svgd():
    command = [
        "train.py",
        "experiment=linear_svgd",
        "++trainer.fast_dev_run=true",
        "logger=csv",
        "trainer=cpu",
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
        "model.kl_reg=0.001",
        f"model.hyper={hyper}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize("hyper", ["mlp", "per_graph", "invariant"])
def test_beaysian_velocity_svgd_hyper_cpu(hyper):
    command = [
        "train.py",
        "experiment=linear_svgd",
        "logger=csv",
        "trainer=cpu",
        f"model.hyper={hyper}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


