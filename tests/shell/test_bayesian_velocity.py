"""Test the velocity portion of the code."""

import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@pytest.mark.slow
def test_bayesian_velocity_basic():
    command = [
        "train.py",
        "experiment=linear_bayes",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "++trainer.fast_dev_run=true",
        "logger=csv",
        "model.l1_reg=0.001",
        "model.kl_reg=0.001",
        "model.n_ens=100",
        "model.eval_batch_size=100",
        "trainer=cpu",
    ]
    run_command(command)


@pytest.mark.slow
def test_bayesian_velocity_svgd():
    command = [
        "train.py",
        "experiment=linear_svgd",
        "model.n_ens=100",
        "model.eval_batch_size=100",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "++trainer.fast_dev_run=true",
        "logger=csv",
        "trainer=cpu",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize("hyper", ["mlp", "per_graph", "invariant"])
def test_bayesian_velocity_hyper_cpu(hyper):
    command = [
        "train.py",
        "experiment=hyper_bayes",
        "logger=csv",
        "trainer=cpu",
        "model.n_ens=100",
        "model.eval_batch_size=100",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        "model.l1_reg=0.001",
        "model.kl_reg=0.001",
        f"model.hyper={hyper}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)


@pytest.mark.slow
@pytest.mark.parametrize("hyper", ["mlp", "per_graph", "invariant"])
def test_bayesian_velocity_svgd_hyper_cpu(hyper):
    command = [
        "train.py",
        "experiment=hyper_svgd",
        "logger=csv",
        "trainer=cpu",
        "model.n_ens=100",
        "model.eval_batch_size=100",
        "datamodule.sparsity=0.99",
        "datamodule.p=10",
        "datamodule.vars_to_deidentify=[0]",
        f"model.hyper={hyper}",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
