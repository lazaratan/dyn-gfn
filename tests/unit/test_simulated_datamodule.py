import os

import numpy as np
import pytest
import torch

from src.datamodules.distribution_datamodule import DistributionDataModule
from src.datamodules.simulated_datamodule import (
    SimulatedDataModule,
    SimulatedVelocityDataModule,
    UnidentifiableSimulatedDataModule,
    UnidentifiableSimulatedVelocityDataModule,
    random_negative_definite_matrix,
    random_sparse_negative_definite_matrix,
)


@pytest.mark.parametrize("p", [10, 100])
@pytest.mark.parametrize("max_eig", [0, -1e-4, -1e-2])
@pytest.mark.parametrize("seed", [None, 42])
def test_random_negative_definite_matrix(p, max_eig, seed):
    arr = random_negative_definite_matrix(p, max_eig, seed)
    arr2 = random_negative_definite_matrix(p, max_eig, seed)
    arr3 = random_negative_definite_matrix(p, max_eig, seed=1)
    tolerance = 1e-5
    assert np.max(np.linalg.eigvalsh(arr)) <= max_eig + tolerance
    if seed is not None:
        assert np.allclose(arr, arr.T)
        assert np.allclose(arr, arr2)
        assert not np.allclose(arr, arr3)


@pytest.mark.parametrize("p", [10, 100])
@pytest.mark.parametrize("max_eig", [0, -1e-4, -1e-2])
@pytest.mark.parametrize("seed", [None, 42])
def test_random_sparse_negative_definite_matrix(p, max_eig, seed):
    arr = random_sparse_negative_definite_matrix(p, max_eig, sparsity=0.5, seed=seed)
    arr2 = random_sparse_negative_definite_matrix(p, max_eig, sparsity=0.5, seed=seed)
    arr3 = random_sparse_negative_definite_matrix(p, max_eig, sparsity=0.5, seed=1)
    tolerance = 1e-5
    assert np.max(np.linalg.eigvalsh(arr)) <= max_eig + tolerance
    if seed is not None:
        assert np.sum(np.abs(arr) > 0) < (0.6 * p**2)
        assert np.sum(np.abs(arr) > 0) > (0.4 * p**2)
        assert np.allclose(arr, arr.T)
        assert np.allclose(arr, arr2)
        assert not np.allclose(arr, arr3)


@pytest.mark.parametrize("system", ["linear", "lorenz", "rossler"])
@pytest.mark.parametrize("train_val_test_split", [[1, 1, 1], 1, 2])
@pytest.mark.parametrize("vars_to_deidentify", [[0], [0, 1, 0]])
def test_unidentifiable_velocity_dataset(
    tmpdir, system, train_val_test_split, vars_to_deidentify
):
    T = 100
    p = 10
    batch_size = 1
    datamodule = UnidentifiableSimulatedVelocityDataModule(
        system=system,
        T=T,
        burn_in=0,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        vars_to_deidentify=vars_to_deidentify,
        p=p,
        data_dir=tmpdir,
    )
    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert os.path.exists(os.path.join(tmpdir, datamodule.__class__.__name__))

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, dx, gc = batch

    assert len(x) == batch_size
    assert len(dx) == batch_size
    assert len(gc) == batch_size
    assert x.dtype == torch.float32
    assert torch.allclose(x[0][0] / x[0][-1], dx[0][0] / dx[0][-1])
    assert x.shape[1] == p
    assert dx.shape[1] == p
    assert dx.dtype == torch.float32
    assert gc.dtype == torch.int64


@pytest.mark.parametrize("system", ["linear", "lorenz", "rossler"])
@pytest.mark.parametrize("train_val_test_split", [[1, 1, 1], 1, 2])
@pytest.mark.parametrize("vars_to_deidentify", [[0], [0, 1, 0]])
def test_unidentifiable_dataset(
    tmpdir, system, train_val_test_split, vars_to_deidentify
):
    T = 100
    p = 10
    batch_size = 1
    datamodule = UnidentifiableSimulatedDataModule(
        system=system,
        T=T,
        burn_in=0,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        vars_to_deidentify=vars_to_deidentify,
        p=p,
        data_dir=tmpdir,
    )
    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert os.path.exists(os.path.join(tmpdir, datamodule.__class__.__name__))

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    data, times, gc = batch

    assert len(data) == batch_size
    assert len(times) == batch_size
    assert len(gc) == batch_size
    assert data.shape[1] == T
    assert data.shape[2] == p
    assert data.dtype == torch.float32
    assert times.dtype == torch.float32
    assert gc.dtype == torch.int64


@pytest.mark.parametrize(
    "system", ["linear", "lorenz", "rossler", "tumor", "glycolytic", "cardiovascular"]
)
@pytest.mark.parametrize("sigma", [0, 0.5, 1])
def test_generated_datamodule_long(tmpdir, system, sigma):
    if system == "tumor":
        pytest.xfail("tumor system is unstable")
    T = 5000
    train_val_test_split = 1
    batch_size = 1
    datamodule = SimulatedDataModule(
        system=system,
        T=T,
        sigma=sigma,
        burn_in=0,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        data_dir=tmpdir,
        seed=42,
    )

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert os.path.exists(os.path.join(tmpdir, datamodule.__class__.__name__))

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    data, times, gc = batch

    assert len(data) == batch_size
    assert len(times) == batch_size
    assert len(gc) == batch_size
    assert data.shape[1] == T
    assert torch.all(torch.isfinite(data))
    assert torch.all(data < 1e3)
    assert torch.all(data > -1e3)
    assert data.dtype == torch.float32
    assert times.dtype == torch.float32
    assert gc.dtype == torch.int64


@pytest.mark.parametrize(
    "system", ["linear", "lorenz", "rossler", "tumor", "glycolytic", "cardiovascular"]
)
@pytest.mark.parametrize("train_val_test_split", [[1, 1, 1], 1])
def test_generated_datamodule(tmpdir, system, train_val_test_split):
    T = 100
    batch_size = 1
    datamodule = SimulatedDataModule(
        system=system,
        T=T,
        burn_in=0,
        sigma=0.5,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        data_dir=tmpdir,
    )

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert os.path.exists(os.path.join(tmpdir, datamodule.__class__.__name__))

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    data, times, gc = batch

    assert len(data) == batch_size
    assert len(times) == batch_size
    assert len(gc) == batch_size
    assert data.shape[1] == T
    assert torch.all(torch.isfinite(data))
    assert data.dtype == torch.float32
    assert times.dtype == torch.float32
    assert gc.dtype == torch.int64


@pytest.mark.parametrize(
    "system", ["linear", "lorenz", "rossler", "tumor", "glycolytic", "cardiovascular"]
)
@pytest.mark.parametrize("train_val_test_split", [[1, 1, 1], 1])
def test_generated_velocity_datamodule(tmpdir, system, train_val_test_split):
    T = 100
    batch_size = 10
    datamodule = SimulatedVelocityDataModule(
        system=system,
        T=T,
        burn_in=0,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        data_dir=tmpdir,
    )

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert os.path.exists(os.path.join(tmpdir, datamodule.__class__.__name__))

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, dx, gc = batch

    assert len(x) == batch_size
    assert len(dx) == batch_size
    assert len(gc) == batch_size
    assert x.dtype == torch.float32
    assert dx.dtype == torch.float32
    assert gc.dtype == torch.int64


def test_unknown_system():
    with pytest.raises(ValueError, match=r"Unknown system: .*"):
        datamodule = SimulatedDataModule(system="asdf")


@pytest.mark.parametrize(
    "system,p",
    [
        ("linear", None),
        ("linear", 10),
        ("lorenz", None),
        ("lorenz", 10),
        ("rossler", None),
        ("rossler", 10),
        ("tumor", None),
        ("tumor", 5),
        ("glycolytic", None),
        ("glycolytic", 7),
        ("cardiovascular", None),
        ("cardiovascular", 4),
    ],
)
def test_parse_system_defaults(tmpdir, system, p):
    T = 100
    batch_size = 1
    datamodule = SimulatedDataModule(
        system=system,
        T=T,
        burn_in=0,
        train_val_test_split=1,
        batch_size=batch_size,
        p=p,
        data_dir=tmpdir,
    )

    batch = next(iter(datamodule.train_dataloader()))
    data, times, gc = batch

    REQUIRED_DIMS = {
        "tumor": 5,
        "glycolytic": 7,
        "cardiovascular": 4,
    }
    correct_p = REQUIRED_DIMS.get(system, 10)
    assert data.shape[2] == correct_p
    assert gc.shape[1] == correct_p
    assert gc.shape[2] == correct_p


@pytest.mark.parametrize("train_val_test_split", [1000, 1, [4000, 500, 500]])
# @pytest.mark.parametrize("train_val_test_split", [[1, 1, 1], 1])
def test_generated_distribution_datamodule(tmpdir, train_val_test_split):
    batch_size = 1
    datamodule = DistributionDataModule(
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        p=10,
        data_dir=tmpdir,
    )

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    data, times, gc = batch

    assert len(data) == batch_size
    assert len(times) == batch_size
    assert len(gc) == batch_size
    # assert data.shape[1] == T
    assert data.dtype == torch.float32
    assert times.dtype == torch.float32
    assert gc.dtype == torch.float32
