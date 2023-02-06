"""simulated_datamodule.py.

Implements a set of common time ODE and SDE systems:
    * lorenz
    * rossler
    * tumor p = 5
    * glycolytic p = 7
    * cardiovascular p = 4

Notes:
    Could be more efficient, right now we generate trajectories one at a time,
    could be parallelized.

    Currently duplicates some data in memory, specifically the ground truth
    causality mapping (GC) and the timepoints.  This allows for a separate GC
    for each time series, but may be inificient for many trajectories coming
    from the same system.
"""
import math
import os
import time
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import sdeint
import torch
from pytorch_lightning import LightningDataModule
from scipy.integrate import odeint
from torch.utils.data import DataLoader, TensorDataset, random_split

from src import utils

log = utils.get_logger(__name__)


def random_negative_definite_matrix(p, max_eig=-1e-4, seed=None):
    """Generates a random negative definite matrix.

    First generates a random symmetric matrix, calculates the eigenvalues, then
    subtracts the maximum - max_eig.

    Args:
        p: dimension of the matrix to create
        max_eig: maximum eigenvalue of negative definite matrix
        seed: randomness seed for repeatable behavior
    """
    if seed is not None:
        np.random.seed(seed)

    arr = np.random.randn(p, p)
    arr = arr + arr.T
    max_eig_val = np.max(np.linalg.eigvalsh(arr))
    arr -= np.eye(p) * (max_eig_val - max_eig)
    return arr


def random_sparse_negative_definite_matrix(p, max_eig=-1e-4, sparsity=0.5, seed=None):
    """Generates a random negative definite matrix.

    First generates a random symmetric matrix, calculates the eigenvalues, then
    subtracts the maximum - max_eig.

    Args:
        p: dimension of the matrix to create
        max_eig: maximum eigenvalue of negative definite matrix
        sparsity: sparsity of matrix to generate
        seed: randomness seed for repeatable behavior
    """
    if seed is not None:
        np.random.seed(seed)

    arr = np.random.randn(p, p) * 0.1
    arr = arr + arr.T
    sparsity_mask = np.triu(np.random.rand(p, p) > sparsity)
    sym_mask = sparsity_mask + sparsity_mask.T
    arr *= sym_mask
    max_eig_val = np.max(np.linalg.eigvalsh(arr))
    arr -= np.eye(p) * (max_eig_val - max_eig)
    return arr


def linear(x, t, A):
    return A @ x


def simulate_linear(
    p, T, sigma=0.5, delta_t=0.1, sd=0.0, burn_in=0, sparsity=0.5, A=None, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma] * p)

    x0 = np.random.normal(scale=1, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    if A is None:
        A = random_sparse_negative_definite_matrix(p, sparsity=sparsity, seed=42)
        A = A * 0.1
    lin = partial(linear, A=A)
    # X = odeint(lin, x0, t)
    X = sdeint.itoint(lin, GG, x0, t)

    GC = (np.abs(A) > 0).astype(np.int64)
    return X[burn_in:], GC


def sigmoid_linear(x, t, A):
    return 1.0 / (1.0 + np.exp(-(A @ x)))
    # return np.tanh(A @ x)


def simulate_sigmoid_linear(
    p, T, sigma=0.5, delta_t=1, sd=0.0, burn_in=0, sparsity=0.5, A=None, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma] * p)

    x0 = np.random.normal(scale=1, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    if A is None:
        A = random_sparse_negative_definite_matrix(p, sparsity=sparsity, seed=42)
        A = A * 0.1
    lin = partial(sigmoid_linear, A=A)
    # X = odeint(lin, x0, t)
    X = sdeint.itoint(lin, GG, x0, t)

    GC = (np.abs(A) > 0).astype(np.int64)
    return X[burn_in:], GC


def lorenz(x, t, F=5):
    """Partial derivatives for Lorenz-96 ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(
    p, T, sigma=0.5, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma] * p)

    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    lor = partial(lorenz, F=F)
    X = sdeint.itoint(lor, GG, x0, t)

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def lotkavolterra(x, t, r, alpha):
    """Partial derivatives for Lotka-Volterra ODE.
    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions"""
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = r[i] * x[i] * (1 - np.dot(alpha[i], x))

    return dxdt


def simulate_lotkavolterra(
    p, T, r, alpha, delta_t=0.1, sd=0.01, burn_in=1000, seed=None
):
    if seed is not None:
        np.random.seed(seed)
    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p) + 0.25
    x0 = np.array([0.0222, 0.0014, 0.0013, 0.0008])
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(
        lotkavolterra,
        x0,
        t,
        args=(
            r,
            alpha,
        ),
    )
    X += np.random.normal(scale=sd, size=(T + burn_in, p))
    # Set up Granger causality ground truth.
    GC = (alpha != 0) * 1
    np.fill_diagonal(GC, 1)
    return X[burn_in:], GC


def rossler(x, t, a=0, eps=0.1, b=4, d=2):
    """Partial derivatives for rossler ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    dxdt[0] = a * x[0] - x[1]
    dxdt[p - 2] = x[(p - 3)]
    dxdt[p - 1] = eps + b * x[(p - 1)] * (x[(p - 2)] - d)
    for i in range(1, p - 2):
        dxdt[i] = np.sin(x[(i - 1)]) - np.sin(x[(i + 1)])
    return dxdt


def simulate_rossler(
    p,
    T,
    sigma=0.5,
    a=0,
    eps=0.1,
    b=4,
    d=2,
    delta_t=0.05,
    sd=0.1,
    burn_in=1000,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        return np.diag([sigma] * p)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(rossler, x0, t, args=(a, eps, b, d))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))
    # X = sdeint.itoint(rossler, GG, x0, t)

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    GC[0, 0] = 1
    GC[0, 1] = 1
    GC[p - 2, p - 3] = 1
    GC[p - 1, p - 1] = 1
    GC[p - 1, p - 2] = 1
    for i in range(1, p - 2):
        # GC[i, i] = 1
        GC[i, (i + 1)] = 1
        GC[i, (i - 1)] = 1

    return 400 * X[burn_in:], GC


def tumor_vaccine(
    x,
    t,
    c2=300,
    t1=3,
    a0=0.1946,
    a1=0.3,
    c1=100,
    c3=300,
    delta0=0.00001,
    delta1=0.00001,
    d=0.0007,
    f=0.62,
    r=0.01,
):
    """Partial derivatives for rossler ODE."""

    dxdt = np.zeros(5)

    c0 = 1 / 369
    dxdt[0] = (
        a0 * x[0] * (1 - c0 * x[0])
        - delta0 * x[0] * x[2] / (1 + c1 * x[1])
        - delta0 * x[0] * x[4]
    )
    dxdt[1] = a1 * (x[0] ** 2) / (c2 + x[0] ** 2) - d * x[1]
    dxdt[2] = (
        f * x[2] * x[0] / (1 + c3 * x[0] * x[1])
        - r * x[2]
        - delta0 * x[3] * x[2]
        - delta1 * x[2]
    )
    dxdt[3] = r * x[2] - delta1 * x[3]

    if math.isclose(t, t1, abs_tol=0.5):
        dxdt[4] = 5000 - delta1 * x[4]
    else:
        dxdt[4] = -delta1 * x[4]

    return dxdt


def simulate_tumor(
    p, T, c2=300, t1=3, delta_t=0.05, sd=0.1, burn_in=0, seed=None, sigma=1.0
):
    assert p == 5
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.zeros(5)
    x0[0] = 3
    x0[1] = 0
    x0[2] = 100
    x0[3] = 0
    x0[4] = 0
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)

    def GG(x, t):
        return np.diag([sigma] * p)

    tumor = partial(tumor_vaccine, c2=c2, t1=t1)
    X = sdeint.itoint(tumor, GG, x0, t)

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    GC[0, 0] = 1
    GC[0, 1] = 1
    GC[p - 2, p - 3] = 1
    GC[p - 1, p - 1] = 1
    GC[p - 1, p - 2] = 1
    for i in range(1, p - 2):
        # GC[i, i] = 1
        GC[i, (i + 1)] = 1
        GC[i, (i - 1)] = 1

    return X[burn_in:], GC


def glycolytic(
    x,
    t,
    k1=0.52,
    K1=100,
    K2=6,
    K3=16,
    K4=100,
    K5=1.28,
    K6=12,
    K=1.8,
    kappa=13,
    phi=0.1,
    q=4,
    A=4,
    N=1,
    J0=2.5,
):
    """Partial derivatives for Glycolytic oscillator model.

    source:
    https://www.pnas.org/content/pnas/suppl/2016/03/23/1517384113.DCSupplemental/pnas.1517384113.sapp.pdf

    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions
    """
    dxdt = np.zeros(7)

    dxdt[0] = J0 - (K1 * x[0] * x[5]) / (1 + (x[5] / k1) ** q)
    dxdt[1] = (
        (2 * K1 * x[0] * x[5]) / (1 + (x[5] / k1) ** q)
        - K2 * x[1] * (N - x[4])
        - K6 * x[1] * x[4]
    )
    dxdt[2] = K2 * x[1] * (N - x[4]) - K3 * x[2] * (A - x[5])
    dxdt[3] = K3 * x[2] * (A - x[5]) - K4 * x[3] * x[4] - kappa * (x[3] - x[6])
    dxdt[4] = K2 * x[1] * (N - x[4]) - K4 * x[3] * x[4] - K6 * x[1] * x[4]
    dxdt[5] = (
        (-2 * K1 * x[0] * x[5]) / (1 + (x[5] / k1) ** q)
        + 2 * K3 * x[2] * (A - x[5])
        - K5 * x[5]
    )
    dxdt[6] = phi * kappa * (x[3] - x[6]) - K * x[6]

    return dxdt


def simulate_glycolytic(
    p, T, sigma=0.5, delta_t=0.001, sd=0.01, burn_in=0, seed=None, scale=True
):
    assert p == 7
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        return np.diag([sigma] * p)

    x0 = np.zeros(p)
    x0[0] = np.random.uniform(0.15, 1.6)
    x0[1] = np.random.uniform(0.19, 2.16)
    x0[2] = np.random.uniform(0.04, 0.2)
    x0[3] = np.random.uniform(0.1, 0.35)
    x0[4] = np.random.uniform(0.08, 0.3)
    x0[5] = np.random.uniform(0.14, 2.67)
    x0[6] = np.random.uniform(0.05, 0.1)

    # Use scipy to solve ODE.
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(glycolytic, x0, t)
    X += np.random.normal(scale=sd, size=(T + burn_in, 7))

    # For some reason SDEint does not work well for this system
    # X = sdeint.itoint(glycolytic, GG, x0, t)

    # Set up ground truth.
    GC = np.zeros((p, p), dtype=int)
    GC[0, :] = np.array([1, 0, 0, 0, 0, 1, 0])
    GC[1, :] = np.array([1, 1, 0, 0, 1, 1, 0])
    GC[2, :] = np.array([0, 1, 1, 0, 1, 1, 0])
    GC[3, :] = np.array([0, 0, 1, 1, 1, 1, 1])
    GC[4, :] = np.array([0, 1, 0, 0, 1, 1, 0])
    GC[5, :] = np.array([1, 1, 0, 0, 0, 1, 0])
    GC[6, :] = np.array([0, 0, 0, 1, 0, 0, 1])

    if scale:
        X = np.transpose(
            np.array(
                [
                    (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
                    for i in range(X.shape[1])
                ]
            )
        )

    return 10 * X[burn_in:], GC


def cardiovascular(x, t, I_ext=None, Rmod=None, Ca=4, Cv=111, tau=20, k=0.1838, Pas=70):
    """Partial derivatives for Glycolytic oscillator model.

    source:
    https://www.pnas.org/content/pnas/suppl/2016/03/23/1517384113.DCSupplemental/pnas.1517384113.sapp.pdf

    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions
    """
    if I_ext is None:
        I_ext = np.random.choice([-2, 0])
    if Rmod is None:
        Rmod = np.random.choice([0.5, 0])
    dxdt = np.zeros(4)

    def f(S, maxx=3, minn=0.66):
        return S * (maxx - minn) + minn

    def R(S, Rmod, Rmax=2.134, Rmin=0.5335):
        return S * (Rmax - Rmin) + Rmin + Rmod

    dxdt[0] = I_ext
    dxdt[1] = (1 / Ca) * ((x[1] - x[2]) / R(x[3], Rmod) - x[0] * f(x[3]))
    dxdt[2] = (1 / Cv) * (-Ca * dxdt[1] + I_ext)
    dxdt[3] = (1 / tau) * (1 - 1 / (1 + np.exp(-k * (x[1] - Pas))) - x[3])

    return dxdt


def simulate_cardiovascular(
    p,
    T,
    delta_t=0.001,
    sd=0.01,
    burn_in=0,
    seed=None,
    scale=True,
    sigma: float = 1.0,
):
    assert p == 4
    if seed is not None:
        np.random.seed(seed)

    x0 = np.zeros(4)
    x0[0] = np.random.uniform(90, 100)
    x0[1] = np.random.uniform(75, 85)
    x0[2] = np.random.uniform(3, 7)
    x0[3] = np.random.uniform(0.15, 0.25)

    I_ext, Rmod = np.random.choice([-2, 0]), np.random.choice([0.5, 0])

    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)

    def GG(x, t):
        return np.diag([sigma] * p)

    cardio = partial(cardiovascular, I_ext=I_ext, Rmod=Rmod)
    X = sdeint.itoint(cardio, GG, x0, t)

    # Set up ground truth.
    GC = np.zeros((4, 4), dtype=int)
    GC[0, :] = np.array([0, 0, 0, 0])
    GC[1, :] = np.array([1, 1, 1, 1])
    GC[2, :] = np.array([0, 1, 0, 0])
    GC[3, :] = np.array([0, 0, 1, 1])

    if scale:
        X = np.transpose(
            np.array(
                [
                    (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
                    for i in range(X.shape[1])
                ]
            )
        )

    return 10 * X[burn_in:], GC


class SimulatedDataModule(LightningDataModule):
    """Lorenz_96 data simulator.

    Simulates a Lorenz-type SDE with specified number of nodes for a specified amount of time.
    """

    SIMULATORS = {
        "linear": simulate_linear,
        "sigmoid_linear": simulate_sigmoid_linear,
        "lorenz": simulate_lorenz_96,
        # "lotka": simulate_lotkavolterra,
        "rossler": simulate_rossler,
        "tumor": simulate_tumor,
        "glycolytic": simulate_glycolytic,
        "cardiovascular": simulate_cardiovascular,
    }

    # Some systems have a required dimensionality
    REQUIRED_DIMS = {
        "tumor": 5,
        "glycolytic": 7,
        "cardiovascular": 4,
    }

    def __init__(
        self,
        data_dir: str = "data/",
        system: str = "lorenz",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        p: Optional[int] = None,
        T: int = 100,
        sigma: float = 1.0,
        sparsity: float = 0.5,
        delta_t: float = 0.05,
        sd: float = 0.0,
        system_kwargs: Optional[dict] = None,
        burn_in=1000,
        seed=None,
        **kwargs,
    ):
        """Initialize a simulated data object.

        Args:
            p: Optional[int] dimensionality of the system, defaults to 10 or required number
            train_val_test_split: int | Tuple[int, int, int] if a single int is
            supplied, creates a dataset without test splits. If a tuple of 3
            ints is supplied simulates a system with a specified train / val /
            test split sizes.
        """
        super().__init__()
        if system_kwargs is None:
            system_kwargs = {}
        if "fast_dev_run" in kwargs and kwargs["fast_dev_run"]:
            log.info("fast_dev_run detected, reducing dataset size")
            train_val_test_split = (1, 1, 1)
            T = 10
            burn_in = 10

        self.save_hyperparameters(logger=True)
        n = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(list(train_val_test_split))
        )
        name = f"{system}-{p}-{n}-{T}-{sigma}-{sparsity}-{delta_t}-{sd}-{system_kwargs}-{burn_in}-{seed}.pt"
        data_path = os.path.join(self.folder, name)
        if os.path.exists(data_path):
            log.info(f"Loading data from {data_path}")
            data_dict = torch.load(data_path)
            self.data, self.GC = data_dict["data"], data_dict["GC"]
        else:
            sim_fn, p = self.parse_system(system, p)
            if system == "linear" or system == "sigmoid_linear":
                A = random_sparse_negative_definite_matrix(
                    p=p, max_eig=-1e-2, sparsity=sparsity, seed=seed
                )
                system_kwargs["A"] = A
                print(A)
            start = time.time()
            trajectories = [
                sim_fn(
                    p,
                    T,
                    sigma=sigma,
                    delta_t=delta_t,
                    sd=sd,
                    burn_in=burn_in,
                    seed=seed + i if seed is not None else None,
                    **system_kwargs,
                )
                for i in range(n)
            ]
            self.data, self.GC = zip(*trajectories)
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.GC = torch.tensor(self.GC)
            end = time.time()
            log.info(
                f"Simulated dataset {self.data.shape} in {end - start:0.1f} seconds"
            )
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            torch.save({"data": self.data, "GC": self.GC}, data_path)
        # TODO this is what is used in previous work, but its off by one?
        self.times = torch.linspace(0, T, T).repeat(n, 1)
        # self.times = torch.linspace(0, 1, T).repeat(n, 1)
        self.split_dataset([self.data, self.times, self.GC])

    def split_dataset(self, tensors):
        dataset = TensorDataset(*tensors)

        if isinstance(self.hparams.train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def parse_system(self, system, p):
        if system not in self.SIMULATORS.keys():
            raise ValueError(
                f"Unknown system: {system} valid systems {self.SIMULATORS.keys()}"
            )
        req_p = self.REQUIRED_DIMS.get(system, None)
        if p is None:
            if req_p is None:
                p = req_p = 10
            else:
                p = req_p
        elif p != req_p and req_p is not None:
            log.warning(
                f"Supplied p: {p} is not equal to required p {req_p} for {system} system"
            )
            p = req_p

        return self.SIMULATORS[system], p

    @property
    def folder(self) -> str:
        return os.path.join(self.hparams.data_dir, self.__class__.__name__)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class UnidentifiableSimulatedDataModule(SimulatedDataModule):
    """takes a Simulated dataset and makes it unidentifiable."""

    def __init__(
        self,
        vars_to_deidentify: list = [],
        data_dir: str = "data/",
        system: str = "lorenz",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        p: Optional[int] = None,
        T: int = 100,
        sigma: float = 1.0,
        sparsity: float = 0.5,
        delta_t: float = 0.05,
        sd: float = 0.0,
        system_kwargs: Optional[dict] = None,
        burn_in=1000,
        seed=None,
        **kwargs,
    ):
        """Initialize a simulated data object.

        Args:
            p: Optional[int] dimensionality of the system, defaults to 10 or required number
            train_val_test_split: int | Tuple[int, int, int] if a single int is
            supplied, creates a dataset without test splits. If a tuple of 3
            ints is supplied simulates a system with a specified train / val /
            test split sizes.
        """
        super(SimulatedDataModule, self).__init__()
        if system_kwargs is None:
            system_kwargs = {}
        if "fast_dev_run" in kwargs and kwargs["fast_dev_run"]:
            log.info("fast_dev_run detected, reducing dataset size")
            train_val_test_split = (1, 1, 1)
            T = 10
            burn_in = 10
        self.save_hyperparameters(logger=True)
        p = p - len(vars_to_deidentify)
        n = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(list(train_val_test_split))
        )
        name = f"{system}-{p}-{n}-{T}-{sigma}-{sparsity}-{delta_t}-{sd}-{system_kwargs}-{burn_in}-{seed}.pt"
        data_path = os.path.join(self.folder, name)
        if os.path.exists(data_path):
            log.info(f"Loading data from {data_path}")
            data_dict = torch.load(data_path)
            self.data, self.GC = data_dict["data"], data_dict["GC"]
        else:
            sim_fn, p = self.parse_system(system, p)
            if system == "linear" or system == "sigmoid_linear":
                A = random_sparse_negative_definite_matrix(
                    p=p, max_eig=-1e-2, sparsity=sparsity, seed=seed
                )
                system_kwargs["A"] = A
                print(A)
            start = time.time()
            trajectories = [
                sim_fn(
                    p,
                    T,
                    sigma=sigma,
                    delta_t=delta_t,
                    sd=sd,
                    burn_in=burn_in,
                    seed=seed + i if seed is not None else None,
                    **system_kwargs,
                )
                for i in range(n)
            ]
            self.data, self.GC = zip(*trajectories)
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.GC = torch.tensor(self.GC)
            end = time.time()
            log.info(
                f"Simulated dataset {self.data.shape} in {end - start:0.1f} seconds"
            )
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            torch.save({"data": self.data, "GC": self.GC}, data_path)
        self.times = torch.linspace(0, T, T).repeat(n, 1)
        # self.times = torch.linspace(0, 1, T).repeat(n, 1)

        self.alter_data()

        self.split_dataset([self.data, self.times, self.GC])

    def alter_data(self):
        # Alter the data
        n_add = len(self.hparams.vars_to_deidentify)
        factors = np.random.rand(n_add)
        new_dims = []
        new_gc_dims = []
        for var, f in zip(self.hparams.vars_to_deidentify, factors):
            simulated = self.data[..., var]
            new_dims.append(simulated * f)
            new_gc_dims.append(self.GC[..., var])
        new_dims = torch.stack(new_dims, -1)
        self.data = torch.cat([self.data, new_dims], axis=-1)
        gc = torch.cat([self.GC, torch.stack(new_gc_dims, -1)], -1)

        # Lets hack this for now, include the parents encoded as (-1 - parent)
        # across the target rows.
        targets = torch.tensor(self.hparams.vars_to_deidentify)  # N_add
        targets = torch.reshape(targets, (1, -1, 1))
        self.GC = torch.cat(
            [gc, -torch.ones(gc.shape[0], n_add, self.hparams.p) - targets], 1
        ).type(torch.int64)


class SimulatedVelocityDataModule(SimulatedDataModule):
    DRIFT_FUNCTIONS = {
        "linear": linear,
        "sigmoid_linear": sigmoid_linear,
        "lorenz": lorenz,
        # "lotka": simulate_lotkavolterra,
        "rossler": rossler,
        "tumor": tumor_vaccine,
        "glycolytic": glycolytic,
        "cardiovascular": cardiovascular,
    }

    def __init__(
        self,
        data_dir: str = "data/",
        system: str = "lorenz",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        p: Optional[int] = None,
        T: int = 100,
        sigma: float = 1.0,
        sparsity: float = 0.5,
        delta_t: float = 0.05,
        sd: float = 0.0,
        system_kwargs: Optional[dict] = None,
        burn_in=1000,
        seed=None,
        **kwargs,
    ):
        """Initialize a simulated data object with velocity information.

        Args:
            p: Optional[int] dimensionality of the system, defaults to 10 or required number
            train_val_test_split: int | Tuple[int, int, int] if a single int is
            supplied, creates a dataset without test splits. If a tuple of 3
            ints is supplied simulates a system with a specified train / val /
            test split sizes.
        """
        super(SimulatedDataModule, self).__init__()
        if system_kwargs is None:
            system_kwargs = {}
        if "fast_dev_run" in kwargs and kwargs["fast_dev_run"]:
            log.info("fast_dev_run detected, reducing dataset size")
            train_val_test_split = (1, 1, 1)
            T = 10
            burn_in = 10

        self.save_hyperparameters(logger=True)
        n = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(list(train_val_test_split))
        )
        name = f"{system}-{p}-{n}-{T}-{sigma}-{sparsity}-{delta_t}-{sd}-{system_kwargs}-{burn_in}-{seed}.pt"
        data_path = os.path.join(self.folder, name)
        if os.path.exists(data_path):
            log.info(f"Loading data from {data_path}")
            data_dict = torch.load(data_path)
            self.data, self.velocity, self.GC = (
                data_dict["data"],
                data_dict["velocity"],
                data_dict["GC"],
            )
            if system == "linear":
                print("A")
                print(data_dict["A"])
                print((np.abs(data_dict["A"]) > 0).sum())
                self.A = data_dict["A"]
        else:
            sim_fn, p = self.parse_system(system, p)
            if system == "linear" or system == "sigmoid_linear":
                A = random_sparse_negative_definite_matrix(
                    p=p, max_eig=-1e-2, sparsity=sparsity, seed=seed
                )
                system_kwargs["A"] = A
                print(A)
                print((np.abs(A) > 0).sum())
                self.A = A
            start = time.time()
            trajectories = [
                sim_fn(
                    p,
                    T,
                    sigma=sigma,
                    delta_t=delta_t,
                    sd=sd,
                    burn_in=burn_in,
                    seed=seed + i if seed is not None else None,
                    **system_kwargs,
                )
                for i in range(n)
            ]
            self.data, self.GC = zip(*trajectories)
            # Run the data through the velocity function again with specified gaussian noise.
            self.data = np.array(self.data)
            self.data = self.data.reshape(-1, p)
            drift_fn = partial(self.DRIFT_FUNCTIONS[system], **system_kwargs)
            self.drift = torch.tensor(
                np.array([drift_fn(d, 0) for d in self.data]), dtype=torch.float32
            )
            self.velocity = self.drift + torch.randn(self.drift.shape) * sigma
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.GC = torch.tensor(self.GC).repeat(T, 1, 1)
            end = time.time()
            log.info(
                f"Simulated dataset {self.data.shape} in {end - start:0.1f} seconds"
            )
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            d = {"data": self.data, "velocity": self.velocity, "GC": self.GC}
            if system == "linear" or system == "sigmoid_linear":
                d.update({"A": A})
            torch.save(d, data_path)
        # TODO this is what is used in previous work, but its off by one?
        self.split_dataset([self.data, self.velocity, self.GC])

    def split_dataset(self, tensors):
        dataset = TensorDataset(*tensors)

        if isinstance(self.hparams.train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=np.array(self.hparams.train_val_test_split) * self.hparams.T,
                generator=torch.Generator().manual_seed(42),
            )


class UnidentifiableSimulatedVelocityDataModule(SimulatedVelocityDataModule):
    """takes a Simulated dataset and makes it unidentifiable."""

    def __init__(
        self,
        vars_to_deidentify: list = [],
        data_dir: str = "data/",
        system: str = "lorenz",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        p: Optional[int] = None,
        T: int = 100,
        sigma: float = 1.0,
        sparsity: float = 0.5,
        delta_t: float = 0.05,
        sd: float = 0.0,
        system_kwargs: Optional[dict] = None,
        burn_in=1000,
        seed=None,
        **kwargs,
    ):
        """Initialize a simulated data object.

        Args:
            p: Optional[int] dimensionality of the system, defaults to 10 or required number
            train_val_test_split: int | Tuple[int, int, int] if a single int is
            supplied, creates a dataset without test splits. If a tuple of 3
            ints is supplied simulates a system with a specified train / val /
            test split sizes.
        """
        super(SimulatedDataModule, self).__init__()
        if system_kwargs is None:
            system_kwargs = {}
        if "fast_dev_run" in kwargs and kwargs["fast_dev_run"]:
            log.info("fast_dev_run detected, reducing dataset size")
            train_val_test_split = (1, 1, 1)
            T = 10
            burn_in = 10
        self.save_hyperparameters(logger=True)
        p = p - len(vars_to_deidentify)
        n = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(list(train_val_test_split))
        )
        name = f"{system}-{p}-{n}-{T}-{sigma}-{sparsity}-{delta_t}-{sd}-{system_kwargs}-{burn_in}-{seed}.pt"
        data_path = os.path.join(self.folder, name)
        if os.path.exists(data_path):
            log.info(f"Loading data from {data_path}")
            data_dict = torch.load(data_path)
            self.data, self.velocity, self.GC = (
                data_dict["data"],
                data_dict["velocity"],
                data_dict["GC"],
            )
            if system == "linear" or system == "sigmoid_linear":
                print("A")
                print(data_dict["A"])
        else:
            sim_fn, p = self.parse_system(system, p)
            if system == "linear" or system == "sigmoid_linear":
                print(seed)
                A = random_sparse_negative_definite_matrix(
                    p=p, max_eig=-1e-2, sparsity=sparsity, seed=seed
                )
                system_kwargs["A"] = A
                print(A)
            start = time.time()
            trajectories = [
                sim_fn(
                    p,
                    T,
                    sigma=sigma,
                    delta_t=delta_t,
                    sd=sd,
                    burn_in=burn_in,
                    seed=seed + i if seed is not None else None,
                    **system_kwargs,
                )
                for i in range(n)
            ]
            self.data, self.GC = zip(*trajectories)
            # Run the data through the velocity function again with specified gaussian noise.
            self.data = np.array(self.data)
            self.data = self.data.reshape(-1, p)
            drift_fn = partial(self.DRIFT_FUNCTIONS[system], **system_kwargs)
            self.drift = torch.tensor(
                np.array([drift_fn(d, 0) for d in self.data]), dtype=torch.float32
            )
            self.velocity = self.drift + torch.randn(self.drift.shape) * sigma
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.GC = torch.tensor(self.GC).repeat(T, 1, 1)
            end = time.time()
            log.info(
                f"Simulated dataset {self.data.shape} in {end - start:0.1f} seconds"
            )
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            d = {"data": self.data, "velocity": self.velocity, "GC": self.GC}
            if system == "linear" or system == "sigmoid_linear":
                d.update({"A": A})
            torch.save(d, data_path)
        self.alter_data()

        self.split_dataset([self.data, self.velocity, self.GC])

    def alter_data(self):
        # Alter the data
        n_add = len(self.hparams.vars_to_deidentify)
        # factors = np.random.rand(n_add) # CHANGE THIS TO CONSTANT be cosntant -1 np.ones(n_add)
        factors = -1 * np.ones(n_add)
        new_dims = []
        new_vel_dims = []
        new_gc_dims = []
        for var, f in zip(self.hparams.vars_to_deidentify, factors):
            simulated = self.data[..., var]
            simulated_vel = self.velocity[..., var]
            new_dims.append(simulated * f)
            new_vel_dims.append(simulated_vel * f)
            new_gc_dims.append(self.GC[..., var])
        new_dims = torch.stack(new_dims, -1)
        new_vel_dims = torch.stack(new_vel_dims, -1)
        self.data = torch.cat([self.data, new_dims], axis=-1)
        self.velocity = torch.cat([self.velocity, new_vel_dims], axis=-1)
        gc = torch.cat([self.GC, torch.stack(new_gc_dims, -1)], -1)

        # Lets hack this for now, include the parents encoded as (-1 - parent)
        # across the target rows.
        targets = torch.tensor(self.hparams.vars_to_deidentify)  # N_add
        targets = torch.reshape(targets, (1, -1, 1))
        self.GC = torch.cat(
            [gc, -torch.ones(gc.shape[0], n_add, self.hparams.p) - targets], 1
        ).type(torch.int64)
