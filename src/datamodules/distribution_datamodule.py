from typing import Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from src import utils

from .components.tnet_dataset import SCData

log = utils.get_logger(__name__)


class DistributionDataModule(LightningDataModule):
    """Implements loader for datasets taking the form of a sequence of distributions over time."""

    def __init__(
        self,
        data_dir: str = "data/",
        system: str = "TREE",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        p: Optional[int] = None,
        T: int = 100,
        system_kwargs: dict = {},
        seed=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        dataset = SCData.factory(system, system_kwargs)
        self.data = dataset.get_data()
        self.labels = dataset.get_times()

        self.grn = torch.zeros((p, p), dtype=torch.float32)
        if hasattr(dataset, "grn"):
            self.grn = torch.tensor(dataset.grn, dtype=torch.float32)
        else:
            log.info("No GRN found")

        self.timepoint_data = [
            self.data[self.labels == lab] for lab in dataset.get_unique_times()
        ]
        self.min_count = min(len(d) for d in self.timepoint_data)
        self.nice_data = np.array([d[: self.min_count] for d in self.timepoint_data])
        self.nice_data = torch.tensor(self.nice_data, dtype=torch.float32).transpose(
            0, 1
        )
        # TODO add support for jagged
        self.times = torch.tensor(
            dataset.get_unique_times(), dtype=torch.float32
        ).repeat(self.min_count, 1)
        self.grn = self.grn.repeat(self.min_count, 1, 1)
        t = len(dataset.get_unique_times())
        self.even_times = torch.linspace(1, t, t).repeat(self.min_count, 1)
        dataset = TensorDataset(self.nice_data, self.even_times, self.grn)

        if isinstance(self.hparams.train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

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
