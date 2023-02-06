import os
from typing import Optional, Tuple, Union

import numpy as np
import scanpy as sc
import torch
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from src import utils

log = utils.get_logger(__name__)


class RNAVelocityDataModule(LightningDataModule):
    def __init__(
        self,
        adata_path: Optional[str] = None,
        data_dir: str = "data/",
        p: Optional[int] = 3,
        train_val_test_split: Union[
            int, Tuple[int, int, int], Tuple[float, float, float]
        ] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        if adata_path is None:
            adata_path = "/h/lazar/data/adata_unidentified_velocity.h5ad"
        adata = sc.read_h5ad(adata_path)
        if "fast_dev_run" in kwargs and kwargs["fast_dev_run"]:
            log.info("fast_dev_run detected, reducing dataset size")
            train_val_test_split = (1, 1, 1)
        if not isinstance(train_val_test_split, int) and isinstance(
            train_val_test_split[0], float
        ):
            # Split according to proportions
            assert np.isclose(sum(train_val_test_split), 1)
            train_frac, val_frac, test_frac = train_val_test_split
            train_num = round(train_frac * adata.shape[0])
            val_num = round(val_frac * adata.shape[0])
            test_num = adata.shape[0] - train_num - val_num
            assert train_num >= 0 and val_num >= 0 and test_num >= 0
            train_val_test_split = (train_num, val_num, test_num)
        if p == 3:
            genes = ["CDK1", "CDC25A", "CDC25C"]
            gc = [[1, 0, 1], [1, 1, 0], [0, 0, 1]]
        elif p == 5:
            genes = ["CDK1", "CDC25A", "CDC25C", "MCM2", "MCM5"]
            gc = [
                [1, 0, 1, 0, 0],
                [1, 1, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [-2, -2, -2, -2, -2],
                [-2, -2, -2, -2, -2],
            ]
        else:
            raise ValueError("p={p} is not supported")
        adata_sub = adata[:, genes]
        genes = adata_sub.var.index
        x = adata_sub.layers["Ms"]
        y = adata_sub.layers["velocity"]
        ss = StandardScaler().fit(x)
        self.data = torch.tensor(ss.transform(x), dtype=torch.float32)
        self.velocity = torch.tensor(ss.transform(y), dtype=torch.float32)
        self.GC = torch.tensor(gc, dtype=torch.float32).repeat(self.data.shape[0], 1, 1)
        self.split_dataset(train_val_test_split, [self.data, self.velocity, self.GC])

    def split_dataset(self, train_val_test_split, tensors):
        dataset = TensorDataset(*tensors)

        if isinstance(train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

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
