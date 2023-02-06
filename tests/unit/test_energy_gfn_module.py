import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.energy_gfn_module import (
    FixedGraphGFlowNetModule,
    LinearTrainableCausalGraphGFlowNetModule,
)


@pytest.mark.parametrize(
    "gfn_class", [FixedGraphGFlowNetModule, LinearTrainableCausalGraphGFlowNetModule]
)
def test_hypergrid_train_base(gfn_class):
    gfn = gfn_class()
    trainer = pl.Trainer(
        max_epochs=1,
    )
    dl = DataLoader(np.arange(10)[:, None])
    v_dl = DataLoader(np.arange(100)[:, None])
    trainer.fit(gfn, train_dataloaders=dl, val_dataloaders=v_dl)
