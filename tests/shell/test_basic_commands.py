import pytest

from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""


def test_fast_dev_run():
    """Test running for 1 train, val and test batch."""
    command = ["train.py", "logger=csv", "++trainer.fast_dev_run=true"]
    run_command(command)


@pytest.mark.slow
def test_cpu():
    """Test running 1 epoch on CPU."""
    command = [
        "train.py",
        "logger=csv",
        "trainer.max_epochs=1",
        "trainer.check_val_every_n_epoch=1",
        "trainer=cpu",
    ]
    run_command(command)


# use RunIf to skip execution of some tests, e.g. when no gpus are available
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_gpu():
    """Test running 1 epoch on GPU."""
    command = [
        "train.py",
        "logger=csv",
        "trainer.max_epochs=1",
        "trainer.check_val_every_n_epoch=1",
        "trainer=gpu",
    ]
    run_command(command)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_mixed_precision():
    """Test running 1 epoch with pytorch native automatic mixed precision (AMP)."""
    command = [
        "train.py",
        "logger=csv",
        "trainer.max_epochs=1",
        "trainer.check_val_every_n_epoch=1",
        "trainer=gpu",
        "++trainer.precision=16",
    ]
    run_command(command)

