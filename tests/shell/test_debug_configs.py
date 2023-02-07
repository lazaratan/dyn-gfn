import pytest

from tests.helpers.run_command import run_command

# @pytest.mark.slow
# def test_debug_default():
#    command = ["train.py", "logger=csv", "debug=default"]
#    run_command(command)


def test_debug_limit_batches():
    command = ["train.py", "logger=csv", "debug=limit_batches"]
    run_command(command)


def test_debug_overfit():
    command = ["train.py", "logger=csv", "debug=overfit", "trainer.max_epochs=2"]
    run_command(command)


# @pytest.mark.slow
# def test_debug_profiler():
#    command = ["train.py", "logger=csv", "debug=profiler"]
#    run_command(command)


def test_debug_step():
    command = ["train.py", "logger=csv", "debug=step"]
    run_command(command)


def test_debug_test_only():
    command = ["train.py", "logger=csv", "debug=test_only"]
    run_command(command)
