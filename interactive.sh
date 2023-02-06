#!/bin/bash
module purge
module load miniconda/3
conda activate structure_recovery_gpu
eval "$(python train.py -sc install=bash)"
