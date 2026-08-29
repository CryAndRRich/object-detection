#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$(which python) -m accelerate.commands.launch \
--config_file configs/acc.yaml \
train_remove_denoising.py 