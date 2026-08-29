#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -e

$(which python) -m accelerate.commands.launch \
--config_file configs/acc.yaml \
train_add_condition.py 

# $(which python) -m accelerate.commands.launch \
# --config_file configs/acc.yaml \
# train_add_denosing.py 