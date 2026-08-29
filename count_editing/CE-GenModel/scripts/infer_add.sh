#!/bin/bash
set -e

echo "--------------run addition --------------------"
python inference_add.py \
--condition_types subject fill \
--denoising_lora_dir ./ckpt/CE-Gen_ckpt/DenoisingAdd \
--denoising_lora_name subject_fill_union \
--denoising_lora_weight 1.0 \
--condition_lora_dir ./ckpt/CE-Gen_ckpt/ConditionAdd \
--test_dir ./examples \
--exam_size 512 \
--work_dir ./output_add \
--turn 1 \
--use_mask \
--mask_padding_value 0 \
--pretrained_model_name_or_path /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/ckpt/FLUX.1-schnell \
--transformer /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/ckpt/FLUX.1-schnell/transformer

