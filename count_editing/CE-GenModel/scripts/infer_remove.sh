#!/bin/bash

echo "--------------run removing --------------------"
python inference_remove.py \
--condition_types subject fill \
--denoising_lora_dir ./ckpt/CE-Gen_ckpt/DenoisingRemove \
--denoising_lora_name subject_fill_remove_union \
--denoising_lora_weight 1.0 \
--test_dir ./examples \
--exam_size 512 \
--work_dir ./output_remove \
--pretrained_model_name_or_path /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/ckpt/FLUX.1-schnell \
--transformer /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/ckpt/FLUX.1-schnell/transformer

