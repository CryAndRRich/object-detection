# Project Title

This module is built upon the foundation of [UniCombine](https://huggingface.co/Xuan-World/UniCombine).

## 🔧 Environment Setup

To install the required dependencies, run the following commands:

```bash
conda create -n unicombine python=3.12
conda activate unicombine
pip install -r requirements.txt
```

Due to some issues with the _diffusers_ library, you need to update the code manually. You can find the location of your library by running:
```bash
pip show diffusers
```

Then, add the following entry to the `_SET_ADAPTER_SCALE_FN_MAPPING` dictionary located in the `diffusers/loaders/peft.py` file:
```python
"UniCombineTransformer2DModel": lambda model_cls, weights: weights
```


## Download Model Checkpoint

Place all the model weights in the `CE-GenModel/ckpt` directory.

**1. FLUX.1-schnell**
```bash
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ./ckpt/FLUX.1-schnell
```

**2. CE-GEN weight**
* [Download via Google Drive](https://drive.google.com/drive/folders/1UL5PblS5ELHPUv01Q1SNgLjIs-EHbVps?usp=sharing)


## 🧪 Testing

To evaluate the models, navigate to the generative model directory and execute the corresponding inference scripts:

### ➕ Adding Task
Run the following commands to test the object addition capabilities:
```bash
cd CE-GenModel
bash scripts/infer_add.sh
```
### ➕ Removing Task
Run the following commands to test the object removing capabilities:
```bash
cd CE-GenModel
bash scripts/infer_remove.sh
```

## Traing
### Adding task
1. Load FLUX.1-schnell-training-assistant-LoRA (optional) 

Download it if you want to train your LoRA on the FLUX-schnell.

```bash
huggingface-cli download ostris/FLUX.1-schnell-training-adapter --local-dir ./ckpt/FLUX.1-schnell-training-adapter
```

> Schnell is a step distilled model, meaning it can generate an image in just a few steps. 
> However, this makes it impossible to train on it directly because every step you train breaks down the compression more and more. 
> With this adapter enabled during training, that doesnt happen. 
> It is activated during the training process, and disabled during sampling. 
> After the LoRA is trained, this adapter is no longer needed.

2. Download Dataset
* **CE-130 Dataset:** [Download via Google Drive](https://drive.google.com/file/d/1iBoz4_rzd8hHnAuR3hF6WO5UTfUrQUZF/view?usp=drive_link)
* **CE-GEN train dataset**: [Download via Google Drive](https://drive.google.com/file/d/1iBoz4_rzd8hHnAuR3hF6WO5UTfUrQUZF/view?usp=sharing)

3. Fine tune condition subject LoRA
```bash
python train_add_condition.py
```

4. Train denoising LoRA
```bash
python train_add_denoising.py
```

### Removing task
1. Load FLUX.1-schnell-training-assistant-LoRA (optional) 

Download it if you want to train your LoRA on the FLUX-schnell.

```bash
huggingface-cli download ostris/FLUX.1-schnell-training-adapter --local-dir ./ckpt/FLUX.1-schnell-training-adapter
```

> Schnell is a step distilled model, meaning it can generate an image in just a few steps. 
> However, this makes it impossible to train on it directly because every step you train breaks down the compression more and more. 
> With this adapter enabled during training, that doesnt happen. 
> It is activated during the training process, and disabled during sampling. 
> After the LoRA is trained, this adapter is no longer needed.

2. Download Dataset
* **CE-130 Dataset:** [Download via Google Drive](https://drive.google.com/file/d/1iBoz4_rzd8hHnAuR3hF6WO5UTfUrQUZF/view?usp=drive_link)
* **CE-GEN train dataset**: [Download via Google Drive](https://drive.google.com/file/d/1iBoz4_rzd8hHnAuR3hF6WO5UTfUrQUZF/view?usp=sharing)

3. Train denoising LoRA
```bash
python train_remove_denoising.py
```


