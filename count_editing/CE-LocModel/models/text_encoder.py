import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", output_dim=128, freeze_backbone=True):
        super().__init__()
        
        # 1. Load Pre-trained CLIP from Hugging Face
        # We use the "Text Model" part of CLIP only
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.backbone = CLIPTextModel.from_pretrained(model_name)
        
        # 2. Freeze the backbone (Optional but Recommended)
        # CLIP is already very smart. We don't want to destroy its weights 
        # while training our simple bounding box predictor.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3. Projection Layer
        # CLIP base outputs 512 dims. We project this to your desired 'output_dim' (e.g. 128)
        # to match the size of your visual features.
        self.hidden_size = self.backbone.config.hidden_size # Usually 512
        self.projection = nn.Linear(self.hidden_size, output_dim)
        self.activation = nn.Mish() # Matches the rest of your architecture

    def forward(self, text_list):
        """
        text_list: A list of strings, e.g. ["mug", "bottle", "plate"]
        Returns: Tensor [Batch, output_dim]
        """
        device = self.projection.weight.device
        
        # 1. Tokenize
        # Padding=True ensures all sequences in batch are same length
        # Truncation=True cuts off super long text
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        # 2. Pass through CLIP
        # We generally use 'pooler_output' which represents the whole sentence
        # Guard was inverted upstream: `not ... .requires_grad` disabled grad for
        # an UNfrozen backbone (silently making freeze_backbone=False a no-op, as
        # no backbone param would ever receive a gradient) and enabled it for a
        # frozen one. Grad should follow requires_grad, not negate it.
        backbone_trainable = next(self.backbone.parameters()).requires_grad
        with torch.set_grad_enabled(backbone_trainable):
            outputs = self.backbone(**inputs)
        
        # The pooler_output is the embedding of the [EOS] token
        # Shape: [Batch, 512]
        pooled_output = outputs.pooler_output
        
        # 3. Project to embedding size
        return self.activation(self.projection(pooled_output))