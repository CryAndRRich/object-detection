import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, feature_map):
        """
        Args:
            feature_map: (Batch, Channels, Height, Width)
        Returns:
            spatial_features: (Batch, Channels * 2) -> (x, y) for each channel
        """
        N, C, H, W = feature_map.shape
        
        # 1. Create coordinate grid
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feature_map.device),
            torch.linspace(-1, 1, W, device=feature_map.device)
        )
        pos_x = pos_x.reshape(H * W)
        pos_y = pos_y.reshape(H * W)
        
        # 2. Flatten feature map to (N, C, H*W)
        feature_flat = feature_map.reshape(N, C, -1)
        
        # 3. Softmax over spatial dimensions
        attention = F.softmax(feature_flat, dim=-1)
        
        # 4. Compute expected coordinates (Weighted Sum)
        expected_x = torch.sum(pos_x * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(pos_y * attention, dim=-1, keepdim=True)
        
        # 5. Concatenate and flatten
        expected_xy = torch.cat([expected_x, expected_y], dim=-1)
        return expected_xy.reshape(N, -1)