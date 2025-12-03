import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # there is an issue with OpenMP, not sure if this is a macOS specific problem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests 
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from depth_anything_3.api import DepthAnything3

#====================================================
class DepthTeacher:

    #====================================================
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE") # base pretrained DepthAnythingV3
        self.model.to(self.device) # set to device
        self.image_to_tensor = transforms.transforms.ToTensor()

        # matches the one used in their code, not sure if this is dataset-dependent...
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # matches their model
        self.patch_size = 14

    #====================================================
    def preprocess(self, image: np.ndarray, res: int):
        image = Image.fromarray(image).convert("RGB")

        # scale to new resolution
        orig_w, orig_h = image.size
        scale = res / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # make sure it is divisible by the patch size
        w, h = image.size 
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        image_tensor = self.normalize(self.image_to_tensor(image))

        # expects (B, N, C, H, W)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        return image_tensor

    #====================================================
    def forward(self, image: str, res: int) -> torch.Tensor:
        image_tensor = self.preprocess(image, res).to(self.device)
        output = self.model.forward(
            image_tensor,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[]
        )['depth']
            
        # remove B, N dims
        while len(output.shape) > 2:
            output = output.squeeze(0)
        return output.cpu() # force to cpu
    
    #====================================================
    @staticmethod
    def visualize(rgb: Image, depth_map: np.ndarray):    
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        rgb_np = np.array(rgb)

        # rgb image
        axes[0].imshow(rgb_np)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')
        
        depth_np = np.array(depth_map)
        depth_vis = depth_np.copy()
        
        # inverse depth for vis
        valid_mask = depth_vis > 0
        if valid_mask.sum() > 0:
            depth_vis[valid_mask] = 1.0 / depth_vis[valid_mask]
            
            depth_min = np.percentile(depth_vis[valid_mask], 2)
            depth_max = np.percentile(depth_vis[valid_mask], 98)
            
            if depth_min == depth_max:
                depth_min = depth_min - 1e-6
                depth_max = depth_max + 1e-6
            
            # Normalize to [0,1]
            depth_vis = ((depth_vis - depth_min) / (depth_max - depth_min)).clip(0, 1)
            depth_vis = 1 - depth_vis
        else:
            depth_vis = np.zeros_like(depth_vis)
        
        # depth image
        colormap = cm.get_cmap("Spectral")
        depth_colored = colormap(depth_vis)[:, :, :3]
        axes[1].imshow(depth_colored)
        axes[1].set_title("Predicted Depth (inverse depth)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()