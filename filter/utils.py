import os
from pyclbr import Class
import clip
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import Optional, List, Tuple
from einops import rearrange
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
from transformers import ViTImageProcessor, ViTModel
from PIL import Image, ImageFont, ImageDraw


class ClipSimilarity(nn.Module):
    def __init__(self, name:str="ViT-L/14", 
                 device:str="cuda", 
                 clip_img_thresh=0.7, clip_thresh=0.2, clip_dir_thresh=0.2, clip_diff_thresh=0.1):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-L/14@336px": 336,
        }.get(name, 224)
        self.device = device

        self.model, _ = clip.load(
            name, device=self.device, download_root="/data/yinzijin/checkpoints/clip"
        )
        self.model.eval().requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).to(self.device)
        )
        self.register_buffer(
            "std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(self.device)
        )
        
        self.clip_img_thresh = clip_img_thresh
        self.clip_thresh = clip_thresh
        self.clip_dir_thresh = clip_dir_thresh
        self.clip_diff_thresh = clip_diff_thresh
        
        self.lemmatizer = WordNetLemmatizer()
    
    @torch.no_grad()
    def preprocess_image(self, path: str):
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        image = torch.from_numpy(image).to(self.device)
        image = image[None, :, :]
        image = rearrange(image, "b h w c -> b c h w")
        return image
        
      
    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    @torch.no_grad()
    def encode_image(
        self, image: torch.Tensor
    ) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(
            image.float(), size=self.size, mode="bicubic", align_corners=False
        )
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
    
    @torch.no_grad()
    def text_similarity(
        self,
        text_0: List[str],
        text_1: List[str],
        get_feats: Optional[bool] = False,
        lemmatize: Optional[bool] = False,
    ) -> torch.Tensor:
        if lemmatize:
            text_0 = [self.lemmatizer.lemmatize(t0) for t0 in text_0]
            text_1 = [self.lemmatizer.lemmatize(t1) for t1 in text_1]

        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim = text_features_0 @ text_features_1.T
        if get_feats:
            return sim, text_features_0, text_features_1
        return sim
    
    @torch.no_grad()
    def forward(
        self,
        image_0_path: str,
        image_1_path: str,
        text_0: List[str],
        text_1: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        image_0 = self.preprocess_image(image_0_path)
        image_1 = self.preprocess_image(image_1_path)
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(
            image_features_1 - image_features_0, text_features_1 - text_features_0
        )
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        
        sim_0 = sim_0.detach().cpu().numpy()
        sim_1 = sim_1.detach().cpu().numpy()
        sim_direction = sim_direction.detach().cpu().numpy()
        sim_image = sim_image.detach().cpu().numpy()
        
        return sim_0, sim_1, sim_direction, sim_image

    @torch.no_grad()
    def pred_consistency(
        self, image: torch.Tensor, text_0: List[str], text_1: List[str]
    ) -> bool:
        sim_0, sim_1, _, _ = self.forward(image, image, text_0, text_1)
        return (sim_1 > sim_0)[0].item()
    
    
class DinoSimilarity(nn.Module):
    def __init__(self, name:str="/data/yinzijin/checkpoints/dino-vitb16", 
                 device:str="cuda", 
                 dino_img_thresh=0.2):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(name)
        self.model = ViTModel.from_pretrained(name).to(device)
        self.device = device
        self.dino_img_thresh = dino_img_thresh
    
    @torch.no_grad()
    def preprocess_image(self, path: str):
        image = Image.open(path).convert("RGB")
        image = self.processor(images=image, size={"height": 224, "width": 224}, return_tensors="pt").to(self.device)
        return image
    
    @torch.no_grad()
    def forward(self, image_0_path: str, image_1_path: str,):
    
        image_0 = self.preprocess_image(image_0_path)
        image_1 = self.preprocess_image(image_1_path)
        
        outputs_0 = self.model(**image_0)
        outputs_1 = self.model(**image_1)
        
        features_0 = outputs_0.last_hidden_state
        features_1 = outputs_1.last_hidden_state

        features_0 = features_0.mean(dim=1)
        features_1 = features_1.mean(dim=1)
        
        sim_image = F.cosine_similarity(features_0, features_1)
        sim_image = sim_image.detach().cpu().numpy()
        return sim_image
        
    
def draw_text_on_image(image_path, text, scale):
    # Open an image file
    with Image.open(image_path) as img:
        width, height = img.size

        # Determine the size of the text based on the image size and scale
        text_size = int(min(width, height) * scale)
        # Use a truetype font from PIL, adjust the path to the font file as needed
        # Here we're using the DejaVuSans which is typically installed with matplotlib
        font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Determine text position
        text_position = (20, 0) # horizontal, vertical
        # Add text to image
        draw.text(text_position, text, font=font, fill=(255, 105, 180))

    return img  