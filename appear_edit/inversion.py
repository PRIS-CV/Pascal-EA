import os
import yaml
import json
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from utils import seed_everything
import torchvision.transforms as T
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

@torch.no_grad()
def inversion(device, unet, vae, tokenizer, text_encoder, scheduler, config, save_path=None):
    # Prepare text embedding
    text_input = tokenizer(config["inversion"]["prompt"], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = tokenizer("", padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt')
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    cond = text_embeddings[1].unsqueeze(0)
        
    # Load and encode source image
    image_pil = Image.open(config["data"]["image_path"]).convert("RGB").resize((512, 512))
    image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        image = 2 * image - 1
        posterior = vae.encode(image).latent_dist
        latent = posterior.mean * 0.18215
    
    # DDIM Inversion Steps
    scheduler.set_timesteps(config["inversion"]["n_timesteps"])
    timesteps = reversed(scheduler.timesteps)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(timesteps)):
            cond_batch = cond.repeat(latent.shape[0], 1, 1)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
            
            eps = unet(latent, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (latent - sigma_prev * eps) / mu_prev
            latent = mu * pred_x0 + sigma * eps
            if save_path is not None:
                # latents: [1, 4, 64, 64]
                torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))


def main():
    # 1. Load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config["general"]["device"]
    seed_everything(config["general"]["seed"])
    
    with open(config["model"]["unet_config_path"]) as f:
        unet_config = json.load(f)
        
    save_path = os.path.join(config["data"]["latents_path"], os.path.splitext(os.path.basename(config["data"]["image_path"]))[0])
    os.makedirs(save_path, exist_ok=True)
    print(f'[INFO] save latents to {save_path}')
    
    # 2. Load models
    vae = AutoencoderKL.from_pretrained(config["model"]["model_path"], subfolder="vae", revision="fp16", torch_dtype=torch.float16).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(config["model"]["model_path"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config["model"]["model_path"], subfolder="text_encoder", revision="fp16", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel(**unet_config).from_pretrained(config["model"]["model_path"], subfolder="unet", revision="fp16", torch_dtype=torch.float16).to(device)
    scheduler = DDIMScheduler.from_pretrained(config["model"]["model_path"], subfolder="scheduler")
    print(f'[INFO] loaded stable diffusion!')
    
    
    # 3. DDIM Inversion
    inversion(device, unet, vae, tokenizer, text_encoder, scheduler, config, save_path)


if __name__ == "__main__":
    main()
