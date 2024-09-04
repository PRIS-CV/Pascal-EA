import os
import cv2
import json
import yaml
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL

from model.unet_2d_condition import UNet2DConditionModel
from utils import register_attention_control_efficient, register_conv_control_efficient, register_time, register_optimization
from utils import seed_everything, load_source_latents_t, preprocess_mask, attn_visualization, phrase2idx
from loss import compute_ca_loss


def inference(device, unet, vae, tokenizer, text_encoder, scheduler, mask, config, batchsize=1, guidance_scale=7.5):
    # Cross-attention save path
    if config["inference"]["save_attn"]:
        attn_save_dir = os.path.join(config["data"]["attn_path"], os.path.splitext(os.path.basename(config["data"]["image_path"]))[0])
        os.makedirs(attn_save_dir, exist_ok=True)
    
    # Prepare conditional embeddings
    cond_input = tokenizer([config["inference"]["prompt"]]*batchsize, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    cond_embeddings = text_encoder(cond_input.input_ids.to(device))[0]
    uncond_input = tokenizer([""]*batchsize, padding='max_length', max_length=tokenizer.model_max_length,return_tensors='pt')
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    
    empty_input = tokenizer([""]*batchsize, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    empty_embeddings = text_encoder(empty_input.input_ids.to(device))[0]
    
    mix_embeddings = torch.cat([empty_embeddings, text_embeddings], dim=0)

    # Loss configurations
    obj_position = phrase2idx(config["inference"]["prompt"], config["inference"]["phrase"])
    loss_scale = config["loss"]["scale"]
    loss_threshold = config["loss"]["threshold"]
    loss_max_iter = config["loss"]["max_iter"]
    loss_max_index_step = config["loss"]["max_index_step"]
    
    # Denoise loop
    latents_path = os.path.join(config["data"]["latents_path"], 
                                os.path.splitext(os.path.basename(config["data"]["image_path"]))[0], 
                                f'noisy_latents_{int(scheduler.timesteps[0])}.pt')
    latents = torch.load(latents_path).to(device)
    loss = torch.tensor(10000)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
        # set t 
        register_time(unet, t.item())
            
        iteration = 0
        while loss.item() / loss_scale > loss_threshold and iteration < loss_max_iter and i < loss_max_index_step:
            register_optimization(unet, True)

            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            pred, attns = unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)
            noise_pred = pred['sample']
            
            # compute mask loss
            attn_maps_down = attns["down"]
            attn_maps_mid = attns["mid"]
            attn_maps_up = attns["up"]
            
            loss = compute_ca_loss(attn_maps_mid, attn_maps_up, mask=mask, obj_position=obj_position) \
                    * loss_scale          

            # update latents with guidance loss
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
            latents = latents - grad_cond ** 2
            
            iteration += 1
            torch.cuda.empty_cache()
        
        # Save cross-attenntion map if need
        if config["inference"]["save_attn"]:
            # cross attention visualization
            attn_show = attn_visualization(attn_maps_up, obj_position, device)
            attn_show_np = attn_show.detach().cpu().numpy().astype(np.uint8)
            attn_show_img = cv2.applyColorMap(attn_show_np, cv2.COLORMAP_VIRIDIS)
            attn_save_path = os.path.join(attn_save_dir, f"attn_{t:03}.png")
            cv2.imwrite(attn_save_path, attn_show_img)
        
        # inference loop
        with torch.no_grad():
            register_optimization(unet, optimization=False)
            
            source_latents = load_source_latents_t(int(t), 
                                    os.path.join(config["data"]["latents_path"], os.path.splitext(os.path.basename(config["data"]["image_path"]))[0]),
                                    device)
            latent_model_input = torch.cat([source_latents] + ([latents] * 2))
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            pred, attns = unet(latent_model_input, t, encoder_hidden_states=mix_embeddings)
            noise_pred = pred['sample']
            # perform guidance
            _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # compute the denoising step with the reference model
            latents = scheduler.step(noise_pred, t, latents)['prev_sample']
            # recover unmasked source latents
            latents = latents * mask + source_latents * (1 - mask)
            torch.cuda.empty_cache()
    
    # Decode latents
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        pil_images = T.ToPILImage()(image[0])
        return pil_images


def main():
    # 1. Load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
        
    with open(os.path.join(config["data"]["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    print(config)
    
    with open(config["model"]["unet_config_path"]) as f:
        unet_config = json.load(f)
    
    device = config["general"]["device"]
    seed_everything(config["general"]["seed"])

    
    # 2. Load models
    unet = UNet2DConditionModel(**unet_config).from_pretrained(config["model"]["model_path"], subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(config["model"]["model_path"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config["model"]["model_path"], subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(config["model"]["model_path"], subfolder="vae").to(device)
    scheduler = DDIMScheduler.from_pretrained(config["model"]["model_path"], subfolder="scheduler")
    scheduler.set_timesteps(config["inference"]["n_timesteps"])

    
    # 3. Load mask and image
    mask = Image.open(config["data"]["mask_path"]).convert('RGB')
    mask = mask.resize((64, 64), resample=Image.Resampling.LANCZOS)
    mask = preprocess_mask(mask)
    mask = mask.to(device)
    image = Image.open(config["data"]["image_path"]).convert('RGB')
    
    
    # 4. Set feature and self-attention injection
    conv_injection_t = int(config["inference"]["n_timesteps"] * config["model"]["feature_inject_threshold"])
    qk_injection_t = int(config["inference"]["n_timesteps"] * config["model"]["selfattn_inject_threshold"])
    qk_injection_timesteps = scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
    conv_injection_timesteps = scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
    register_attention_control_efficient(unet, qk_injection_timesteps)
    register_conv_control_efficient(unet, conv_injection_timesteps)
    
    # 5. Inference
    edited_image = inference(device, unet, vae, tokenizer, text_encoder, scheduler, mask, config)

    
    # 6. Save example images
    save_path = os.path.join(config["data"]["output_path"], os.path.splitext(os.path.basename(config["data"]["image_path"]))[0])
    os.makedirs(save_path, exist_ok=True)
    
    image_save_path = os.path.join(save_path, 'output.png')
    edited_image = edited_image.resize(image.size)
    edited_image.save(image_save_path)
        
if __name__ == "__main__":
    main()