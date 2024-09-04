import os
import cv2
import torch
import yaml
import argparse
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from utils import seed_everything, rescale_maximum, set_control_image_color, reposition_and_paste, resize_and_paste


def object_transform(img, mask, semseg, mode, scale):
    img = np.array(img)
    mask = np.array(mask)
    semseg = np.array(semseg)
    print(img.shape)
    print(mask.shape)
    print(semseg.shape)
    img_rescale_list, mask_rescale_list, semseg_rescale_list = rescale_maximum(img.copy(), mask.copy(), semseg.copy()) # 裁切

    if mode == 'resize':
        tf_img, tf_mask, tf_semseg = resize_and_paste(img, mask, semseg, img_rescale_list, mask_rescale_list, semseg_rescale_list, scale)
    elif mode == 'reposition':
        tf_img, tf_mask, tf_semseg  = reposition_and_paste(img, mask, semseg, img_rescale_list, mask_rescale_list, semseg_rescale_list, scale)
    else:
        raise ValueError("Unsupported editing type, only support 'resize' and 'reposition'.")
    
    tf_img = Image.fromarray(tf_img)
    tf_mask = Image.fromarray(tf_mask)
    tf_semseg = Image.fromarray(tf_semseg)
    return tf_img, tf_mask, tf_semseg


def inpainting_inference(pipe, img, mask, tf_img, tf_mask, prompt="", n_timesteps=50, guidance_scale=7.5, f=5):
    mask = mask.convert('L')
    tf_mask = tf_mask.convert('L')
    control_mask = set_control_image_color(tf_mask)

    img_np = np.array(img)
    tf_img_np = np.array(tf_img)
    inpaint_mask_np = np.array(mask)
    tf_mask_np = np.array(tf_mask)
    indices = np.argwhere(tf_mask_np >= 125.5)
    
    # 将obj中对应位置的像素替换到bg中对应位置上
    for index in indices:
        inpaint_mask_np[index[0], index[1]] = 0
        img_np[index[0], index[1], :] = tf_img_np[index[0], index[1], :]
        
    inpaint_mask_np = cv2.dilate(inpaint_mask_np, np.ones((f, f), np.uint8), iterations=1)
    inpaint_mask = Image.fromarray(inpaint_mask_np)
    inpaint_mask.save("./inpaint_mask.png")
    inpaint_image = Image.fromarray(img_np)
    
    # inpainting
    output = pipe(
        prompt=prompt,
        image=inpaint_image,
        mask_image=inpaint_mask,
        control_image=control_mask,
        num_images_per_prompt=1,
        num_inference_steps=n_timesteps,
        guidance_scale=guidance_scale,
        strength=1,
    )
                
    output_img = output.images[0]
    
    return output_img
                    

def main():
    # 1. Load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = config["general"]["device"]
    seed_everything(config["general"]["seed"])
    
    # 2. Load models
    controlnet = ControlNetModel.from_pretrained(config["model"]["controlnet_path"], torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        config["model"]["sd_path"],
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 3. Load mask and image
    mask = Image.open(config["data"]["mask_path"]).convert('RGB')
    image = Image.open(config["data"]["image_path"]).convert('RGB')
    semseg = Image.open(config["data"]["semseg_path"]).convert('RGB')
    
    # 4. Inference
    tf_image, tf_mask, tf_semseg = object_transform(image, mask, semseg, mode=config["inference"]["mode"], scale=config["inference"]["scale"])
    output_image = inpainting_inference(pipe, image, mask, tf_image, tf_mask, 
                prompt=config["inference"]["prompt"], n_timesteps=config["inference"]["n_timesteps"], guidance_scale=config["inference"]["guidance_scale"])
    
    
    # 5. Save inpainted images, and transformed semseg label and object mask
    os.makedirs(config["data"]["image_save_path"], exist_ok=True)
    os.makedirs(config["data"]["semseg_save_path"], exist_ok=True)
    os.makedirs(config["data"]["mask_save_path"], exist_ok=True)
    
    image_save_path = os.path.join(config["data"]["image_save_path"], os.path.basename(config["data"]["image_path"]))
    mask_save_path = os.path.join(config["data"]["mask_save_path"], os.path.basename(config["data"]["mask_path"]))
    semseg_save_path = os.path.join(config["data"]["semseg_save_path"], os.path.basename(config["data"]["semseg_path"]))

    
    output_image = output_image.resize(image.size)
    output_image.save(image_save_path)
    tf_mask.save(mask_save_path)
    tf_semseg.save(semseg_save_path)
    

if __name__ == "__main__":
    main()
