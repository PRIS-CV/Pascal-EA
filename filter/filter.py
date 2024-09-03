import os
import shutil
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from utils import ClipSimilarity, draw_text_on_image

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def main():
    parser = ArgumentParser()
    parser.add_argument('--image-dir-1', type=str, required=True)
    parser.add_argument('--image-dir-2', type=str, required=True)
    parser.add_argument('--save-image-dir', type=str, required=True)
    parser.add_argument('--dis-image-dir', type=str, required=True)
    parser.add_argument('--text-path-1', type=str, required=True)
    parser.add_argument('--text-path-2', type=str, required=True)
    args = parser.parse_args()
    
    image_dir_1 = args.image_dir_1
    image_dir_2 = args.image_dir_2
    
    text_path_1 = args.text_path_1
    text_path_2 = args.text_path_2
    
    save_image_dir = args.save_image_dir
    dis_image_dir = args.dis_image_dir
    
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(dis_image_dir, exist_ok=True)
    
    with open(text_path_1, 'r') as fp:
        texts_info_1 = json.load(fp)
    with open(text_path_2, 'r') as fp:
        texts_info_2 = json.load(fp)
    
    instances = sorted(texts_info_1.keys())
    
    clip_estimator = ClipSimilarity(name="ViT-L/14", device=device,
                                   clip_img_thresh=0.7, 
                                   clip_thresh=0.15, 
                                   clip_dir_thresh=0.07, 
                                   clip_diff_thresh=0.05)

    num_selected = 0
    num_discarded = 0
    
    clip_sim_1_list = list()
    clip_sim_2_list = list()
    clip_sim_dir_list = list()
    clip_sim_diff_list = list()
    clip_sim_image_list = list()
    
    for ins in tqdm(instances):
        image_1_path = os.path.join(image_dir_1, ins)
        image_2_path = os.path.join(image_dir_2, ins)
        
        text_1 = texts_info_1[ins]
        text_2 = "violet " + texts_info_2[ins]
        
        clip_sim_1, clip_sim_2, clip_sim_dir, clip_sim_image = clip_estimator(image_1_path, image_2_path, [text_1], [text_2])
        
        clip_sim_1_list.append(clip_sim_1)
        clip_sim_2_list.append(clip_sim_2)
        clip_sim_dir_list.append(clip_sim_dir)
        clip_sim_diff_list.append(clip_sim_1 - clip_sim_2)
        clip_sim_image_list.append(clip_sim_image)
        
        if (clip_sim_image >= clip_estimator.clip_img_thresh  # image-image similarity
            and clip_sim_2 >= clip_estimator.clip_thresh  # image-text similarity
            and clip_sim_dir >= clip_estimator.clip_dir_thresh  # clip directional similarity
        ):
            image_save_path = os.path.join(save_image_dir, ins)
            num_selected += 1
        else:
            image_save_path = os.path.join(dis_image_dir, ins)
            num_discarded += 1
        shutil.copy(image_2_path, image_save_path)
        
        text = f"clip sim: {clip_sim_2[0]:.2f}, clip sim dir: {clip_sim_dir[0]:.2f}"
        image_show = draw_text_on_image(image_save_path, text, 0.05)
        image_show.save(image_save_path)
        
        
    
    print(f"The number of selected images : {num_selected}")
    print(f"The number of discarded images : {num_discarded}")
    
    print(f"Mean clip similarity 1 : {np.mean(clip_sim_1_list)}")
    print(f"Mean clip similarity 2 : {np.mean(clip_sim_2_list)}")
    print(f"Mean clip directional similarity : {np.mean(clip_sim_dir_list)}")
    print(f"Mean clip similarity difference: {np.mean(clip_sim_diff_list)}")
    print(f"Mean clip image-image similarity : {np.mean(clip_sim_image_list)}")        

if __name__ == "__main__":
    main()

