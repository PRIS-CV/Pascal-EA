import os
import cv2
import json
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from argparse import ArgumentParser


def extract_most_salient_object(coco, labels_to_names, anns, image_size):
    max_area = 0
    max_mask = None
    max_category = None
    for instance_ann in anns:
        if instance_ann['category_id'] <= 90 and instance_ann['category_id'] != 1:
            bbox = instance_ann['bbox']
            bbox = [bbox[0] / image_size[1], (bbox[0]+bbox[2]) / image_size[1], bbox[1] / image_size[0], (bbox[1]+bbox[3]) / image_size[0]]
            area = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
            if area > max_area:
                max_area = area
                max_category = labels_to_names[instance_ann['category_id']]
                max_mask = coco.annToMask(instance_ann)*instance_ann['category_id']
    
    mask_result = None
    if max_mask is not None :
        if max_area >= 0.1:  # as area larger than 10% of the whole image, add to list
            mask_result = np.zeros_like(max_mask)
            mask_result[max_mask > 0] = 255
    return mask_result, max_category


def main():
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--data-save-root', type=str, required=True)
    args = parser.parse_args()
    
    for split_ in ["train2017", "val2017"]:
        
        img_dir = os.path.join(args.data_root, "images", split_)
        ann_dir = os.path.join(args.data_root, "annotations", split_)

        img_save_dir = os.path.join(args.data_save_root, "images", split_)
        ann_save_dir = os.path.join(args.data_save_root, "annotations", split_)
        obj_save_dir = os.path.join(args.data_save_root, "masks_object", split_)
        bck_save_dir = os.path.join(args.data_save_root, "masks_background", split_)
        meta_file_path = os.path.join(args.data_save_root, "meta_{split_}.json")
        
        os.makedirs(obj_save_dir, exist_ok=True)
        os.makedirs(bck_save_dir, exist_ok=True)
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(ann_save_dir, exist_ok=True)


        coco = COCO(os.path.join(args.data_save_root, "annotations", f"{split_}.json"))
        ids = list(coco.anns.keys())
        image_ids = sorted(list(set([coco.anns[this_id]['image_id'] for this_id in ids])))
                
        cats = coco.loadCats(coco.getCatIds())
        labels_to_names = {}
        for cat in cats:
            labels_to_names[cat['id']] = cat['name']

        meta_info = dict()
        for image_id in tqdm(image_ids):
            img_name = coco.loadImgs(image_id)[0]["file_name"]
            img_path = os.path.join(img_dir, img_name)
            
            image = Image.open(img_path).convert("RGB")
            image_size = [image.size[1], image.size[0]]
            annIds = coco.getAnnIds(imgIds=image_id)
            coco_anns = coco.loadAnns(annIds) # coco is [x, y, width, height]
            
            salient_object_mask, salient_object_class = extract_most_salient_object(coco, labels_to_names, coco_anns, image_size)
            
            if salient_object_mask is not None:
                background_mask = 255 - salient_object_mask
                cv2.imwrite(os.path.join(obj_save_dir, img_name), salient_object_mask)
                cv2.imwrite(os.path.join(bck_save_dir, img_name), background_mask)
                
                shutil.copy(img_path, os.path.join(img_save_dir, img_name))
                shutil.copy(os.path.join(ann_dir, img_name.split('.')[0]+'_labelTrainIds.png'), os.path.join(ann_save_dir, img_name.split('.')[0]+'_labelTrainIds.png'))
                
                meta_info[img_name] = salient_object_class

        ## Save the file
        with open(meta_file_path, 'w') as fp:
            text = json.dumps(meta_info, indent=4)
            fp.write(text)
    

if __name__ == "__main__":
    main()