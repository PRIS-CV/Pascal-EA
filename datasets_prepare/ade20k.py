import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser


def extract_most_salient_object(masks, class_values):
    contours = list()
    new_class_values = list()
    for j in range(len(masks)):
        # find contours of all objects
        _contours, _ = cv2.findContours(masks[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(_contours)
        new_class_values.extend([class_values[j]]*len(_contours))
    assert len(contours) == len(new_class_values)
    
    # calculate object area, and find the largest object
    max_area = 0
    max_contour = None
    max_index = None
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            max_index = i
            max_contour = contours[i]
    
    # filter out non-salient object
    mask_result = None
    if max_contour is not None and max_index is not None:
        total_area = masks[j].shape[0] * masks[j].shape[1]
        if max_area / total_area >= 0.1:
            mask_result = np.zeros_like(masks[j])
            cv2.drawContours(mask_result, [max_contour], -1, 255, thickness=cv2.FILLED)
            
    return mask_result, new_class_values[max_index]

ade150_classes = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag')

def main():
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--data-save-root', type=str, required=True)
    args = parser.parse_args()
    
    for split_ in ["training", "validation"]:
        img_dir = os.path.join(args.data_root, "images", split_)
        ann_dir = os.path.join(args.data_root, "annotations", split_)
  
        img_save_dir = os.path.join(args.data_save_root, "images", split_)
        ann_save_dir = os.path.join(args.data_save_root, "annotations", split_)
        obj_save_dir = os.path.join(args.data_save_root, "masks_object", split_)
        bck_save_dir = os.path.join(args.data_save_root, "masks_background", split_)
        meta_file_path = os.path.join(args.data_save_root, "meta_{split_}.json")


        instances = sorted(os.listdir(ann_dir))
        os.makedirs(obj_save_dir, exist_ok=True)
        os.makedirs(bck_save_dir, exist_ok=True)
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(ann_save_dir, exist_ok=True)

        ade20k_dict = {}
        for idx, class_name in enumerate(ade150_classes):
            ade20k_dict[idx] = class_name
        print(ade20k_dict)
        ade20k_selected_classes = ['table', 'sofa', 'house', 'car', 'truck', 'bicycle', 'monitor']


        meta_info = dict()
        for img in tqdm(instances):
            # Load the instance and semantic segmentation labels
            semseg_mask = np.array(Image.open(os.path.join(ann_dir, img)))
            
            class_values = sorted(list(np.unique(semseg_mask)))
            new_class_values = []
            # discard some categories
            for v in class_values:
                if v != 255 and ade20k_dict[v] in ade20k_selected_classes:
                    new_class_values.append(v)
            objects = [np.array(semseg_mask == i).astype(np.uint8) for i in new_class_values] 
            
            # Extract the salient object
            if len(objects) != 0:
                salient_object_mask, salient_class_value = extract_most_salient_object(objects, new_class_values)
                if salient_object_mask is not None:
                    background_mask = 255 - salient_object_mask
                    cv2.imwrite(os.path.join(obj_save_dir, img), salient_object_mask)
                    cv2.imwrite(os.path.join(bck_save_dir, img), background_mask)
                    shutil.copy(os.path.join(img_dir, img.replace('png', 'jpg')), os.path.join(img_save_dir, img)) # jpg to png format
                    shutil.copy(os.path.join(ann_dir, img), os.path.join(ann_save_dir, img))
                
                    # record the category of the salient object
                    class_name = ade20k_dict[salient_class_value]
                    meta_info[img] = class_name

        # Save the meta information file
        with open(meta_file_path, 'w') as fp:
            text = json.dumps(meta_info, indent=4)
            fp.write(text)
  

if __name__ == "__main__":
    main()