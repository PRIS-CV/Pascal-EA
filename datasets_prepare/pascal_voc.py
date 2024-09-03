import os
import cv2
import json
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
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

pas21_dict = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
pas21_discarded_classes = [5, 15, 16, 255]

def main():
    parser = ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--data-save-root', type=str, required=True)
    args = parser.parse_args()
    
    for split_ in ["train", "val"]:
        ann_dir = os.path.join(args.data_root, "annotations_mmseg", split_)
        img_dir = os.path.join(args.data_root, "images", split_)

        obj_save_dir = os.path.join(args.data_save_root, "masks_object", split_)
        bck_save_dir = os.path.join(args.data_save_root, "masks_background", split_)
        img_save_dir = os.path.join(args.data_save_root, "images", split_)
        ann_save_dir = os.path.join(args.data_save_root, "annotations_mmseg", split_)
        meta_file_path = os.path.join(args.data_save_root, "meta_{split_}.json")

        instances = sorted(os.listdir(ann_dir))
        os.makedirs(obj_save_dir, exist_ok=True)
        os.makedirs(bck_save_dir, exist_ok=True)
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(ann_save_dir, exist_ok=True)

        meta_info = dict()
        for img in tqdm(instances):
            # Load the instance and semantic segmentation labels
            semseg_mask = np.array(Image.open(os.path.join(ann_dir, img)))
            semseg_mask[semseg_mask == 0] = 255 # discard background
            class_values = sorted(list(np.unique(semseg_mask)))
            
            # discard some categories
            for i in pas21_discarded_classes:
                if i in class_values:
                    class_values.remove(i) 
            objects = [np.array(semseg_mask == i).astype(np.uint8) for i in class_values] 
            
            if len(objects) != 0:
                # Extract the salient object
                salient_object_mask, salient_class_value = extract_most_salient_object(objects, class_values)
                if salient_object_mask is not None:
                    background_mask = 255 - salient_object_mask

                    cv2.imwrite(os.path.join(obj_save_dir, img), salient_object_mask)
                    cv2.imwrite(os.path.join(bck_save_dir, img), background_mask)
                    shutil.copy(os.path.join(img_dir, img.replace('png', 'jpg')), os.path.join(img_save_dir, img))
                    shutil.copy(os.path.join(ann_dir, img), os.path.join(ann_save_dir, img))

                    # record the category of the salient object
                    class_name = pas21_dict[salient_class_value]
                    meta_info[img] = class_name

        # Save the meta information file
        with open(meta_file_path, 'w') as fp:
            text = json.dumps(meta_info, indent=4)
            fp.write(text)    
    

if __name__ == "__main__":
    main()
