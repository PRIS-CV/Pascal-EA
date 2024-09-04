import cv2
import torch
import random
import numpy as np
from PIL import Image


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_bbox(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, None
    bbox = []
    for i, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if w<5 or h<5 :
            continue
        bbox.append((x, y, w, h))

    return bbox

def rescale_maximum(img, mask, semseg):
    bbox = get_bbox(mask)
    if bbox is None:
        return None, None, None
    img_list = []
    mask_list = []
    semseg_list = []
    for i in bbox:
        x,y,w,h = i
        H,W,C = img.shape
        img_new = img[y:y+h,x:x+w,:]
        mask_new = mask[y:y+h,x:x+w,:]
        semseg_new = semseg[y:y+h,x:x+w,:]

        img_list.append(img_new)
        mask_list.append(mask_new)
        semseg_list.append(semseg_new)

    return img_list, mask_list, semseg_list
   
def set_control_image_color(control_img: Image):
    control_img = np.array(control_img)
    color_control_img = np.zeros((control_img.shape[0], control_img.shape[1], 3), dtype=np.uint8)
    color_control_img[control_img == 0, :] = [0, 0, 0]
    color_control_img[control_img > 0, :] = [180, 120, 120]
    color_control_img = color_control_img.astype(np.uint8)
    return Image.fromarray(color_control_img)

def resize_and_paste(img, mask, semseg, img_obj_list, mask_obj_list, semseg_obj_list, scale):
    img_new = img.copy()
    mask_new = mask.copy()
    semseg_new = semseg.copy()
    bbox = get_bbox(mask)

    semseg_new[mask == 255] = 0
    mask_new[:] = 0

    for index in range(len(img_obj_list)):
        img_rescale = img_obj_list[index]
        mask_rescale = mask_obj_list[index]
        semseg_rescale = semseg_obj_list[index]
        class_index = np.unique(semseg_rescale)[-1]
        
        height, width = img_rescale.shape[:2]
        # print(scale)
        if random.choice([0, 1]):
            t_scale = 1 + random.uniform(scale/2, scale) # 正的
        else:
            t_scale = 1 + random.uniform(-scale, -scale/2) # 负的

        height, width = int(height * t_scale), int(width * t_scale)
        img_rescale = cv2.resize(img_rescale, (width, height))
        mask_rescale = cv2.resize(mask_rescale, (width, height))  
        mask_rescale[mask_rescale > 0] = 255  # resize会插值，需要重新二值化

        # w_new, h_new = height, width
        x,y,w,h = bbox[index]
        H,W,C = mask.shape

        center_x, center_y = x+w//2, y+h//2
        start_point_x = max(center_x - mask_rescale.shape[1]//2, 0) # center - w
        start_point_y = max(center_y - mask_rescale.shape[0]//2, 0) # center - h
        end_point_x = min(center_x + mask_rescale.shape[1]//2, W) # center+w
        end_point_y = min(center_y + mask_rescale.shape[0]//2, H) # center+h

        img_new[start_point_y:end_point_y, start_point_x:end_point_x] = img_rescale[:end_point_y-start_point_y, :end_point_x-start_point_x]
        mask_new[start_point_y:end_point_y, start_point_x:end_point_x] = mask_rescale[:end_point_y-start_point_y, :end_point_x-start_point_x]
        
        semseg_new[start_point_y:end_point_y, start_point_x:end_point_x] = mask_rescale[:end_point_y-start_point_y, :end_point_x-start_point_x] / 255 * class_index

    return img_new, mask_new, semseg_new

def reposition_and_paste(img, mask, semseg, img_obj_list, mask_obj_list, semseg_obj_list, scale):
    img_new = img.copy()
    mask_new = mask.copy()
    semseg_new = semseg.copy()
    
    bbox = get_bbox(mask)
    
    semseg_new[mask == 255] = 0
    mask_new[:] = 0
    
    
    for index in range(len(img_obj_list)):
        img_rescale = img_obj_list[index]
        mask_rescale = mask_obj_list[index]
        semseg_rescale = semseg_obj_list[index]

        # w_new, h_new = height, width
        x,y,w,h = bbox[index]
        H,W,C = mask.shape

        region_range_x = int(w * scale)
        region_range_y = int(h * scale)
        
        if random.choice([0, 1]):
            tx = random.randint(int(region_range_x/2), region_range_x) # 正的
            ty = random.randint(int(region_range_y/2), region_range_y)
        else:
            # print(region_range_x, region_range_y)
            tx = random.randint(-region_range_x, int(-region_range_x/2)) # 负的
            ty = random.randint(-region_range_y, int(-region_range_y/2))
            
        # print(f'x移动{tx}，y移动{ty}')
        center_x, center_y = x+w//2, y+h//2
        start_point_x = max(center_x - mask_rescale.shape[1]//2 + tx, 0) # center - w + tx
        start_point_y = max(center_y - mask_rescale.shape[0]//2 + ty, 0) # center - h + ty
        end_point_x = min(center_x + mask_rescale.shape[1]//2 + tx, W) # center+w + tx
        end_point_y = min(center_y + mask_rescale.shape[0]//2 + ty, H) # center+h + ty

        # img = cv2.GaussianBlur(img, (49, 49), 0)
        img_new[start_point_y:end_point_y, start_point_x:end_point_x] = img_rescale[:end_point_y-start_point_y, :end_point_x-start_point_x]
        mask_new[start_point_y:end_point_y, start_point_x:end_point_x] = mask_rescale[:end_point_y-start_point_y, :end_point_x-start_point_x]
        semseg_new[start_point_y:end_point_y, start_point_x:end_point_x] = semseg_rescale[:end_point_y-start_point_y, :end_point_x-start_point_x]
        
    return img_new, mask_new, semseg_new