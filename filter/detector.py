import os
import cv2
import json
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from mmseg.apis import init_model, inference_model

class InferDataset(Dataset):
    def __init__(self, info_path, real_img_root, syn_img_root, mask_root, label_root):
        with open(info_path, 'r') as fp:
            text_info = json.load(fp)
        self.filenames = sorted(text_info.keys())
        
        self.real_img_root = real_img_root
        self.syn_img_root = syn_img_root
        self.mask_root = mask_root
        self.label_root = label_root

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, item):
        filename = self.filenames[item]
        real_img = Image.open(os.path.join(self.real_img_root, filename)).convert('RGB')
        syn_img = Image.open(os.path.join(self.syn_img_root, filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_root, filename))
        label = Image.open(os.path.join(self.label_root, filename))
        
        return self.img_transform(real_img), \
            self.img_transform(syn_img), \
            torch.from_numpy(np.array(mask)).int(), \
            torch.from_numpy(np.array(label)).long(), \
            filename
    
    def __len__(self):
        return len(self.filenames)
    

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    
    
    parser.add_argument('--meta_file_path', type=str, required=True)
    parser.add_argument('--real-img-path', type=str, required=True)
    parser.add_argument('--syn-img-path', type=str, required=True)
    parser.add_argument('--object-mask-path', type=str, required=True)
    parser.add_argument('--real-label-path', type=str, required=True)
    parser.add_argument('--filtered-label-path', type=str, required=True)
    parser.add_argument('--tolerance-margin', type=float, default=1.25)
    
    args = parser.parse_args()
    
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    
    
    dataset = InferDataset(info_path=args.info_file_path, 
                           real_img_root=args.real_img_path, 
                           syn_img_root=args.syn_img_path, 
                           mask_root=args.object_mask_path, 
                           label_root=args.real_label_path)
    
    real_dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    dataloader= DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
        
    #os.makedirs(args.diff_save_path, exist_ok=True)
    os.makedirs(args.filtered_label_path, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        
    # Calculate the class-wise mean loss on real images
    class_wise_mean_loss = [(0, 0) for _ in range(20)]
    for i, (real_img, _, _, label, filenames) in enumerate(tqdm(real_dataloader)):
        real_img, label = real_img.to(args.device), label.to(args.device)
        classes = torch.unique(label).tolist()
        
        with torch.no_grad():
            preds = model.predict(real_img)
            preds = torch.cat([pred.seg_logits.data.unsqueeze(0) for pred in preds])

        loss = criterion(preds, label-1)
        
        for class_ in classes:
            if class_ == 0:
                continue
            pixel_num, loss_sum = class_wise_mean_loss[class_-1]
            class_wise_mean_loss[class_-1] = (pixel_num + torch.sum(label == class_).item(), loss_sum + torch.sum(loss[label == class_]).item())
    
    class_wise_mean_loss = [loss_sum / (pixel_num + 1e-5) for pixel_num, loss_sum in class_wise_mean_loss]
    print('Class-wise mean loss:')
    print(class_wise_mean_loss)

    
    # Filter out noisy synthetic pixels
    for i, (real_img, syn_img, mask, label, filenames) in enumerate(tqdm(dataloader)):
        real_img, syn_img, mask, label = real_img.to(args.device), syn_img.to(args.device), mask.to(args.device), label.to(args.device)
        
        with torch.no_grad():
            real_preds = model.predict(real_img)
            real_preds = torch.cat([real_pred.seg_logits.data.unsqueeze(0) for real_pred in real_preds])
            
            syn_preds = model.predict(syn_img)
            syn_preds = torch.cat([syn_pred.seg_logits.data.unsqueeze(0) for syn_pred in syn_preds])

        real_loss = criterion(real_preds, label-1)
        syn_loss = criterion(syn_preds, label-1)
        
        
        # Save the differences of losses of real images and synthetic images
        diff_loss = torch.abs(real_loss - syn_loss)
        diff_loss = diff_loss.squeeze()
        diff_loss = 255 * diff_loss / diff_loss.max()
        diff_loss = diff_loss.detach().cpu().numpy().astype(np.uint8)
        #diff_loss_show = cv2.applyColorMap(diff_loss, cv2.COLORMAP_VIRIDIS)
        #cv2.imwrite(os.path.join(args.diff_save_path, filenames[0]), diff_loss_show)
        
        # Filter out
        label_filtered = torch.zeros_like(label)
        label_filtered[:] = label[:]
        for class_ in classes:
            if class_ == 0:
                continue
            filtered_region = (mask == 255) & (syn_loss > real_loss) & (syn_loss > class_wise_mean_loss[class_-1] * args.tolerance_margin)
            label_filtered[filtered_region] = 255
        
        label_filtered = label_filtered.cpu().numpy().astype(np.uint8)
        
        label_filtered = Image.fromarray(label_filtered[0])
        label_filtered.save(os.path.join(args.filtered_label_path, filenames[0]))
            


if __name__ == '__main__':
    main()
