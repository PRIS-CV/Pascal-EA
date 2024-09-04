import math
import torch
import torch.nn.functional as F

def compute_ca_loss(attn_maps_mid, attn_maps_up, mask, obj_position):
    loss = 0
    
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated
        #print(attn_map.shape)
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
        mask = F.interpolate(mask, (H,W))
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        
        activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
        obj_loss = torch.mean((1 - activation_value) ** 2)
        loss += obj_loss

    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated
        #print(attn_map.shape)
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
        mask = F.interpolate(mask, (H,W))
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        
        activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(dim=-1)
        obj_loss = torch.mean((1 - activation_value) ** 2)
        loss += obj_loss   
    loss = loss / (len(attn_maps_up[0]) + len(attn_maps_mid))
    return loss

def compute_ca_up_loss(attn_maps_up, mask, obj_position):
    loss = 0
    num_attns = 0
    for attn_maps in attn_maps_up:
        for attn_map_integrated in attn_maps:
            attn_map = attn_map_integrated
            #print(attn_map.shape)
            #
            b, i, j = attn_map.shape
            H = W = int(math.sqrt(i))
            ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
            
            mask = F.interpolate(mask, (H,W))
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            
            activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(dim=-1)
            obj_loss = torch.mean((1 - activation_value) ** 2)
            loss += obj_loss
            num_attns += 1
    
    loss = loss / num_attns
    return loss