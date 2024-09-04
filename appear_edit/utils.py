import os
import math
import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def preprocess_mask(mask, batch_size: int = 1):
    if not isinstance(mask, torch.Tensor):
        # preprocess mask
        if isinstance(mask, Image.Image) or isinstance(mask, np.ndarray):
            mask = [mask]

        if isinstance(mask, list):
            if isinstance(mask[0], Image.Image):
                mask = [np.array(m.convert("L")).astype(np.float32) / 255.0 for m in mask]
            if isinstance(mask[0], np.ndarray):
                mask = np.stack(mask, axis=0) if mask[0].ndim < 3 else np.concatenate(mask, axis=0)
                mask = torch.from_numpy(mask)
            elif isinstance(mask[0], torch.Tensor):
                mask = torch.stack(mask, dim=0) if mask[0].ndim < 3 else torch.cat(mask, dim=0)

    # Batch and add channel dim for single mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)

    # Batch single mask or add channel dim
    if mask.ndim == 3:
        # Single batched mask, no channel dim or single mask not batched but channel dim
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)

        # Batched masks no channel dim
        else:
            mask = mask.unsqueeze(1)

    # Check mask shape
    if batch_size > 1:
        if mask.shape[0] == 1:
            mask = torch.cat([mask] * batch_size)
        elif mask.shape[0] > 1 and mask.shape[0] != batch_size:
            raise ValueError(
                f"`mask_image` with batch size {mask.shape[0]} cannot be broadcasted to batch size {batch_size} "
                f"inferred by prompt inputs"
            )

    if mask.shape[1] != 1:
        raise ValueError(f"`mask_image` must have 1 channel, but has {mask.shape[1]} channels")

    # Check mask is in [0, 1]
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("`mask_image` should be in [0, 1] range")

    # Binarize mask and convert to boolean
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    return mask


def register_time(unet, t):
    conv_module = unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    
def register_optimization(unet, optimization):
    conv_module = unet.up_blocks[1].resnets[1]
    setattr(conv_module, 'optimization', optimization)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'optimization', optimization)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'optimization', optimization)
    module = unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'optimization', optimization)


def load_source_latents_t(t, latents_path, device):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path).to(device)
    return latents

def register_attention_control_efficient(unet, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            if self.optimization:
                return self.processor(
                    self,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                batch_size, sequence_length, dim = hidden_states.shape
                h = self.heads
                is_cross = encoder_hidden_states is not None
                encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
                if not is_cross and self.injection_schedule is not None and (
                        self.t in self.injection_schedule or self.t == 1000):
                    q = self.to_q(hidden_states)
                    k = self.to_k(encoder_hidden_states)

                    source_batch_size = int(q.shape[0] // 3)
                    # inject unconditional
                    q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                    k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                    # inject conditional
                    q[2 * source_batch_size:] = q[:source_batch_size]
                    k[2 * source_batch_size:] = k[:source_batch_size]

                    q = self.head_to_batch_dim(q)
                    k = self.head_to_batch_dim(k)
                else:
                    q = self.to_q(hidden_states)
                    k = self.to_k(encoder_hidden_states)
                    q = self.head_to_batch_dim(q)
                    k = self.head_to_batch_dim(k)

                v = self.to_v(encoder_hidden_states)
                v = self.head_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~attention_mask, max_neg_value)

                # attention, what we cannot get enough of
                attn = sim.softmax(dim=-1)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.batch_to_head_dim(out)

                return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_control_efficient(unet, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            if self.optimization:
                hidden_states = input_tensor

                if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                    hidden_states = self.norm1(hidden_states, temb)
                else:
                    hidden_states = self.norm1(hidden_states)

                hidden_states = self.nonlinearity(hidden_states)

                if self.upsample is not None:
                    # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                    if hidden_states.shape[0] >= 64:
                        input_tensor = input_tensor.contiguous()
                        hidden_states = hidden_states.contiguous()
                    input_tensor = self.upsample(input_tensor)
                    hidden_states = self.upsample(hidden_states)
                elif self.downsample is not None:
                    input_tensor = self.downsample(input_tensor)
                    hidden_states = self.downsample(hidden_states)

                hidden_states = self.conv1(hidden_states)

                if self.time_emb_proj is not None:
                    if not self.skip_time_act:
                        temb = self.nonlinearity(temb)
                    temb = self.time_emb_proj(temb)[:, :, None, None]

                if temb is not None and self.time_embedding_norm == "default":
                    hidden_states = hidden_states + temb

                if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                    hidden_states = self.norm2(hidden_states, temb)
                else:
                    hidden_states = self.norm2(hidden_states)

                if temb is not None and self.time_embedding_norm == "scale_shift":
                    scale, shift = torch.chunk(temb, 2, dim=1)
                    hidden_states = hidden_states * (1 + scale) + shift

                hidden_states = self.nonlinearity(hidden_states)

                hidden_states = self.dropout(hidden_states)
                hidden_states = self.conv2(hidden_states)

                if self.conv_shortcut is not None:
                    input_tensor = self.conv_shortcut(input_tensor)

                output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

                return output_tensor
            
            else:
                hidden_states = input_tensor

                hidden_states = self.norm1(hidden_states)
                hidden_states = self.nonlinearity(hidden_states)

                if self.upsample is not None:
                    # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                    if hidden_states.shape[0] >= 64:
                        input_tensor = input_tensor.contiguous()
                        hidden_states = hidden_states.contiguous()
                    input_tensor = self.upsample(input_tensor)
                    hidden_states = self.upsample(hidden_states)
                elif self.downsample is not None:
                    input_tensor = self.downsample(input_tensor)
                    hidden_states = self.downsample(hidden_states)

                hidden_states = self.conv1(hidden_states)

                if temb is not None:
                    temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

                if temb is not None and self.time_embedding_norm == "default":
                    hidden_states = hidden_states + temb

                hidden_states = self.norm2(hidden_states)

                if temb is not None and self.time_embedding_norm == "scale_shift":
                    scale, shift = torch.chunk(temb, 2, dim=1)
                    hidden_states = hidden_states * (1 + scale) + shift

                hidden_states = self.nonlinearity(hidden_states)

                hidden_states = self.dropout(hidden_states)
                hidden_states = self.conv2(hidden_states)
                if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                    source_batch_size = int(hidden_states.shape[0] // 3)
                    # inject unconditional
                    hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                    # inject conditional
                    hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

                if self.conv_shortcut is not None:
                    input_tensor = self.conv_shortcut(input_tensor)

                output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

                return output_tensor

        return forward

    conv_module = unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
    

def phrase2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    print(prompt_list)
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            print(word)
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

def attn_visualization(attn_maps, obj_position):
    attn_map_list = list()
    for attn_map in attn_maps:
        for attn in attn_map:
            #print(attn_map.shape)
            b, i, j = attn.shape
            H = W = int(math.sqrt(i))
            ca_map_obj = attn[:, :, obj_position].reshape(b, H, W)
            ca_map_obj = torch.mean(ca_map_obj, dim=0)[None,None,:,:]
            ca_map_obj = F.interpolate(ca_map_obj, (64, 64))
            attn_map_list.append(ca_map_obj)
    attns_out = torch.cat(attn_map_list, dim=1)
    attns_out = attns_out.squeeze()
    attns_out = attns_out.sum(0) / attns_out.shape[0]
    attns_out = 255 * attns_out / attns_out.max()
    return attns_out