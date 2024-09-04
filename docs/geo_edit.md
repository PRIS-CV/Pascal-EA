# Object geometry editing
We also edit object geometry attributes (size, position) by diffusion-based inpainting methods. We employ [stable-diffusion-inpainting](https://modelscope.cn/models/ai-modelscope/stable-diffusion-inpainting) and [controlnet](https://huggingface.co/lllyasviel/control_v11p_sd15_seg).

## Inference
Then, run the following command to apply editing on the source image.
```shell
python inference.py --config_path config.yaml
```
Then, the inpainted image, transformed object mask and transformed semseg label will be stored under `data.image_save_path`, `data.mask_save_path`, `data.semseg_save_path`.

In config file, `inference.mode` means the geometry editing types, "resize" means adjusting size of objects, "reposition" means moving objects. `inference.scale` is a hyperparameter that means amplitude of geometry editing, its value usually from `0.1-0.3`.

## Dataset editing
Above is the instruction of editing individual image. If generate synthetic evaluation dataset, you should write your own scripts. You may need **object masks**, **source images**, and **semantic segmentation labels** accquired from previous dataset preparation step.
