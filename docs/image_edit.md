# Edit image
We edit images under framework of the PnP (CVPR 2023), the codes are heavily from [pnp-diffusers](https://github.com/MichalGeyer/pnp-diffusers), thanks for their contribution!


## DDIM inversion
First, we need to perform DDIM inversion and store the intermediate noisy latents of the source image. Please run the following command.
```shell
cd image_edit
python inversion.py --config_path config.yaml
```
In `config.yaml`, `inversion.prompt` describes the content of the source image, `data.latents_path/<image_name>` indicates where intermediate noisy latents are stored, `data.image_path` is path to the source image.

## Inference
Then, run the following command to apply editing on the source image with text guidance.

```shell
cd image_edit
python pnp.py --config_path config.yaml
```
In `config.yaml`, `data.mask_path` is path to the object mask, `data.output_path` is save path of the edited image, `data.attn_path` is save path of intermediate cross-attention map. `inference.prompt` describes the content of the edited image, and `inference.phrase` describes the object you want to edit. `loss.scale`, `loss.threshold`, `loss.max_iter` and `loss.max_index_step` means configurations of our mask-guidance energy function. 

For additional hyperparameters details, please see `config.yaml`.


## Dataset editing
Above is the instruction of editing individual image, evaluation dataset editing need iteractively editing. You should write your own scripts to generate new evaluate datasets. You may need **object masks**, **source images**, and **edited text prompts** accquired from previous steps(dataset preparation and text edit).