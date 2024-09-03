# Filter out low-quality images and detect artifacts
Diffusion process may induce low-quality images, and harmful artifacts or regions. In this stage, we use clip directional similarity metric and segmentation models to gradually filter out harmful synthetic samples and artifact regions.


## Step 1: Filter out low-quality images
Please run the following command to filter out low-quality images.
```shell
cd filter
python filter.py --image-dir-1 <> --image-dir-2 <> --save-image-dir <> --dis-image-dir <>> --text-path-1 <>> --text-path-2 <>
```
`--image-dir-1` and `--image-dir-2` are directories to original and synthetic images.
Then the high-quality images will be stored in `--save-image-dir`, low-quality images will be filtered out in `--dis-image-dir`


## Step 2: Detect artifacts region
Please run the following command to detect artifact regions with segmentation model.
```shell
python detector.py --cnofig <config> --checkpoint <checkpoint> --meta_file_path <> --real-img-path <> --real-label-path <> --syn-img-path <> --object-mask-path <> --filtered-label-path <>
```
We use the classical [pre-trained Upernet-R101 model](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/upernet) to calculate per-pixel loss on real images and synthetic image, and then filter out noisy synthetic regions. At last, the filtered semseg label will be stored in `--filtered-label-path`