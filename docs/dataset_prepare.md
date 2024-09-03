# Prepare datasets
Before generate variations of exisiting segmentation dataset, you should download original datasets first. Then in the dataset preprocess stage, we select one salient object from each sample to conduct local editing in order to guarantee the quality of edited images. We support experiments on PASCAL VOC 2021, ADE20K, COCO Stuff 164k and Cityscapes.

## Pascal VOC
1. Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). Please run dataset preparation scripts from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-voc).
2. Before dataset prepapre, please re-organize the file structure like this:
    ```none
    data
    ├── pascal_voc
    │   ├── images
    │   │   │── train
    │   │   │── val
    │   ├── annotations_mmseg
    │   │   ├── train
    │   │   └── val
    ``` 
3. Then please run following command to convert files into proper format.
    ```shell
    python datasets_prepare/pascal_voc.py --data-root $your_data_path --data-save-root $your_data_save_path
    ```


## ADE20K
1. The ADE20K dataset could be download from [here](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
2. Then please run following command to convert files into proper format.
   ```shell
   python datasets_prepare/ade20k.py --data-root $your_data_path --data-save-root $your_data_save_path
   ```

## COCO Stuff 164k
1. For downloading data and converting annotations, please refer to guidance from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-164k)
2. After prepare the dataset, the files should be 
    ```none
    data
    ├── coco_stuff164k
    │   ├── images
    │   │   │── train2017
    │   │   │── val2017
    │   ├── annotations
    │   │   ├── train2017
    │   │   ├── val2017
    │   │   ├── train2017.json 
    │   │   └── val2017.json
    ```
3. Then please run following command to convert files into proper format.
   ```shell
   python datasets_prepare/coco_stuff164k.py --data-root $your_data_path --data-save-root $your_data_save_path
   ```


After dataset preprocess, the files of each dataset should be like this:
```none
data
├── datasets_name
│   ├── images
│   │   │── train
│   │   │── val
│   ├── annotations # semantic segmentation annotations in mmseg format
│   │   ├── train
│   │   └── val
│   ├── masks_object # mask within [0, 255]
│   │   ├── train
│   │   └── val
│   ├── meta_train.json # includes categories info of selected objects
│   ├── meta_val.json
```