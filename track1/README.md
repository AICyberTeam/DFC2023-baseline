# DFC 2023 Track 1 
## Introduction
Track 1 is an building instance segmentation task that focusing on detecting individual buildings and fine-grained classification of building roofs in remote sensing images.
This track has a large number of neighboring building clusters, which challenges the participants' ability to involve the algorithmic process.
## Dataset Format
We provide individual RGB and SAR (Synthetic Aperture Radar) remote sensing images.
For better use of multi-modal data, we provide a python [script](make_merge.py) to generate 4-channel images concatenated in the channel dimension, in 4-channel (R,G,B,SAR) tif format.
You can run the following command to generate the merge direcory:
```
python ./make_merge.py $DATASET_ROOT
```

All the images are in size of 512x512.
The data format follows the MS COCO format, and the annotation is in json format.
The topology of the dataset directory is as follows：

    ```
    DFC_Track_1
    ├── annotations
    │   └── roof_fine_train.json
    ├── train
    │   └── rgb
    │   │   ├── P0001.tif
    │   │   └── ...
    │   │   └── P0009.tif
    │   └── sar
    │   │   ├── P0001.tif
    │   │   └── ...
    │   │   └── P0009.tif
    │   └── merge
    │       ├── P0001.tif
    │       └── ...
    │       └── P0009.tif
    ├── val
    │   └── rgb
    │   │   ├── P0011.tif
    │   │   └── ...
    │   │   └── P0019.tif
    │   └── sar
    │   │   ├── P0011.tif
    │   │   └── ...
    │   │   └── P0019.tif
    │   └── merge
    │       ├── P0011.tif
    │       └── ...
    │       └── P0019.tif
    └── test
        └── rgb
        │   ├── P0021.tif
        │   └── ...
        │   └── P0029.tif
        └── sar
        │   ├── P0021.tif
        │   └── ...
        │   └── P0029.tif
        └── merge
            ├── P0021.tif
            └── ...
            └── P0029.tif
    ```
## Baselines
We choose the classical mask rcnn algorithm as the contest baseline model. Among the input image modalities are RGB and SAR.
We use [MMDetection](https://github.com/open-mmlab/mmdetection) (version 2.25.1) to test the baseline model performance. \
The performance report of Mask R-CNN on the validation set of track 1 (fine-grained classfication and instance segmentation) is as follows:

| Model      | Modality |  mAP  |  mAP_50  |
| ---------- | -------- | :---: | :------: |
| Mask R-CNN | RGB      |  16.3 |   28.1   |
| Mask R-CNN | RGB+SAR  |  10.7 |   19.9   |

The performance report of Mask R-CNN on the validation set of track 2 (building instance segmentation) is as follows:

| Model      | Modality |  mAP  |  mAP_50  |
| ---------- | -------- | :---: | :------: |
| Mask R-CNN | RGB      |  22.8 |   49.1   |
| Mask R-CNN | RGB+SAR  |  18.6 |   43.6   |

Since we use pre-trained weights in RGB modality, four-channel images can not use pre-trained weights directly.
Therefore, the former has higher scores than the latter.
Simple four-channel concatenation does not make better use of the pre-training weights, and some multimodal late fusion methods can be expected to improve the performance.
All the training configs are provided in [configs](configs).

## Submission Format
The submission should be a json format file which consisting of a list with all detected objects.
Each element in the list is a python dictionary object containing four keys: "image_id", "bbox", "category_id" and "segmentation".
The format of values in the dictionary is consistant with COCO format.
We provide the one-to-one relationship between the test image name and image_id in [json](./image_id) format, so you can use it to inference the results.
If you want to quickly generate commit results in this format, you can run the following command from mmdetection.
```
# out: results.bbox.json and results.segm.json (only segm is required)
python tools/test.py $CONFIG $checkpoint --format-only --eval-options "jsonfile_prefix=$SAVE_PATH"
python tools/test.py \
       configs/mask_rcnn_roof_fine.py \
       checkpoint/mask_rcnn_r50_fpn_roof_fine/latest.pth \
       --format-only \
       --eval-options "jsonfile_prefix=./results"
```
The json results of baselines can be referenced in .
