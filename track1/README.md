# DFC 2023 Track 1 
## Introduction
Track 1 is an building instance segmentation task that focusing on detecting individual buildings and fine-grained classification of building roofs in remote sensing images.
This track has a large number of neighboring building clusters, which challenges the participants' ability to involve the algorithmic process.
## Dataset Format
We provide individual RGB and SAR (Synthetic Aperture Radar) remote sensing images, as well as 4-channel images (.MERGE) concatenated in the channel dimension, in 4-channel (R,G,B,SAR) tif format.

All the images are in size of 512x512.
The data format follows the MS COCO format, and the annotation is in json format.
The topology of the dataset directory is as follows：

    ```
    DFC_Track_1
    ├── annotations
    │   └── roof_fine_train_merge.json
    │   └── roof_fine_val_merge.json
    │   └── roof_fine_test_merge.json
    ├── train
    │   └── rgb
    │   │   ├── P0001_RGB.tif
    │   │   └── ...
    │   │   └── P0009_RGB.tif
    │   └── sar
    │   │   ├── P0001_SAR.tif
    │   │   └── ...
    │   │   └── P0009_SAR.tif
    │   └── merge
    │       ├── P0001_MERGE.tif
    │       └── ...
    │       └── P0009_MERGE.tif
    ├── val
    │   └── rgb
    │   │   ├── P0011_RGB.tif
    │   │   └── ...
    │   │   └── P0019_RGB.tif
    │   └── sar
    │   │   ├── P0011_SAR.tif
    │   │   └── ...
    │   │   └── P0019_SAR.tif
    │   └── merge
    │       ├── P0011_MERGE.tif
    │       └── ...
    │       └── P0019_MERGE.tif
    └── test
        └── rgb
        │   ├── P0021_RGB.tif
        │   └── ...
        │   └── P0029_RGB.tif
        └── sar
        │   ├── P0021_SAR.tif
        │   └── ...
        │   └── P0029_SAR.tif
        └── merge
            ├── P0021_MERGE.tif
            └── ...
            └── P0029_MERGE.tif
    ```
## Baselines
We choose the classical mask rcnn algorithm as the contest baseline model. Among the input image modalities are RGB and SAR.
We use MMDetection to test the baseline model performance.
The performance report of Mask R-CNN on the fine-grained building instance segmentation and roof classification is as follows:

| Model      | Modality | mAP | mAP_50 |
| ---------- | -------- | --- | ------ |
| Mask R-CNN | RGB      | 13.2|  22.9  |
| Mask R-CNN | RGB+SAR  | 7.9 |  15.4  |

## Submission Format
The submission should be a json format file.
We provide demos of json files.
If you want to quickly generate commit results in this format, you can run the following command from mmdetection.
```
# out: results.bbox.json and results.segm.json
python tools/test.py $CONFIG $checkpoint --format-only --eval-options "jsonfile_prefix=./results"
python tools/test.py \
       configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       --format-only \
       --options "jsonfile_prefix=./results"
```
