# DFC 2023 Track 2 
## Introduction
Track 2 is a multi task track for building instance segmentation and height prediction, focusing on detecting individual buildings, fine-grained classification of building roofs, and height information.
This track has a large number of neighboring building clusters, which challenges the participants' ability to involve the algorithmic process.
## Dataset Format
We provide individual RGB and SAR (Synthetic Aperture Radar) remote sensing images.
For better use of multi-modal data, we provide a python [script](make_merge.py) to generate 4-channel images concatenated in the channel dimension, in 4-channel (R,G,B,SAR) tif format.
You can run the following command to generate the merge direcory:
```
python ./make_merge.py $DATASET_ROOT
```
All the images are in size of 512x512.
In the instance segmentation sub-task, the data format follows the MS COCO format, and the annotation uses the json format. 
In the height prediction sub-task, the data annotation adopts the height ground true formed by the pixel by pixel elevation value corresponding to the RGB images, 
and the data uses tif format.

The topology of the dataset directory is as follows：

    ```
    DFC_Track_2
    ├── annotations
    │   └── buildings_only_train.json
    │   └── buildings_only_val.json
    │   └── buildings_only_test.json
    ├── train
    │   └── rgb
    │   │   ├── P0001.tif
    │   │   └── ...
    │   │   └── P0009.tif
    │   └── sar
    │   │   ├── P0001.tif
    │   │   └── ...
    │   │   └── P0009.tif
    │   └── height
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
    │   └── height
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
        └── ndsm
            ├── P0021.tif
            └── ...
            └── P0029.tif
    ```

## Baselines
We choose the classical mask rcnn with multimodal multitask learning (height prediction) framework as the contest baseline model. Among the input image modalities are RGB and SAR.
We use [MMDetection](https://github.com/open-mmlab/mmdetection) (version 2.25.1) to test the baseline model performance. \
The performance report of multimodal multitask learning (MML) framework on the validation set of track 2 (instance segmentation and height prediction) is as follows:

| Model      | Modality |  mAP  |  mAP_50  |  Delta_1  |  Delta_2  |  Delta_3  |
| ---------- | -------- | :---: | :------: |  :-----:  |  :-----:  |  :-----:  |
|    MML     | RGB+SAR  |  15.0 |   40.8   |   29.9    |   35.1    |    39.4   |

## Submission Format
The documents submitted should be a folder. 
The topology of the submitted floder directory is as follows：

    ```
    DFC_Track_2_submit
    ├── results.json
    └── height
        ├── P0011.tif
        └── P0012.tif
        └── ...
        └── P0019.tif
    ```

