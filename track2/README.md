# DFC 2023 Track 2 
## Introduction
Track 2 is a multi task track for building instance segmentation and height prediction, focusing on detecting individual buildings, fine-grained classification of building roofs, and height information.
This track has a large number of neighboring building clusters, which challenges the participants' ability to involve the algorithmic process.
## Dataset Format
We provide individual RGB images in 3-channel tif format.
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
    │   │   ├── P0001_RGB.tif
    │   │   └── ...
    │   │   └── P0009_RGB.tif
    │   └── sar
    │   │   ├── P0001_RGB.tif
    │   │   └── ...
    │   │   └── P0009_RGB.tif
    │   └── height
    │       ├── P0001_height.tif
    │       └── ...
    │       └── P0009_height.tif
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

