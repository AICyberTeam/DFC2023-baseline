# DFC 2023 Track 2 
## Introduction
Track 2 is a multi task track for building instance segmentation and DSM prediction, focusing on detecting individual buildings, fine-grained classification of building roofs, and elevation (DSM) information.
This track has a large number of neighboring building clusters, which challenges the participants' ability to involve the algorithmic process.
## Dataset Format
We provide individual RGB images in 3-channel tif format.
All the images are in size of 512x512.
In the instance segmentation sub-task, the data format follows the MS COCO format, and the annotation uses the json format. 
In the DSM prediction sub-task, the data annotation adopts the DSM ground true formed by the pixel by pixel elevation value corresponding to the RGB images, 
and the data uses tif format.
The topology of the dataset directory is as follows：

    ```
    DFC_Track_2
    ├── annotations
    │   └── roof_fine_train.json
    │   └── roof_fine_val.json
    │   └── roof_fine_test.json
    ├── train
    │   └── rgb
    │   │   ├── P0001_RGB.tif
    │   │   └── ...
    │   │   └── P0009_RGB.tif
    │   └── dsm
    │       ├── P0001_DSM.tif
    │       └── ...
    │       └── P0009_DSM.tif
    ├── val
    │   └── rgb
    │   │   ├── P0011_RGB.tif
    │   │   └── ...
    │   │   └── P0019_RGB.tif
    │   └── dsm
    │       ├── P0011_DSM.tif
    │       └── ...
    │       └── P0019_DSM.tif
    └── test
        └── rgb
        │   ├── P0021_RGB.tif
        │   └── ...
        │   └── P0029_RGB.tif
        └── dsm
            ├── P0021_DSM.tif
            └── ...
            └── P0029_DSM.tif
    ```
## Baselines


## Submission Format
The documents submitted should be a folder. 
The topology of the submitted floder directory is as follows：

    ```
    DFC_Track_2_submit
    ├── roof_fine_results.json
    └── dsm_results
        ├── P0011_Pred.tif
        └── P0012_Pred.tif
        └── ...
        └── P0019_Pred.tif
    ```

