# Pointcloud-Segmentation

## Overview
In this project, I used [Kitti360](https://www.cvlibs.net/datasets/kitti-360/demo.php) dataset to give pointcloud semantic labels using segmentation obtained from a camera image of the scene. I used an implementation of PSPNet to generate semantic labels on the image. After applying semantic labels on multiple pointclouds, I used ICP registration to generate the mapping of the complete scene.

This repository does the following tasks.
1. Intrinsic calibration of perspective camera (Point Gray Flea 2).
2. Extrinsic calibration of the perspective camera with a Velodyne Lidar (HDL-64E).
3. Projecting lidar points on to the image plane (run ```python src/utils.py```).
4. Generating a colored pointcloud (run ```python src/gen_color_pcd.py```).
5. Applying semantic labels on the pointcloud and registering multiple pointclouds using ICP based registration(run ```python src/pointcloud_segmentation.py```).

This repository consists of comparison of baseline edge detection algorithms like Canny and Sobel.

with [Probability of boundary detection algorithm](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf). A simpler version of PB algorithm has been implemented which considers texture, color and intensity discontinuities. This algorithm predicts per pixel probability of the boundary detected. The original image and the output of implementation is shown below:

<img src="BSDS500/Images/1.jpg" align="center" alt="Original" width="400"/> <img src="Results/color_gradient_maps/1.jpg" align="center" alt="PBLite" width="400"/>

The algorithm of PBLite detection is shown below:

<img src="Results/hw0.png" align="center" alt="PBLite"/>

The main steps for implementing the same are:

## Step 1: Feature extraction using Filtering
The filter banks implemented for low-level feature extraction are Oriented Derivative if Gaussian Filters, Leung-Malik Filters (multi-scale) and Gabor Filter.

<img src="Results/DOG.png" align="center" alt="DoG" width="250"/> <img src="Results/LM.png" align="center" alt="PBLite" width="250"/> <img src="Results/Gabor.png" align="center" alt="PBLite" width="250"/>

## Step 2: Extracting texture, color and brightness using clustering
Filter banks can be used for extraction of texture properties but here all the three filter banks are combined which results into vector of filter responses. As filter response vectors are generated, they are clustered together using k-means clustering. For Texton Maps k = 64 is used; Color and Brightness Maps k= 16 is used.


<img src="Results/texture_maps/1.jpg" align="center" alt="DoG" width="250"/> <img src="Results/color_maps/1.jpg" align="center" alt="PBLite" width="250"/> <img src="Results/intensity_maps/1.jpg" align="center" alt="PBLite" width="250"/>

The gradient measurement is performed to know how much all features distribution is changing at a given pixel. For this purpose, half-disc masks are used.

<img src="Results/texture_gradient_maps/1.jpg" align="center" alt="PBLite" width="250"/> <img src="Results/color_gradient_maps/1.jpg" align="center" alt="PBLite" width="250"/> <img src="Results/intensity_gradient_maps/1.jpg" align="center" alt="PBLite" width="250"/>

## Step 3: Pb-Score
The gradient maps which are generated are combined with classical edge detectors like Canny and Sobel baselines for weighted average.

<img src="Results/final_output/1.jpg" align="center" alt="output" />

## Run Instructions
```
python Wrapper.py
```
# File structure
    ├── Code
    |  ├── Wrapper.py
    |  ├── utils.py
    ├── BSDS500
    ├── Results
    |  ├── color_gradient_maps
    |  ├── color_maps
    |  ├── intensity_gradient_maps
    |  ├── intensiy_maps
    |  ├── final_output
    |  ├── texture_maps
    |  ├── texture_gradient_maps
    