# Scale-Invariant Feature Transform (SIFT) Implementation

This repository contains a Python script for implementing the Scale-Invariant Feature Transform (SIFT) algorithm. The SIFT algorithm is commonly used for detecting and describing local features in images, making it robust to scale, rotation, and illumination changes.

## Requirements
- Python
- NumPy
- Matplotlib
- PIL (Pillow)

Install the required dependencies using the following command:
```bash
pip install numpy matplotlib
```
## Instructions

### Part 1: Image Processing and Gaussian Blur

The script includes functions for Gaussian blurring, generating Gaussian kernels, and creating Difference of Gaussian (DoG) images. Additionally, there is a function to localize keypoints accurately.

### Part 2: SIFT Keypoint Detection and Visualization

1. **Load an image using the PIL library and convert it to grayscale.**
2. **Specify parameters such as `sigma_min`, `num_intervals`, `k`, and `num_octaves`.**
3. **Generate octaves, blurred images, and DoG images.**
4. **Detect keypoints using the `findKeypoints` function.**
5. **Visualize the DoG images, scale-space extrema detection, and accurate keypoint localization.**


