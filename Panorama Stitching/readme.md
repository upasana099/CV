# Panoram Stitching with SIFT and OpenCV

This Python script uses the Scale-Invariant Feature Transform (SIFT) algorithm and OpenCV to perform image stitching. It matches keypoints between two images and creates a panorama by applying a homography transformation.

## Features

- **Keypoint Detection:** Utilizes the SIFT algorithm to detect keypoints in grayscale images.
- **Keypoint Matching:** Matches keypoints between two images using the FLANN-based matcher.
- **Homography Estimation:** Computes a homography matrix using the RANSAC algorithm.
- **Image Warping:** Warps the second image to align with the first based on the computed homography.
- **Panorama Creation:** Combines the two images to create a panorama.

## Usage

1. **Run the Script:** Execute the script in a Python environment.
2. **Input Images:** Modify the file paths in the script to specify the input images (`img1_color`, `img2_color`).
3. **View Output:** The script displays images with keypoints, matches, and the final panorama.

## Output Images

Keypoints in image 1:

![Keypoints in Image 1-1](https://github.com/upasana099/CV/assets/89516193/e7f14ca8-4cbc-4a08-b5b4-11bc1226d9ab)

Keypoints in image 2:

![Keypoints in Image 2-1](https://github.com/upasana099/CV/assets/89516193/4e5c7f91-63ed-401d-aa9c-89ba79d0a68d)

Feature Matches:

![Feature Matches-1](https://github.com/upasana099/CV/assets/89516193/19117b05-9e2d-4364-9696-6813e2d49954)

Stitched Image:
![Panorama-1](https://github.com/upasana099/CV/assets/89516193/6a13ef4a-892c-4e2b-9a1f-34d69b93a9c9)


## Dependencies

- OpenCV: 4.x

## Setup

```bash
pip install opencv-python
```
