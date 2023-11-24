# Image Transformation and Feature Detection

This repository contains a Python script for performing various image transformations using OpenCV, as well as applying Harris Corner Detection and SIFT Feature Detection to the transformed images. The script is organized into two parts.

## Part 1: Image Transformation

### Requirements
- Python
- OpenCV
- NumPy
- Matplotlib (for image visualization)

### Instructions
1. Make sure you have the required dependencies installed (`pip install opencv-python numpy matplotlib`).
2. Run the script (`python image_transformation.py`) to perform the following transformations:
   - Rotation by 10 degrees
   - Scaling up by 20%
   - Scaling down by 20%
   - Affine transformation
   - Perspective transformation

The transformed images are saved as:
- `rotated.png`
- `scaled_up.png`
- `scaled_down.png`
- `affine.png`
- `perspective.png`

## Part 2: Feature Detection

### Requirements
- Python
- OpenCV

### Instructions
1. Ensure you have the required dependencies installed (`pip install opencv-python`).
2. Run the script (`python feature_detection.py`) to apply Harris Corner Detection and SIFT Feature Detection to all transformed images and the original image.

The results are saved as:
- Harris Corner Detection: `harris_original.png`, `harris_rotated.png`, ...
- SIFT Feature Detection: `sift_original.png`, `sift_rotated.png`, ...

