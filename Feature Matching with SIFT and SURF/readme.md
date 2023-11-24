# Feature Matching with SIFT and SURF

This Python script demonstrates feature matching using SIFT (Scale-Invariant Feature Transform) and SURF (Speeded-Up Robust Features) algorithms. It uses the OpenCV library for image processing and feature extraction.

## Setup

Make sure to install the required libraries before running the script:

```bash
pip install opencv-python
```
# Usage

1. **Load two images (`book.jpg` and `table.jpg`) for feature matching.**

2. **SIFT and SURF Features Extraction:**
   - SIFT features are extracted using the `cv2.SIFT_create()` method.
   - SURF features are extracted using the `cv2.xfeatures2d.SURF_create()` method with a specified Hessian threshold.

3. **Matching using Brute-Force (BF) and FLANN (Fast Library for Approximate Nearest Neighbors) algorithms:**
   - Brute-Force matcher (`cv2.BFMatcher()`) is used for direct feature matching.
   - FLANN matcher (`cv2.FlannBasedMatcher()`) is used for approximate nearest neighbor search.

4. **Ratio Test Filtering:**
   - Matches are filtered using a ratio test to select good matches.

5. **Visualization:**
   - Matched keypoints are visualized in the images, and the number of matches is printed.

# Results

The script generates four output images:

- `SIFT_BF.jpg`: SIFT features matched using Brute-Force.
- `SIFT_FLANN.jpg`: SIFT features matched using FLANN.
- `SURF_BF.jpg`: SURF features matched using Brute-Force.
- `SURF_FLANN.jpg`: SURF features matched using FLANN.

# Dependencies

- OpenCV: [https://opencv.org/](https://opencv.org/)
