# Stereo Epipolar Geometry Visualization

This Python script uses OpenCV and Matplotlib to visualize stereo epipolar geometry between three images of a globe: left (`globe_left.jpg`), center (`globe_center.jpg`), and right (`globe_right.jpg`). The script performs the following tasks:

1. **Load Images:**
   - Loads three grayscale images representing the left, center, and right views of a globe.
  
![globe_left](https://github.com/upasana099/CV/assets/89516193/db7a45be-54ad-4e6c-9267-96196e0c465f)

![globe_center](https://github.com/upasana099/CV/assets/89516193/101f4bc9-f211-40e0-ad01-52773e08f1c6)


![globe_right](https://github.com/upasana099/CV/assets/89516193/b4fa41d4-2eed-4bbb-a138-614df4ce7368)



2. **SIFT Feature Matching:**
   - Uses the SIFT (Scale-Invariant Feature Transform) algorithm to detect and compute keypoints and descriptors in the left and center images.
   - Performs feature matching between the left and center images using the Brute-Force matcher.
   - Applies the ratio test to select good matches.

3. **Fundamental Matrix Estimation:**
   - Finds the fundamental matrix (`F`) using the RANSAC-based method (`FM_LMEDS`) and a confidence level of 0.99.
   - Selects inlier points based on the fundamental matrix.

4. **Epipolar Lines Drawing:**
   - Draws epipolar lines corresponding to the inlier points on the left and center images.
   - Displays the left image with drawn epipolar lines from the center image (`img5`).
   - Displays the center image with drawn epipolar lines from the left image (`img3`).
   - Adds an extra visualization of the center-right stereo pair (`img_center_right`).

5. **Visualization:**
   - Displays the resulting visualizations using Matplotlib.
  
## Output :

![Epipolar Lines and Points_1](https://github.com/upasana099/CV/assets/89516193/5be6cc2e-b085-4172-8f1c-a2e39ae01dab)



![Epipolar Lines and Points_2](https://github.com/upasana099/CV/assets/89516193/b69d4f59-6030-4c62-8753-52ff1b72e88c)


## Instructions

1. Ensure that the required images (`globe_left.jpg`, `globe_center.jpg`, `globe_right.jpg`) are available in the specified file paths.

2. Run the script
3. Observe the stereo epipolar geometry visualizations.
  
