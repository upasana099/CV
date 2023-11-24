# Visualize 3D Points and Draw Pyramid on Chessboard Images

This Python script utilizes OpenCV and NumPy to visualize 3D points and draw a pyramid on a set of chessboard images. The script performs the following tasks:

1. **Load Calibration Data:**
   - Loads camera calibration data, including the calibration matrix (`calibration_matrix.npy`) and distortion coefficients (`distortion_coeffs.npy`).

2. **Define 3D Points:**
   - Defines the 3D points representing the base and apex of a pyramid.

3. **Visualize 3D Points:**
   - Displays a 3D scatter plot of the base and apex points using Matplotlib.

4. **Draw Pyramid on Chessboard Images:**
   - For a set of chessboard images (`left01.jpg` to `left08.jpg`), the script detects chessboard corners, finds rotation and translation vectors using camera calibration data, and draws a pyramid on the images.

5. **Display Images:**
   - Displays each image with the drawn pyramid.
   - Press 's' to save the output images with appended '_output.jpg'.


## Output 

![triangular](https://github.com/upasana099/CV/assets/89516193/e71791fa-e084-4e1d-90db-4ec9fd2eb10c)


## Instructions

1. Ensure that the required calibration data files (`calibration_matrix.npy` and `distortion_coeffs.npy`) are available in the script directory.

2. Run the script using the following command:

3. Observe the 3D scatter plot of points and view the images with drawn pyramids.

4. Press 's' while viewing an image to save the corresponding output image.
