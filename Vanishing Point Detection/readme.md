# Vanishing Point Detection

This Python script performs vanishing point detection using Hough Transform and DBSCAN clustering. The script reads an input image, applies Canny Edge Detection, and then utilizes the Hough Transform to detect lines in the image. The intersections of these lines are found and clustered using DBSCAN to identify the vanishing point.


<img width="617" alt="texas" src="https://github.com/upasana099/CV/assets/89516193/88a5141a-03cb-4f62-ba1a-581dd7453079">



## Usage

1. **Input Image:** The script reads an input image, e.g., 'texas.png', and converts it to grayscale.

2. **Canny Edge Detection:** The grayscale image is processed using Canny Edge Detection to highlight edges.

3. **Hough Transform:** The script applies the Hough Transform to detect lines in the Canny edges image.

4. **Draw Detected Lines:** Detected lines are drawn on a copy of the original image.

5. **Find Intersections:** The script finds intersections between lines using the Hough Transform.

6. **DBSCAN Clustering:** DBSCAN clustering is applied to find the largest cluster of intersections.

7. **Least Squares to Find Vanishing Point:** Least squares method is used to find the best estimate for the vanishing point.

8. **Draw Vanishing Point:** The vanishing point is marked on the original image.

## Output 

Canny Edge:

![Canny Edges](https://github.com/upasana099/CV/assets/89516193/969c3ddc-d9ea-487f-b31a-ac68e5ef0f68)


Detected Lines :

![Detected Lines](https://github.com/upasana099/CV/assets/89516193/cc676d97-a798-4f6d-ae85-3741020a9bc2)


Vanishing Point :

![Vanishing Point](https://github.com/upasana099/CV/assets/89516193/0af5ba79-c5f8-4f02-b3db-a262eaf46d48)



## Dependencies

- OpenCV (`cv2`)
- NumPy
- scikit-learn (`sklearn`)

## Instructions

Ensure the required dependencies are installed (`opencv-python`, `numpy`, `scikit-learn`).

   ```bash
   pip install opencv-python numpy scikit-learn
```

