import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN


def find_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    
    A_matrix = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    
    b_vector = np.array([[rho1], [rho2]])
    
    # Check for singularity
    if np.linalg.det(A_matrix) == 0:
    
        return None
    
    # intersection 
    x, y = np.linalg.solve(A_matrix, b_vector)
    return np.array([float(x), float(y)])

input_image = cv.imread('/home/upasana/Desktop/CV/texas.png')

# grayscale for processing
gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

# Canny Edge Detection 
blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
canny_edges = cv.Canny(blurred_image, 100, 200)
cv.imwrite('Canny Edges.png', canny_edges)
cv.waitKey(0)

# Hough Transform 
lines = cv.HoughLines(canny_edges, 1, np.pi / 180, 150, None, 0, 0)


image_with_lines = np.copy(input_image)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        point1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        point2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        cv.line(image_with_lines, point1, point2, (0, 0, 0), 3, cv.LINE_AA)

cv.imwrite('Detected Lines.png', image_with_lines)
cv.waitKey(0) 

# image point that represents the vanishing point
intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        intersection = find_intersection(lines[i], lines[j])
        if intersection is not None:
            intersections.append(intersection)

intersections_array = np.array(intersections)

# DBSCAN to find the largest cluster
dbscan = DBSCAN(eps=10, min_samples=2).fit(intersections_array)
labels = dbscan.labels_


largest_cluster_label = max(set(labels), key=list(labels).count)

# least squares to find the best intersection
A_matrix = np.array([[1, -pt[0]] for pt in intersections], dtype=np.float64)
b_vector = np.array([pt[1] for pt in intersections], dtype=np.float64)


AtA_inv = np.linalg.inv(A_matrix.T.dot(A_matrix))
Atb = A_matrix.T.dot(b_vector)
t = AtA_inv.dot(Atb)


cluster_points = intersections_array[labels == largest_cluster_label]
vanishing_point = np.mean(cluster_points, axis=0).astype(int)

# Draw the vanishing point on the original image
cv.circle(input_image, (vanishing_point[0], vanishing_point[1]), 15, (0, 0, 255), -1)
cv.imwrite('Vanishing Point.png', input_image)
cv.waitKey(0)

cv.destroyAllWindows()
