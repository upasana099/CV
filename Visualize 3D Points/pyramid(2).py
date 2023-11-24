import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load calibration data
mtx = np.load('calibration_matrix.npy')
dist = np.load('distortion_coeffs.npy')

# Define the axis for a square base pyramid (base at z=0)
# base = np.float32([[0, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0], [1.5, 0, 0]])
# apex = np.float32([[0.75, 0.75, -1]])  # Apex position remains the same

# Define the new base points
base = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0]])

# Define the new apex point
apex = np.float32([[1, 1, -1.5]])

# # Plot the base points
# plt.scatter(base[:, 0], base[:, 1], color='blue', label='Base Points')

# # Plot the apex point
# plt.scatter(apex[0][0], apex[0][1], color='red', label='Apex Point')x

# # Add labels and legend
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()

# # Show the plot
# plt.show()


def visualize_3d_points(base, apex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the base points in blue
    ax.scatter(base[:, 0], base[:, 1], base[:, 2], c='b', marker='o')
    # Plot the apex point in red
    ax.scatter(apex[0][0], apex[0][1], apex[0][2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# # Concatenate the base and apex points into a single array
# pyramid_points = np.concatenate((base, apex), axis=0)

# Call the function to visualize the points
visualize_3d_points(base, apex)


# # For a pyramid 
# def draw_pyramid(img, corners, imgpts):
#     imgpts = np.int32(imgpts).reshape(-1, 2)
#     base = imgpts[:4]
#     img = cv.drawContours(img, [base], -1, (0, 0, 255), -3)
#     for i in range(4):
#         img = cv.line(img, tuple(base[i]), tuple(imgpts[4]), (0, 0, 0), 3)

#     # Optionally draw the bottom of the base if you want it to be visible
#     img = cv.line(img, tuple(base[0]), tuple(base[1]), (0, 0, 0), 3)
#     img = cv.line(img, tuple(base[1]), tuple(base[2]), (0, 0, 0), 3)
#     img = cv.line(img, tuple(base[2]), tuple(base[3]), (0, 0, 0), 3)
#     img = cv.line(img, tuple(base[3]), tuple(base[0]), (0, 0, 0), 3)

#     return img


def draw_pyramid(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    base = imgpts[:4]

    # Create a copy of the original image to draw on
    img_draw = img.copy()

    # Draw the base on the copy with red color
    img_draw = cv.drawContours(img_draw, [base], -1, (0, 0, 255), -3)

    # Blend the original image with the drawn image to create a transparency effect
    img = cv.addWeighted(img, 0.7, img_draw, 0.0, 0)

    for i in range(4):
        img = cv.line(img, tuple(base[i]), tuple(imgpts[4]), (0, 255, 0), 2)

    # Draw the bottom of the base
    img = cv.line(img, tuple(base[0]), tuple(base[1]), (0, 255, 0), 2)
    img = cv.line(img, tuple(base[1]), tuple(base[2]), (0, 255, 0), 2)
    img = cv.line(img, tuple(base[2]), tuple(base[3]), (0, 255, 0), 2)
    img = cv.line(img, tuple(base[3]), tuple(base[0]), (0, 255, 0), 2)

    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Define the axis for the pyramid with a height of 1 and a smaller base
axis = np.float32([[0, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0], [1.5, 0, 0],
                   [0.75, 0.75, -1]])  # Apex position remains the same

# List of image file names
image_files = ['/home/upasana/Desktop/CV/left01.jpg', '/home/upasana/Desktop/CV/left02.jpg', '/home/upasana/Desktop/CV/left03.jpg' , '/home/upasana/Desktop/CV/left04.jpg', '/home/upasana/Desktop/CV/left05.jpg','/home/upasana/Desktop/CV/left06.jpg','/home/upasana/Desktop/CV/left07.jpg','/home/upasana/Desktop/CV/left08.jpg']

for fname in image_files:
    print("Processing:", fname)
    img = cv.imread(fname)
    if img is None:
        print("Error: Unable to load image:", fname)
        continue
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print("Corners2:\n", corners2)  # Print the detected corners

        # Draw the corners on the imagef
        img = cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow('Corners', img)
        cv.waitKey(0)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        print("Rvecs:\n", rvecs)  # Print the rotation vectors
        print("Tvecs:\n", tvecs)  # Print the translation vectors
        # Project 3D points to the image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        print("Image Points (imgpts):\n", imgpts)  # Print the projected image points
        img = draw_pyramid(img, corners2, imgpts)
        cv.imshow('img', img)
        print("Displaying window for:", fname)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:-4] + '_output.jpg', img)  # Save the output image
    else:
        print("Chessboard corners not found for image:", fname)

cv.destroyAllWindows()
