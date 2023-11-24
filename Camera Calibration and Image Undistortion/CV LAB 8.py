import numpy as np
import cv2

# bject points

# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

objp = np.zeros((11*7,3), np.float32)
objp[:,:2] = np.mgrid[0:11, 0:7].T.reshape(-1,2)


# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real-world space
imgpoints = [] # 2d points in the image plane

# Load each image from the specified directory
# image_files = ['/home/upasana/Desktop/CV/left{:02d}.jpg'.format(i) for i in list(range(1, 10)) + list(range(11, 15))]
# image_files = ['/home/upasana/Desktop/CV/caliberation_data/IMG_65{:02d}.jpg'.format(i) for i in range(2, 29)]
image_files = ['/home/upasana/Desktop/CV/rightcamera/Im_R_{}.png'.format(i) for i in range(1, 21)]

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print("Error loading image:", fname)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    # ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    ret, corners = cv2.findChessboardCorners(gray, (11,7), None)


    # If found, add object points and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

         # Draw the corners on the image and display 
        cv2.drawChessboardCorners(img, (11,7), corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Calculate re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

re_projection_error = mean_error/len(objpoints)

# Print and save the results
print("Calibration Matrix:\n", mtx)
print("\nDistortion Coefficients:\n", dist)
print("\nRe-projection Error:", re_projection_error)

# np.save('calibration_matrix.npy', mtx)
# np.save('distortion_coeffs.npy', dist)
# np.save('re_projection_error.npy', re_projection_error)


for fname in image_files:
   
    img = cv2.imread(fname)

   
    if img is None:
        print("Error loading image:", fname)
        continue

    # Undistort the image
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    # Stack the original and undistorted images side by side
    side_by_side = np.hstack((img, undistorted_img))

   
    cv2.imshow('Original vs Undistorted - {}'.format(fname.split("/")[-1]), side_by_side)
    
    
    cv2.waitKey(500)
    cv2.destroyAllWindows()



