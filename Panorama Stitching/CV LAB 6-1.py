import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Minimum number of matches required to consider the object found
MIN_MATCH_COUNT = 10

# Load the images
img1_gray = cv2.imread('/home/upasana/Desktop/CV/boston1.jpeg', cv2.IMREAD_GRAYSCALE)
img1_color = cv2.imread('/home/upasana/Desktop/CV/boston1.jpeg')
img2_color = cv2.imread('/home/upasana/Desktop/CV/boston2.jpeg')
img2_gray = cv2.imread('/home/upasana/Desktop/CV/boston2.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors using SIFT
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# Draw keypoints on the images
img1_with_keypoints = cv2.drawKeypoints(img1_gray, kp1, outImage=None, color=(0, 255, 0))
img2_with_keypoints = cv2.drawKeypoints(img2_gray, kp2, outImage=None, color=(0, 255, 0))

# Display the images with keypoints
cv2.imshow('Keypoints in Image 1', img1_with_keypoints)
cv2.imshow('Keypoints in Image 2', img2_with_keypoints)

# Setup FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Store all the good matches using Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

# If enough good matches are found
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)
    # Print the homography matrix
    print("Homography Matrix:")
    print(M)

    matchesMask = mask.ravel().tolist()

    # Warp the second image using the computed homography matrix
    height_panorama = max(img1_color.shape[0], img2_color.shape[0])
    width_panorama = img1_color.shape[1] + img2_color.shape[1]
    warped_img2 = cv2.warpPerspective(img2_color, M, (width_panorama, height_panorama))
    
    # Overlay img1 onto the panorama canvas
    warped_img2[0:img1_color.shape[0], 0:img1_color.shape[1]] = img1_color
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# Draw matches
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
img3 = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None, **draw_params)

# Display the matches       
cv2.imshow('Feature Matches', img3)

# # Warp the second image to fit the first
# height_panorama = max(img1_color.shape[0], img2_color.shape[0])
# width_panorama = img1_color.shape[1] + img2_color.shape[1]

# # Use warp perspective using the homography matrix M
# warped_img2 = cv2.warpPerspective(img2_color, M, (width_panorama, height_panorama))
# warped_img2[0:img1_color.shape[0], 0:img1_color.shape[1]] = img1_color

# Display the panorama
cv2.imshow('Panorama', warped_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
