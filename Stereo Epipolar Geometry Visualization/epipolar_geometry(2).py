import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load images
img1 = cv.imread('/home/upasana/Desktop/CV/globe_left.jpg', 0)   # left images
img2 = cv.imread('/home/upasana/Desktop/CV/globe_center.jpg', 0) # center images
img3 = cv.imread('/home/upasana/Desktop/CV/globe_right.jpg', 0)  # right images

sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#BF
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
pts1 = []   
pts2 = []


#Draw and Visualize the sift and flann matches
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv.imshow('Matches',img3)
# cv.waitKey(0)

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 1*n.distance: #left center , 1 threshold
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS,0.99)#,0.99)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]






def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


# Find the keypoints and descriptors with SIFT for the right image
kp3, des3 = sift.detectAndCompute(img3,None)

# Match descriptors between center and right images
matches = bf.knnMatch(des2,des3,k=2)

good = []
pts2 = []   
pts3 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance: #center right , 0.75 threshold
        good.append(m)
        pts3.append(kp3[m.trainIdx].pt)
        pts2.append(kp2[m.queryIdx].pt)

pts2 = np.int32(pts2)
pts3 = np.int32(pts3)

F, mask = cv.findFundamentalMat(pts2,pts3,cv.FM_LMEDS,0.99)
# We select only inlier points
pts2 = pts2[mask.ravel()==1]
pts3 = pts3[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts3.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img_center_right, _ = drawlines(img2,img3,lines1,pts2,pts3)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.subplot(133),plt.imshow(img_center_right)  # Add this line
plt.show()
plt.show()

