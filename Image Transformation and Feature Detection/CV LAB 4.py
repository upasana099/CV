import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image  
image = cv2.imread('UnityHall.png')
print(image.shape)
assert image is not None, "file could not be read, check with os.path.exists()"


# # Convert from BGR to RGB
# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # Show the image using Matplotlib
# plt.figure(figsize=(10, 10))
# plt.title('Image')
# plt.imshow(img_rgb)
# # plt.axis('off')  # Hide axes for better visualization
# plt.show()


## PART 1

# 1) Rotation by 10 degrees
rows, cols = image.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
rotated = cv2.warpAffine(image, M, (cols, rows))

# 2) Scale up (+20%)
res_up = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_LINEAR)

# 3) Scale down (-20%)
res_down = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation = cv2.INTER_LINEAR)

# 4) Affine transformation
# Specify three points from the input image and their corresponding locations in the output image
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
A = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(image, A, (cols, rows))

# 5) Perspective transformation
pts1 = np.float32([[60, 60], [500, 50], [20, 275], [500, 275]])
pts2 = np.float32([[50, 100], [450, 0], [50, 300], [430, 250]])
P = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(image, P, (cols, rows))

# Saving transformed images
cv2.imwrite('rotated.png', rotated)
cv2.imwrite('scaled_up.png', res_up)
cv2.imwrite('scaled_down.png', res_down)
cv2.imwrite('affine.png', affine)
cv2.imwrite('perspective.png', perspective)



## PART 2

# Harris Corner Detection
def apply_harris(image, filename):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Convert the grayscale image back to a color image
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Mark the detected corners in red
    gray_color[dst > 0.001 * dst.max()] = [0, 0, 255]
    
    cv2.imwrite(filename, gray_color)

# SIFT Feature Detection
def apply_sift(image, filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    cv2.drawKeypoints(image, kp, image, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(filename, image)

# Applying to all transformed images plus the original one
images = [image, rotated, res_up, res_down, affine, perspective]
names = ['original', 'rotated', 'scaled_up', 'scaled_down', 'affine', 'perspective']

for i, img in enumerate(images):
    apply_harris(img.copy(), f'harris_{names[i]}.png')
    apply_sift(img.copy(), f'sift_{names[i]}.png')
