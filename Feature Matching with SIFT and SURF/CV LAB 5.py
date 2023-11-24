import cv2
import numpy as np

# Load the images
book_img = cv2.imread('/home/upasana/Desktop/CV/book.jpg', 0)
table_img = cv2.imread('/home/upasana/Desktop/CV/table.jpg', 0)

# Verify that the images are loaded correctly
if book_img is None:
    print("Failed to load book.jpg")
    exit()

if table_img is None:
    print("Failed to load table.jpg")
    exit()

## SIFT features
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(book_img, None)
kp2_sift, des2_sift = sift.detectAndCompute(table_img, None)


hessian_threshold = 400
surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
# # SURF features
# surf = cv2.xfeatures2d.SURF_create()
kp1_surf, des1_surf = surf.detectAndCompute(book_img, None)
kp2_surf, des2_surf = surf.detectAndCompute(table_img, None)

# Brute-Force matcher
bf = cv2.BFMatcher()

# FLANN parameters and matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Function for ratio test filtering
def filter_matches(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.55 * n.distance:
            good_matches.append(m)
    return good_matches


def draw_matched_keypoints(image, keypoints, matches, color=(0, 0, 255)):
    ''' Draw only the keypoints that are matched as '+' signs '''
    img = image.copy()
    for match in matches:
        keypoint = keypoints[match.queryIdx]
        x, y = keypoint.pt
        x = int(x)
        y = int(y)
        cv2.line(img, (x-5, y), (x+5, y), color, 2)  # Horizontal line
        cv2.line(img, (x, y-5), (x, y+5), color, 2)  # Vertical line
    return img


# 1) SIFT and Brute-Force
matches_sift_bf = bf.knnMatch(des1_sift, des2_sift, k=2)
good_matches_sift_bf = filter_matches(matches_sift_bf)
img_sift_bf_matched_kps = draw_matched_keypoints(book_img, kp1_sift, good_matches_sift_bf)
img_sift_bf = cv2.drawMatches(img_sift_bf_matched_kps, kp1_sift, table_img, kp2_sift, good_matches_sift_bf, None, matchColor=(0,255,0))
cv2.imwrite("SIFT_BF.jpg", img_sift_bf)
print("Number of SIFT & BF matches:", len(good_matches_sift_bf))

# 2) SIFT and FLANN
matches_sift_flann = flann.knnMatch(des1_sift, des2_sift, k=2)
good_matches_sift_flann = filter_matches(matches_sift_flann)
img_sift_flann_matched_kps = draw_matched_keypoints(book_img, kp1_sift, good_matches_sift_flann)
img_sift_flann = cv2.drawMatches(img_sift_flann_matched_kps, kp1_sift, table_img, kp2_sift, good_matches_sift_flann, None, matchColor=(0,255,0))
cv2.imwrite("SIFT_FLANN.jpg", img_sift_flann)
print("Number of SIFT & FLANN matches:", len(good_matches_sift_flann))

# 3) SURF and Brute-Force
matches_surf_bf = bf.knnMatch(des1_surf, des2_surf, k=2)
good_matches_surf_bf = filter_matches(matches_surf_bf)
img_surf_bf_matched_kps = draw_matched_keypoints(book_img, kp1_surf, good_matches_surf_bf)
img_surf_bf = cv2.drawMatches(img_surf_bf_matched_kps, kp1_surf, table_img, kp2_surf, good_matches_surf_bf, None, matchColor=(0,255,0))
cv2.imwrite("SURF_BF.jpg", img_surf_bf)
print("Number of SURF & BF matches:", len(good_matches_surf_bf))

# 4) SURF and FLANN
matches_surf_flann = flann.knnMatch(des1_surf, des2_surf, k=2)
good_matches_surf_flann = filter_matches(matches_surf_flann)
img_surf_flann_matched_kps = draw_matched_keypoints(book_img, kp1_surf, good_matches_surf_flann)
img_surf_flann = cv2.drawMatches(img_surf_flann_matched_kps, kp1_surf, table_img, kp2_surf, good_matches_surf_flann, None, matchColor=(0,255,0))
cv2.imwrite("SURF_FLANN.jpg", img_surf_flann)
print("Number of SURF & FLANN matches:", len(good_matches_surf_flann))



cv2.waitKey(0)
cv2.destroyAllWindows()

