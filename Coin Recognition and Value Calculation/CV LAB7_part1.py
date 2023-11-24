import cv2
import numpy as np

# Read all images
image = cv2.imread('/home/upasana/Desktop/CV/coin1.jpeg', cv2.IMREAD_COLOR)
five_cent_reference = cv2.imread('/home/upasana/Desktop/CV/fivecent.jpeg', cv2.IMREAD_COLOR)
quarter_reference = cv2.imread('/home/upasana/Desktop/CV/quater.jpeg', cv2.IMREAD_COLOR)

# Convert all images to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_five_cent = cv2.cvtColor(five_cent_reference, cv2.COLOR_BGR2GRAY)
gray_quarter = cv2.cvtColor(quarter_reference, cv2.COLOR_BGR2GRAY)

# Use GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_main = clahe.apply(gray)
enhanced_five_cent = clahe.apply(gray_five_cent)
enhanced_quarter = clahe.apply(gray_quarter)

sift = cv2.SIFT_create()

# Extract SIFT features from the enhanced images
kp_main, des_main = sift.detectAndCompute(enhanced_main, None)
kp_five_cent, des_five_cent = sift.detectAndCompute(enhanced_five_cent, None)
kp_quarter, des_quarter = sift.detectAndCompute(enhanced_quarter, None)

# Setup FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Detect circles in the image
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=80, param2=50, minRadius=300, maxRadius=450)

total_value = 0

# Check if any circle is detected
if circles is not None:
    # Convert circle coordinates to integer
    circles = np.round(circles[0, :]).astype("int")
    print(f"Number of circles detected: {len(circles)}")

    # Draw the circles and print their radii
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 255), 20)  # Circle outline
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -3)  # Center of the circle
        cv2.putText(image, f"Radius: {r} pixels", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
        print(f"Circle with center ({x}, {y}) has a radius of {r} pixels")


        # Create a blank mask
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, -1)
        kp_region, des_region = sift.detectAndCompute(gray, mask=mask)

        # Match with five cent reference
        matches_five_cent = flann.knnMatch(des_five_cent, des_region, k=2)
        good_matches_five_cent = [m for m, n in matches_five_cent if m.distance < 0.8 * n.distance]

        # Match with quarter reference
        matches_quarter = flann.knnMatch(des_quarter, des_region, k=2)
        good_matches_quarter = [m for m, n in matches_quarter if m.distance < 0.8 * n.distance]

        if len(good_matches_five_cent) > len(good_matches_quarter) and len(good_matches_five_cent) > 70:
            total_value += 0.05  # five-cent coin
            print(f"Coin at center ({x}, {y}) is a five-cent coin.")
        elif len(good_matches_quarter) > 70:
            total_value += 0.25  # quarter
            print(f"Coin at center ({x}, {y}) is a quarter.")
        else:
            print(f"Coin at center ({x}, {y}) is not recognized.")

        # Displaying matches with five-cent reference
        matched_image_five_cent = cv2.drawMatches(five_cent_reference, kp_five_cent, image, kp_region, good_matches_five_cent, None)
        # cv2.putText(matched_image_five_cent, f"Matches: {len(good_matches_five_cent)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output_width = 800
        aspect_ratio = matched_image_five_cent.shape[1] / matched_image_five_cent.shape[0]  # width / height
        output_height = int(output_width / aspect_ratio)
        resized_matched_image_five_cent = cv2.resize(matched_image_five_cent, (output_width, output_height))
        cv2.imshow(f"Matches with five-cent for coin at ({x}, {y})", resized_matched_image_five_cent)
        print(f"Number of matches with five-cent for coin at ({x}, {y}): {len(good_matches_five_cent)}")
        cv2.waitKey(0)

        # Displaying matches with quarter reference
        matched_image_quarter = cv2.drawMatches(quarter_reference, kp_quarter, image, kp_region, good_matches_quarter, None)
        # cv2.putText(matched_image_quarter, f"Matches: {len(good_matches_quarter)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        resized_matched_image_quarter = cv2.resize(matched_image_quarter, (output_width, output_height))
        cv2.imshow(f"Matches with quarter for coin at ({x}, {y})", resized_matched_image_quarter)
        print(f"Number of matches with quarter for coin at ({x}, {y}): {len(good_matches_quarter)}")
        cv2.waitKey(0)

    

    
    # Resize and display final results
    output_width = 800
    aspect_ratio = image.shape[1] / image.shape[0]
    output_height = int(output_width / aspect_ratio)
    resized_image = cv2.resize(image, (output_width, output_height))

    # Displaying the total dollar value on the image
    text = f"Total value: ${total_value:.2f}"
    cv2.putText(resized_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Detected Circles", resized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(text)
else:
    print("No circles were detected.")
