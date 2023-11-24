# Coin Recognition and Value Calculation

This Python script uses OpenCV to recognize and calculate the total value of coins in an image. It applies the Scale-Invariant Feature Transform (SIFT) algorithm to match features in the image with reference coins, specifically five-cent and quarter coins.

## Features

- **Image Reading:** Reads the main image and reference coin images (five-cent and quarter).
- **Image Preprocessing:** Converts images to grayscale, applies GaussianBlur, and enhances contrast using CLAHE.
- **SIFT Feature Extraction:** Extracts SIFT features from the main and reference coin images.
- **Circle Detection:** Uses HoughCircles to detect circular objects in the main image.
- **Feature Matching:** Matches SIFT features of detected circles with reference coins.
- **Coin Recognition:** Identifies whether a coin is a five-cent or a quarter.
- **Value Calculation:** Calculates the total dollar value based on recognized coins.

## Usage

1. **Run the Script:** Execute the script in a Python environment.
2. **Input Images:** Modify the file paths in the script to specify the input images (`image`, `five_cent_reference`, `quarter_reference`).

## Output

- **Detected Circles:** Displays the main image with detected circles, annotated with radii and recognized coin types.
- **Matches with Five-Cent:** Displays feature matches with the five-cent reference for each detected coin.
- **Matches with Quarter:** Displays feature matches with the quarter reference for each detected coin.


![total_value](https://github.com/upasana099/CV/assets/89516193/d6ce71bc-b5ff-4783-b15c-04704f6d815c)



## Dependencies

- OpenCV: 4.x

## Setup

```bash
pip install opencv-python
```
