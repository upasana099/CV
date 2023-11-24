import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def gaussian_blur(img, sigma):
    size = int(6 * sigma + 1)
    if size % 2 == 0:  # Ensure kernel size is odd
        size += 1
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()

    # Pad the image for boundary handling and compute the convolution
    padding = size // 2
    img_padded = np.pad(img, ((padding, padding), (padding, padding)), mode='edge')
    img_filtered = np.array([[np.sum(img_padded[i: i + size, j: j + size] * kernel) 
                            for j in range(img.shape[1])] 
                            for i in range(img.shape[0])])
    return img_filtered


def generateGaussianKernels(sigma_min, num_intervals, k):
    sigmas = [sigma_min * (k ** i) for i in range(num_intervals + 3)]
    return sigmas

def generateGaussianImages(image, sigmas):
    return [gaussian_blur(image, sigma) for sigma in sigmas]



def generateDoG(gaussian_images):
    return [gaussian_images[i+1] - gaussian_images[i] for i in range(len(gaussian_images)-1)]

def localize_keypoint(dog_images, x, y, s):
    dx = (dog_images[s][x, y+1] - dog_images[s][x, y-1]) / 2.0
    dy = (dog_images[s][x+1, y] - dog_images[s][x-1, y]) / 2.0
    ds = (dog_images[s+1][x, y] - dog_images[s-1][x, y]) / 2.0

    dxx = dog_images[s][x, y+1] + dog_images[s][x, y-1] - 2 * dog_images[s][x, y]
    dyy = dog_images[s][x+1, y] + dog_images[s][x-1, y] - 2 * dog_images[s][x, y]
    dss = dog_images[s+1][x, y] + dog_images[s-1][x, y] - 2 * dog_images[s][x, y]

    dxy = (dog_images[s][x+1, y+1] - dog_images[s][x+1, y-1] - dog_images[s][x-1, y+1] + dog_images[s][x-1, y-1]) / 4.0
    dxs = (dog_images[s+1][x, y+1] - dog_images[s+1][x, y-1] - dog_images[s-1][x, y+1] + dog_images[s-1][x, y-1]) / 4.0
    dys = (dog_images[s+1][x+1, y] - dog_images[s+1][x-1, y] - dog_images[s-1][x+1, y] + dog_images[s-1][x-1, y]) / 4.0

    J = np.array([dx, dy, ds])
    HD = np.array([[dxx, dxy, dxs], 
                   [dxy, dyy, dys], 
                   [dxs, dys, dss]])
    
    offset = -np.linalg.inv(HD).dot(J)

    return offset, J, HD[:2,:2], x, y, s

def is_extrema(dog, s, x, y):
    pixel = dog[s][y, x]
    
    # Define possible displacements for checking neighboring pixels
    ds_values = [-1, 0, 1]
    dx_values = [-1, 0, 1]
    dy_values = [-1, 0, 1]
    
    # Determine if pixel is a local minimum or maximum
    is_max = all(
        pixel > dog[s + ds][y + dy, x + dx]
        for ds in ds_values if 0 <= s + ds < len(dog)
        for dx in dx_values if 0 <= x + dx < dog[s].shape[1]
        for dy in dy_values if 0 <= y + dy < dog[s].shape[0]
        if not (ds == 0 and dx == 0 and dy == 0)  # Exclude the central pixel itself
    )
    
    is_min = all(
        pixel < dog[s + ds][y + dy, x + dx]
        for ds in ds_values if 0 <= s + ds < len(dog)
        for dx in dx_values if 0 <= x + dx < dog[s].shape[1]
        for dy in dy_values if 0 <= y + dy < dog[s].shape[0]
        if not (ds == 0 and dx == 0 and dy == 0)  # Exclude the central pixel itself
    )

    return is_max or is_min

def compute_hessian(dog, x, y, s):
    """Compute the 2x2 Hessian matrix."""
    dxx = dog[s][x, y+1] + dog[s][x, y-1] - 2 * dog[s][x, y]
    dyy = dog[s][x+1, y] + dog[s][x-1, y] - 2 * dog[s][x, y]
    dxy = (dog[s][x+1, y+1] - dog[s][x+1, y-1] - dog[s][x-1, y+1] + dog[s][x-1, y-1]) / 4.0
    return np.array([[dxx, dxy], [dxy, dyy]])


def generateOctaves(image, num_octaves):
    """Generate octaves by downsampling the image."""
    octaves = [image]
    for i in range(num_octaves - 1):
        image = image[::2, ::2]  # downsample by taking every second pixel
        octaves.append(image)
    return octaves


def findKeypoints(dog_images_per_octave, contrast_threshold=0.03, edge_ratio=30):
    keypoints = []

    # Iterate over the octaves
    for dog_images in dog_images_per_octave:
        
        # Iterate over the scales of the DoG images within the current octave
        for s in range(1, len(dog_images) - 1):
            
            for y in range(1, dog_images[s].shape[0] - 1):
                for x in range(1, dog_images[s].shape[1] - 1):

                    # Check if the current pixel is an extrema
                    if is_extrema(dog_images, s, x, y):
                        try:
                        # # Contrast thresholding
                        # if abs(dog_images[s][y,x]) < contrast_threshold:
                        #     continue
                            D = dog_images[s][y, x]
                            offset, J, _, _, _, _ = localize_keypoint(dog_images, x, y, s)
                            gradient = J
                            D_refined = D + 0.5 * np.dot(gradient, offset)
                            # print(D_refined)
                            # Contrast thresholding
                            if abs(D_refined) < contrast_threshold:
                                continue


                            # Edge response elimination
                            H = compute_hessian(dog_images, x, y, s)
                            detH = np.linalg.det(H)
                            traceH = np.trace(H)
                            if detH == 0:
                                    print(f"Warning: Determinant is zero at {(x, y, s)}")
                                    continue

                            edge_response = traceH**2 / detH
                            if edge_response > edge_ratio:
                                    continue
                            # if traceH**2 * edge_ratio > (edge_ratio + 1)**2 * (detH):
                            #     continue

                            keypoints.append((x, y, s))
                        except Exception as e:
                            print(f"Error at {(x, y, s)}: {e}")
    return keypoints



# Load image and convert to grayscale
image = Image.open('C:\\Users\\upasa\\OneDrive - Worcester Polytechnic Institute (wpi.edu)\\CV LAB\\lenna.png')

image = np.array(image)
if image.ndim == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
image = image / 255.0

# Parameters
sigma_min = 0.5
num_intervals = 3
k = np.power(2, 1.0/num_intervals)
num_octaves = 3

# Generate octaves
octaves = generateOctaves(image, num_octaves)

# For each octave, generate blurred images and DoG
dog_images_per_octave = []
for image in octaves:
    sigmas = generateGaussianKernels(sigma_min, num_intervals, k)
    gaussian_images = generateGaussianImages(image, sigmas)
    dog_images = generateDoG(gaussian_images)
    dog_images_per_octave.append(dog_images)

# Detect keypoints across octaves
keypoints = findKeypoints




# Load image and convert to grayscale
image = Image.open('C:\\Users\\upasa\\OneDrive - Worcester Polytechnic Institute (wpi.edu)\\CV LAB\\lenna.png')



image = np.array(image)
if image.ndim == 3:
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

image = image / 255.0

# Parameters
sigma_min = 0.5
num_intervals = 3
k = np.power(2, 1.0/num_intervals)

# Generate blurred images and DoG
sigmas = generateGaussianKernels(sigma_min, num_intervals, k)
gaussian_images = generateGaussianImages(image, sigmas)
dog_images = generateDoG(gaussian_images)



# Detect keypoints
keypoints = findKeypoints(dog_images_per_octave)


s = 1
extrema_coords = [(x,y) for x,y,scale in keypoints if scale == s]


num_dog_images = len(dog_images)
plt.figure(figsize=(15, 5))

for i, dog_img in enumerate(dog_images):
    plt.subplot(1, num_dog_images, i+1)
    plt.imshow(dog_img, cmap='gray')
    plt.title(f"DoG Image at scale {i}")

plt.tight_layout()
plt.show()


# Scale-space extrema detection
plt.figure(figsize=(6, 6))
plt.imshow(dog_images[s], cmap='gray')
# Uncomment the next line if you have the extrema_coords variable to show.
# plt.scatter([y for x,y in extrema_coords], [x for x,y in extrema_coords], c='blue', s=30, marker='+')
plt.title("Scale-space extrema detection")
plt.show()

# Accurate Keypoint Localization
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.scatter([kp[1] for kp in keypoints], [kp[0] for kp in keypoints], c='blue', s=30, marker='+')
plt.title("Accurate Keypoint Localization")
plt.show()

print(f"Number of detected keypoints: {len(keypoints)}")
