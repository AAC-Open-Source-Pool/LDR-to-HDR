import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Function to convert an image to PU21 perceptually uniform space
def convert_to_pu21(image):
    """Convert an image to PU21 space by applying gamma correction."""
    # Apply gamma correction (raising image values to the power of 1/2.2)
    return np.power(image, 1 / 2.2)

# Function to compute the Visual Saliency Index (VSI) between two images
def compute_pu21_vsi(img1, img2):
    """Compute a VSI score (using SSIM as a proxy) between two images in PU21 space."""
    # If the images are in color, convert them to grayscale
    if img1.ndim == 3:  # If img1 has 3 dimensions (color image)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    if img2.ndim == 3:  # If img2 has 3 dimensions (color image)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Normalize the images so that their pixel values are between 0 and 1
    img1 = img1 / 255.0  # Divide by 255 to normalize the values
    img2 = img2 / 255.0  # Divide by 255 to normalize the values

    # Convert both images to PU21 space using the convert_to_pu21 function
    pu_img1 = convert_to_pu21(img1)
    pu_img2 = convert_to_pu21(img2)

    # Define the range of data (for SSIM, this is typically 1.0 for normalized images)
    data_range = 1.0  # Since the images are already normalized

    # Use SSIM (Structural Similarity Index) to measure similarity between the images
    # This will act as a proxy for VSI
    ssim_score, _ = ssim(pu_img1, pu_img2, full=True, data_range=data_range)

    return ssim_score  # Return the SSIM score, which we're using as VSI

# Main function that loads images and computes the VSI score
if __name__ == "__main__":
    import sys

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python pu21_vsi.py <image1_path> <image2_path>")
        sys.exit(1)  # Exit if incorrect arguments

    # Get image paths from command-line arguments
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    # Load the images from the given paths
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)  # Read image1 in color
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)  # Read image2 in color

    # Check if images were loaded correctly
    if image1 is None:
        print(f"Error: Could not load image from {image1_path}")
        sys.exit(1)  # Exit if image1 could not be loaded
    if image2 is None:
        print(f"Error: Could not load image from {image2_path}")
        sys.exit(1)  # Exit if image2 could not be loaded

    # Compute the PU21-VSI score between the two images
    pu21_vsi_score = compute_pu21_vsi(image1, image2)

    # Print the result (VSI score)
    print(f"PU21-VSI Score: {pu21_vsi_score}")
