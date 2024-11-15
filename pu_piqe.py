import sys
import cv2
import numpy as np
from skimage import color, util
from skimage.metrics import structural_similarity as ssim

# Function to convert an image to PU space by applying gamma correction
def convert_to_pu_space(image, gamma=2.2):
    """Convert an image to perceptually uniform (PU) space using gamma correction."""
    # Normalize image to [0, 1] by dividing by 255, and then apply gamma correction
    return np.power(image / 255.0, 1 / gamma)

# Function to calculate a simple PIQE score as a proxy
def calculate_piqe(image):
    """Calculate a proxy for the PIQE score on a grayscale image."""
    # If the image is in color (RGB), convert it to grayscale
    gray = color.rgb2gray(image) if image.ndim == 3 else image

    # Simulate noise in the image (this is a very rough approximation of PIQE)
    variance_map = util.img_as_float(util.random_noise(gray, mode='gaussian'))
    
    # Calculate the variance (spread) of pixel values as a proxy for image quality
    return np.var(variance_map)

# Function to compute PU-PIQE score for an image
def compute_pu_piqe(image_path):
    """Compute PU-PIQE score for an image after converting to PU space."""
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        raise ValueError(f"Could not load image at the provided path: {image_path}")
    
    # Convert the image to perceptually uniform (PU) space
    pu_image = convert_to_pu_space(image)

    # Calculate the PIQE score on the PU space image
    pu_piqe_score = calculate_piqe(pu_image)
    
    return pu_piqe_score

# Main function to read image and compute PU-PIQE score
if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python pu_piqe.py <image_path>")
        sys.exit(1)  # Exit if the number of arguments is incorrect

    # Get the image path from command-line arguments
    image_path = sys.argv[1]
    
    try:
        # Compute PU-PIQE score for the provided image
        pu_piqe_score = compute_pu_piqe(image_path)
        
        # Print the result (PU-PIQE score)
        print(f"PU-PIQE Score: {pu_piqe_score}")
    
    except ValueError as e:
        # Handle the error if the image couldn't be loaded
        print(e)
