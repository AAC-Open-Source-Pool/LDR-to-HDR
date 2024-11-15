import sys
import cv2
import numpy as np
import imageio.v2 as imageio  # Import imageio for reading HDR images

def psnr(img1, img2):
    # Calculate Mean Squared Error (MSE) between two images
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Return infinity if no difference
    max_pixel = 1.0  # Images are assumed to be in range [0, 1]
    return 10 * np.log10(max_pixel ** 2 / mse)

def convert_to_pu_space(image, gamma=2.2):
    # Convert image to PU space using gamma correction
    return np.power(image / 255.0, 1 / gamma)

def compute_pu21_psnr(sdr_image_path, hdr_image_path):
    # Read and process the SDR image
    sdr_img = cv2.imread(sdr_image_path, cv2.IMREAD_COLOR)
    if sdr_img is None:
        raise ValueError(f"Could not open or find the SDR image at {sdr_image_path}")
    sdr_img = cv2.cvtColor(sdr_img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize the SDR image to [0, 1]

    # Read and process the HDR image
    hdr_img = imageio.imread(hdr_image_path, format='HDR-FI')
    if hdr_img is None:
        raise ValueError(f"Could not open or find the HDR image at {hdr_image_path}")
    hdr_img = hdr_img / hdr_img.max()  # Normalize HDR image to [0, 1]

    # Convert both images to PU space
    pu_sdr = convert_to_pu_space(sdr_img)
    pu_hdr = convert_to_pu_space(hdr_img)

    # Compute PSNR in PU space
    return psnr(pu_sdr, pu_hdr)

if __name__ == "__main__":
    # Check if arguments are passed correctly
    if len(sys.argv) != 3:
        print("Usage: python pu21_psnr.py <sdr_image_path> <hdr_image_path>")
        sys.exit(1)
    
    sdr_image_path = sys.argv[1]
    hdr_image_path = sys.argv[2]
    try:
        # Compute and print the PU21-PSNR score
        pu21_psnr_score = compute_pu21_psnr(sdr_image_path, hdr_image_path)
        print(f"PU21-PSNR Score: {pu21_psnr_score}")
    except ValueError as e:
        print(e)
