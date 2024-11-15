import cv2
import numpy as np
from skimage import metrics, io
import sys

def load_and_tone_map(hdr_path, target_shape):
    """
    Load an HDR image and apply tone mapping to convert it to an 8-bit SDR format.
    """
    hdr_image = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    if hdr_image is None:
        print("Error: Cannot open HDR image at", hdr_path)
        return None
    
    # Resize HDR image to match the shape of the SDR image
    hdr_image = cv2.resize(hdr_image, (target_shape[1], target_shape[0]))

    # Tone mapping using Mantiuk algorithm (to make HDR look realistic)
    tonemap = cv2.createTonemapMantiuk(gamma=2.2)
    ldr_image = tonemap.process(hdr_image)
    
    # Convert HDR to 8-bit by scaling and clipping
    ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)
    return ldr_image

def pu_piqe(image_path):
    """
    Compute PIQE score for a single grayscale image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Cannot open image at", image_path)
        return
    
    # Blur image and calculate difference for PIQE (using simple method)
    image_blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
    image_diff = cv2.absdiff(image, image_blurred)
    piqe_score = np.mean(image_diff)
    print("PIQE Score:", piqe_score)

def pu21_psnr(jpeg_path, hdr_path):
    """
    Compute PSNR between a JPEG image and an HDR image.
    """
    jpeg_image = cv2.imread(jpeg_path)
    if jpeg_image is None:
        print("Error: Cannot open JPEG image at", jpeg_path)
        return
    
    # Convert HDR to SDR to compare with JPEG
    hdr_image = load_and_tone_map(hdr_path, jpeg_image.shape)
    if hdr_image is None:
        return
    
    # Calculate PSNR (peak signal to noise ratio)
    psnr_value = metrics.peak_signal_noise_ratio(jpeg_image, hdr_image, data_range=255)
    print("PSNR Value:", psnr_value, "dB")

def pu21_vsi(jpeg_path, hdr_path):
    """
    Compute VSI score between a JPEG image and an HDR image.
    """
    jpeg_image = io.imread(jpeg_path, as_gray=True)
    hdr_image = load_and_tone_map(hdr_path, jpeg_image.shape)
    if hdr_image is None:
        return
    
    # Convert HDR to grayscale and compute VSI using SSIM approximation
    hdr_image_gray = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)
    
    # Use SSIM to approximate VSI
    vsi_value = metrics.structural_similarity(jpeg_image, hdr_image_gray, data_range=255)
    print("VSI Score (approximated by SSIM):", vsi_value)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python metrics.py <metric> <image_path1> [image_path2]")
    else:
        metric = sys.argv[1]
        image_path1 = sys.argv[2]
        
        if metric == "pu_piqe":
            pu_piqe(image_path1)
        elif metric == "pu21_psnr" and len(sys.argv) == 4:
            image_path2 = sys.argv[3]
            pu21_psnr(image_path1, image_path2)
        elif metric == "pu21_vsi" and len(sys.argv) == 4:
            image_path2 = sys.argv[3]
            pu21_vsi(image_path1, image_path2)
        else:
            print("Invalid metric or arguments.")
