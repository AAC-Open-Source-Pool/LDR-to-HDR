LDR to HDR Image Conversion
This repository contains code and resources to convert Low Dynamic Range (LDR) images to High Dynamic Range (HDR) images. The conversion leverages multiple LDR exposures or a single LDR image to synthesize HDR data, enhancing visual details in both highlights and shadows.

Table of Contents
About
Features
Requirements

About
The LDR to HDR conversion aims to expand the luminance range of images, allowing for a more comprehensive representation of real-world lighting conditions. This project utilizes techniques for tone mapping and exposure fusion to generate HDR images from standard LDR photos.

Features
Exposure Fusion: Combine multiple LDR images taken at different exposures.
Single-Image HDR Approximation: Generate HDR from a single LDR image.
Tone Mapping: Convert HDR data into displayable formats, preserving contrast and dynamic range.
Color Correction: Minimize color distortion during HDR synthesis.
Image Processing Utilities: Helper functions for image alignment and noise reduction.

Requirements

Python 3.8 or higher
OpenCV
NumPy
PyTorch
tqdm
Shutil
Streamlit
Argsparse
SciKitLearn

