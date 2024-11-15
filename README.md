# LDR to HDR Image Conversion

This repository contains code and resources to convert Low Dynamic Range (LDR) images to High Dynamic Range (HDR) images. The conversion leverages multiple LDR exposures or a single LDR image to synthesize HDR data, enhancing visual details in both highlights and shadows.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example Results](#example-results)
- [Contributing](#contributing)
- [License](#license)

## About
High dynamic range (HDR) imaging provides the capability of handling real world lighting as opposed to the low dynamic range (LDR) which struggles to accurately represent images with higher dynamic range. However, most imaging content is still available only in LDR. This implementation presents a method for generating HDR images from LDR images based on Convolutional Neural Networks . The model attempts to reconstruct missing information that was lost from the original image . The image is reconstructed from learned features .The model is trained in a supervised method using a dataset of HDR images. 

## Features
- **Exposure Fusion**: Combine multiple LDR images taken at different exposures.
- **Single-Image HDR Approximation**: Generate HDR from a single LDR image.
- **Tone Mapping**: Convert HDR data into displayable formats, preserving contrast and dynamic range.
- **Color Correction**: Minimize color distortion during HDR synthesis.
- **Image Processing Utilities**: Helper functions for image alignment and noise reduction.

## Requirements
- Python 3.8 or higher
- OpenCV
- NumPy
- tqdm
- SciKitLearn
- Shutil
- Streamlit

