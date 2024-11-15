from __future__ import division, print_function
import argparse
import os
import numpy as np
import torch
import cv2
from smooth import smoothen_luminance
from model import ExpandNet
from util import (
    process_path,
    split_path,
    map_range,
    str2bool,
    cv2torch,
    torch2cv,
    resize,
    tone_map,
    create_tmo_param_from_args,
)

# Get arguments function
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ldr', nargs='+', type=process_path, help='LDR image(s)')
    parser.add_argument('--out', type=lambda x: process_path(x, True), default=None, help='Output location.')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size (to limit memory use).')
    parser.add_argument('--resize', type=str2bool, default=False, help='Use resized input.')
    parser.add_argument('--use_exr', type=str2bool, default=False, help='Produce .EXR instead of .HDR files.')
    parser.add_argument('--width', type=int, default=960, help='Image width resizing.')
    parser.add_argument('--height', type=int, default=540, help='Image height resizing.')
    parser.add_argument('--tag', default=None, help='Tag for outputs.')
    parser.add_argument('--use_gpu', type=str2bool, default=torch.cuda.is_available(), help='Use GPU for prediction.')
    parser.add_argument('--tone_map', choices=['exposure', 'reinhard', 'mantiuk', 'drago', 'durand'], default=None, help='Tone Map resulting HDR image.')
    parser.add_argument('--stops', type=float, default=0.0, help='Stops (loosely defined here) for exposure tone mapping.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma curve value (if tone mapping).')
    parser.add_argument('--use_weights', type=process_path, default='weights.pth', help='Weights to use for prediction')
    parser.add_argument('--ldr_extensions', nargs='+', type=str, default=['.jpg', '.jpeg', '.tiff', '.bmp', '.png'], help='Allowed LDR image extensions')
    opt = parser.parse_args()
    return opt

# Load pretrained network
def load_pretrained(opt):
    net = ExpandNet()
    state_dict = torch.load(opt.use_weights, map_location=lambda s, l: s, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net

# Preprocess the input image
def preprocess(x, opt):
    x = x.astype('float32')
    if opt.resize:
        x = resize(x, size=(opt.width, opt.height))
    x = map_range(x)
    return x

# Generate output filename
def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = f'{tag}_{extra_tag}'
    if out is not None:
        root = out
    return os.path.join(root, f'{name}_{tag}.{ext}')

# Function to create HDR images
def create_images(opt):
    net = load_pretrained(opt)
    
    if (len(opt.ldr) == 1) and os.path.isdir(opt.ldr[0]):
        # Treat this as a directory of LDR images
        opt.ldr = [
            os.path.join(opt.ldr[0], f)
            for f in os.listdir(opt.ldr[0])
            if any(f.lower().endswith(x) for x in opt.ldr_extensions)
        ]

    # Loop through the LDR images and process
    for ldr_file in opt.ldr:
        loaded = cv2.imread(ldr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        
        if loaded is None:
            print(f'Could not load {ldr_file}')
            continue

        ldr_input = preprocess(loaded, opt)

        if opt.resize:
            out_name = create_name(ldr_file, 'resized', 'jpg', opt.out, opt.tag)
            cv2.imwrite(out_name, (ldr_input * 255).astype(int))

        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()

        prediction = map_range(torch2cv(net.predict(t_input, opt.patch_size).cpu()), 0, 1)

        extension = 'exr' if opt.use_exr else 'hdr'
        out_name = create_name(ldr_file, 'prediction', extension, opt.out, opt.tag)
        print(f'Writing {out_name}')
        cv2.imwrite(out_name, prediction)

        if opt.tone_map is not None:
            tmo_img = tone_map(prediction, opt.tone_map, **create_tmo_param_from_args(opt))
            out_name = create_name(ldr_file, f'prediction_{opt.tone_map}', 'jpg', opt.out, opt.tag)
            cv2.imwrite(out_name, (tmo_img * 255).astype(int))

# Main function to call appropriate functions
def main():
    opt = get_args()
    create_images(opt)

if __name__ == '__main__':
    main()
