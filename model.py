import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ExpandNet(nn.Module):
    def __init__(self):
        super(ExpandNet, self).__init__()

        # Basic convolutional layer with activation
        def layer(nIn, nOut, k, s, p, d=1):
            return nn.Sequential(
                nn.Conv2d(nIn, nOut, k, s, p, d), nn.SELU(inplace=True)
            )

        # Initializing network components
        self.nf = 64
        self.local_net = nn.Sequential(
            layer(3, 64, 3, 1, 1), layer(64, 128, 3, 1, 1)
        )

        self.mid_net = nn.Sequential(
            layer(3, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            nn.Conv2d(64, 64, 3, 1, 2, 2),
        )

        self.glob_net = nn.Sequential(
            layer(3, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 4, 1, 0),
        )

        self.end_net = nn.Sequential(
            layer(256, 64, 1, 1, 0), nn.Conv2d(64, 3, 1, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        # Passing through the local, mid, and global networks
        local = self.local_net(x)
        mid = self.mid_net(x)
        
        # Resize the image for global network
        resized = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        b, c, h, w = local.shape
        glob = self.glob_net(resized).expand(b, 64, h, w)
        
        # Combine results and pass through final layer
        fuse = torch.cat((local, mid, glob), -3)
        return self.end_net(fuse)

    # Predict function for low memory usage by using patches
    def predict(self, x, patch_size):
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension if not present
            if x.size(-3) == 1:
                # Handle grayscale images by expanding channels
                x = x.expand(1, 3, *x.size()[-2:])
            
            # Process global features
            resized = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
            glob = self.glob_net(resized)

            # Define overlap for patch processing
            overlap = 20
            skip = int(overlap / 2)

            result = x.clone()
            x = F.pad(x, (skip, skip, skip, skip))  # Pad input to handle edges
            padded_height, padded_width = x.size(-2), x.size(-1)

            # Calculate how many patches we need to cover the image
            num_h = int(np.ceil(padded_height / (patch_size - overlap)))
            num_w = int(np.ceil(padded_width / (patch_size - overlap)))

            # Iterate over the patches
            for h_index in range(num_h):
                for w_index in range(num_w):
                    h_start = h_index * (patch_size - overlap)
                    w_start = w_index * (patch_size - overlap)
                    h_end = min(h_start + patch_size, padded_height)
                    w_end = min(w_start + patch_size, padded_width)
                    
                    # Slice the patch from the image
                    x_slice = x[:, :, h_start:h_end, w_start:w_end]
                    loc = self.local_net(x_slice)
                    mid = self.mid_net(x_slice)
                    exp_glob = glob.expand(1, 64, h_end - h_start, w_end - w_start)
                    
                    # Combine local, mid, and global features
                    fuse = torch.cat((loc, mid, exp_glob), 1)
                    res = self.end_net(fuse).data

                    # Stitch the result back into the output
                    h_start_stitch = h_index * (patch_size - overlap)
                    w_start_stitch = w_index * (patch_size - overlap)
                    h_end_stitch = min(h_start + patch_size - overlap, padded_height)
                    w_end_stitch = min(w_start + patch_size - overlap, padded_width)

                    # Insert the result into the correct position in the output tensor
                    res_slice = res[:, :, skip:-skip, skip:-skip]
                    result[:, :, h_start_stitch:h_end_stitch, w_start_stitch:w_end_stitch].copy_(res_slice)

                    # Clear unnecessary variables to save memory
                    del fuse, loc, mid, res

            return result[0]  # Return the result after processing all patches
