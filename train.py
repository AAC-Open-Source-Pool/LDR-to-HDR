import os
import argparse
from tqdm import tqdm
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from util import (
    slice_gauss,
    map_range,
    cv2torch,
    random_tone_map,
    DirectoryDataset,
    str2bool,
)
from model import ExpandNet


def parse_args():
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size.')
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=200,
        help='Checkpoint model every x epochs.',
    )
    parser.add_argument('-d', '--data_root_path', default='hdr_data1', help='Path to hdr data.')
    parser.add_argument('--save_path', default='checkpoints', help='Path for saving checkpoints.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--loss_freq', type=int, default=20, help='Report average loss every x iterations.')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU for training.')

    return parser.parse_args()


class ExpandNetLoss(nn.Module):
    """Custom loss function combining L1 loss and Cosine Similarity."""
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)  # Cosine similarity
        self.l1_loss = nn.L1Loss()  # L1 loss
        self.loss_lambda = loss_lambda  # Scaling factor for cosine term

    def forward(self, x, y):
        """Compute total loss as the sum of L1 loss and weighted cosine similarity loss."""
        cosine_term = (1 - self.similarity(x, y)).mean()  # Cosine similarity loss
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term  # Total loss


def transform(hdr):
    """Preprocess HDR image by slicing, resizing, and applying tone mapping."""
    hdr = slice_gauss(hdr, crop_size=(384, 384), precision=(0.1, 1))  # Slice and preprocess
    hdr = cv2.resize(hdr, (256, 256))  # Resize to 256x256
    hdr = map_range(hdr)  # Normalize the image
    ldr = random_tone_map(hdr)  # Apply random tone mapping for LDR image
    return cv2torch(ldr), cv2torch(hdr)  # Convert images to torch tensors


def train(opt):
    """Main training loop."""
    # Initialize model, loss function, optimizer, and dataset
    model = ExpandNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)
    loss = ExpandNetLoss()
    dataset = DirectoryDataset(data_root_path=opt.data_root_path, preprocess=transform)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
        drop_last=True,
    )

    # Use GPU if specified
    if opt.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    # Create save directory if not exists
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    else:
        print('WARNING: save_path already exists. Checkpoints may be overwritten.')

    avg_loss = 0
    for epoch in tqdm(range(1, 10_001), desc='Training'):
        for i, (ldr_in, hdr_target) in enumerate(tqdm(loader, desc=f'Epoch {epoch}')):
            if opt.use_gpu:
                ldr_in = ldr_in.cuda()  # Move input to GPU
                hdr_target = hdr_target.cuda()  # Move target to GPU
            hdr_prediction = model(ldr_in)  # Forward pass
            total_loss = loss(hdr_prediction, hdr_target)  # Compute loss
            optimizer.zero_grad()  # Zero gradients
            total_loss.backward()  # Backpropagate
            optimizer.step()  # Update model parameters
            avg_loss += total_loss.item()  # Update average loss
            if ((i + 1) % opt.loss_freq) == 0:
                # Report average loss every specified frequency
                rep = (
                    f'Epoch: {epoch:>5d}, '
                    f'Iter: {i + 1:>6d}, '
                    f'Loss: {avg_loss / opt.loss_freq:>6.2e}'
                )
                tqdm.write(rep)  # Output loss report
                avg_loss = 0

        # Save checkpoint every specified frequency
        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(opt.save_path, f'epoch_{epoch}.pth'),
            )


if __name__ == '__main__':
    opt = parse_args()  # Parse command-line arguments
    train(opt)  # Start the training process
