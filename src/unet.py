import torch
import torch.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss, KLDivLoss
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.functional import binary_cross_entropy_with_logits, pad
from torch.optim import Adam

from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from functools import wraps
from einops import rearrange
from fastcore.all import noop

import math
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net Paper - https://arxiv.org/pdf/1505.04597

    The U-Net is composed of convolutional layers that have skip connections between them - contains a contracting and an expanding path.
    One part of them concatenating the inital max pool of their layer to the last convolutional layer - this has a stride of two for downsampling.
    The other part of them taking the residual and concatenating it with the upsampled layer in the up blocks.

    During training for small datasets they used elastic deformations, e.g applying displacement controlled by their max and smoothness.

    Timestep embeddings
    The model conditions on the timesteps t.
    We can use sinusoidal embedding vectors as seen in other transformer models. These positional embeddings can thenbe added to other layers of the network.
    This embedding passes through a feed forward network, then reshapes the vector into a tensor that matches the input size of the up-sampling blocks and adds it to them.
    """
    def __init__(self):
        super(UNet, self).__init__()

        # Down blocks
        self.down_block_one = UNetDownBlock(1, 4, num_groups=1)
        self.down_block_two = UNetDownBlock(4, 10, num_groups=1)
        self.down_block_three = UNetDownBlock(10, 20, num_groups=2)
        self.down_block_four = UNetDownBlock(20, 40, num_groups=4)

        # Convolutional layer at the bottom of the U-Net
        self.mid_conv = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.mid_groupnorm = nn.GroupNorm(4, 40)
        self.mid_relu = nn.ReLU()

        # Up blocks
        self.up_block_one = UNetUpBlock(60, 20, 60, num_groups=2) # down4 channels + down3 channels
        self.up_block_two = UNetUpBlock(30, 10, 30, num_groups=1) # down2 channels + up1 channels
        self.up_block_three = UNetUpBlock(14, 5, 14, num_groups=1) # down1 channels + up2 channels
        self.up_block_four = UNetUpBlock(6, 5, 6, num_groups=1) # input channels + up3 channels

        # Final convolution to produce output. This layer injects negative values into the output.
        self.final_conv = nn.Conv2d(5, 1, kernel_size=3, padding=1)

    def forward(self, x, timestep_embeddings):
        # Pad since we are just testing on small images - making them 32x32 so we can downsample
        x = pad(x, (2,2,2,2), 'constant', 0)

        # Down-blocks of the U-Net compress the image down to a smaller
        # representation
        down_block_one = self.down_block_one(x)
        down_block_two = self.down_block_two(down_block_one)
        down_block_three = self.down_block_three(down_block_two)
        down_block_four = self.down_block_four(down_block_three)

        # Middle layer for extra transformations prior to upsampling
        middle = self.mid_conv(down_block_four)
        middle = self.mid_groupnorm(middle)
        middle = self.mid_relu(middle)

        # These upblocks take both the latent representation and the time embeddings provided
        # The time mlp and relu allow the unet to condition the noise reduction on the timesteps provided as information to the model
        up_block_one = self.up_block_one(middle, down_block_three, timestep_embeddings)
        up_block_two = self.up_block_two(up_block_one, down_block_two, timestep_embeddings)
        up_block_three = self.up_block_three(up_block_two, down_block_one, timestep_embeddings)
        up_block_four = self.up_block_four(up_block_three, x, timestep_embeddings)

        # Final convolutional layer
        x = self.final_conv(up_block_four)
        # Removing the padding that we added initially due to using the small pictures using for this test dataset
        return x[:,:,2:-2,2:-2]

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=4):
        super(UNetDownBlock, self).__init__()
        
        # Block of convolutions followed by a max pool
        self.down_block_one = self.down_block(in_channels, out_channels, kernel_size, padding, num_groups)
        self.down_block_two = self.down_block(out_channels, out_channels, kernel_size, padding, num_groups)
        self.down_block_three = self.down_block(out_channels, out_channels, kernel_size, padding, num_groups)
        self.maxpool = nn.MaxPool2d(2, padding=0)

    def down_block(self, in_channels, out_channels, kernel_size, padding, num_groups):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.groupnorm = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU()

        return nn.Sequential(self.conv, self.groupnorm, self.relu)

    def forward(self, x):
        # Keep track of the first convolution as the skip connection
        # Then pass it as the skip value so it can be applied before the relu in the downblock
        skip = self.down_block_one(x)
        x = self.down_block_two(skip)
        x = self.down_block_three(x)
        return self.maxpool(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimensions, kernel_size=3, padding=1, num_groups=4):
        super(UNetUpBlock, self).__init__()
        # Upsampling block
        self.upsample =  nn.Upsample(scale_factor=2, mode='nearest')

        # Convolutions following upsampling
        self.up_block_one = self.up_block(in_channels, out_channels, kernel_size, padding, num_groups)
        self.up_block_two = self.up_block(out_channels, out_channels, kernel_size, padding, num_groups)
        self.up_block_three = self.up_block(out_channels, out_channels, kernel_size, padding, num_groups)

        # Parameters to scale and shift the time embedding
        self.timestep_feedforward = nn.Linear(time_dimensions, time_dimensions)
        self.timestep_nonlinearity = nn.ReLU()

    def up_block(self, in_channels, out_channels, kernel_size, padding, num_groups):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.groupnorm = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU()

        return nn.Sequential(self.conv, self.groupnorm, self.relu)

    def forward(self, x, skip, timestep_embeddings):
        x_up = self.upsample(x)
        # Concatenate the skip connection passed in from the down
        x = torch.cat([skip, x_up], dim=1)

        # Cut embedding to be the size of the current channels
        timestep_embeddings = timestep_embeddings[:, :x.shape[1]]
        # Apply the time mlp and relu embeddings - these are applied to the timestep embeddings that we sample randomly during training
        # They allow the model to condition its probability distribution on the timestep
        timestep_embeddings = self.timestep_feedforward(timestep_embeddings)
        timestep_embeddings = self.timestep_nonlinearity(timestep_embeddings)
        # Since they are based on the batch size only we have to expand each one to the size of each image and hence using the
        # None twice to expand dimensions and broadcast the embeddings
        x += timestep_embeddings[:, :, None, None].expand(x.shape)

        skip = self.up_block_one(x)
        x = self.up_block_two(skip)
        return self.up_block_three(x)

def linear_schedule(min, max, max_timesteps):
    return torch.linspace(min, max, max_timesteps + 1)[:-1]

def get_timestep_embedding(max_timesteps, embedding_dim: int):
    timesteps = torch.arange(0, max_timesteps)
    # Make sure we divide the dimensions by two so that we can apply the embeddings to both sin and cosine
    half_dimensions = embedding_dim // 2
    # We take the max number of timesteps and if we apply log then we can get logarithmic steps that will give us short and high frenquency components
    # This means that they will capture information at many different ranges
    embeddings = np.log(max_timesteps) / (half_dimensions - 1)
    # Now that we have a spacing we create a linearly spaced tensor that is of size of half_dimensions
    embeddings = torch.exp(torch.arange(half_dimensions, dtype=torch.float32) * -embeddings)
    # Here we multiply each timestep with each embedding value - so that we get combinations of different time and frequency components
    embeddings = timesteps[:, None].to(torch.float32) * embeddings[None, :]
    # Apply sine and cosine to the embeddings and concatenate them - this will introduce periodicity into the embeddings
    # This periodicity in the cosine and sine embeddigs allows you to calculate relative positions quite effectively
    embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

    # If the embedding dimension is odd, pad it otherwise we will always have a size less than one
    return pad(embeddings, (0, 1, 0, 0)) if embedding_dim % 2 == 1 else embeddings

# Parameters
max_timesteps = 1000
epochs = 3
learning_rate = 1e-4
batch_size = 32

# Compute variance schedule
betas = linear_schedule(1e-4, 0.02, max_timesteps).to("cuda")
# Calculating the constants based on the implementation
alphas = 1 - betas
alpha_bar = torch.exp(torch.cumsum(torch.log(alphas), dim=0))
# These constants will dictate how much of the image and the noise respectively will be combined
sqrt_alphabar = torch.sqrt(alpha_bar)
sqrt_1malphabar = torch.sqrt(1 - alpha_bar)

# Generate timestep embeddings - the embedding dimension is hardcoded and based on the number of channels at the bottom layer of the U-Net
# Here we take the number of timesteps that will create all the betas / alphas for noise generation and then we create sinusoidal embeddings
time_embeddings = get_timestep_embedding(
  max_timesteps,
  embedding_dim = 100
).to("cuda")

# The original MNIST dataset images and images in general begin in the range of 0-255 after scaling them to 0-1 range
# The problem is here however that we dont have a mean of 0 and a std of 1 around this data so by scaling it by a mean of 0.5 and a std of 0.5
# This range is now moved from 0-1 to -1 - 1 which is completely normalised and will improve stability in our training as well as performance
# This is because this range prevents exploding and vanishing gradients during training
dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

training_dataset = FashionMNIST(root="dataset", train=True, download=True, transform=dataset_transforms)
testing_dataset = FashionMNIST(root="dataset", train=False, download=True, transform=dataset_transforms)
# Concatenate training and test data and create a dataloader to provide an iterable for training
dataset = ConcatDataset([training_dataset, testing_dataset])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialise the main model, optimisers and loss - put them to the device
unet = UNet().to("cuda")
optimizer = Adam(unet.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss().to("cuda")

# Batched training loop - can control the batch size in the hyperparameters above
for epoch in range(epochs):
    loss_sum = 0
    num_batches_complete = 0
    for _, (batched_images, _) in enumerate(dataloader):
        # Move the batch to our GPU for training
        batch_size = batched_images.shape[0]
        batched_images = batched_images.to("cuda")

        # Sample noise random noise the same shape as our batch - Batch, Height, Width
        noise = torch.randn_like(batched_images).to("cuda")

        # Get a randomly sampled timestep for each image in our batch
        timesteps = torch.randint(0, max_timesteps, size=(batch_size,)).to("cuda")

        # Grab the positional time embeddings for all the random timesteps - if we have B timesteps each of size N we will get a B x N output
        timestep_embeddings = time_embeddings[timesteps].to("cuda")

        # TODO update these based on the diffusion paper
        # Calculate how much noise to add to the image batch by broadcasting the two tensors to the size of the image and the noise
        sqrt_alphabar_ts = sqrt_alphabar[timesteps]
        sqrt_1malphabar_ts = sqrt_1malphabar[timesteps]
        X_t = sqrt_alphabar_ts[:, None, None, None] * batched_images + sqrt_1malphabar_ts[:, None, None, None] * noise

        # Use the U-Net to predict the amount of noise in the image give the timestep and the noised image
        noise_prediction = unet(X_t, timestep_embeddings)

        # Compute the loss between the real noise and predicted noise - making sure we keep the noise loss separate to .backward
        loss = mse_loss(noise, noise_prediction)
        loss_sum += float(loss)

        # Update the weights in the U-Net via a step of gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Increase number of batches complete
        num_batches_complete += 1
            
    print(f"Epoch: {epoch}. Mean loss: {loss_sum/num_batches_complete}")

torch.save(
    unet.state_dict(), f'./training/saved_models/unet_{epoch + 1}_epochs.pth'
)