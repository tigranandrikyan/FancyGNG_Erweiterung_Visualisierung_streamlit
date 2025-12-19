### IMPORTANT: before rerunning this file, delete the folder ./out_fancy_pca_caltech101 ###

import os
import numpy as np
import constants
import fancy_pca as FP
from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
from PIL import Image
import ssl
import parser

# SSL fix for HTTPS downloads (needed for Caltech101)
ssl._create_default_https_context = ssl._create_unverified_context

# Output directory
output_path = './out_fancy_pca_caltech101_std_0'
os.makedirs(output_path, exist_ok=True)

# Load Caltech101 dataset with transformations
caltech_dataset = Caltech101(
    root='./data_caltech101',
    download=True,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # returns values between 0 and 1
    ])
)

print(f"Number of images in dataset: {len(caltech_dataset)}")

# Split dataset into train and test sets
train_data, test_data = random_split(caltech_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# DataLoader for training
data_loader = DataLoader(train_data, batch_size=1, shuffle=False)

# FancyPCA instance
fancy_pca_transform = FP.FancyPCA()

# Iterate over dataset and apply Fancy PCA
for data_index, (image_tensor, label) in enumerate(data_loader):
    for i in range(len(image_tensor)):  # Here len(image_tensor) is always 1 (batch_size=1)
        # Create output directory for current class, e.g., ./out_fancy_pca_caltech101/42
        os.makedirs(os.path.join(output_path, str(label[i].item())), exist_ok=True)

        # Convert (1, 3, 224, 224) → (224, 224, 3)
        # Check if tensor is 2D or 3D
        image_np = image_tensor[i].squeeze(0)  # Remove batch dimension (if present)

        # If tensor is only 2D (no channel), add channels
        if len(image_np.shape) == 2:  # If image has only HxW
            image_np = image_np.unsqueeze(0).repeat(3, 1, 1)  # Add 3 channels (RGB)

        # Now tensor has shape (3, H, W), permute to (H, W, 3)
        image_np = image_np.permute(1, 2, 0).numpy()  # Convert to (H, W, 3)

        height, width, _ = image_np.shape
        image_name = f"caltech_img_{data_index}"  # Generate image name like caltech_img_0, caltech_img_1, ...

        # Prepare image as in parser.parse()
        data_array = np.vstack(image_np * constants.MAX_COLOR_VALUE) / constants.MAX_COLOR_VALUE
        # Store image size as list of tuples – FancyPCA expects this format
        size_images = [(width, height)]

        # Augmentation loop
        for aug_count in range(constants.AUG_COUNT):
            print(f"Fancy_PCA: {image_name}, {aug_count + 1}/{constants.AUG_COUNT}")
            fancy_pca_images = fancy_pca_transform.fancy_pca(data_array)

            # Save augmented image in output directory, sorted by class folder
            parser.save_data(
                fancy_pca_images,
                image_name + f"_{aug_count + 1}",
                size_images,
                aug_count,
                0,  # index 0, because size_images has only one entry
                path=output_path + "/" + str(label[i].item())  # target path, e.g., ./out_fancy_pca_caltech101/42
            )
