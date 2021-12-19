import os
import PIL
import h5py

import numpy as np

import torchvision.transforms as tvtrans

from pathlib import Path


# 18000 samples
##  Train est 
##  Val set 
# mean (118.25, 165.38, 92.55) std (40.48, 35.02, 51.05)

main_folder = Path('/media/cornell/Data/tq42/Hyphal_2020/data/segmentation')
dataset_folder = main_folder
train_set_filepath = dataset_folder / 'val_set.hdf5'

# Load data
with h5py.File(train_set_filepath, 'r') as f:
    image_ds = f['images']
    train_images = image_ds[:, ]

train_images_red = train_images[..., 0]
train_images_green = train_images[..., 1]
train_images_blue = train_images[..., 2]

train_images_mean = (np.mean(train_images_red), np.mean(train_images_green), np.mean(train_images_blue))
train_images_std = (np.std(train_images_red), np.std(train_images_green), np.std(train_images_blue))

print(f'{train_images.shape[0]} training samples')
print(f'train mean {train_images_mean} std {train_images_std}')