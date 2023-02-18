####### need to adjust - running out of memory #######


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time

# Specify the device to use for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set the path to the image patches and masks
image_dir = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\data\patches'
mask_dir = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\data\masks'

# Set the image dimensions and number of channels
img_height, img_width, channels = 224, 224, 3

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_names = [name for name in os.listdir(image_dir) if name.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, f"dilated_mask{image_name.split('image')[1]}")
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image patch and mask
        image = cv2.resize(image, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height))

        # Normalize the image patch and mask
        image = image / 255.0
        mask = mask / 255.0

        if self.transform:
            image = self.transform(image)

        return image, mask

# Define the transformations to apply to the images and masks
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transforms = transforms.ToTensor()

# Create the dataset and dataloaders
train_dataset = CustomDataset(image_dir, mask_dir, transform=image_transforms)
test_dataset = CustomDataset(image_dir, mask_dir, transform=image_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the encoder part of the network
        self.conv1 = DoubleConv(3, 64)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)

        # Define the decoder part of the network
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        down1 = self.down1(conv1)
        conv2 = self.conv2(down1)

        up1 = self.upconv1(conv2)
        cat1 = torch.cat([up1, conv1], dim=1)
        dec1 = self.decoder1(cat1)

        conv3 = self.conv3(dec1)

        # Forward pass
        outputs = model(batch_X)

        # Resize the target tensor to the same size as the output tensor
        batch_Y = F.interpolate(batch_Y, size=outputs.size()[2:], mode='nearest')

        # Compute the loss
        loss = criterion(outputs, batch_Y)


        return torch.sigmoid(conv3)



# Create the model
model = UNet()

# Move the model to the GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Define the number of epochs and batch size
epochs = 20
batch_size = 16

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    start_time = time.time()
    epoch_loss = 0
    epoch_acc = 0

    for batch in train_loader:
        # Get a batch of training data
        batch_X, batch_Y = batch
        batch_X = batch_X.float()
        batch_Y = batch_Y.float()
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        # Reset the optimizer gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        # Compute the loss and accuracy
        loss = criterion(outputs, batch_Y)
        acc = ((outputs > 0.5).float() == batch_Y).float().mean()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)
        epoch_acc += acc.item() * batch_X.size(0)

    # Compute the epoch loss and accuracy
    epoch_loss /= len(train_loader.dataset)
    epoch_acc /= len(train_loader.dataset)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Training loss: {epoch_loss:.4f}")
    print(f"Training accuracy: {epoch_acc:.4f}")

    # Evaluate the model on the validation set
    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch in test_loader:
            # Get a batch of validation data
            batch_X, batch_Y = batch
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            # Forward pass
            outputs = model(batch_X)

            # Compute the loss and accuracy
            loss = criterion(outputs, batch_Y)
            acc = ((outputs > 0.5).float() == batch_Y).float().mean()

            val_loss += loss.item() * batch_X.size(0)
            val_acc += acc.item() * batch_X.size(0)

        # Compute the validation loss and accuracy
        val_loss /= len(test_loader.dataset)
        val_acc /= len(test_loader.dataset)

        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

    # Save the best model
    if epoch == 0:
        best_loss = val_loss
        torch.save(model.state_dict(), "unet_model.pt")
    else:
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "unet_model.pt")
