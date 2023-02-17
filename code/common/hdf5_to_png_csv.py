import h5py
import csv
from PIL import Image
import numpy as np

# This script splits .HDF5 files as some deep learning models require different inputs. 

# Specify the path to the HDF5 file
dataset_filepath = '/path/to/mask/files/'

# Open the HDF5 file
with h5py.File(dataset_filepath, 'r') as file:

    # Get the datasets containing the image, label, and mask data
    images = file['images']
    labels = file['labels']
    masks = file['masks']

    # Export images as PNG files
    for i in range(images.shape[0]):
        # Get the image data as a numpy array
        image_data = images[i, :, :, :]
        # Create a PIL Image object from the data
        image = Image.fromarray(image_data, mode='RGB')
        # Save the image as a PNG file
        image.save('image{}.png'.format(i))

    # Export masks as PNG files
    for i in range(masks.shape[0]):
        # Get the mask data as a numpy array
        mask_data = masks[i, :, :]
        # Convert the mask data from 0/1 to 0/255
        mask_data = np.uint8(mask_data * 255)
        # Create a PIL Image object from the data
        mask = Image.fromarray(mask_data, mode='L')
        # Save the mask as a PNG file
        mask.save('mask{}.png'.format(i))

    # Export labels as a CSV file
    with open('labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['image_id', 'label'])
        # Write the label data for each image
        for i in range(labels.shape[0]):
            writer.writerow([i, labels[i]])
