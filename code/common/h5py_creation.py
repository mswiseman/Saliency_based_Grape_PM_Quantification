import h5py
import numpy as np
import cv2
import glob

# This script creates an HDF5 file for classification.
# For segementation, add an 8-bit unsigned integer binary mask object.
# If concatenating rows with another HDF5 dataset, make sure the number of dimensions and object names are identical. 

# where all the prelabeled patches are
img_dir = r'.\data\labeled'

# stores all the image paths
list_of_images = glob.glob(img_dir + '/' + '*.png')
print(len(list_of_images))


# Load labels from csv
labels = np.genfromtxt(r'.\data\labels.csv', delimiter=',')

# preprocess the images by converting them to grayscale
images = []
for image_path in list_of_images:
    image = cv2.imread(image_path)
    if image is None:
        continue
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    images.append(image)
images = np.array(images)

# create the HDF5 file
with h5py.File('dataset.hdf5', 'w') as f:
    # write the images and labels to the HDF5 file
    f.create_dataset('images', data=images)
    f.create_dataset('labels', data=labels)

    # print the shape of the images and labels to verify they were stored correctly
    print('Images shape:', images.shape)
    print('Labels shape:', labels.shape)
