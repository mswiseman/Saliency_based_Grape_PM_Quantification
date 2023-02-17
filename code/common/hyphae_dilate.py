import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# Powdery mildew hyphae average about ~5 pixels in diameter when using the blackbird imaging system; however, some annotations were made using 1 px line, 
# so this thickens the line to a more realistic width. 

# Specify the path to the directory containing the mask PNG files
mask_dir = '/path/to/mask/files/'

# Create a structuring element to use for dilation
selem = np.ones((5, 5), dtype=np.uint8)

# Loop through all the files in the directory
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') and f.startswith('mask')]
if not mask_files:
    print('No mask*.png files found. Please check that your path and file types are correct.')
else:
    for file_name in mask_files:
        # Load the mask image
        mask_path = os.path.join(mask_dir, file_name)
        mask = Image.open(mask_path).convert('L')
        mask_data = np.array(mask)

        # Dilate the mask image
        dilated_mask_data = binary_dilation(mask_data, structure=selem)

        # Create a PIL Image object from the dilated mask data
        dilated_mask = Image.fromarray(dilated_mask_data.astype(np.uint8) * 255, mode='L')

        # Save the dilated mask image as a PNG file
        dilated_mask_path = os.path.join(mask_dir, 'dilated_' + file_name)
        dilated_mask.save(dilated_mask_path)
