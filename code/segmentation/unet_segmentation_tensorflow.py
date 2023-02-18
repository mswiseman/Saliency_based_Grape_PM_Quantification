import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time

# Specify the physical devices to use
devices = tf.config.list_physical_devices('GPU')
if devices:
    # Set the GPU as the device for training
    tf.config.set_visible_devices(devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

# Set the path to the image patches and masks
image_dir = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantification-main\data\patches'
mask_dir = r'C:\Users\michele.wiseman\Desktop\Saliency_based_Grape_PM_Quantificati on-main\data\masks'

# Set the image dimensions and number of channels
img_height, img_width, channels = 224, 224, 3

# Load the image patches and masks
print('Loading images and masks...')
X = []
Y = []
for image_name in os.listdir(image_dir):
    if image_name.endswith('.png'):
        # Load the image patch and its corresponding mask
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, f"dilated_mask{image_name.split('image')[1]}")
        #print(f"Image path: {image_path}")
        #print(f"Image path: {mask_path}")
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image patch and mask
        image = cv2.resize(image, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height))

        # Normalize the image patch and mask
        image = image / 255.0
        mask = mask / 255.0

        # Add the image patch and mask to the lists
        X.append(image)
        Y.append(mask)

print('Loaded {} images and masks.'.format(len(X)))

# Convert the lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Split the data into training and validation sets
print('Splitting data into training and validation sets...')
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the current GPU usage every 60 seconds
#while True:
#    os.system('nvidia-smi --query-gpu=utilization.gpu --format=csv')
#    time.sleep(60)

# Define the U-Net model architecture
print('Building model...')
inputs = layers.Input((img_height, img_width, channels))
s = layers.Lambda(lambda x: x / 255)(inputs)

c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = layers.Dropout(0.1)(c1)
c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = layers.Dropout(0.1)(c2)
c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = layers.Dropout(0.2)(c3)
c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = layers.Dropout(0.2)(c4)
c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = layers.Dropout(0.3)(c5)
c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = layers.Dropout(0.2)(c6)
c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = layers.Dropout(0.2)(c7)
c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = layers.Dropout(0.1)(c8)
c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = layers.Dropout(0.1)(c9)
c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
batch_size = 64
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    start_time = time.time()
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, Y_val))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Training loss: {history.history['loss'][0]:.4f}")
    print(f"Validation loss: {history.history['val_loss'][0]:.4f}")
    print(f"Training accuracy: {history.history['accuracy'][0]:.4f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][0]:.4f}")

# Save the trained model
model.save('unet_model.h5')
