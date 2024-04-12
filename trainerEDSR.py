import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
import cv2
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
import tensorflow.keras.backend as K 

# Define the directory paths for HR and LR video chunks
hr_video_chunks_img_dir = "E:\SR-Video-Stream\hr_frame_imges"
lr_video_chunks_img_dir = "E:\SR-Video-Stream\lr_frame_imges"

# Set the number of video chunks
num_chunks = 8

#print("num GPU available:",len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
physical_devices = tf.config.list_physical_devices('GPU')  # Get the list of available GPUs
tf.config.experimental.set_memory_growth(physical_devices[0], True)

WIDTH,HIGHT=(854,480)
# Define SRCNN model architecture
def create_sr_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(None, None, 1)))
    model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    for _ in range(16):
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same'))
    return model

# Prepare your LR and HR training data
def prepare_training_data(chunk_id):
    # Define the directory paths for HR and LR video chunks imges
    hr_video_chunks_imges = os.path.join(hr_video_chunks_img_dir, f"chunk_{chunk_id}")
    lr_video_chunks_imges = os.path.join(lr_video_chunks_img_dir, f"chunk_{chunk_id}")

        # Load the LR and HR image datasets using flow_from_directory
    lr_data = tf.keras.preprocessing.image_dataset_from_directory(
        lr_video_chunks_imges,
        image_size=(HIGHT, WIDTH),
        batch_size=32,
        shuffle=False,
        seed=1,
        label_mode=None,
        color_mode='rgb'
        #interpolation='bicubic'
        #subset="training",
        #validation_split=0.2
    )

    hr_data = tf.keras.preprocessing.image_dataset_from_directory(
        hr_video_chunks_imges,
        image_size=(HIGHT, WIDTH),
        batch_size=32,
        shuffle=False,
        seed=1,
        label_mode=None,
        color_mode='rgb'
        #interpolation='bicubic'
        #subset="validation",
        #validation_split=0.2
    )

    # Convert LR and HR images from RGB to YCbCr
    lr_data = lr_data.map(rgb_to_ycbcr)
    hr_data = hr_data.map(rgb_to_ycbcr)

    return lr_data, hr_data

def rgb_to_ycbcr(image):
    ycbcr_image = tf.image.rgb_to_yuv(image)
    return ycbcr_image[..., :1] # Extract only the Y channel

def train_sr_model():
    # Set the number of video chunks
    num_chunks = 8

    # Create an instance of your SR model for each video chunk
    sr_models = []
    for chunk_id in range(1, num_chunks + 1):
        sr_model = create_sr_model()
        sr_models.append(sr_model)

    # Compile the models with appropriate loss and optimizer
    for sr_model in sr_models:
        sr_model.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam')

    for chunk_id in range(1, num_chunks + 1):
        # Prepare the LR and HR training data for the current chunk
        lr_data, hr_data = prepare_training_data(chunk_id)
        print(f"Chunk {chunk_id} :")

         # Separate the input features and target values
        lr_images = []
        hr_images = []
        for lr_batch, hr_batch in zip(lr_data, hr_data):
            lr_images.append(lr_batch)
            hr_images.append(hr_batch)
        lr_images = np.concatenate(lr_images)
        hr_images = np.concatenate(hr_images)

        #Display the first LR image for preview
        first_lr_image = lr_images[0].astype(np.uint8)
        first_hr_image = hr_images[0].astype(np.uint8)
        
        plt.imshow(first_lr_image)
        plt.axis('off')
        #plt.show()
        plt.imshow(first_hr_image)
        plt.axis('off')
        #plt.show()
       
        # Print the shape of the first LR image
        print("Shape of the LR image:", lr_images.shape)
        print("Shape of the HR image:", hr_images.shape)
        
        # Train the model
        sr_model = sr_models[chunk_id - 1]
        sr_model.summary()
        sr_model.fit(
            lr_images,
            hr_images,
            batch_size=1,
            epochs=10,  # Adjust the number of epochs as needed
            callbacks=None,  # Add your desired callbacks here
            verbose=2
        )

        # Save the trained model for the current chunk
        sr_models[chunk_id - 1].save(f'E:\SR-Video-Stream\model\model_{chunk_id}.h5')
        
        K.clear_session()

# Call the training function to train the SR model
train_sr_model()
