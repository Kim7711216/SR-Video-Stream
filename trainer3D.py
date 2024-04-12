import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
import cv2
import os
import numpy as np
from keras.optimizers import Adam

# Define the directory paths for HR and LR video chunks
hr_video_chunks_dir = "E:\SR-Video-Stream\hr_video_chunks"
lr_video_chunks_dir = "E:\SR-Video-Stream\lr_video_chunks"

# Set the number of video chunks
num_chunks = 8
chunk_size=10

#print("num GPU available:",len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

# Define your SR model architecture
def create_sr_model():
    model = Sequential()
    # add model layers
    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None, None, None, 3)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), kernel_initializer='glorot_uniform',
                     activation='linear', padding='same', use_bias=True))

    # define optimizer
    adam = Adam(learning_rate=0.0003)

    # compile model
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

# Prepare your LR and HR training data
def prepare_training_data(chunk_id):
    # Define the directory paths for HR and LR video chunks
    hr_video_chunks_dir = "E:\SR-Video-Stream\hr_video_chunks"
    lr_video_chunks_dir = "E:\SR-Video-Stream\lr_video_chunks"

    # Initialize lists to store LR and HR frames
    lr_frames = []
    hr_frames = []

    # Construct the file paths for the current chunk
    lr_chunk_path = os.path.join(lr_video_chunks_dir, f"chunk_{chunk_id}.mp4")
    hr_chunk_path = os.path.join(hr_video_chunks_dir, f"chunk_{chunk_id}.mp4")

    # Open the LR video and get its properties
    lr_video = cv2.VideoCapture(lr_chunk_path)
    lr_width = int(lr_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    lr_height = int(lr_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lr_bits_per_channel = 8

    # Open the HR video and get its properties
    hr_video = cv2.VideoCapture(hr_chunk_path)
    hr_width = int(hr_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    hr_height = int(hr_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hr_bits_per_channel = 8

    # Calculate maximum intensity based on bits per channel
    lr_max_intensity = 2 ** lr_bits_per_channel - 1
    hr_max_intensity = 2 ** hr_bits_per_channel - 1

    # Load LR frames from the video chunk
    while True:
        ret, lr_frame = lr_video.read()
        if not ret:
            break
        lr_frame = np.array(lr_frame, dtype=np.float16)
        lr_resized = cv2.resize(lr_frame, (lr_width, lr_height))
        lr_resized /= lr_max_intensity
        lr_frames.append(lr_resized)

    # Load HR frames from the video chunk
    while True:
        ret, hr_frame = hr_video.read()
        if not ret:
            break
        hr_frame = np.array(hr_frame, dtype=np.float16)
        hr_resized = cv2.resize(hr_frame, (hr_width, hr_height))
        hr_resized /= hr_max_intensity
        hr_frames.append(hr_resized)

    # Convert the frame lists to NumPy arrays
    lr_data = np.array(lr_frames, dtype=np.float16)
    hr_data = np.array(hr_frames, dtype=np.float16)

    return lr_data, hr_data

def load_video_frames(video_path):
    video = cv2.VideoCapture(video_path)

    # Initialize a list to store the video chunks
    video_chunks = []

    # Read frames until the video is exhausted
    while True:
        # Initialize a list to store the frames of the current chunk
        frames = []

        # Read frames until the chunk is complete
        for _ in range(chunk_size):
            ret, frame = video.read()  
            if not ret:
                break

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Append the frame to the current chunk
            frames.append(frame_rgb)

        # Append the current chunk to the list of video chunks
        if frames:
            video_chunks.append(frames)

        if not ret:
            break

    # Release the video object
    video.release()

    return video_chunks
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
        sr_model.compile(loss='mean_squared_error', optimizer='adam')

    # Iterate over the video chunks and train the models
    for chunk_id in range(1, num_chunks + 1):
        # Prepare the LR and HR training data for the current chunk
        lr_data, hr_data = prepare_training_data(chunk_id)
        print(f"Chunk {chunk_id} :")
        print("LR shape:", lr_data.shape)
        print("HR shape:", hr_data.shape)

        # Determine the total number of frames and dimensions
        num_frames = lr_data.shape[0]
        height = lr_data.shape[1]
        width = lr_data.shape[2]
        channels = lr_data.shape[3]

        # Reshape the LR and HR data into (num_frames, height, width, channels)
        lr_data_reshaped = lr_data.reshape((num_frames, height, width, channels))
        hr_data_reshaped = hr_data.reshape((num_frames, height, width, channels))

        # Define batch size and number of iterations per epoch
        batch_size = 1
        iterations_per_epoch = num_frames // batch_size

        # Train the model in mini-batches
        for epoch in range(10):  # Adjust the number of epochs as needed
            print(f"Epoch {epoch+1}/{10}")
            for iteration in range(iterations_per_epoch):
                start_idx = iteration * batch_size
                end_idx = (iteration + 1) * batch_size

                # Select a mini-batch of LR and HR frames
                lr_batch = lr_data_reshaped[start_idx:end_idx]
                hr_batch = hr_data_reshaped[start_idx:end_idx]

                # Add the batch dimension to lr_batch and hr_batch
                lr_batch = np.expand_dims(lr_batch, axis=0)
                hr_batch = np.expand_dims(hr_batch, axis=0)

                # Train the model on the mini-batch
                sr_models[chunk_id - 1].train_on_batch(lr_batch, hr_batch)

        # Save the trained model for the current chunk
        sr_models[chunk_id - 1].save(f'E:\SR-Video-Stream\model\model_{chunk_id}.h5')

# Call the training function to train the SR model
train_sr_model()
