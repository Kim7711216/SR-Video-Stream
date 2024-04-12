import os
import requests
import cv2
import numpy as np
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

# Image Processing Functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img

def shave(image, border):
    img = image[border: -border, border: -border]
    return img


# Defining SRCNN Model
def model():

    # define model type
    SRCNN = Sequential()

    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))

    # define optimizer
    adam = Adam(lr=0.0003)

    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN

srcnn = model()
weights_path=r"E:\SR-Video-Stream\model\3051crop_weight_200.h5"
#weights_path=r"E:\SR-Video-Stream\model\model_1.h5"
srcnn.load_weights(weights_path)

# Define the base URL of your Flask server
base_url = 'http://localhost:5000'

# Define the folder path to save the LR videos
download_chunks_dir="E:\SR-Video-Stream\client_video_downlaod"

# Create the save folder if it doesn't exist
os.makedirs(download_chunks_dir, exist_ok=True)

# Create an empty array to store the processed frames
processed_frames = []
    
def perform_pretrained_SRCNN_super_resolution(id):

    # Load LR video
    lr_video_path=os.path.join(download_chunks_dir, f'LR_chunks_{id}.mp4')
    lr_video = cv2.VideoCapture(lr_video_path)

    # Process each frame of the low-resolution video
    success, frame = lr_video.read()

    while success:
        # Preprocess the frame for input to the model
        # preprocess the image with modcrop
        ref = modcrop(frame, 3)

        # convert the image to YCrCb - (srcnn trained on Y channel)
        temp = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

        # create image slice and normalize
        Y = numpy.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

        # perform super-resolution with srcnn
        predict_frame = srcnn.predict(Y)

        # post-process output
        predict_frame *= 255
        predict_frame[predict_frame[:] > 255] = 255
        predict_frame[predict_frame[:] < 0] = 0
        predict_frame = predict_frame.astype(np.uint8)

        # copy Y channel back to image and convert to BGR
        temp = shave(temp, 6)
        temp[:, :, 0] = predict_frame[0, :, :, 0]
        output_frame = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

        cv2.imshow('pretained SRCNN Preview', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Append the processed frame to the array
        processed_frames.append(output_frame)

        # Read the next frame from the low-resolution video
        success, frame = lr_video.read()

    # Release the video capture and destroy the window
    lr_video.release()
    cv2.destroyAllWindows()
    
# Retrieve the LR videos
for chunk_id in range(1, 9):
    video_url = f'{base_url}/video_chunks/{chunk_id}'
    video_response = requests.get(video_url, stream=True)
    video_save_path = os.path.join(download_chunks_dir, f'LR_chunks_{chunk_id}.mp4')
    with open(video_save_path, 'wb') as file:
        bytes_written = 0
        for chunk in video_response.iter_content(chunk_size=1024*1024):  # Download in 1 MB chunks
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                bytes_written += len(chunk)
    print(f'LR video {chunk_id} retrieved.')
    print(f'LR video {chunk_id} saved to disk. File size: {bytes_written} bytes.')
    
    perform_pretrained_SRCNN_super_resolution(chunk_id)

temp=r"E:\SR-Video-Stream\client_storage"
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter.fourcc(*'mp4v')

# Write the processed frames to the output video
for i, frame in enumerate(processed_frames):
    #cv2.imshow('pretained SRCNN Preview', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Save the frame as an image in the output folder
    cv2.imwrite(f'{temp}/frame_{i}.jpg', frame)


# Create a list of image file paths in the output folder
image_files = [os.path.join(temp, f) for f in os.listdir(temp) if f.endswith('.jpg')]

# Sort the image files numerically
image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

# Load the first image to get the frame size
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape

#output_video_path=r"E:\SR-Video-Stream\client_storage\output.mp4"
output_video = cv2.VideoWriter('pretrained client output.mp4', fourcc, 24, (width, height))

# Write the frames to the output video
for image_file in image_files:
    frame = cv2.imread(image_file)
    
    # Perform interpolation to double the width and height of the frame
    #frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame if needed
    cv2.imshow('pretrained SRCNN Preview', frame)
    
    # Check for the 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
output_video.release()

# Delete the JPG files
for image_file in image_files:
    #os.remove(image_file)
    pass