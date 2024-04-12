import os
import requests
import cv2
import numpy as np
import tensorflow as tf
import numpy

# Define the base URL of your Flask server
base_url = 'http://localhost:5000'

# Define the folder path to save the LR videos
download_models_dir = "E:\SR-Video-Stream\client_model_download" 
download_chunks_dir="E:\SR-Video-Stream\client_video_downlaod"

# Create the save folder if it doesn't exist
os.makedirs(download_models_dir, exist_ok=True)
os.makedirs(download_chunks_dir, exist_ok=True)

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

def perform_super_resolution(id):
    # Load LR video
    lr_video_path=os.path.join(download_chunks_dir, f'LR_chunks_{id}.mp4')
    lr_video = cv2.VideoCapture(lr_video_path)

    # Load model data
    model_path=os.path.join(download_models_dir, f'model_{id}.h5')
    srcnn = tf.keras.models.load_model(model_path)

        # Get the dimensions of the low-resolution video frames
    width = int(lr_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(lr_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a window to display the enhanced video
    cv2.namedWindow('Enhanced Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Enhanced Video', width, height)

    # Process each frame of the low-resolution video
    success, frame = lr_video.read()
    while success:
        # Preprocess the frame for input to the model
        ref = modcrop(frame, 3)

        # convert the image to YCrCb - (srcnn trained on Y channel)
        temp = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

        # create image slice and normalize
        Y = numpy.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

        # perform super-resolution with srcnn
        predict_frame = srcnn.predict(Y)

        # post-process output
        predict_frame *= 255.0
        predict_frame[predict_frame[:] > 255] = 255
        predict_frame[predict_frame[:] < 0] = 0
        predict_frame = predict_frame.astype(np.uint8)

        # copy Y channel back to image and convert to BGR
        temp = shave(temp, 6)
        temp = temp[:predict_frame.shape[1], :predict_frame.shape[2], :]

        # Normalize pixel values
        predict_frame_norm = predict_frame[0, :, :, 0] / 255.0

        # Resize the normalized frame to match the dimensions of the temporary frame
        resized_frame = cv2.resize(predict_frame_norm, (temp.shape[1], temp.shape[0]))

        # Copy Y channel back to image and convert to BGR
        temp[:, :, 0] = resized_frame*255.0

        sr_frame = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

        # Display the enhanced frame
        cv2.imshow('Enhanced Video', sr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Append the processed frame to the array
        processed_frames.append(sr_frame)

        rgb_video=cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        # Display the enhanced frame
        #cv2.imshow('video', rgb_video)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Read the next frame from the low-resolution video
        success, frame = lr_video.read()

    # Release the video capture and destroy the window
    lr_video.release()
    cv2.destroyAllWindows()

# Create an empty array to store the processed frames
processed_frames = []

# Retrieve the LR videos
for chunk_id in range(1, 9):
    video_url = f'{base_url}/video_chunks/{chunk_id}'
    model_url = f'{base_url}/model_chunks/{chunk_id}'
    video_response = requests.get(video_url, stream=True)
    model_response = requests.get(model_url, stream=True)
    video_save_path = os.path.join(download_chunks_dir, f'LR_chunks_{chunk_id}.mp4')
    model_save_path = os.path.join(download_models_dir, f'model_{chunk_id}.h5')
    with open(video_save_path, 'wb') as file:
        bytes_written = 0
        for chunk in video_response.iter_content(chunk_size=1024*1024):  # Download in 1 MB chunks
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                bytes_written += len(chunk)
    print(f'LR video {chunk_id} retrieved.')
    print(f'LR video {chunk_id} saved to disk. File size: {bytes_written} bytes.')

    with open(model_save_path, 'wb') as file:
        bytes_written = 0
        for chunk in model_response.iter_content(chunk_size=1024*1024):  # Download in 1 MB chunks
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                bytes_written += len(chunk)
    print(f'model_{chunk_id} retrieved.')
    print(f'model_{chunk_id} saved to disk. File size: {bytes_written} bytes.')

    perform_super_resolution(chunk_id)

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
output_video = cv2.VideoWriter('client output.mp4', fourcc, 24, (width, height))

# Write the frames to the output video
for image_file in image_files:
    frame = cv2.imread(image_file)
    
    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame if needed
    cv2.imshow('pretrained SRCNN Preview', frame)
    
    # Check for the 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
output_video.release()

# Delete the JPG files
#for image_file in image_files:
    #os.remove(image_file)
    
