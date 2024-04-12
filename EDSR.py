import os
import requests
import numpy as np
import tensorflow as tf
import cv2
from cv2 import dnn_superres

# Define the base URL of your Flask server
base_url = 'http://localhost:5000'

# Define the folder path to save the LR videos
download_chunks_dir="E:\SR-Video-Stream\client_video_downlaod"

# Create the save folder if it doesn't exist
os.makedirs(download_chunks_dir, exist_ok=True)

def perform_EDSR_resolution(id):
    # initialize super resolution object
    edsr_model = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = 'E:\SR-Video-Stream\model\EDSR_x4.pb'
    edsr_model.readModel(model_path)
    edsr_model.setModel('edsr', 4)
    edsr_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    edsr_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Load LR video
    lr_video_path=os.path.join(download_chunks_dir, f'LR_chunks_{id}.mp4')
    lr_video = cv2.VideoCapture(lr_video_path)

    # Process each frame of the low-resolution video
    success, frame = lr_video.read()
    while success:
         # Preprocess the frame for input to the model
        lr_frame = frame / 255.0  # Normalize pixel values to [0, 1]

        # Perform super-resolution using the model
        sr_frame = edsr_model.upsample(lr_frame)

        # Display the enhanced frame
        cv2.imshow('Enhanced Video', sr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
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
    
    perform_EDSR_resolution(chunk_id)