import os
import cv2

def load_video_frames(video_path, output_dir):
    video = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables
    count = 1  # Counter for frames

    while True:
        # Read the next frame
        ret, frame = video.read()

        if not ret:
            break

        # Save the frame as an image in the output directory
        frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)

        count += 1

    # Release the video object
    video.release()

    # Print the number of images saved
    print(f"Saved {count-1} images for {output_dir}")

hr_video_chunks_dir = "E:\SR-Video-Stream\hr_video_chunks"
lr_video_chunks_dir = "E:\SR-Video-Stream\lr_video_chunks"
hr_img_dir="E:\SR-Video-Stream\hr_frame_imges"
lr_img_dir="E:\SR-Video-Stream\lr_frame_imges"

# Example usage
num_chunks=8
target_bitrate=100

# Iterate over the HR video chunks and downscale each chunk to LR
for chunk_id in range(1, num_chunks + 1):
    hr_chunk_path = os.path.join(hr_video_chunks_dir, f"chunk_{chunk_id}.mp4")
    lr_chunk_path = os.path.join(lr_video_chunks_dir, f"chunk_{chunk_id}.mp4")

    hr_chunk_img_path = os.path.join(hr_img_dir, f"chunk_{chunk_id}")
    lr_chunk_img_path = os.path.join(lr_img_dir, f"chunk_{chunk_id}")
   
    load_video_frames(hr_chunk_path, hr_chunk_img_path)
    load_video_frames(lr_chunk_path, lr_chunk_img_path)