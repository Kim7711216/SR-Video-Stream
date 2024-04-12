import cv2
import os
import subprocess

# Define the directory paths for HR and LR video chunks
hr_video_chunks_dir = "E:\SR-Video-Stream\hr_video_chunks"
lr_video_chunks_dir = "E:\SR-Video-Stream\lr_video_chunks"
#temp_path="E:\SR-Video-Stream\_temp"


scaling_factor = 2

# Function to downscale a video chunk from HR to LR
def downscale_video(hr_chunk_path, lr_chunk_path):
    # Open the HR video chunk
    hr_video = cv2.VideoCapture(hr_chunk_path)

    # Get the HR video properties
    fps = hr_video.get(cv2.CAP_PROP_FPS)
    width = int(hr_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(hr_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the dimensions of the LR video frames
    lr_width = width // scaling_factor
    lr_height = height // scaling_factor

    # Create a VideoWriter object for the LR video chunk
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Update the codec as per your desired output format
    lr_video_writer = cv2.VideoWriter(lr_chunk_path, fourcc, fps, (width, height))

    # Read and downscale each frame in the HR video chunk
    while True:
        ret, frame = hr_video.read()
        if not ret:
            break

        # Downscale the frame to LR dimensions
        lr_frame = cv2.resize(frame, (lr_width, lr_height))
        frame = cv2.resize(lr_frame, (width, height), interpolation=cv2.INTER_NEAREST)
        # Write the LR frame to the LR video chunk
        lr_video_writer.write(frame)

    # Release the video objects and writers
    hr_video.release()
    lr_video_writer.release()


"""def interpolate_lr_video(temp_video_path, hr_video_path, lr_video_path):
    # Open the temporary video
    temp_video = cv2.VideoCapture(temp_video_path)

    # Open the HR video
    hr_video = cv2.VideoCapture(hr_video_path)

    # Get the HR video properties
    fps = hr_video.get(cv2.CAP_PROP_FPS)
    width = int(hr_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(hr_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the LR video
    lr_video_writer = cv2.VideoWriter(lr_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Read each frame from the temporary video and interpolate to HR dimensions
    while True:
        ret_temp, frame_temp = temp_video.read()
        ret_hr, frame_hr = hr_video.read()
        if not ret_temp or not ret_hr:
            break

        # Upscale the frame to HR dimensions using interpolation
        hr_frame = cv2.resize(frame_temp, (width, height), interpolation=cv2.INTER_CUBIC)

        # Write the HR frame to the LR video
        lr_video_writer.write(hr_frame)

    # Release the temporary and HR videos, and close the LR video writer
    temp_video.release()
    hr_video.release()
    lr_video_writer.release()"""

def downscale_video_chunk(input, output, target_bitrate):
    # Define the FFmpeg command to transcode the video with the desired bitrate
    ffmpeg_command = [
        "ffmpeg",
        "-i", input,
        "-c:v", "libx264",
        "-crf", "33",
        "-c:a", "copy",
        output
    ]

    # Run the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)

num_chunks=8
target_bitrate=100

# Iterate over the HR video chunks and downscale each chunk to LR
for chunk_id in range(1, num_chunks + 1):
    #hr_chunk_path = os.path.join(hr_video_chunks_dir, f"chunk_{chunk_id}.mp4")
    #lr_chunk_path = os.path.join(lr_video_chunks_dir, f"chunk_{chunk_id}.mp4")
    #temp_chunk_path = os.path.join(temp_path, f"chunk_{chunk_id}.mp4")
    #downscale_video_chunk(hr_chunk_path, temp_chunk_path)
    #interpolate_lr_video(temp_chunk_path, hr_chunk_path, lr_chunk_path)
    #downscale_video_chunk(hr_chunk_path, lr_chunk_path, target_bitrate)
    pass

hr_path=r"E:\SR-Video-Stream\video\raw.mp4"
lr_path=r"E:\SR-Video-Stream\lr_video\low resolution.mp4"
lr_path2=r"E:\SR-Video-Stream\lr_video\low bitrate.mp4"
#downscale_video_chunk(hr_path, lr_path, target_bitrate)
downscale_video(hr_path, lr_path)
downscale_video_chunk(lr_path, lr_path2, target_bitrate)