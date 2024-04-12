import cv2
import os

# Divide the HR video into chunks
def divide_video_into_chunks(video_path,output_dir, chunk_duration):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS:", fps)
    print("Total Frames:", total_frames)

    # Calculate the number of frames per chunk based on the desired duration
    frames_per_chunk = int(chunk_duration * fps)

    # Initialize variables
    current_frame = 0
    chunk_id = 1

    while current_frame < total_frames:
        # Set the starting and ending frame for the current chunk
        start_frame = current_frame
        end_frame = min(current_frame + frames_per_chunk, total_frames)

        # Extract frames from the video chunk
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for frame_index in range(start_frame, end_frame):
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)

        # Save the video chunk as a new video file
        chunk_filename = f"chunk_{chunk_id}.mp4"
        chunk_path = os.path.join(output_dir, chunk_filename)

        # Write the frames to the video file
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(chunk_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

        # Update variables for the next chunk
        current_frame = end_frame
        chunk_id += 1

    # Release the video object
    video.release()

# Define the path to the HR LR video file
hr_video_path = "E:/SR-Video-Stream/video/raw.mp4"
lr_video_path = "E:/SR-Video-Stream/lr_video/low bitrate.mp4"

# Define the directory where the video chunks will be stored
hr_video_chunks_dir = "E:/SR-Video-Stream/hr_video_chunks"
lr_video_chunks_dir = "E:/SR-Video-Stream/lr_video_chunks"

video = cv2.VideoCapture(hr_video_path)
if not video.isOpened():
    print("Failed to open the video file.")

if os.path.exists(hr_video_path):
    print("File exists.")
else:
    print("File does not exist.")

# Divide the HR video into chunks (10 seconds per chunk)
divide_video_into_chunks(hr_video_path,hr_video_chunks_dir, 10)

video = cv2.VideoCapture(lr_video_path)
if not video.isOpened():
    print("Failed to open the video file.")

if os.path.exists(lr_video_path):
    print("File exists.")
else:
    print("File does not exist.")

# Divide the HR video into chunks (10 seconds per chunk)
divide_video_into_chunks(lr_video_path,lr_video_chunks_dir, 10)