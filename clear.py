#clear dir
import os
import shutil

def remove_directory_contents(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

remove_directory_contents(r"E:\SR-Video-Stream\lr_video_chunks")
remove_directory_contents(r"E:\SR-Video-Stream\hr_video_chunks") 
remove_directory_contents(r"E:\SR-Video-Stream\client_storage") 
remove_directory_contents(r"E:\SR-Video-Stream\client_video_downlaod")
remove_directory_contents(r"E:\SR-Video-Stream\client_model_download")
remove_directory_contents(r"E:\SR-Video-Stream\lr_video") 
shutil.rmtree(r"E:\SR-Video-Stream\hr_frame_imges")
shutil.rmtree(r"E:\SR-Video-Stream\lr_frame_imges")


