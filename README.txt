enviromnment:
ffmpeg and GPAC MP4Box is installed and path are set in windows
anaconda python 3.10 for keras run on GPU

prepare video "raw.mp4" inside folder "video"

1. run chucks.py: divide the video into chunks and will save video chucks into folder "hr_video_chunks"

2. run lr_gen.py: downscale a video chunk from folder "hr_video_chunks" into lower quaily video chunk by lowering bit rate and will save video chucks into folder "lr_video_chunks" 

3.run brake_video.py: turning "hr_video_chunks" and  "lr_video_chunks" video into images and will save images into folder "hr_frame_imges"    and "lr_frame_imges"

4. run trainer.py: training model for each video chunks use hr images and lr images from folder "hr_frame_imges" and "lr_frame_imges"
and save models into folder "model"

5. run server.py: start flask server for sending lr video and trained model for each chucks from folder "lr_video_chunks" and "model"

6. run client.py receive lr video and trained model and saved in "client_model_download" and "client_video_downlaod" and do prediction in quaily enhencement, it will display the predicted images frame by frame, the result of traning with my own images and model is bad where is need to impove.

6.1 pretrained_client.py: similar to client.py but only receive lr video and use pretrained model downloaded from internet, the result of it is as expected, it will store frames enheced in folder "client_storage" and reconstruct video for "output.mp4" 


NOT related the own implementation programs:
trainer.py: similar to trainer2D.py but use 3Dcnn, found very diffcult to use video with 3Dcnn in early development so change to trainer2D.py use 2dcnn for images, try to impove only when have time

ESDR.py: early development similar to pretrained_client.py but dnn_superres is not supported in my machine

test_img.py: test trined model result on single image instead of whole video for saving time

play.py: test camera and the codec for reconstruct vidoe from images


You can ignore the NOT related the own implementation programs, which is not affecting the main workflow of implementation or approaches is given up, keep it for reference only
