from flask import Flask,  send_from_directory,  send_file
import os
import cv2

app = Flask(__name__)

# Define the directory where the video chunks will be stored
video_chunks_dir = "E:/SR-Video-Stream/lr_video_chunks"
model_files_dir="E:/SR-Video-Stream/model"

@app.route('/video_chunks/<chunk_id>', methods=['GET'])
def get_video_chunk(chunk_id):
    # Construct the path to the video chunk file
    chunk_file = os.path.join(video_chunks_dir, f"chunk_{chunk_id}.mp4")
    # Send the video chunk file as the response
        #return send_from_directory(path=chunk_file,directory=video_chunks_dir,filename=f"chunk_{chunk_id}.mp4" ,mimetype='video/mp4')
    return send_file(chunk_file, as_attachment=True)
    
@app.route('/model_chunks/<chunk_id>', methods=['GET'])
def get_model_chunk(chunk_id):
    # Construct the path to the model file
    model_file = os.path.join(model_files_dir, f"model_{chunk_id}.h5")

    # Send the model file as the response
        #return send_from_directory(path=model_file, directory=model_files_dir, filename=f"model_{chunk_id}.h5",mimetype='application/octet-stream', as_attachment=True, download_name=f"model_{chunk_id}.h5")
    return send_file(model_file, as_attachment=True)
if __name__ == '__main__':
     # Iterate through the video chunks
    #for chunk_id in range(1, 9):
        #get_model_chunk(chunk_id)
        #get_video_chunk(chunk_id)

    app.run(host='0.0.0.0', port=5000)
