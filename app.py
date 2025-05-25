from flask import Flask, render_template, request, jsonify
from os import makedirs, path, rename, remove
import numpy as np
from pandas import DataFrame
from cv2 import VideoWriter, imdecode, IMREAD_COLOR, VideoWriter_fourcc, VideoCapture, CAP_PROP_FRAME_COUNT
from base64 import b64decode
from glob import glob
#from werkzeug.middleware.profiler import ProfilerMiddleware

# Import functions from your data_processing script
import data_processing
import get_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
#app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

files = glob('./slices/*')
for f in files:
    remove(f)

files = glob(path.join(app.config['UPLOAD_FOLDER'], '*'))
for f in files:
    remove(f)

SEQUENCE_LENGTH = 9  # Should match what the model was trained on

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/interpreter", methods=['GET', 'POST'])
def interpreter_page():
    text_output = ""
    is_processing = False
    
    if request.method == "POST":
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename != '':
                filename = video_file.filename
                video_path = path.join(app.config['UPLOAD_FOLDER'], filename)
                video_file.save(video_path)

                slice_video(filename)  # Slice the video if needed
                
                text_output = get_prediction.predict(filename, 'default_user')
                
                is_processing = True
                # Clean up uploaded file
                # os.remove(video_path) # Optional: remove after processing
            else:
                text_output = "No video file selected."
                is_processing = True
    
    return render_template("interpreter.html", 
                           text_output=text_output,
                           is_processing=is_processing)

@app.route("/interpret_live_frames", methods=['POST'])
def interpret_live_frames():
    data = request.get_json()
    frames_base64 = data.get('frames') # Expecting a list of base64 encoded frames
    print(frames_base64)
    user_id = data.get('userId', 'default_user') # Simple user ID for buffer management

    video_path = path.join(app.config['UPLOAD_FOLDER'], f"live_{user_id}.mp4")
    if not frames_base64 or len(frames_base64) == 0:
        return jsonify({'text': "No frames received."}), 400
    
    frames = []
    for frame_b64 in frames_base64:
        frame_data = b64decode(frame_b64)
        np_arr = np.frombuffer(frame_data, np.uint8)
        img = imdecode(np_arr, IMREAD_COLOR)
        if img is not None:
            frames.append(img)

    height, width, layers = frames[0].shape
    fps = 30

    fourcc = VideoWriter_fourcc(*'mp4v')
    out = VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    print(f"Received {len(frames_base64)} frames for live interpretation from user {user_id}.")
    slice_video(video_path)

    text = get_prediction.predict(video_path, user_id)
        
    return jsonify({'text': text})

def slice_video(video_path):
    video = VideoCapture(video_path)

    if not video.get(CAP_PROP_FRAME_COUNT) <= SEQUENCE_LENGTH:
        offsets = range(0, int(video.get(CAP_PROP_FRAME_COUNT)) - SEQUENCE_LENGTH + 1)
        for offset in offsets:
            out = VideoWriter(
                f'slices/slice_{video_path}_{offset}.mp4',
                VideoWriter_fourcc(*'mp4v'),
                30,
                (int(video.get(3)), int(video.get(4)))
            )
            video.set(1, offset)
            for frame in range(SEQUENCE_LENGTH):
                ret, frame = video.read()
                if not ret:
                    break
                out.write(frame)
            out.release()

        return
            
    rename(path.join(app.config['UPLOAD_FOLDER'], video_path), 
              f'slices/slice_{video_path}_0.mp4')
    
if __name__ == "__main__":
    app.run(debug=True)
