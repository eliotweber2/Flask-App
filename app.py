from flask import Flask, render_template, request, jsonify
from os import makedirs, path
import numpy as np
from pandas import DataFrame

# Import functions from your data_processing script
import data_processing
import get_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
                
                text_output = get_prediction.predict(video_path, 'default_user')
                
                is_processing = True
                # Clean up uploaded file
                # os.remove(video_path) # Optional: remove after processing
            else:
                text_output = "No video file selected."
                is_processing = True
    
    return render_template("interpreter.html", 
                           text_output=text_output,
                           is_processing=is_processing)

# This session store is a simplified way to handle frame buffers for multiple users.
# In a production app, use a more robust solution (e.g., Redis, Flask-Session with server-side storage).
user_frame_buffers = {} 

@app.route("/interpret_live_frames", methods=['POST'])
def interpret_live_frames():
    data = request.get_json()
    frames_base64 = data.get('frames') # Expecting a list of base64 encoded frames
    print(frames_base64)
    user_id = data.get('userId', 'default_user') # Simple user ID for buffer management

    if not frames_base64 or len(frames_base64) == 0:
        return jsonify({'text': "No frames received."}), 400
    
    

    # --- Placeholder for actual live frame to landmark conversion ---
    # The client-side camera.js is set to send SEQUENCE_LENGTH frames.
    
    # This function needs to be adapted if you send frames one by one and buffer on server
    # For now, assume `frames_base64` corresponds to one sequence worth of visual data
    print(f"Received {len(frames_base64)} frames for live interpretation from user {user_id}.")

    text = get_prediction.predict(video_path, user_id)
        
    return jsonify({'text': text})

if __name__ == "__main__":
    app.run(debug=True)
