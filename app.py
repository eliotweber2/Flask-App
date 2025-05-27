from flask import Flask, render_template, request, jsonify
from os import makedirs, path
import numpy as np
from pandas import DataFrame
#from memory_profiler import profile

# Import functions from your data_processing script
import get_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
#@profile
def home():
    return render_template("index.html")

@app.route("/interpreter", methods=['GET', 'POST'])
def interpreter_page():
    text_output = ""
    is_processing = False
    
    if request.method == "POST":
        print("Received POST request")
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

if __name__ == "__main__":
    app.run(debug=True)
