from flask import Flask, render_template, request, jsonify, redirect, url_for
from os import makedirs, path
from threading import Thread
#from memory_profiler import profile

# Import functions from your data_processing script
import get_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processing_results = {}

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
                print("Saving file...")
                video_path = path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
                video_file.save(video_path)
                print("File saved.")
                Thread(target=get_prediction.predict, args=(video_path, 'default_user')).start()
                return redirect(url_for('interpreter_page', text_output='', is_processing=is_processing, filename=video_file.filename))
                # Process the video file
                
                # Clean up uploaded file
                # os.remove(video_path) # Optional: remove after processing
            else:
                text_output = "No video file selected."
                is_processing = True
    print("Processing video if filename is provided")
    filename = request.args.get('filename')
    if filename:
        result = processing_results.get(filename)
        if result is None:
            is_processing = True
            text_output = ""
        else:
            is_processing = True
            text_output = result
    return render_template("interpreter.html", 
                           text_output=text_output,
                           is_processing=is_processing,
                           filename=filename)

@app.route("/check_result")
def check_result():
    filename = request.args.get('filename')
    result = processing_results.get(filename)
    return jsonify({"ready": result is not None})

if __name__ == "__main__":
    app.run(debug=True)
