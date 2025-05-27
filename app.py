from flask import Flask, render_template, request, jsonify, redirect, url_for
from os import makedirs, path, listdir
from celery import Celery
#from memory_profiler import profile

# Import functions from your data_processing script
import get_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processing_results = {}

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend="redis://red-d0r2flqdbo4c73emjt4g:6379",
        broker="redis://red-d0r2flqdbo4c73emjt4g:6379"
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

@app.route("/")
#@profile
def home():
    return render_template("index.html")

@celery.task()
def run_prediction(video_path, user, filename):
    print(f"Processing video: {video_path} for user: {user}")
    print(path.exists(video_path))
    print(listdir(app.config['UPLOAD_FOLDER']))
    result = get_prediction.predict(video_path, user)
    print(f"Processing result for {filename}: {result}")
    processing_results[filename] = result
    return result

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
                print(f"Video saved to {video_path}")
                # Start celery task
                task = run_prediction.delay(video_path, 'default_user', filename)
                processing_results[filename] = None  # Mark as processing
                return redirect(url_for('interpreter_page', filename=filename))
            else:
                text_output = "No video file selected."
                is_processing = True

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
    print(result)
    return jsonify({"ready": result is not None})

if __name__ == "__main__":
    app.run(debug=True)
