from flask import Flask, render_template, request, jsonify, redirect, url_for
from os import makedirs, path, listdir, environ
from celery import Celery
import redis
import get_prediction


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit
app.config['UPLOAD_FOLDER'] = '/data/uploads'
if not path.exists(app.config['UPLOAD_FOLDER']):
    makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

redis_client = redis.StrictRedis.from_url('redis://red-d0r56hre5dus73fkg89g:6379')


processing_results = {}

def make_celery(app):
    redis_url = environ.get("REDIS_URL")
    celery = Celery(
        app.import_name,
        backend=redis_url,
        broker=redis_url
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

@app.route("/")
def home():
    return render_template("index.html")

@celery.task()
def run_prediction(filename, user):
    # Retrieve file bytes from Redis
    file_bytes = redis_client.get(f"video:{filename}")
    if file_bytes is None:
        result = "File not found in Redis."
    else:
        # Save to a temp file or process directly
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        result = get_prediction.predict(temp_path, user)
    processing_results[filename] = result
    redis_client.set(f"result:{filename}", result)
    
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
                file_bytes = video_file.read()

                redis_client.set(f"video:{filename}", file_bytes)
                # Start celery task
                task = run_prediction.delay(filename, 'default_user')
                processing_results[filename] = None
                return redirect(url_for('interpreter_page', filename=filename))
            else:
                text_output = "No video file selected."
                is_processing = True

    filename = request.args.get('filename')
    if filename:
        result = redis_client.get(f"result:{filename}")
        if result is None:
            is_processing = True
            text_output = ""
        else:
            is_processing = True
            text_output = result.decode() if isinstance(result, bytes) else result

    return render_template("interpreter.html",
                           text_output=text_output,
                           is_processing=is_processing,
                           filename=filename)

@app.route("/check_result")
def check_result():
    filename = request.args.get('filename')
    result = redis_client.get(f"result:{filename}")

    redis_client.flushdb()
    return jsonify({"ready": result is not None})

if __name__ == "__main__":
    app.run(debug=True)
