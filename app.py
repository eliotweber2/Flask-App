from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd

# Import functions from your data_processing script
import data_processing
import get_prediction

SEQUENCE_LENGTH = 14  # Should match what the model was trained on
NUM_FEATURES = 93
LABELS = ["Hello", "Thank you", "Yes", "No", "Please", "Sorry", "Good", "Bye"] # Example labels
INT_TO_LABEL = {i: label for i, label in enumerate(LABELS)}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = get_prediction.load_sign_language_model('sign_language_model_Attention_LSTM.h5')

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
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video_file.save(video_path)
                
                # 1. Placeholder: Extract Landmarks from video file
                # This returns a list of frame landmark dicts
                landmark_frames_list = get_prediction.process_video_file_to_landmarks(video_path)
                
                if not landmark_frames_list:
                    text_output = "Could not extract landmarks from video."
                else:
                    # 2. Prepare data for the model using data_processing.py
                    # prepare_sequences expects a DataFrame
                    df_data = [{'video_id': 'uploaded_video', 'label': 'unknown', 'landmarks': landmark_frames_list}]
                    landmarks_df = pd.DataFrame(df_data)
                    
                    # Ensure 'landmarks' column contains the list of dicts, not string
                    # print(f"Debug: Type of landmarks in DF for prepare_sequences: {type(landmarks_df['landmarks'].iloc[0])}")

                    X, _ = data_processing.prepare_sequences(
                        landmarks_df, 
                        sequence_length=SEQUENCE_LENGTH, 
                        include_pairwise=True, # Assuming your NUM_FEATURES includes pairwise
                        pad_value=0.0
                    )
                    
                    if X.shape[0] > 0 and X.shape[2] == NUM_FEATURES:
                        # 3. Get Model Prediction
                        predictions = model.predict(X)
                        predicted_class_indices = np.argmax(predictions, axis=1)
                        
                        # Map prediction to text
                        # For a single sequence from a file, we take the first prediction
                        predicted_label = INT_TO_LABEL.get(predicted_class_indices[0], "Unknown prediction")
                        text_output = f"Interpreted: {predicted_label}"
                    elif X.shape[0] > 0 and X.shape[2] != NUM_FEATURES:
                        text_output = f"Error: Feature mismatch. Expected {NUM_FEATURES}, got {X.shape[2]}."
                    else:
                        text_output = "Could not prepare sequence for model."
                
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
    user_id = data.get('userId', 'default_user') # Simple user ID for buffer management

    if not frames_base64 or len(frames_base64) == 0:
        return jsonify({'text': "No frames received."}), 400

    # --- Placeholder for actual live frame to landmark conversion ---
    # The client-side camera.js is set to send SEQUENCE_LENGTH frames.
    
    # This function needs to be adapted if you send frames one by one and buffer on server
    # For now, assume `frames_base64` corresponds to one sequence worth of visual data
    print(f"Received {len(frames_base64)} frames for live interpretation from user {user_id}.")
    landmark_frames_list = simulate_landmark_extraction(len(frames_base64), SEQUENCE_LENGTH)

    if not landmark_frames_list:
        return jsonify({'text': "Could not extract landmarks from live frames."})

    # Prepare data for the model
    df_data = [{'video_id': f'live_{user_id}', 'label': 'unknown', 'landmarks': landmark_frames_list}]
    landmarks_df = pd.DataFrame(df_data)

    X, _ = data_processing.prepare_sequences(
        landmarks_df, 
        sequence_length=SEQUENCE_LENGTH, 
        include_pairwise=True,
        pad_value=0.0
    )

    interpreted_text = "Processing..."
    if X.shape[0] > 0 and X.shape[2] == NUM_FEATURES:
        predictions = model.predict(X)
        predicted_class_idx = np.argmax(predictions[0])
        interpreted_text = INT_TO_LABEL.get(predicted_class_idx, "Unknown Sign")
    elif X.shape[0] > 0 and X.shape[2] != NUM_FEATURES:
        interpreted_text = f"Feature mismatch (got {X.shape[2]})"
    else:
        interpreted_text = "Could not process sequence."
        
    return jsonify({'text': interpreted_text})

if __name__ == "__main__":
    app.run(debug=True)
