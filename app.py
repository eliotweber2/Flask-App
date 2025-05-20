from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import cv2 # Needed for video processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
import tensorflow as tf # Ensure tensorflow is imported

# Import functions from your data_processing script
import data_processing

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
SEQUENCE_LENGTH = 14  # Should match what the model was trained on
NUM_FEATURES = 93     # (21 landmarks * 3 coords) + 30 pairwise features (15 dist + 15 angles)
                        # Adjust if your feature extraction is different (e.g., no pairwise: 63)
LABELS = ["Hello", "Thank you", "Yes", "No", "Please", "Sorry", "Good", "Bye"] # Example labels
NUM_CLASSES = len(LABELS)
INT_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# Load the trained model (architecture from paste.py)
def load_sign_language_model(model_path='model/sign_language_model.h5'):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}. Using uninitialized model.")
    else:
        print(f"Warning: Model file not found at {model_path}. Using uninitialized model.")
    
    # Compile the model - necessary even for inference if loaded this way without optimizer state
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_sign_language_model()

# --- Placeholder for Landmark Extraction ---
# In a real app, this would use a library like MediaPipe to process video frames
def simulate_landmark_extraction(num_frames, sequence_length):
    """
    Simulates extracting landmarks for a number of frames.
    Returns data in the list-of-dictionaries format that prepare_sequences can use after parsing.
    Each frame dictionary should represent landmarks for that frame.
    """
    all_frames_landmarks = []
    for _ in range(num_frames):
        # Simulate landmarks for one frame (e.g., one hand, 21 landmarks)
        # Format: {'result': 'DETECTION_SUCCESS', 'landmarks': [ [[id,x,y,z], ...], ... ]}
        # For simplicity, creating dummy landmarks
        hand_landmarks = []
        for i in range(21): # 21 landmarks per hand
            hand_landmarks.append([i, np.random.rand(), np.random.rand(), np.random.rand()])
        
        # Simulate one or two hands randomly
        num_hands = np.random.randint(1, 3)
        current_frame_hands_data = [hand_landmarks for _ in range(num_hands)]

        if current_frame_hands_data:
            all_frames_landmarks.append({
                'result': 'DETECTION_SUCCESS',
                'landmarks': current_frame_hands_data
            })
        else:
             all_frames_landmarks.append({'result': 'DETECTION_FAILURE', 'landmarks': []})

    # Ensure the sequence length matches what's needed, pad if necessary
    # data_processing.pad_seq expects a list of frames
    if len(all_frames_landmarks) < sequence_length:
        all_frames_landmarks = data_processing.pad_seq(all_frames_landmarks, sequence_length)
    
    return all_frames_landmarks[:sequence_length] # Return only the required sequence length

def process_video_file_to_landmarks(video_path, sequence_length):
    """
    Placeholder: Simulates extracting landmarks from a video file.
    In a real scenario, you'd use OpenCV to read frames and a landmark detector.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Simulating landmark extraction for video {video_path} with approx {frame_count} frames.")
    # For this simulation, we just generate a fixed number of landmark sets based on sequence_length
    # A real implementation would extract from actual video frames.
    return simulate_landmark_extraction(max(frame_count, sequence_length), sequence_length)

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
                landmark_frames_list = process_video_file_to_landmarks(video_path, SEQUENCE_LENGTH)
                
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
