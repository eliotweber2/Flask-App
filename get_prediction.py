from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout, Bidirectional
from os import path
from numpy import random

import data_processing
import video_loader
import obj_detect

from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT

detector = obj_detect.Landmark_Creator()

SEQUENCE_LENGTH = 14  # Should match what the model was trained on
NUM_FEATURES = 93     # (21 landmarks * 3 coords) + 30 pairwise features (15 dist + 15 angles)
                        # Adjust if your feature extraction is different (e.g., no pairwise: 63)
LABELS = ["Hello", "Thank you", "Yes", "No", "Please", "Sorry", "Good", "Bye"] # Example labels
NUM_CLASSES = len(LABELS)

# Load the trained model (architecture from paste.py)
def load_sign_language_model(model_path='sign_language_model_Attention_LSTM.h5'):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=32)))
    model.add(Dropout(0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    if path.exists(model_path):
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

def format_landmarks(frame_lst):
    formatted_landmarks = '||||'.join(
        ['|||'.join(
            ['||'.join([
                '|'.join([str(coord) for coord in landmark[1:4]])
            for landmark in landmark_lst ])
        for landmark_lst in frame])
    for frame in frame_lst])  
    return formatted_landmarks

def create_stream_interface(video_path):
    print(video_path,path.exists(video_path))
    return video_loader.StreamInterface(
        open=lambda: VideoCapture(video_path),
        read_frame=lambda cap: cap.read(),
        error=lambda error: print(f"Error: {error}"),
        close=lambda cap: cap.release()
    )

def process_video_file_to_landmarks(video_path):
    landmark_lst = []
    def process_frame(frame):
        landmarks = detector.process_image(frame)
        if landmarks['result'] == 'DETECTION_SUCCESS':
            landmark_lst.append(landmarks['landmarks'])

    video_loader.read_and_process(lambda: create_stream_interface(video_path), lambda frame: process_frame(frame),n_skip=1)  

    return format_landmarks(landmark_lst)