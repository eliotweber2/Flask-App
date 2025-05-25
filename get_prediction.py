from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout, Bidirectional
from keras.config import enable_unsafe_deserialization
from sklearn.preprocessing import LabelEncoder
from pickle import load
from os import path, listdir, remove
from numpy import argmax
from pandas import DataFrame

import video_loader
import obj_detect
import data_processing
import create_models

from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, CAP_PROP_FRAME_COUNT

SEQUENCE_LENGTH = 9  # Should match what the model was trained on
NUM_FEATURES = 93     # (21 landmarks * 3 coords) + 30 pairwise features (15 dist + 15 angles)
                        # Adjust if your feature extraction is different (e.g., no pairwise: 63)

with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = load(f)

print(label_encoder.classes_)  # Print the classes for debugging

NUM_CLASSES = len(label_encoder.classes_)  # Number of classes in the dataset

def predict(filename, user_id):
    slices = [slice for slice in listdir('slices') if filename in slice]
    predicted_labels = []
    for slice in slices:
        slice_path = path.join("slices", slice)

                    # Save or process the frame as needed
        landmarks = process_video_file_to_landmarks(slice_path)
        if landmarks == '':
            continue
        
        df_data = [{'video_id': f'live_{user_id}', 'label': 'unknown', 'landmarks': landmarks}]
        landmarks_df = DataFrame(df_data)

        if path.exists(slice_path):
            remove(slice_path)
        
        X, _ = data_processing.prepare_sequences(
                landmarks_df, 
                sequence_length=SEQUENCE_LENGTH, 
                include_pairwise=True,
                pad_value=0.0
            )
                
        for model in [attention_model, cnn_model, transformer_model]:
            # Prepare the data for prediction
            
            # Make predictions
            predictions = model.predict(X)
            
            # Decode the predictions
            predicted_class = argmax(predictions, axis=1)
            predicted_label = label_encoder.inverse_transform(predicted_class)
            
            predicted_labels.append(predicted_label[0])

            print(f"Predicted label for {slice}: {predicted_label[0]}")

    print(f"Predictions for {filename}: {predictions}")   
    
    return max(set(predicted_labels), key=predicted_labels.count) if len(predicted_labels) > 0 else "No predictions made."

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

enable_unsafe_deserialization()

attention_model = create_models.create_attention_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
print("Attention model created successfully.")
cnn_model = create_models.create_cnn_lstm_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
print("CNN-LSTM model created successfully.")
transformer_model = create_models.create_transformer_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
print("Transformer model created successfully.")

detector = obj_detect.Landmark_Creator() 