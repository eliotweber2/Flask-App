from keras.config import enable_unsafe_deserialization
from pickle import load
from os import path
from numpy import argmax
from pandas import DataFrame

import video_loader
import obj_detect
import data_processing
import create_models

from cv2 import VideoCapture

SEQUENCE_LENGTH = 9
NUM_FEATURES = 93

with open('model/label_encoder_per_frame.pkl', 'rb') as f:
    label_encoder = load(f)

print(label_encoder.classes_)

NUM_CLASSES = len(label_encoder.classes_)

def predict(video_path, user_id):
    attention_model = create_models.create_attention_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    print("Attention model created successfully.")
    cnn_model = create_models.create_cnn_lstm_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    print("CNN-LSTM model created successfully.")
    transformer_model = create_models.create_transformer_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    print("Transformer model created successfully.")
    predictions = []
    landmarks = process_video_file_to_landmarks(video_path)
    df_data = [{'video_id': f'live_{user_id}', 'label': 'unknown', 'landmarks': landmarks}]
    landmarks_df = DataFrame(df_data)
    X, _ = data_processing.prepare_sequences(
            landmarks_df, 
            sequence_length=SEQUENCE_LENGTH, 
            include_pairwise=False,
            pad_value=0.0
        )
    
    print("Input shape to model:", X.shape)
    for sequence in X:
        seq_predictions = []
        for model in [attention_model, cnn_model, transformer_model]:
            if model is None:
                continue
            try:
                y_pred = model.predict(X)
                y_pred_classes = argmax(y_pred, axis=1)
                predicted_labels = label_encoder.inverse_transform(y_pred_classes)
                seq_predictions.append(predicted_labels[0])
            except Exception as e:
                print(f"Error during prediction with {model.name}: {e}")
        seq_prediction =  max(set(seq_predictions), key=seq_predictions.count) if seq_predictions else 'unknown'
        if seq_prediction != (predictions[-1] if len(predictions) > 0 else None):
            predictions.append(seq_prediction)
    
    return ' '.join(predictions) if len(predictions) > 0 else 'unknown'
                
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

detector = obj_detect.Landmark_Creator() 