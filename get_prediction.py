from keras.config import enable_unsafe_deserialization
from pickle import load
from os import path
import numpy as np
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
    # Load the trained ensemble models
    attention_model = create_models.create_attention_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    print("Attention model created successfully.")
    cnn_model = create_models.create_cnn_lstm_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    print("CNN-LSTM model created successfully.")
    transformer_model = create_models.create_transformer_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    print("Transformer model created successfully.")

    models = [attention_model, cnn_model, transformer_model]
    
    if not models:
        print("No trained models found!")
        return 'unknown'
    
    # Use the loaded label encoder instead of the global one
    if label_encoder is None:
        print("Label encoder not found!")
        return 'unknown'
    
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
    
    if X.size == 0:
        print("No sequences generated from video")
        return 'unknown'
    
    try:
        # Use ensemble prediction
        ensemble_pred = create_models.ensemble_prediction_per_frame(models, X)
        
        if ensemble_pred.size > 0:
            # Get predictions for each frame
            frame_predictions = []
            for frame_idx in range(ensemble_pred.shape[1]):
                frame_pred_class = np.argmax(ensemble_pred[0, frame_idx, :])
                frame_pred_label = label_encoder.inverse_transform([frame_pred_class])[0]
                frame_predictions.append(frame_pred_label)
            
            # Filter consecutive duplicates
            filtered_predictions = []
            for pred in frame_predictions:
                if not filtered_predictions or pred != filtered_predictions[-1]:
                    filtered_predictions.append(pred)
            
            return ' '.join(filtered_predictions) if filtered_predictions else 'unknown'
        
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")
        return 'unknown'
    
    return 'unknown'

    
def ensemble_prediction_per_frame(models, X_data):
    """Ensemble predictions by averaging. Handles different sequence lengths by taking majority vote per original frame."""
    if not models:
        print("Error: No models provided for ensemble prediction.")
        return np.array([])

    original_seq_len = X_data.shape[1]
    n_samples = X_data.shape[0]
    
    # Get predictions from all models
    all_predictions = []
    for name, model in models:
        pred_raw = model.predict(X_data, verbose=0)
        model_seq_len = pred_raw.shape[1]
        
        if model_seq_len == original_seq_len:
            # Direct use for models with same sequence length
            all_predictions.append(pred_raw)
        else:
            # Upsample predictions to match original sequence length
            # Simple interpolation approach
            upsampled_pred = np.zeros((n_samples, original_seq_len, pred_raw.shape[2]))
            for i in range(n_samples):
                for class_idx in range(pred_raw.shape[2]):
                    # Interpolate each class probability
                    original_indices = np.linspace(0, model_seq_len - 1, original_seq_len)
                    upsampled_pred[i, :, class_idx] = np.interp(
                        original_indices, 
                        np.arange(model_seq_len), 
                        pred_raw[i, :, class_idx]
                    )
            all_predictions.append(upsampled_pred)
            print(f"Upsampled {name} predictions from {model_seq_len} to {original_seq_len} frames")

    if not all_predictions:
        print("Error: No valid predictions for ensembling.")
        return np.array([])

    # Average all predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    return ensemble_pred
                
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
