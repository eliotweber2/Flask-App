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
NUM_FEATURES = 372

with open('model/label_encoder_per_frame.pkl', 'rb') as f:
    label_encoder = load(f)
print(label_encoder.classes_)
NUM_CLASSES = len(label_encoder.classes_)

def predict(video_path, user_id):
    # Load the trained ensemble models
    attention_model = create_models.create_attention_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    attention_model.load_weights('sign_language_model_Per_Frame_LSTM_per_frame.weights.h5')
    print("Attention model created successfully.")

    cnn_model = create_models.create_cnn_lstm_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    cnn_model.load_weights('sign_language_model_Per_Frame_CNN_LSTM_per_frame.weights.h5')
    print("CNN-LSTM model created successfully.")

    transformer_model = create_models.create_transformer_model(NUM_CLASSES, SEQUENCE_LENGTH, NUM_FEATURES)
    transformer_model.load_weights('sign_language_model_Per_Frame_Transformer_per_frame.weights.h5')
    print("Transformer model created successfully.")

    models = [attention_model, cnn_model, transformer_model]
    
    if not models:
        print("No trained models found!")
        return 'unknown'
    
    # Use the loaded label encoder instead of the global one
    if label_encoder is None:
        print("Label encoder not found!")
        return 'unknown'
    
    landmarks = process_video_file_to_landmarks(video_path)
    df_data = [{'video_id': f'live_{user_id}', 'label': 'unknown', 'landmarks': landmarks}]
    landmarks_df = DataFrame(df_data)
    
    X, _ = data_processing.prepare_sequences(
        landmarks_df, 
        sequence_length=SEQUENCE_LENGTH, 
        include_pairwise=True,
        pad_value=0.0
    )
    
    print("Input shape to model:", X.shape)
    
    if X.size == 0:
        print("No sequences generated from video")
        return 'unknown'
    
    try:
        # Use ensemble prediction
        ensemble_pred = ensemble_prediction_per_frame(models, X, 0.8)
        
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

    
def ensemble_prediction_per_frame(models, X_data, min_confidence=0.0):
    """Ensemble predictions by averaging. Handles different sequence lengths by taking majority vote per original frame.
    Only includes model predictions for a frame if the model's max probability for that frame >= min_confidence.
    """
    if not models:
        print("Error: No models provided for ensemble prediction.")
        return np.array([])

    original_seq_len = X_data.shape[1]
    n_samples = X_data.shape[0]
    n_classes = None

    # Get predictions from all models
    all_predictions = []
    for model in models:
        pred_raw = model.predict(X_data, verbose=0)
        model_seq_len = pred_raw.shape[1]
        n_classes = pred_raw.shape[2]

        if model_seq_len == original_seq_len:
            pred = pred_raw
        else:
            # Upsample predictions to match original sequence length
            upsampled_pred = np.zeros((n_samples, original_seq_len, n_classes))
            for i in range(n_samples):
                for class_idx in range(n_classes):
                    original_indices = np.linspace(0, model_seq_len - 1, original_seq_len)
                    upsampled_pred[i, :, class_idx] = np.interp(
                        original_indices,
                        np.arange(model_seq_len),
                        pred_raw[i, :, class_idx]
                    )
            pred = upsampled_pred
            print(f"Upsampled predictions from {model_seq_len} to {original_seq_len} frames")

        # Compute per-frame confidence (max probability per frame)
        frame_confidence = np.max(pred, axis=2)  # shape: (n_samples, original_seq_len)

        # Mask out predictions below min_confidence
        mask = frame_confidence >= min_confidence  # shape: (n_samples, original_seq_len)
        masked_pred = np.zeros_like(pred)
        for i in range(n_samples):
            for t in range(original_seq_len):
                if mask[i, t]:
                    masked_pred[i, t, :] = pred[i, t, :]
                else:
                    masked_pred[i, t, :] = np.nan  # Use NaN so we can ignore in averaging

        all_predictions.append(masked_pred)

    if not all_predictions:
        print("Error: No valid predictions for ensembling.")
        return np.array([])

    # Stack and average, ignoring NaNs (frames with no confident prediction)
    stacked = np.stack(all_predictions, axis=0)  # shape: (n_models, n_samples, seq_len, n_classes)
    with np.errstate(invalid='ignore'):
        ensemble_pred = np.nanmean(stacked, axis=0)  # shape: (n_samples, seq_len, n_classes)

    # If all models are below confidence for a frame, result will be NaN for that frame
    # Optionally, you can fill NaNs with zeros or uniform probabilities if desired:
    ensemble_pred = np.nan_to_num(ensemble_pred, nan=0.0)

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
