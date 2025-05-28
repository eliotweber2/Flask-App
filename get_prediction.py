def predict(video_path, user_id):
    # Load the trained ensemble models
    models, label_encoder = create_models.load_ensemble_per_frame_models()
    
    if not models:
        print("No trained models found!")
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
    
    # Use ensemble prediction
    ensemble_pred = create_models.ensemble_prediction_per_frame(models, X)
    
    if ensemble_pred.size > 0:
        # Get predictions for each frame
        frame_predictions = []
        for frame_idx in range(ensemble_pred.shape[1]):
            frame_pred_class = argmax(ensemble_pred[0, frame_idx, :])
            frame_pred_label = label_encoder.inverse_transform([frame_pred_class])[0]
            frame_predictions.append(frame_pred_label)
        
        # Filter consecutive duplicates
        filtered_predictions = []
        for pred in frame_predictions:
            if not filtered_predictions or pred != filtered_predictions[-1]:
                filtered_predictions.append(pred)
        
        return ' '.join(filtered_predictions)
    
    return 'unknown'
