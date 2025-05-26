from numpy import array, linalg, sqrt, random
from pandas import DataFrame
# Assuming your custom modules like video_loader, obj_detect, camera are in the PYTHONPATH
# from video_loader import read_and_process, StreamInterface
# from obj_detect import Landmark_Creator
# from camera import camera_stream_factory

# Fixed format_landmarks function to properly parse landmarks
def parse_landmarks(landmarks_str):
    """Parse landmarks from string format to structured format."""
    if not isinstance(landmarks_str, str):
        # If it's not a string, assume it might be already parsed or is an error
        # For this function's original intent, we return [] if not a parsable string.
        # print(f"Debug: parse_landmarks received non-string input: {type(landmarks_str)}")
        return []

    try:
        frames = landmarks_str.split('||||')
        parsed_frames = []

        for frame_str in frames:
            if not frame_str:
                continue

            hands = frame_str.split('|||')
            parsed_hands_in_frame = []

            for hand_idx, hand_str in enumerate(hands):
                if not hand_str:  
                    continue
  
                landmarks = hand_str.split('||')
                parsed_landmarks_in_hand = []

                for landmark_idx, landmark_coords_str in enumerate(landmarks):
                    coords = landmark_coords_str.split('|')
                    if len(coords) != 3:
                        # print(f"Warning: Malformed coordinate string '{landmark_coords_str}' in frame. Skipping.")
                        continue
                    try:
                        # Create landmark in format [id, x, y, z]
                        parsed_landmarks_in_hand.append([landmark_idx, float(coords[0]), float(coords[1]), float(coords[2])])
                    except ValueError:
                        # print(f"Warning: Could not convert coords to float: {coords}. Skipping landmark.")
                        continue
               
                if parsed_landmarks_in_hand:
                    # Ensure a hand has the expected number of landmarks (e.g., 21 for MediaPipe Hands)
                    # This check can be added if a fixed number is always expected per hand.
                    # if len(parsed_landmarks_in_hand) == 21:
                    parsed_hands_in_frame.append(parsed_landmarks_in_hand)
                    # else:
                    #     print(f"Warning: Hand did not have 21 landmarks, got {len(parsed_landmarks_in_hand)}. Skipping hand.")


            if parsed_hands_in_frame:
                parsed_frame_dict = {
                    'result': 'DETECTION_SUCCESS', # Assuming if hands are parsed, detection was a success
                    'landmarks': parsed_hands_in_frame
                }
                parsed_frames.append(parsed_frame_dict)
            # else:
                # If a frame string was present but no valid hands were parsed, it results in an empty frame for landmarks.
                # Depending on requirements, one might append a "DETECTION_FAILURE" or skip.
                # For now, if no hands, the frame is effectively empty of landmarks.
                # To ensure consistency, if a frame was meant to be there but empty, add a specific structure:
                # parsed_frames.append({'result': 'DETECTION_FAILURE', 'landmarks': []})


        return parsed_frames
    except Exception as e:
        print(f"Error parsing landmarks string: {e}, Input: '{landmarks_str[:100]}...'") # Print first 100 chars
        return []

# Enhanced landmark normalization for better accuracy
def normalize_landmarks_enhanced(landmarks_data_frame): # landmarks_data_frame is one frame's data
    """Enhanced normalization with scale invariance and hand orientation correction."""
    if not landmarks_data_frame or landmarks_data_frame.get('result') != 'DETECTION_SUCCESS' or not landmarks_data_frame.get('landmarks'):
        return None

    normalized_frame_hands = []
    for hand_landmarks_list in landmarks_data_frame['landmarks']: # Iterating through hands in the frame
        # Get wrist as reference point (point 0)
        wrist = None
        for point in hand_landmarks_list: # point is [id, x, y, z]
            if point[0] == 0:  # Wrist point
                wrist = point[1:4] # [x, y, z]
                break
       
        if wrist is None: # Should not happen if landmarks are well-formed from MediaPipe
            # print("Warning: Wrist point (0) not found in hand. Skipping hand normalization.")
            continue # Skip this hand

        index_mcp, middle_mcp, pinky_mcp = None, None, None
        for point in hand_landmarks_list:
            if point[0] == 5: index_mcp = point[1:4]
            elif point[0] == 9: middle_mcp = point[1:4]
            elif point[0] == 17: pinky_mcp = point[1:4]

        current_hand_normalized_landmarks = []

        # Fallback to simple normalization if key MCPs are missing for advanced normalization
        if index_mcp is None or pinky_mcp is None or middle_mcp is None: # Require all three for advanced
            # print("Warning: Not all MCPs (5,9,17) found. Using simple wrist-relative normalization for this hand.")
            scale = 1.0 # No scale normalization in this simplified fallback
            for point in hand_landmarks_list:
                point_id = point[0]
                x = (point[1] - wrist[0]) / scale
                y = (point[2] - wrist[1]) / scale
                z = (point[3] - wrist[2]) / scale
                current_hand_normalized_landmarks.append([point_id, x, y, z])
            if current_hand_normalized_landmarks:
                 normalized_frame_hands.append(current_hand_normalized_landmarks)
            continue # Move to the next hand in the frame

        # Calculate hand scale (distance between index and pinky MCP)
        scale = sqrt(
            (index_mcp[0] - pinky_mcp[0])**2 +
            (index_mcp[1] - pinky_mcp[1])**2 +
            (index_mcp[2] - pinky_mcp[2])**2
        )
        if scale < 1e-6: scale = 1.0 # Avoid division by zero or excessively small scale

        # --- Advanced Normalization (Simplified version from your code, without rotation matrix application) ---
        # Your code calculates rotation_matrix but then doesn't use it for transforming points.
        # It uses 'centered / scale'. If rotation is intended, points should be multiplied by inv(rotation_matrix).
        # For now, sticking to your implemented logic (scaling and centering).
       
        for point in hand_landmarks_list:
            point_id = point[0]
            centered_x = point[1] - wrist[0]
            centered_y = point[2] - wrist[1]
            centered_z = point[3] - wrist[2]
           
            norm_x = centered_x / scale
            norm_y = centered_y / scale
            norm_z = centered_z / scale
            current_hand_normalized_landmarks.append([point_id, norm_x, norm_y, norm_z])
       
        if current_hand_normalized_landmarks:
            normalized_frame_hands.append(current_hand_normalized_landmarks)

    if not normalized_frame_hands: # If no hands were successfully normalized in the frame
        return None

    return {
        'result': 'DETECTION_SUCCESS',
        'landmarks': normalized_frame_hands, # List of hands, each hand is a list of [id, x, y, z]
        'video_id': landmarks_data_frame.get('video_id'), # Propagate original video_id if available
        'label': landmarks_data_frame.get('label') # Propagate original label if available
    }


# Calculate pairwise distances between landmarks for improved features
def calculate_pairwise_features(normalized_landmarks_data): # normalized_landmarks_data is one normalized frame
    """Calculate pairwise distances and angles between key landmarks for each hand."""
    all_hands_additional_features = []

    if not normalized_landmarks_data or not normalized_landmarks_data.get('landmarks'):
        return [] # Return empty list if no landmarks

    for hand_normalized in normalized_landmarks_data['landmarks']: # Iterate through hands in the frame
        hand_features = []
        key_points_indices = [0, 4, 8, 12, 16, 20]  # Wrist, thumb tip, index tip, etc.
        key_coords = {}
       
        for point in hand_normalized: # point is [id, x, y, z]
            if point[0] in key_points_indices:
                key_coords[point[0]] = array(point[1:4]) # Store as numpy array for easier math

        # Skip if not all key points are present for this hand
        if len(key_coords) != len(key_points_indices):
            # print(f"Warning: Not all key points for pairwise features found in a hand. Expected {len(key_points_indices)}, got {len(key_coords)}. Pairwise features for this hand will be limited or empty.")
            # To ensure fixed length, we might need to pad here or handle it later.
            # For now, if critical points are missing, pairwise features for THIS HAND might be incomplete.
            # The outer padding in prepare_sequences will handle the overall feature vector length.
            pass # Continue, features list will be shorter or empty for this hand

        # Calculate distances
        for i in range(len(key_points_indices)):
            for j in range(i + 1, len(key_points_indices)):
                p1_idx, p2_idx = key_points_indices[i], key_points_indices[j]
                if p1_idx in key_coords and p2_idx in key_coords:
                    dist = linalg.norm(key_coords[p1_idx] - key_coords[p2_idx])
                    hand_features.append(dist)
                else: # If a key point is missing, append a default (e.g., 0) for this distance
                    hand_features.append(0.0)


        # Calculate angles for fingers relative to wrist
        # Ensure point 0 (wrist) is available
        if 0 in key_coords:
            wrist_coord = key_coords[0]
            for finger_tip_idx in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky tips
                if finger_tip_idx in key_coords:
                    finger_tip_coord = key_coords[finger_tip_idx]
                    vec = finger_tip_coord - wrist_coord
                    magnitude = linalg.norm(vec)
                    if magnitude > 1e-9: # Avoid division by zero
                        cos_angles = vec / magnitude
                        hand_features.extend(cos_angles.tolist()) # Add cos_x, cos_y, cos_z
                    else:
                        hand_features.extend([0.0, 0.0, 0.0]) # Default if no magnitude
                else: # If a finger tip is missing
                    hand_features.extend([0.0, 0.0, 0.0])
        else: # If wrist is missing, add default for all angle features
             for _ in range(5 * 3): # 5 finger tips, 3 coords each
                hand_features.append(0.0)

        all_hands_additional_features.extend(hand_features) # Concatenate features from all hands in the frame

    return all_hands_additional_features

# Helper function to pad sequences of landmark frames (original user function)
def pad_seq(sequence_of_frames, desired_sequence_length):  
    """Pads a sequence of frame landmark dictionaries with dummy frames."""
    # A dummy frame with plausible structure for normalize_landmarks_enhanced.s
    # It has one hand with 21 landmarks, all at [0,0,0,0].
    dummy_hand = [[i, 0.0, 0.0, 0.0] for i in range(21)]
    dummy_frame = {
        'result': 'DETECTION_SUCCESS', # So normalize_landmarks_enhanced processes it
        'landmarks': [dummy_hand] # List containing one hand
    }
    while len(sequence_of_frames) < desired_sequence_length:
        sequence_of_frames.append(dummy_frame)
    return sequence_of_frames # Return modified sequence

# Improved sequence preparation for ML model
def prepare_sequences(landmarks_dataframe, sequence_length=14, include_pairwise=True, pad_value=0.0):
    """Prepare sequences for LSTM model with enhanced features and masking-compatible padding."""
    all_sequences = []
    all_labels = []

    # Define expected number of features per frame
    # Base features: 21 landmarks * 3 coords = 63 (assuming one hand, or concatenated if multiple)
    # If multiple hands are detected, this needs to be handled.
    # Assuming for now max 1 hand's landmarks are flattened, or a fixed number.
    # For simplicity, let's assume normalize_landmarks_enhanced & calculate_pairwise_features
    # will output features for ONE dominant hand, or concatenate if multiple, up to a max.
    # Current normalize_landmarks_enhanced returns a list of hands.
    # Current calculate_pairwise_features sums up features from all hands. This can lead to variable length.
   
    # Let's fix feature length: assume max 1 hand, 21 landmarks.
    # If more hands, take first. If no hands, it's padded.
    NUM_LANDMARKS_PER_HAND = 21
    COORDS_PER_LANDMARK = 3
    base_coord_features = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK # 63 for one hand

    pairwise_feature_count = 0
    if include_pairwise:
        # Distances: C(6,2) = 15 (for 6 keypoints)
        # Angles: 5 finger tips * 3 cosines = 15
        pairwise_feature_count = 15 + 15 # 30 for one hand
   
    expected_features_per_frame = base_coord_features + pairwise_feature_count # 63 or 93 for one hand

    print(f"Preparing sequences. Expected features per frame: {expected_features_per_frame}")

    for idx, row in landmarks_dataframe.iterrows():
        video_id = row['video_id']
        label = row['label']

        landmarks_input = row['landmarks'] # This can be a string or a list of frame dicts
        if not landmarks_input:
            continue
        if isinstance(landmarks_input, str):
            # print(f"Parsing string for {video_id}")
            current_video_frames_list = parse_landmarks(landmarks_input)
        elif isinstance(landmarks_input, list):
            # print(f"Using pre-parsed list for {video_id}")
            current_video_frames_list = landmarks_input # Already list of frame dicts
        else:
            print(f"Warning: Unknown landmark format for {video_id}. Skipping.")
            continue
       
        if not current_video_frames_list:
            # print(f"Warning: No landmark frames found for {video_id} after parsing/retrieval. Skipping.")
            continue

        # Pad the sequence of FRR_AME_DICTS if it's shorter than sequence_length
        if len(current_video_frames_list) < sequence_length:
            # print(f"Padding video {video_id} from {len(current_video_frames_list)} to {sequence_length} frames.")
            current_video_frames_list = pad_seq(current_video_frames_list, sequence_length)
       
        # Sliding window for sequences longer than sequence_length or just take the first part
        # The stride logic from your original code:
        # max(1, min(5, (len(current_video_frames_list) - sequence_length) // 3 if (len(current_video_frames_list) - sequence_length) > 0 else 1))
        # If len is already sequence_length due to padding, stride logic will ensure one window.
       
        num_frames_in_video = len(current_video_frames_list)
        stride = 1 # Default stride
        if num_frames_in_video > sequence_length:
             # Calculate a dynamic stride, e.g., to get ~3-5 windows from a longer video
            stride = max(1, (num_frames_in_video - sequence_length + 1) // 3) # Ensure at least 1 if difference is small
            stride = min(stride, sequence_length // 2) # Cap stride to avoid too few windows
            stride = max(1, stride) # Ensure stride is at least 1


        for i in range(0, num_frames_in_video - sequence_length + 1, stride):
            window_of_frames = current_video_frames_list[i : i + sequence_length]
           
            single_sequence_feature_vectors = []
            valid_frames_in_window = 0
            for frame_idx, frame_data_dict in enumerate(window_of_frames):
                # Normalize landmarks for the current frame
                # frame_data_dict is like {'result': 'SUCCESS', 'landmarks': [ [[id,x,y,z],...], ... ]}
                normalized_frame_data = normalize_landmarks_enhanced(frame_data_dict) # Processes all hands in frame

                current_frame_features = []
                if normalized_frame_data and normalized_frame_data.get('landmarks'):
                    # For fixed feature length, consider only the first hand's data
                    # If multiple hands, this logic might need to be more sophisticated
                    # (e.g., average, concatenate up to a max, choose dominant hand)
                    first_hand_normalized = normalized_frame_data['landmarks'][0] # Taking the first hand
                   
                    for point in first_hand_normalized: # point is [id, x, y, z]
                        current_frame_features.extend(point[1:4]) # Add x, y, z
                   
                    # Pad base coordinate features if a hand had < NUM_LANDMARKS_PER_HAND (e.g. due to parsing error)
                    # This assumes first_hand_normalized always aims for NUM_LANDMARKS_PER_HAND
                    # For simplicity, assume normalize_landmarks_enhanced output for a hand is always NUM_LANDMARKS_PER_HAND length.
                    # If not, padding needed here:
                    # while len(current_frame_features) < base_coord_features:
                    # current_frame_features.append(pad_value)
                   
                    if include_pairwise:
                        # Create a temporary dict for calculate_pairwise_features for ONE hand
                        temp_normalized_data_for_pairwise = {'landmarks': [first_hand_normalized]}
                        pairwise_features = calculate_pairwise_features(temp_normalized_data_for_pairwise)
                        current_frame_features.extend(pairwise_features)
                    valid_frames_in_window +=1
                else: # Normalization failed or no landmarks in normalized_frame_data
                    pass # Features will be padded below

                # Ensure fixed feature length for THIS frame by padding/truncating
                if len(current_frame_features) < expected_features_per_frame:
                    current_frame_features.extend([pad_value] * (expected_features_per_frame - len(current_frame_features)))
                elif len(current_frame_features) > expected_features_per_frame:
                    current_frame_features = current_frame_features[:expected_features_per_frame]
               
                single_sequence_feature_vectors.append(current_frame_features)
           
            # Ensure the sequence has the desired length (should be true due to windowing)
            if len(single_sequence_feature_vectors) == sequence_length:
                # Only add if there was at least one non-dummy frame processed, or adjust as needed
                # if valid_frames_in_window > 0: # Heuristic to avoid all-padding sequences if not desired
                all_sequences.append(single_sequence_feature_vectors)
                all_labels.append(label)
            # else:
                # This case should ideally not be hit if windowing and pad_seq are correct.
                # print(f"Warning: Sequence for {video_id} window {i} ended up with length {len(single_sequence_feature_vectors)} instead of {sequence_length}. Discarding.")


    return array(all_sequences), array(all_labels)  