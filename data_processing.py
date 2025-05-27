import numpy as np
import pandas as pd

def parse_landmarks(landmarks_str):
    """
    Parse a single string of MediaPipe-style landmarks into a list of frames,
    where each frame contains one or more hands, and each hand is a list of
    [id, x, y, z] landmark coordinates.
    """
    # Return empty if input is not a string
    if not isinstance(landmarks_str, str):
        return []

    try:
        # Split by '||||' to separate frames
        frames = landmarks_str.split('||||')
        parsed_frames = []

        for frame_str in frames:
            if not frame_str:
                # Skip empty frame segments
                continue

            # Each frame may contain multiple hands, separated by '|||'
            hands = frame_str.split('|||')
            parsed_hands_in_frame = []

            for hand_idx, hand_str in enumerate(hands):
                if not hand_str:
                    # Skip empty hand segments
                    continue

                # Each hand has landmarks separated by '||'
                landmarks = hand_str.split('||')
                parsed_landmarks_in_hand = []

                for landmark_idx, landmark_coords_str in enumerate(landmarks):
                    coords = landmark_coords_str.split('|')
                    if len(coords) != 3:
                        # If coordinate string is malformed, skip
                        continue
                    try:
                        # Convert to floats and prepend the landmark index
                        parsed_landmarks_in_hand.append([
                            landmark_idx,
                            float(coords[0]),
                            float(coords[1]),
                            float(coords[2])
                        ])
                    except ValueError:
                        # Skip landmarks that cannot be parsed as floats
                        continue

                if parsed_landmarks_in_hand:
                    parsed_hands_in_frame.append(parsed_landmarks_in_hand)

            if parsed_hands_in_frame:
                # Build a dictionary representing a detected frame
                parsed_frame_dict = {
                    'result': 'DETECTION_SUCCESS',
                    'landmarks': parsed_hands_in_frame
                }
                parsed_frames.append(parsed_frame_dict)

        return parsed_frames

    except Exception as e:
        # Print error message including first 100 chars of input for debugging
        print(f"Error parsing landmarks string: {e}, Input: '{landmarks_str[:100]}...'")
        return []


def normalize_landmarks_enhanced(landmarks_data_frame):
    """
    Normalize landmark coordinates per hand by centering on the wrist (landmark 0)
    and scaling by the distance between index_mcp (5) and pinky_mcp (17) for
    scale invariance. If key points are missing, fall back to centering only.
    Returns a dict with normalized landmarks or None if no valid hands exist.
    """
    # Check for valid input dictionary and at least one hand of landmarks
    if not landmarks_data_frame or landmarks_data_frame.get('result') != 'DETECTION_SUCCESS' or not landmarks_data_frame.get('landmarks'):
        return None

    normalized_frame_hands = []

    for hand_landmarks_list in landmarks_data_frame['landmarks']:
        # Locate wrist (landmark 0) to use as the origin
        wrist = None
        for point in hand_landmarks_list:
            if point[0] == 0:  # Landmark ID 0 is the wrist
                wrist = point[1:4]  # Keep [x, y, z]
                break

        if wrist is None:
            # If wrist isn't found, skip this hand entirely
            continue

        # Find key MCP points needed for scale calculation
        index_mcp, middle_mcp, pinky_mcp = None, None, None
        for point in hand_landmarks_list:
            if point[0] == 5:
                index_mcp = point[1:4]
            elif point[0] == 9:
                middle_mcp = point[1:4]
            elif point[0] == 17:
                pinky_mcp = point[1:4]

        current_hand_normalized_landmarks = []

        # If any key MCP is missing, skip scale and only center at the wrist
        if index_mcp is None or pinky_mcp is None or middle_mcp is None:
            scale = 1.0  # No scaling
            for point in hand_landmarks_list:
                point_id = point[0]
                # Center coordinates around the wrist
                x = (point[1] - wrist[0]) / scale
                y = (point[2] - wrist[1]) / scale
                z = (point[3] - wrist[2]) / scale
                current_hand_normalized_landmarks.append([point_id, x, y, z])
            if current_hand_normalized_landmarks:
                normalized_frame_hands.append(current_hand_normalized_landmarks)
            continue  # Move on to next hand

        # Compute scale as the Euclidean distance between index_mcp and pinky_mcp
        scale = np.linalg.norm(np.array(index_mcp) - np.array(pinky_mcp))
        if scale < 1e-6:
            # Avoid divide-by-zero if points coincide
            scale = 1.0

        # Normalize each landmark by centering on the wrist and dividing by scale
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

    if not normalized_frame_hands:
        # If no hands were successfully normalized, return None
        return None

    return {
        'result': 'DETECTION_SUCCESS',
        'landmarks': normalized_frame_hands,
        'video_id': landmarks_data_frame.get('video_id'),  # Preserve video_id if present
        'label': landmarks_data_frame.get('label')         # Preserve label if present
    }


def calculate_single_hand_pairwise_features(hand_normalized_landmarks, pad_value=0.0):
    """
    For a single hand's normalized landmarks, compute:
      1) All pairwise distances among 6 key landmarks [0, 4, 8, 12, 16, 20].
      2) The cosine of the vector from the wrist (0) to each finger tip [4, 8, 12, 16, 20] 
         (i.e., x/r, y/r, z/r).
    Return a fixed-length list: (15 distances) + (5 tips * 3 coordinates) = 30 features.
    Missing points are padded with pad_value.
    """
    hand_features = []
    key_points_indices = [0, 4, 8, 12, 16, 20]
    num_key_points = len(key_points_indices)

    # There are C(6, 2) = 15 pairwise distance features
    expected_distances_count = num_key_points * (num_key_points - 1) // 2

    # There are 5 finger tips * 3 coordinates = 15 angle/cosine features
    expected_angles_count = 5 * 3
    total_expected_pairwise_features = expected_distances_count + expected_angles_count

    # Map landmark ID to its (x, y, z) coordinates
    key_coords = {point[0]: np.array(point[1:4]) for point in hand_normalized_landmarks if point[0] in key_points_indices}

    # Compute pairwise distances
    for i in range(num_key_points):
        for j in range(i + 1, num_key_points):
            p1_idx = key_points_indices[i]
            p2_idx = key_points_indices[j]
            if p1_idx in key_coords and p2_idx in key_coords:
                dist = np.linalg.norm(key_coords[p1_idx] - key_coords[p2_idx])
                hand_features.append(dist)
            else:
                # If either keypoint is missing, use pad_value
                hand_features.append(pad_value)

    # Pad distance list if fewer than expected (should not happen, but safeguards)
    while len(hand_features) < expected_distances_count:
        hand_features.append(pad_value)

    # Compute cosine angles from wrist to each fingertip
    if 0 in key_coords:
        wrist_coord = key_coords[0]
        finger_tip_indices = [4, 8, 12, 16, 20]
        for tip_idx in finger_tip_indices:
            if tip_idx in key_coords:
                finger_vec = key_coords[tip_idx] - wrist_coord
                magnitude = np.linalg.norm(finger_vec)
                if magnitude > 1e-9:
                    cos_angles = (finger_vec / magnitude).tolist()
                    hand_features.extend(cos_angles)
                else:
                    # If fingertip coincides with wrist, pad with zeros
                    hand_features.extend([pad_value] * 3)
            else:
                # Missing fingertip: pad with pad_value
                hand_features.extend([pad_value] * 3)
    else:
        # If wrist is missing entirely, pad all angle features
        hand_features.extend([pad_value] * expected_angles_count)

    # Ensure feature length is exactly total_expected_pairwise_features
    while len(hand_features) < total_expected_pairwise_features:
        hand_features.append(pad_value)

    return hand_features[:total_expected_pairwise_features]


def augment_landmarks(landmarks_df, augmentation_factor=3):
    """
    Take a DataFrame with columns ['video_id', 'label', 'landmarks'] and produce
    an augmented DataFrame that includes the original data plus `augmentation_factor`
    variations per video, where random noise is applied to each landmark coordinate.
    """
    augmented_data_list = []

    print("Augmenting landmark data...")
    for idx, row in landmarks_df.iterrows():
        video_id = row['video_id']
        label = row['label']
        landmarks_input = row['landmarks']

        # Parse landmarks string if necessary; else assume list of frames already
        if isinstance(landmarks_input, str):
            original_frames_list = parse_landmarks(landmarks_input)
        elif isinstance(landmarks_input, list):
            original_frames_list = landmarks_input
        else:
            print(f"Warning: Unknown landmark format for video_id {video_id} during augmentation. Skipping.")
            continue

        if not original_frames_list:
            # Skip if parsing failed or no landmarks
            continue

        # Add the original, unmodified data first
        augmented_data_list.append({
            'video_id': video_id,
            'label': label,
            'landmarks': original_frames_list
        })

        # Generate augmented versions
        for i in range(augmentation_factor):
            augmented_frames_for_one_video = []
            for frame_dict in original_frames_list:
                if frame_dict.get('result') != 'DETECTION_SUCCESS' or not frame_dict.get('landmarks'):
                    # If frame was a detection failure, keep as failure
                    augmented_frames_for_one_video.append({'result': 'DETECTION_FAILURE', 'landmarks': []})
                    continue

                new_frame_dict = {'result': 'DETECTION_SUCCESS', 'landmarks': []}

                # For each hand in the frame, apply Gaussian noise
                for hand_landmarks_list in frame_dict['landmarks']:
                    augmented_hand = []
                    # Noise scale grows slightly with each augmentation round
                    noise_scale = 0.02 * (np.random.rand() + 0.5) * (i + 1)
                    for point in hand_landmarks_list:
                        point_id = point[0]
                        # Add noise proportional to absolute coordinate (or fixed if near zero)
                        x = point[1] + np.random.normal(0, noise_scale * abs(point[1]) if abs(point[1]) > 1e-3 else noise_scale)
                        y = point[2] + np.random.normal(0, noise_scale * abs(point[2]) if abs(point[2]) > 1e-3 else noise_scale)
                        z = point[3] + np.random.normal(0, noise_scale * abs(point[3]) if abs(point[3]) > 1e-3 else noise_scale)
                        augmented_hand.append([point_id, x, y, z])
                    new_frame_dict['landmarks'].append(augmented_hand)

                augmented_frames_for_one_video.append(new_frame_dict)

            # Append the augmented video with a new video_id suffix
            if augmented_frames_for_one_video:
                augmented_data_list.append({
                    'video_id': f"{video_id}_aug_{i+1}",
                    'label': label,
                    'landmarks': augmented_frames_for_one_video
                })

    # Return a new DataFrame containing original + all augmentations
    return pd.DataFrame(augmented_data_list)


def pad_seq(sequence_of_frames, desired_sequence_length):

    # Build a dummy hand of 21 zeroed landmarks
    dummy_hand = [[i, 0.0, 0.0, 0.0] for i in range(21)]
    dummy_frame = {
        'result': 'DETECTION_SUCCESS',
        'landmarks': [dummy_hand]
    }
    # Append dummy frames until reaching desired length
    while len(sequence_of_frames) < desired_sequence_length:
        sequence_of_frames.append(dummy_frame)
    return sequence_of_frames


def prepare_sequences(data, segment_length):
    sequences = []
    labels = []

    for item in data:
        landmarks = item['landmarks']
        label = item['label']

        # Pad the full sequence if it's too short
        if len(landmarks) < segment_length:
            landmarks = pad_seq(landmarks, segment_length)

        sequences.append(landmarks)
        labels.append(label)

    return sequences, labels
