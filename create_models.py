from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, Lambda, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention, TimeDistributed
)

from tensorflow import squeeze, expand_dims, reduce_sum
from tensorflow.nn import softmax
#from memory_profiler import profile

def create_attention_lstm_per_frame_model(n_classes, sequence_length, n_features):
    """LSTM-based model for per-frame predictions."""
    input_layer = Input(shape=(sequence_length, n_features), name="input_lstm")
    x = Bidirectional(LSTM(256, return_sequences=True))(input_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.3)(x)
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameLSTM")
    return model

# Create CNN-LSTM model for ensemble

def create_cnn_lstm_per_frame_model(n_classes, sequence_length, n_features):
    """CNN-LSTM model for per-frame predictions.
    Note: Pooling reduces sequence length. Output will be shorter than input.
    """
    input_layer = Input(shape=(sequence_length, n_features), name="input_cnn_lstm")
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2, padding='same')(x) # seq_len /= 2
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x) # seq_len /= 4 (total reduction)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.3)(x)
    
    # Output predictions for the pooled sequence length
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'))(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameCNN_LSTM")
    return model

def create_transformer_per_frame_model(n_classes, sequence_length, n_features):
    """Transformer-based model for per-frame predictions."""
    input_layer = Input(shape=(sequence_length, n_features), name="input_transformer")
    x = LayerNormalization(epsilon=1e-6)(input_layer)
    for _ in range(2): # Number of transformer blocks
        key_dim = max(1, n_features // 8) # Ensure key_dim is positive
        attention_output = MultiHeadAttention(num_heads=8, key_dim=key_dim, dropout=0.1)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        ffn_output = Dense(512, activation='relu')(x)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(n_features)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.1)(x)
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameTransformer")
    return model