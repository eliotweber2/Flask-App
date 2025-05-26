from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, Lambda, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
)

from tensorflow import squeeze, expand_dims, reduce_sum
from tensorflow.nn import softmax
from memory_profiler import profile

def create_attention_model(n_classes, sequence_length, n_features):
    """Create an enhanced model architecture with attention mechanism for higher accuracy."""
    
    # Input layer
    input_layer = Input(shape=(sequence_length, n_features))
    
    # First LSTM block
    x = Bidirectional(LSTM(256, return_sequences=True))(input_layer)
    x = Dropout(0.3)(x)
    
    # Second LSTM block
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Attention mechanism to focus on important frames
    attention = Dense(1, activation='tanh')(x)
    attention_weights = Lambda(lambda x: softmax(squeeze(x, axis=-1), axis=1))(attention)
    
    # Apply attention
    context =  Lambda(
        lambda inputs: reduce_sum(inputs[0] * expand_dims(inputs[1], axis=-1), axis=1)
        )([x, attention_weights])
    
    # Dense layers
    x = Dense(128, activation='relu')(context)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.load_weights('model/sign_language_model_Attention_LSTM.weights.h5')
    
    return model
# Create CNN-LSTM model for ensemble
def create_cnn_lstm_model(n_classes, sequence_length, n_features):
    """Create a CNN-LSTM model for the ensemble."""
    input_layer = Input(shape=(sequence_length, n_features))
    # CNN layers
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.load_weights('model/sign_language_model_CNN_LSTM.weights.h5')
    
    return model
# Create Transformer model for ensemble
def create_transformer_model(n_classes, sequence_length, n_features):
    """Create a Transformer-based model for the ensemble."""
    input_layer = Input(shape=(sequence_length, n_features))
    
    # Normalization and positional embeddings
    x = LayerNormalization(epsilon=1e-6)(input_layer)
    
    # Transformer blocks
    for _ in range(2):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward
        ffn = Dense(512, activation='relu')(x)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(n_features)(ffn)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.load_weights('model/sign_language_model_Transformer.weights.h5')

    return model