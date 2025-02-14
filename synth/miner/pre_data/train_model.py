import os
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler

# Ensure TensorFlow uses GPU with memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU Memory Growth Enabled")

def train_lstm_model(csv_file="data/btc_historical_data.csv", model_path="models/lstm_model.h5", scaler_path="models/scaler.pkl", epochs=50, batch_size=128):
    """Train or update LSTM model using updated BTC historical data."""
    
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(csv_file):
        print(f"❌ Data file {csv_file} not found.")
        return None, None
    
    # Load historical data
    df = pd.read_csv(csv_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_price"] = scaler.fit_transform(df["close"].values.reshape(-1, 1))

    # Prepare sequences
    seq_length = 30
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df["scaled_price"].values[i:i+seq_length])
        y.append(df["scaled_price"].values[i+seq_length])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Load existing model if available, otherwise create a new one
    if os.path.exists(model_path):
        print("🔄 Loading existing model for incremental training...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=Huber(delta=1.0), jit_compile=True)  # GPU-optimized
    else:
        print("🆕 Creating new LSTM model...")
        model = Sequential([
            LSTM(128, activation='tanh', return_sequences=True, input_shape=(30, 1)),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, activation='tanh', return_sequences=False),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Predicts price directly
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=Huber(delta=1.0), jit_compile=True)
    
    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Save updated model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model updated and saved at {model_path}")
    print(f"✅ Scaler saved at {scaler_path}")
    
    return model, scaler
