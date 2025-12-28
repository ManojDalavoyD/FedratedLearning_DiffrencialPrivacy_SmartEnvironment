import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_dim):
    # Lightweight Model suitable for IoT
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear') # Predict Total Energy
    ])
    model.compile(optimizer='adam', loss='mse')
    return model