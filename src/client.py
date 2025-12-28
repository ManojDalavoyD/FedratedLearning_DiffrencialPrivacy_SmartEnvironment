import numpy as np
from src.model import create_model

class VirtualClient:
    def __init__(self, client_id, data, input_dim):
        self.client_id = client_id
        self.X, self.y = data
        self.model = create_model(input_dim)
        
    def set_weights(self, global_weights):
        self.model.set_weights(global_weights)
        
    def train(self, epochs=5):
        # Train locally
        history = self.model.fit(self.X, self.y, epochs=epochs, verbose=0, batch_size=32)
        return self.model.get_weights(), history.history['loss'][-1]