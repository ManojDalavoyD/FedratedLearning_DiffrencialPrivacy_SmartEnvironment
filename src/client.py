import numpy as np
import tensorflow as tf
from src.model import create_model

class VirtualClient:
    def __init__(self, client_id, data, input_dim):
        self.client_id = client_id
        self.X, self.y = data
        self.model = create_model(input_dim)
        
    def set_weights(self, global_weights):
        self.model.set_weights(global_weights)
        
    def get_weights(self):
        return self.model.get_weights()
        
    def apply_dp_noise(self, weights, noise_scale=0.005):
        """
        Applies Differential Privacy by adding Gaussian Noise.
        Formula: W_new = W_old + Noise(0, sigma)
        """
        noisy_weights = []
        for w in weights:
            # 1. Create Random Noise Matrix (Same shape as weight)
            # scale=0.005 is the 'Sigma' (Privacy Budget). Higher = More Privacy but Less Accuracy.
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=w.shape)
            
            # 2. Add Noise to Weights
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
            
        print(f"   🛡️ [Client {self.client_id}] Differential Privacy Noise Added (Sigma={noise_scale})")
        return noisy_weights

    def train(self, epochs=5):
        # 1. Normal Training (Learning from chunks)
        history = self.model.fit(self.X, self.y, epochs=epochs, verbose=0, batch_size=32)
        
        # 2. Get the Clean Weights
        clean_weights = self.model.get_weights()
        
        # --- SCENARIO 2: NO PRIVACY SETUP ---
        # We define 'privacy_weights' as just the clean weights (No Noise)
        privacy_weights = clean_weights 
        
        # NOTE: To switch back to Scenario 3 (With Privacy), use this line instead:
        privacy_weights = self.apply_dp_noise(clean_weights)
        
        return privacy_weights, history.history['loss'][-1]