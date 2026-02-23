import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import parse_and_process_data, partition_data
from src.model import create_model
from src.server import fed_avg
from src.client import VirtualClient
import copy

# --- CONFIGURATION ---
DATA_PATH = "D:/final_year_Project_M/Dataset/Smart_homes_data_along_with_time_stamp.csv" # CHECK PATH
NUM_CLIENTS = 5
ROUNDS = 5
EPOCHS = 2

# --- HELPER: CUSTOM CLIENT TO CONTROL NOISE ---
class ControllableClient(VirtualClient):
    """Extends your client to allow turning noise ON/OFF dynamically."""
    def train_custom(self, epochs, noise_scale):
        # 1. Train
        self.model.fit(self.X, self.y, epochs=epochs, verbose=0, batch_size=32)
        
        # 2. Get Weights
        weights = self.model.get_weights()
        
        # 3. Apply Noise (controlled)
        if noise_scale > 0:
            noisy_weights = self.apply_dp_noise(weights, noise_scale=noise_scale)
            return noisy_weights, self.model.evaluate(self.X, self.y, verbose=0)
        else:
            # Return CLEAN weights
            return weights, self.model.evaluate(self.X, self.y, verbose=0)

def run_simulation(noise_scale, clients_data, input_dim):
    """Runs a complete FL session with specific noise settings."""
    print(f"\n--- Running Simulation with Noise Scale = {noise_scale} ---")
    
    # Init Global Model
    global_model = create_model(input_dim)
    global_weights = global_model.get_weights()
    
    # Init Clients
    clients = [ControllableClient(i, data, input_dim) for i, data in enumerate(clients_data)]
    
    loss_history = []
    
    for r in range(ROUNDS):
        local_weights = []
        local_losses = []
        
        for client in clients:
            client.set_weights(global_weights)
            w, loss = client.train_custom(EPOCHS, noise_scale)
            local_weights.append(w)
            local_losses.append(loss)
            
        global_weights = fed_avg(global_weights, local_weights)
        avg_loss = np.mean(local_losses)
        loss_history.append(avg_loss)
        print(f"Round {r+1} Loss: {avg_loss:.4f}")
        
    return loss_history, global_weights

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    try:
        X_scaled, y, _, _ = parse_and_process_data(DATA_PATH)
        if X_scaled is None: raise Exception("Data not found")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    clients_data = partition_data(X_scaled, y, NUM_CLIENTS)
    input_dim = X_scaled.shape[1]

    # 2. Run Scenario A: CLEAN (No Privacy)
    loss_clean, weights_clean = run_simulation(0.0, clients_data, input_dim)

    # 3. Run Scenario B: PRIVATE (Your DP Algorithm)
    loss_private, weights_private = run_simulation(0.01, clients_data, input_dim) # Increased noise slightly for visibility

    # --- VISUALIZATION 1: PERFORMANCE METRIC ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, ROUNDS+1), loss_clean, 'g-o', label='Standard FL (No Noise)')
    plt.plot(range(1, ROUNDS+1), loss_private, 'r--s', label='Differential Privacy FL (With Noise)')
    plt.title('Impact of Privacy on Model Convergence')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Mean Squared Error (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig('metric_comparison_loss.png')
    print("\n✅ Saved 'metric_comparison_loss.png'")

    # --- VISUALIZATION 2: THE MATRIX OF DIFFERENCE ---
    # We take the weights of the first layer (Input -> Dense 64)
    # This matrix is usually shape (12, 64)
    
    w_clean_layer1 = weights_clean[0]
    w_private_layer1 = weights_private[0]
    
    # Calculate the Difference (The Noise)
    difference_matrix = w_private_layer1 - w_clean_layer1

    plt.figure(figsize=(14, 6))
    
    # Plot 1: Clean Weights
    plt.subplot(1, 3, 1)
    sns.heatmap(w_clean_layer1[:12, :12], cmap="viridis", cbar=False)
    plt.title("Standard Weights (Sample)")
    
    # Plot 2: Private Weights
    plt.subplot(1, 3, 2)
    sns.heatmap(w_private_layer1[:12, :12], cmap="viridis", cbar=False)
    plt.title("Private Weights (Sample)")
    
    # Plot 3: The Difference (Noise)
    plt.subplot(1, 3, 3)
    sns.heatmap(difference_matrix[:12, :12], cmap="coolwarm", center=0)
    plt.title("Difference Matrix (The Privacy Noise)")
    
    plt.tight_layout()
    plt.savefig('matrix_of_difference.png')
    print("✅ Saved 'matrix_of_difference.png'")
    
    plt.show()
