import numpy as np
import joblib  # <--- Essential for saving the scaler
from src.utils import parse_and_process_data, partition_data
from src.model import create_model
from src.client import VirtualClient
from src.server import fed_avg
# REMOVED: from src.suggestion import analyze_usage (Not needed for training)

# --- CONFIG ---
# Ensure this path points to your actual CSV file
DATA_PATH = "D:/final_year_Project_M/Dataset/Smart_homes_data_along_with_time_stamp.csv"
NUM_CLIENTS = 5
ROUNDS = 5
EPOCHS = 3

if __name__ == "__main__":
    # 1. LOAD & PARSE
    print("Loading Data...")
    try:
        X_scaled, y, scaler, features = parse_and_process_data(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("Please check the path or move your CSV to the 'data' folder.")
        exit()
    
    if X_scaled is None:
        exit()
        
    print(f"✅ Data Prepared. Features detected: {len(features)}")
    
    # 2. SETUP FL (Federated Learning)
    clients_data = partition_data(X_scaled, y, NUM_CLIENTS)
    input_dim = X_scaled.shape[1]
    
    # Initialize Global Model
    global_model = create_model(input_dim)
    global_weights = global_model.get_weights()
    
    # Create Virtual Clients (Virtual Homes)
    clients = [VirtualClient(i, data, input_dim) for i, data in enumerate(clients_data)]
    
    # 3. TRAINING LOOP
    print(f"\n--- Starting Federated Training ({ROUNDS} Rounds) ---")
    for r in range(ROUNDS):
        local_weights = []
        losses = []
        
        for client in clients:
            # Server sends global weights to client
            client.set_weights(global_weights)
            
            # Client trains on local data (and applies DP noise if enabled in client.py)
            w, loss = client.train(EPOCHS)
            
            local_weights.append(w)
            losses.append(loss)
            
        # Aggregation: Server averages the weights
        global_weights = fed_avg(global_weights, local_weights)
        
        # Log Progress
        print(f"Round {r+1} | Avg Loss: {np.mean(losses):.4f}")
        
    print("--- Training Complete ---")
    
    # 4. SAVE MODEL & SCALER (Crucial for the Website)
    print("\n--- Saving System Files ---")
    
    # Save Weights (The Brain)
    global_model.set_weights(global_weights)
    global_model.save_weights('final_model.weights.h5')
    
    # Save Scaler (The Translator for inputs)
    joblib.dump(scaler, 'scaler.pkl')
    
    print("✅ Success! Files saved:")
    print("   - final_model.weights.h5")
    print("   - scaler.pkl")
    print("You can now run 'python app.py'")