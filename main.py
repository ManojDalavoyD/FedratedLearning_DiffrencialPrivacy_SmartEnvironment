import numpy as np
from src.utils import parse_and_process_data, partition_data
from src.model import create_model
from src.client import VirtualClient
from src.server import fed_avg
from src.suggestion import analyze_usage

# --- CONFIG ---
DATA_PATH = "D:/final_year_Project_M/Dataset/Smart_homes_data_along_with_time_stamp.csv"
NUM_CLIENTS = 5
ROUNDS = 5
EPOCHS = 3

if __name__ == "__main__":
    # 1. LOAD & PARSE
    X_scaled, y, scaler, features = parse_and_process_data(DATA_PATH)
    
    if X_scaled is None:
        exit()
        
    print(f"Data Prepared. Features detected: {features}")
    
    # 2. SETUP FL
    clients_data = partition_data(X_scaled, y, NUM_CLIENTS)
    input_dim = X_scaled.shape[1]
    
    global_model = create_model(input_dim)
    global_weights = global_model.get_weights()
    
    clients = [VirtualClient(i, data, input_dim) for i, data in enumerate(clients_data)]
    
    # 3. TRAINING
    print(f"\n--- Starting Federated Training ({ROUNDS} Rounds) ---")
    for r in range(ROUNDS):
        local_weights = []
        losses = []
        for client in clients:
            client.set_weights(global_weights)
            w, loss = client.train(EPOCHS)
            local_weights.append(w)
            losses.append(loss)
            
        global_weights = fed_avg(global_weights, local_weights)
        print(f"Round {r+1} | Avg Loss: {np.mean(losses):.4f}")
        
    print("--- Training Complete ---")
    
    # 4. REAL WORLD SUGGESTION DEMO
    # Pick a random row (e.g., Home 1 at 8:00 AM)
    test_idx = 15 
    sample_row = X_scaled[test_idx]
    
    # Predict Total Energy
    global_model.set_weights(global_weights)
    pred = global_model.predict(sample_row.reshape(1, -1), verbose=0)[0][0]
    
    print(f"\n[Snapshot Analysis]")
    print(f"Time Context: Outside Temp is {scaler.inverse_transform([sample_row])[0][-2]:.1f}°C")
    print(f"Predicted Total Load: {pred:.2f} kWh")
    
    # Run the Suggestion Engine
    analyze_usage(sample_row, features)

import joblib # Install this: pip install joblib

# ... (After training loop in main.py) ...

print("\n--- Saving Model for Website ---")
# 1. Save the Model Weights
global_model.set_weights(global_weights)
global_model.save_weights('final_model.weights.h5')

# 2. Save the Scaler (Crucial for Z-Score/Suggestions)
# We need to save the mean and variance the model learned
joblib.dump(scaler, 'scaler.pkl')

print("✅ Saved 'final_model.weights.h5' and 'scaler.pkl'")