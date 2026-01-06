import numpy as np
from src.utils import parse_and_process_data
from src.model import create_model

# CONFIG
DATA_PATH = "D:/final_year_Project_M/Dataset/Smart_homes_data_along_with_time_stamp.csv"
EPOCHS = 15  # Equivalent to 5 Rounds * 3 Epochs

if __name__ == "__main__":
    print("--- 🚀 Running Centralized Benchmark (No Privacy) ---")
    
    # 1. Load ALL Data
    X_scaled, y, _, _ = parse_and_process_data(DATA_PATH)
    input_dim = X_scaled.shape[1]
    
    # 2. Create Model
    model = create_model(input_dim)
    
    # 3. Train on EVERYTHING at once
    history = model.fit(X_scaled, y, epochs=EPOCHS, batch_size=32, verbose=1)
    
    final_loss = history.history['loss'][-1]
    print(f"\n✅ Centralized Baseline Loss: {final_loss:.4f}")
    print("---------------------------------------------------")