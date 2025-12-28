import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def parse_and_process_data(filepath):
    print(f"Loading and parsing {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Error: File not found.")
        return None, None, None

    parsed_rows = []
    
    # 1. PARSE THE STRING LISTS
    # Your CSV has "1.2, 3.4, 5.6" in single cells. We need to split them.
    for idx, row in df.iterrows():
        home_id = row['Home ID']
        
        # Split strings by comma
        times = str(row['Time']).split(',')
        appliances = str(row['Appliance Type']).split(',')
        energies = str(row['Energy Consumption (kWh)']).split(',')
        temps = str(row['Outdoor Temperature (°C)']).split(',')
        sizes = str(row['Household Size']).split(',')
        
        # Zip them together to create individual rows
        for t, app, en, temp, size in zip(times, appliances, energies, temps, sizes):
            try:
                en_val = float(en.strip()) if en.strip() else 0.0
                temp_val = float(temp.strip()) if temp.strip() else 0.0
                size_val = float(size.strip()) if size.strip() else 0.0
                hour = int(t.strip().split(':')[0]) # Extract Hour from HH:MM
                
                parsed_rows.append({
                    'HomeID': home_id,
                    'Hour': hour,
                    'Appliance': app.strip(),
                    'Energy': en_val,
                    'Temp': temp_val,
                    'Size': size_val
                })
            except:
                continue 

    df_parsed = pd.DataFrame(parsed_rows)
    
    # 2. PIVOT (One Row per Home+Hour)
    # We want features as columns: [AC, Fridge, TV...]
    df_pivot = df_parsed.pivot_table(index=['HomeID', 'Hour'], 
                                     columns='Appliance', 
                                     values='Energy', 
                                     aggfunc='sum', 
                                     fill_value=0)
    
    # Add Context Features (Temp, Size)
    context = df_parsed.groupby(['HomeID', 'Hour'])[['Temp', 'Size']].mean()
    final_df = df_pivot.join(context)
    
    feature_names = final_df.columns.tolist()
    
    # 3. NORMALIZE
    scaler = StandardScaler()
    X_raw = final_df.values
    # Target = Total Energy (Sum of all columns)
    y_raw = X_raw.sum(axis=1) 
    
    X_scaled = scaler.fit_transform(X_raw)
    y_scaled = y_raw # We can keep target in Watts/kWh for readability, or scale it too.
    
    return X_scaled, y_scaled, scaler, feature_names

def partition_data(X, y, num_clients):
    # Split data into chunks for each client
    partition_size = len(X) // num_clients
    clients_data = []
    
    for i in range(num_clients):
        start = i * partition_size
        end = (i + 1) * partition_size
        clients_data.append((X[start:end], y[start:end]))
        
    return clients_data