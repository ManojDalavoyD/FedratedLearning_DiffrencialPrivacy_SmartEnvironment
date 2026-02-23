from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
from src.model import create_model 
from src.suggestion import generate_offline_suggestions, generate_live_suggestions, generate_gemini_suggestions
from src.client import VirtualClient
from dotenv import load_dotenv

load_dotenv() # Load .env at startup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide Info/Warning logs

app = Flask(__name__)

import requests # NEW: for Cloud Fetch

# --- CONFIGURATION ---
INPUT_DIM = 12 
FEATURES = ['Air Conditioning', 'Computer', 'Dishwasher', 'Fridge', 'Heater', 
            'Lights', 'Microwave', 'Oven', 'TV', 'Washing Machine', 'Temp', 'Size']

# Cloud Config
FIREBASE_HOST = os.getenv("FIREBASE_HOST")
FIREBASE_SECRET = os.getenv("FIREBASE_SECRET")

# 1. Load the AI Model & Scaler
print("Loading Local TensorFlow Model...")
model = create_model(INPUT_DIM)

# Init Global Client to prevent TF Retracing
# We initialize with dummy data 
live_client = VirtualClient(client_id="LiveHome", 
                          data=(np.zeros((1, INPUT_DIM)), np.zeros((1, 1))), 
                          input_dim=INPUT_DIM)

try:
    model.load_weights('final_model.weights.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ AI Brain & Scaler Loaded Successfully")
    
    # Get initial weights for FL simulation
    global_weights = model.get_weights()
    
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("TIP: Did you run 'python main.py' first?")
    global_weights = [] # Fallback

def read_latest_data():
    """Reads the latest data from Cloud (Priority) or Local CSV (Fallback)."""
    
    # 1. Try Cloud Fetch
    if FIREBASE_HOST and FIREBASE_SECRET:
        try:
            url = f"https://{FIREBASE_HOST}/SmartHome/Sensors.json?auth={FIREBASE_SECRET}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    # Convert to pandas Series-like dict for consistency
                    # Cloud Keys: Temperature, AC_Power, Oven_Status, Lights_Status, Total_Load, Last_Updated
                    return pd.Series({
                        'Timestamp': data.get('Last_Updated'),
                        'Temperature': float(data.get('Temperature', 25)),
                        'AC_Power': float(data.get('AC_Power', 0)),
                        'Oven_Status': int(data.get('Oven_Status', 0)),
                        'Light_Status': int(data.get('Lights_Status', 0)), # Note: Bridge uses Lights_Status, CSV uses Light_Status
                        'Total_Load': float(data.get('Total_Load', 0))
                    })
        except Exception as e:
            print(f"⚠️ Cloud Fetch Failed (Using Local): {e}")

    # 2. Fallback to CSV
    csv_path = 'Dataset/live_smart_home_data.csv'
    if not os.path.exists(csv_path):
        return None
        
    try:
        # Read last 5 lines to get the very latest
        df = pd.read_csv(csv_path) 
        latest = df.iloc[-1]
        return latest
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def map_csv_to_features(row):
    """Maps CSV/Cloud columns to the model's 12 features."""
    # CSV Columns: Timestamp,Temperature,AC_Power,Oven_Status,Light_Status,Total_Load
    
    data_map = {
        'Air Conditioning': row.get('AC_Power', 0),
        'Oven': row.get('Oven_Status', 0),
        'Lights': row.get('Light_Status', 0),
        'Temp': row.get('Temperature', 25.0)
    }
    
    # Fill defaults for missing columns
    feature_values = []
    for f in FEATURES:
        if f in data_map:
            feature_values.append(float(data_map[f]))
        elif f == 'Size':
            feature_values.append(4.0) # Default household size
        else:
            feature_values.append(0.0) # Other appliances assumed 0 if not live
            
    return feature_values

@app.route('/')
def home():
    # DIRECTLY SHOW LIVE DASHBOARD instead of manual input
    return render_template('live_dashboard.html')

@app.route('/live')
def live_dashboard():
    return render_template('live_dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        user_input = []
        for f in FEATURES:
            val = request.form.get(f)
            user_input.append(float(val) if val else 0.0)
            
        input_scaled = scaler.transform(np.array([user_input]))
        prediction_scaled = model.predict(input_scaled, verbose=0)[0][0]
        predicted_kwh = abs(prediction_scaled)
        
        # Calculate approximate total load from input for the prompt
        # User input is in 'user_input' list, corresponding to FEATURES
        # We need to reconstruct a raw_values list or dict
        est_total_load = sum(user_input) # Simple sum for context
        
        # Use Gemini for superior insights
        suggestions = generate_gemini_suggestions(user_input, FEATURES, total_load=est_total_load)
        
        return render_template('result.html', 
                             prediction=f"{predicted_kwh:.2f}", 
                             suggestions=suggestions)
    except Exception as e:
        return f"<h3>Error:</h3> <p>{str(e)}</p>"

# --- DATASET STATISTICS (TRAINING DATA) ---
DATASET_STATS = {}

def load_dataset_stats():
    """Parses the complex string-list CSV to get typical appliance wattages."""
    global DATASET_STATS
    path = 'Dataset/Smart_homes_data_along_with_time_stamp.csv'
    if not os.path.exists(path):
        print("⚠️ Warning: Training dataset not found.")
        return

    print("Loading & Parsing Training Dataset...")
    try:
        df = pd.read_csv(path)
        stats = {} # { 'Appliance': [list of values] }
        
        for index, row in df.iterrows():
            try:
                # Columns contain comma-separated strings: "Wash,TV,..." and "1.5,0.2,..."
                apps = row['Appliance Type'].split(',')
                consumptions = row['Energy Consumption (kWh)'].split(',')
                
                if len(apps) == len(consumptions):
                    for app, cons in zip(apps, consumptions):
                        app_name = app.strip()
                        try:
                            # Convert kWh to Watts (assuming 1 hour usage for typical rating)
                            watts = float(cons) * 1000
                            # Filter outliers (0 to 4500W)
                            if 0 < watts < 4500:
                                if app_name not in stats: stats[app_name] = []
                                stats[app_name].append(watts)
                        except: pass
            except: pass
            
        # Calculate Medians (More robust than Mean for "Typical" profile)
        final_stats = {}
        for app, values in stats.items():
            if values:
                import statistics
                median_val = int(statistics.median(values))
                # Sanity Caps for Display (some dataset values are aggregate/noisy)
                if app == 'Lights' and median_val > 500: median_val = 200
                if app == 'TV' and median_val > 500: median_val = 150
                final_stats[app] = median_val
            
        DATASET_STATS = final_stats
        print(f"✅ Dataset Stats Loaded: {DATASET_STATS}")
        
    except Exception as e:
        print(f"❌ Error parsing dataset: {e}")

# Load on startup
load_dataset_stats()

@app.route('/api/live-data')
def api_live_data():
    try:
        row = read_latest_data()
        if row is None:
            return jsonify({'error': 'No data found'}), 404
            
        # 1. Prepare Data
        raw_values = map_csv_to_features(row)
        
        # 2. AI Prediction (Federated Model)
        # The model predicts the *Next Load* (Watts), not Units.
        input_scaled = scaler.transform(np.array([raw_values]))
        prediction_scaled = model.predict(input_scaled, verbose=0)[0][0]
        # Inverse transform is ideal, but assuming scaler is minmax(0-1) logic or similar range
        # For display, we use it as a 'Model Confidence' or 'Forecast'
        ai_forecast_watts = abs(prediction_scaled) 
        
        # 3. Simulate Federated Learning & Differential Privacy
        # Use GLOBAL live_client to avoid TF Function Retracing
        live_client.update_data((input_scaled, np.array([prediction_scaled])))
        live_client.set_weights(global_weights)
        new_weights, loss = live_client.train(epochs=1)
        dp_msg = f"🛡️ Differential Privacy: Gaussian Noise added (Loss: {loss:.4f})"
        
        # 4. Inferred/Virtual Sensors (NILM Lite) -- MOVED UP
        total_l = row.get('Total_Load', 0) # Fixed: Define total_l before use
        ac_p = row.get('AC_Power', 0)
        oven_p = 2000 if int(row.get('Oven_Status', 0)) == 1 else 0
        light_p = 100 if int(row.get('Light_Status', 0)) == 1 else 0
        
        known_load = ac_p + oven_p + light_p
        unaccounted = max(0, total_l - known_load)
        
        # Simple Disaggregation Logic
        inferred = {
            'TV': 0, 'Fridge': 0, 'Washing Machine': 0, 'Microwave': 0
        }
        
        if unaccounted > 100:
            inferred['Fridge'] = 150 
            unaccounted -= 150
            
        if unaccounted > 100:
            inferred['TV'] = 120
            unaccounted -= 120
            
        # Lowered threshold to catch smaller washing loads
        if unaccounted > 200: 
            inferred['Washing Machine'] = unaccounted
            
        # UPDATE RAW VALUES with Inferred Data so Suggestions see it
        try:
            raw_values[FEATURES.index('Washing Machine')] = inferred['Washing Machine']
            raw_values[FEATURES.index('Fridge')] = inferred['Fridge']
            raw_values[FEATURES.index('TV')] = inferred['TV']
        except ValueError: pass # Safety if feature names change

        # 5. Generate Suggestions (Scientific Audit)
        # Now uses updated raw_values containing Inferred data
        suggestions = generate_gemini_suggestions(raw_values, FEATURES, total_load=total_l) 
        
        # 6. Smart Bill Projection (Component-based Duty Cycles)
        # Instead of multiplying Total Load * 24, we estimate based on typical usage hours.
        
        # Extract individual components (Watts)
        p_ac = raw_values[FEATURES.index('Air Conditioning')]
        p_oven = raw_values[FEATURES.index('Oven')] if raw_values[FEATURES.index('Oven')] > 10 else (2000 if raw_values[FEATURES.index('Oven')] else 0)
        p_light = raw_values[FEATURES.index('Lights')] if raw_values[FEATURES.index('Lights')] > 10 else (100 if raw_values[FEATURES.index('Lights')] else 0)
        p_fridge = inferred.get('Fridge', 0)
        p_tv = inferred.get('TV', 0)
        p_wash = inferred.get('Washing Machine', 0)
        
        # Calculate Daily kWh based on "Typical Indian Home" Usage Hours
        # ASSUMPTIONS (Smart Duty Cycle):
        # 1. Air Conditioner (AC): Runs approx 8 hours (Night + Afternoon)
        # 2. Oven/Microwave: Runs approx 30 mins (Reheating meals)
        # 3. Lights: Runs approx 6 hours (Evening to Night)
        # 4. Fridge: Runs 12 hours (Compressor duty cycle is generally 50%)
        # 5. TV: Runs 5 hours (Entertainment evening)
        # 6. Washing Machine: Runs 1.5 hours (One cycle per day)
        daily_kwh = (
            (p_ac * 8) + 
            (p_oven * 0.5) + 
            (p_light * 6) + 
            (p_fridge * 12) + 
            (p_tv * 5) +
            (p_wash * 1.5)
        ) / 1000.0
        
        # Check against minimum base load (Phantom power ~50W * 24h)
        base_load_kwh = (50 * 24) / 1000.0
        daily_kwh = max(daily_kwh, base_load_kwh)
        
        daily_cost = daily_kwh * 7.50
        
        # Return JSON
        def to_native(obj):
            if isinstance(obj, (np.integer, np.int64)): return int(obj)
            if isinstance(obj, (np.floating, np.float64)): return float(obj)
            return obj

        raw_dict = {k: to_native(v) for k, v in row.to_dict().items()}
        
        response_data = {
            'total_load': to_native(total_l),
            'predicted_units': f"{daily_kwh:.2f}",
            'predicted_cost': f"₹ {daily_cost:.2f}", # Projected Daily Cost
            'ai_forecast': f"{ai_forecast_watts:.2f}", # Raw AI output
            'suggestions': suggestions,
            'dp_status': dp_msg,
            'raw': raw_dict,
            'inferred': {k: to_native(v) for k, v in inferred.items()},
            'dataset_stats': DATASET_STATS
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
