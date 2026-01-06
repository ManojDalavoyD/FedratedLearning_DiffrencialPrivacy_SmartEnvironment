from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib
from src.model import create_model 
# IMPORT THE MANUAL LOGIC
from src.suggestion import generate_offline_suggestions 

app = Flask(__name__)

# --- CONFIGURATION ---
# No API Key needed anymore!

# 1. Load the AI Model & Scaler
print("Loading Local TensorFlow Model...")
INPUT_DIM = 12 
model = create_model(INPUT_DIM)

try:
    # Load the H5 weights (TensorFlow format)
    model.load_weights('final_model.weights.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ AI Brain & Scaler Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("TIP: Did you run 'python main.py' first?")

FEATURES = ['Air Conditioning', 'Computer', 'Dishwasher', 'Fridge', 'Heater', 
            'Lights', 'Microwave', 'Oven', 'TV', 'Washing Machine', 'Temp', 'Size']

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Get User Input
        user_input = []
        for f in FEATURES:
            val = request.form.get(f)
            # Default to 0.0 if empty
            user_input.append(float(val) if val else 0.0)
            
        # 2. Local AI Prediction (Total Energy)
        # Scale the input because the model expects Z-Scores
        input_scaled = scaler.transform(np.array([user_input]))
        
        # Predict
        prediction_scaled = model.predict(input_scaled, verbose=0)[0][0]
        predicted_kwh = abs(prediction_scaled)
        
        # 3. MANUAL SUGGESTIONS (Math Logic)
        # We pass the scaled data (Z-Scores) to our offline logic file
        suggestions = generate_offline_suggestions(input_scaled[0], FEATURES)
        
        return render_template('result.html', 
                             prediction=f"{predicted_kwh:.2f}", 
                             suggestions=suggestions)
                             
    except Exception as e:
        return f"<h3>Error:</h3> <p>{str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True)