import os
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib
import google.generativeai as genai  # <--- NEW IMPORT
from src.model import create_model 

app = Flask(__name__)

# --- CONFIGURATION ---
# 1. Paste your Gemini API Key here
GEMINI_API_KEY =
genai.configure(api_key=GEMINI_API_KEY)

# 2. Load the AI Model & Scaler
INPUT_DIM = 12 
model = create_model(INPUT_DIM)

try:
    model.load_weights('final_model.weights.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ AI Brain & Scaler Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading files: {e}")

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
        user_input_dict = {} # Keep a dictionary for the prompt
        
        for f in FEATURES:
            val = request.form.get(f)
            float_val = float(val) if val else 0.0
            user_input.append(float_val)
            user_input_dict[f] = float_val
            
        # 2. Local AI Prediction (Total Energy)
        input_scaled = scaler.transform(np.array([user_input]))
        prediction_scaled = model.predict(input_scaled, verbose=0)[0][0]
        predicted_kwh = abs(prediction_scaled)
        
        # 3. ASK GEMINI FOR SUGGESTIONS
        # We find high usage items first to make the prompt cheaper/faster
        high_usage_items = []
        ignore = ['Temp', 'Size']
        
        for i, score in enumerate(input_scaled[0]):
            feature = FEATURES[i]
            if feature not in ignore and score > 1.0: # Z-Score > 1 means high
                high_usage_items.append(f"{feature} (Usage Level: High)")
        
        gemini_advice = get_gemini_suggestion(high_usage_items, predicted_kwh, user_input_dict['Temp'])
        
        return render_template('result.html', 
                             prediction=f"{predicted_kwh:.2f}", 
                             suggestions=gemini_advice) # Pass the raw text/list
                             
    except Exception as e:
        return f"Error: {str(e)}"

def get_gemini_suggestion(high_usage_list, total_kwh, temp):
    """
    Sends data to Gemini and gets a natural language response.
    """
    if not high_usage_list:
        return [{"msg": "Great Job!", "action": "Your energy usage is perfectly normal."}]

    # Create the Prompt
    prompt = f"""
    Act as a friendly Home Energy Auditor.
    
    Context:
    - The home is consuming {total_kwh:.2f} kWh (which is predicted to be high).
    - Outdoor Temperature: {temp}°C.
    - The following appliances have abnormally high usage patterns compared to neighbors: {', '.join(high_usage_list)}.
    
    Task:
    Provide 3 short, specific, and actionable tips to reduce energy for these specific appliances. 
    Format the output as a clean list. Do not use markdown (bold/italic). 
    Keep it polite and encouraging.
    """
    
    try:
        # Call Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Convert text response to the list format your HTML expects
        # We manually structure it to look like your old list
        return [{"msg": "AI Architect Suggestion", "action": response.text}]
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        return [{"msg": "Connection Error", "action": "Could not reach Gemini API. Please check internet."}]

if __name__ == '__main__':
    app.run(debug=True)