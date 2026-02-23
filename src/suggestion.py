import json

def generate_gemini_suggestions(raw_data, features, total_load=0):
    """
    Generates Smart Home suggestions using a ROBUST RULE-BASED EXPERT SYSTEM.
    (Previously named Gemini, kept for compatibility, but now purely algorithmic).
    
    This function analyzes raw appliance usage against real-world thresholds 
    to provide dynamic, meaningful advice without calling external APIs.
    """
    suggestions = []
    
    # Map features to a dictionary for easy, safe access
    # features list: ['Air Conditioning', 'Computer', 'Dishwasher', 'Fridge', 'Heater', 'Lights', 'Microwave', 'Oven', 'TV', 'Washing Machine', 'Temp', 'Size']
    data = dict(zip(features, raw_data))
    
    # --- 1. BILL & LOAD ANALYSIS (The "Main Gauge") ---
    # Cost Estimation: Assuming Avg Rate ₹7.50 / kWh
    # Hourly Cost = (Watts / 1000) * 7.50
    hourly_cost = (total_load / 1000.0) * 7.50
    
    if total_load > 4000:
         suggestions.append({
            'appliance': '⚠️ HIGH SYSTEM LOAD',
            'action': f"Total load is {total_load}W. Hourly cost is approx ₹{hourly_cost:.2f}. Check for heavy appliances running simultaneously.",
            'type': 'danger'
        })
    elif total_load > 2000:
         suggestions.append({
            'appliance': 'System Load',
            'action': f"Moderate usage ({total_load}W). Costing approx ₹{hourly_cost:.2f}/hour.",
            'type': 'info'
        })
         
    # --- 2. APPLIANCE-SPECIFIC EXPERT RULES ---
    
    # --- AIR CONDITIONING (The biggest consumer) ---
    ac_power = data.get('Air Conditioning', 0)
    outside_temp = data.get('Temp', 25) # Default 25C if sensor missing
    
    if ac_power > 1500:
        # High AC Usage
        if outside_temp < 24:
            suggestions.append({
                'appliance': 'Air Conditioning',
                'action': f"Outside temp is cool ({outside_temp}°C). Open windows instead of using high-power AC ({ac_power}W).",
                'type': 'warning'
            })
        else:
            suggestions.append({
                'appliance': 'Air Conditioning',
                'action': f"AC is running high ({ac_power}W). Set Temp to 24°C to save ~6% electricity.",
                'type': 'warning'
            })
    elif ac_power > 500:
        # Moderate AC
        suggestions.append({
            'appliance': 'Air Conditioning',
            'action': "AC is running efficiently. Ensure doors/windows are closed.",
            'type': 'success'
        })

    # --- HEATER ---
    heater_power = data.get('Heater', 0)
    if heater_power > 1500:
        if outside_temp > 20:
             suggestions.append({
                'appliance': 'Heater',
                'action': f"It's warm outside ({outside_temp}°C). Do you really need the heater on?",
                'type': 'danger'
            })
        else:
             suggestions.append({
                'appliance': 'Heater',
                'action': "Heater is consuming high power. Check insulation for drafts.",
                'type': 'warning'
            })

    # --- KITCHEN (Oven / Microwave) ---
    oven_status = data.get('Oven', 0)
    # Note: Oven is often binary (1/0) or Watts depending on data source.
    # If using 'Oven_Status' from cloud passed as feature, likely 0 or 1.
    # But map_csv_to_features might convert it. Let's handle both.
    oven_val = float(oven_status)
    if oven_val > 100: # It's Watts
        suggestions.append({
            'appliance': 'Oven',
            'action': "Oven is heating. Avoid opening the door often to retain heat.",
            'type': 'warning'
        })
    elif oven_val == 1: # It's Status
        suggestions.append({
            'appliance': 'Oven',
            'action': "Oven is ON. Ensure it's not empty.",
            'type': 'warning'
        })
        
    # --- LIGHTING ---
    # Smart threshold: Don't nag for a single 10W bulb
    lights_val = data.get('Lights', 0)
    if lights_val > 200:
        suggestions.append({
            'appliance': 'Lighting',
            'action': f"High lighting load ({lights_val}W). Switch to LEDs or turn off unused rooms.",
            'type': 'warning'
        })
    elif lights_val >= 1: # Just ON status
        # If we only know it's ON (value 1) or low watts, maybe just a gentle tip?
        # Actually, user said "do not give warning for small use".
        # So if it's small (e.g. < 50W), we say nothing or 'Good'.
        if lights_val < 50 and lights_val > 1:
             pass # Ignore small loads
        else:
             suggestions.append({
                'appliance': 'Lighting',
                'action': "Lights are ON. Turn off if room is empty.",
                'type': 'info'
            })

    # --- PHANTOM LOAD DETECTION ---
    # Estimate sum of major known appliances
    # Note: This is rough because we might not have all live feeds for every appliance
    known_consumers = ac_power + heater_power + (2000 if oven_val >= 1 else 0) + lights_val
    unaccounted = total_load - known_consumers
    
    if unaccounted > 300:
        suggestions.append({
            'appliance': 'Unknown Devices',
            'action': f"Unaccounted usage of ~{int(unaccounted)}W detected. Check TV, Fridge, or Vampire Power.",
            'type': 'info'
        })
        
    # --- DEFAULT "GOOD JOB" ---
    if len(suggestions) == 0:
        suggestions.append({
            'appliance': 'Home Energy',
            'action': "✅ Usage is optimized. No major wastage detected.",
            'type': 'success'
        })
        
    return suggestions

# Keep these for compatibility if imported elsewhere
def generate_offline_suggestions(scaled_row, features):
    return generate_gemini_suggestions(scaled_row, features, 0)
    
def generate_live_suggestions(raw_data, scaled_row, features):
    return generate_gemini_suggestions(raw_data, features, 0)
