def generate_offline_suggestions(scaled_row, features):
    """
    Analyzes the normalized data (Z-Scores) to find inefficiencies.
    Returns a list of dictionaries: [{'msg': '...', 'action': '...'}]
    """
    advice_list = []
    
    # We don't give advice on context features, only appliances
    ignore_list = ['Temp', 'Size', 'Hour', 'HomeID']
    
    for i, score in enumerate(scaled_row):
        feature = features[i]
        
        # Skip non-appliance features
        if any(x in feature for x in ignore_list):
            continue
            
        # LOGIC:
        # A Z-Score of 0.0 is the average.
        # A Z-Score > 1.0 means usage is High (Top 16% of users).
        # A Z-Score > 2.0 means usage is Very High (Top 2% of users).
        
        if score > 1.0:
            # Calculate percentage above average for display
            pct = int(score * 100)
            severity = "High" if score < 2.0 else "Critical"
            
            msg = f"⚠️ {feature}: Usage is {severity} ({pct}% above avg)"
            
            # --- Rule-Based Advice Database ---
            if "Air" in feature or "AC" in feature:
                action = "Set thermostat to 24°C. Every degree lower increases bill by 6%."
            elif "Heater" in feature:
                action = "Check window insulation for drafts. Lower thermostat by 1-2°C."
            elif "Fridge" in feature:
                action = "Check door seals. Vacuum coils behind the fridge for efficiency."
            elif "Washing" in feature:
                action = "Run only full loads. Use cold water (30°C) to save 90% energy."
            elif "Dishwasher" in feature:
                action = "Skip the 'Heated Dry' cycle. Run only when full."
            elif "Lights" in feature:
                action = "Switch to LED bulbs. Install motion sensors in hallways."
            elif "Computer" in feature or "TV" in feature:
                action = "Enable 'Energy Saver' mode. Unplug when not in use (Vampire load)."
            elif "Oven" in feature:
                action = "Don't open the door while cooking. Use microwave for small meals."
            else:
                action = "Consider checking this device for maintenance or replacement."
            
            advice_list.append({'msg': msg, 'action': action})
            
    # If no high usage found
    if not advice_list:
        advice_list.append({
            'msg': "✅ Excellent Efficiency!", 
            'action': "Your usage matches the most efficient homes in the network."
        })
        
    return advice_list