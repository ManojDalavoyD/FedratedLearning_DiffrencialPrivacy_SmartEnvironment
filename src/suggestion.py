def generate_offline_suggestions(scaled_row, features):
    """
    Generates suggestions using a Data-Driven approach (Dictionary Lookup).
    Logic: Advice = Map(Appliance, State)
    State is determined by Z-Score (Math).
    """
    advice_list = []
    
    # Context features to ignore
    ignore_set = {'Temp', 'Size', 'Hour', 'HomeID'}
    
    # --- 1. THE KNOWLEDGE BASE (Mathematical Mapping) ---
    # Structure: { 'Keyword': (High_Usage_Advice, Low_Usage_Advice) }
    knowledge_base = {
        'Air':      ("⚠️ Critical: Set thermostat to 24°C. Clean filters.", 
                     "✅ Good: AC usage is optimized."),
        'AC':       ("⚠️ Critical: Set thermostat to 24°C. Clean filters.", 
                     "✅ Good: AC usage is optimized."),
        'Heater':   ("⚠️ Warning: Check window insulation for drafts.", 
                     "✅ Efficient: Heating load is normal."),
        'Fridge':   ("⚠️ Maintenance: Check door seals and vacuum coils.", 
                     "✅ Good: Fridge is running efficiently."),
        'Washing':  ("⚠️ Tip: Run only full loads with cold water (30°C).", 
                     "✅ Great: You are using the washer efficiently."),
        'Lights':   ("⚠️ Tip: Switch to LED bulbs or use motion sensors.", 
                     "✅ Good: Lighting consumption is minimal."),
        'Dish':     ("⚠️ Tip: Skip the 'Heated Dry' cycle.", 
                     "✅ Efficient: Dishwasher usage is low."),
        'Comp':     ("⚠️ Alert: Unplug at night to stop 'Vampire Power'.", 
                     "✅ Good: Electronics usage is normal."),
        'TV':       ("⚠️ Alert: Unplug at night to stop 'Vampire Power'.", 
                     "✅ Good: Electronics usage is normal."),
        'Oven':     ("⚠️ Tip: Avoid opening door while cooking.", 
                     "✅ Good: Cooking energy is efficient."),
        'Micro':    ("⚠️ Tip: Use microwave instead of oven for small meals.", 
                     "✅ Good: Microwave usage is efficient.")
    }

    # Default fallback advice
    default_advice = ("⚠️ Warning: Usage is higher than neighbors.", 
                      "✅ Good: Usage is optimal.")

    for i, score in enumerate(scaled_row):
        feature_name = features[i]
        
        # Skip if feature is in ignore set
        if any(x in feature_name for x in ignore_set):
            continue
            
        # --- 2. MATH LOGIC (Z-Score Analysis) ---
        # Score > 0.5 is "High", else "Normal"
        is_high = score > 0.5
        
        # Calculate Percentage Deviation
        pct = int(abs(score * 100))
        direction = "Above" if score > 0 else "Below"
        stat_label = f"{feature_name} ({pct}% {direction} Avg)"
        
        # --- 3. MAPPING RETRIEVAL ---
        # Find the matching key in our knowledge base
        # This replaces the long if-else chain
        advice_pair = default_advice
        for key, val in knowledge_base.items():
            if key in feature_name:
                advice_pair = val
                break
        
        # Select advice based on state (0 = High, 1 = Low)
        # Python Bool False=0, True=1. We invert 'is_high' index logic.
        # High (True) -> Index 0. Low (False) -> Index 1.
        action_text = advice_pair[0] if is_high else advice_pair[1]
        
        # Combine Label + Advice
        full_text = f"{stat_label}: {action_text}"

        advice_list.append({
            'appliance': feature_name,
            'action': full_text,
            'type': 'warning' if is_high else 'success' 
        })
            
    return advice_list