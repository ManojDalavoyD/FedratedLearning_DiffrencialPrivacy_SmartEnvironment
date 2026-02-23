import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from suggestion import generate_gemini_suggestions, generate_rule_based_suggestions

# Dummy data
features = ['Air Conditioning', 'Computer', 'Dishwasher', 'Fridge', 'Heater', 'Lights', 'Microwave', 'Oven', 'TV', 'Washing Machine', 'Temp', 'Size']
raw_data = [1600, 150, 0, 200, 0, 250, 0, 0, 100, 0, 22, 1200]
total_load = sum(raw_data)

print(f"Testing Gemini Suggestions with Load: {total_load}W")

# 1. Test Fallback directly
print("\n--- Direct Fallback Test ---")
try:
    fallback_sug = generate_rule_based_suggestions(raw_data, features, total_load)
    print(f"Fallback returned {len(fallback_sug)} items")
    for s in fallback_sug:
        print(s)
except Exception as e:
    print(f"Fallback Crashed: {e}")

# 2. Test Gemini (which might use fallback)
print("\n--- Gemini Wrapper Test ---")
sug = generate_gemini_suggestions(raw_data, features, total_load)
print(f"Wrapper returned {len(sug)} items")
for s in sug:
    print(s)
