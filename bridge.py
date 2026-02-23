import serial
import time
import csv
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========================================================
# 1. CONFIGURATION
# ========================================================
SERIAL_PORT = 'COM4'   # <--- CHECK YOUR ARDUINO PORT!
BAUD_RATE = 9600       # Must match Arduino code
FIREBASE_HOST = os.getenv("FIREBASE_HOST", "smarthome-f6262-default-rtdb.firebaseio.com")
FIREBASE_SECRET = os.getenv("FIREBASE_SECRET", "dQZWIkIhYX0Dq1g9gpgw8Yx1wSWg4J4QLKx8jgyq")

# ========================================================
# 2. SETUP CSV FILE (In 'Dataset' Folder)
# ========================================================
# This saves the file inside your "Dataset" folder automatically
current_folder = os.getcwd()
csv_filename = os.path.join(current_folder, "Dataset", "live_smart_home_data.csv")

# Check if file exists to decide if we need headers
file_exists = os.path.isfile(csv_filename)

# Open the file
csv_file = open(csv_filename, 'a', newline='')
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(["Timestamp", "Temperature", "AC_Power", "Oven_Status", "Light_Status", "Total_Load"])
    print(f"Created new file at: {csv_filename}")
else:
    print(f"Appending to: {csv_filename}")

# ========================================================
# 3. CONNECT TO ARDUINO
# ========================================================
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"ERROR: Could not connect to {SERIAL_PORT}. Close Arduino IDE Serial Monitor!")
    exit()

print("Listening... (Press Ctrl+C to stop)")

# ========================================================
# 4. MAIN LOOP
# ========================================================
try:
    while True:
        # --- 1. READ FRESH DATA (Anti-Lag) ---
        # Loop to clear the buffer and get only the LATEST line
        last_raw_line = None
        while arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode('utf-8').strip()
                if line: last_raw_line = line
            except: pass
        
        # If no new data, wait a bit and loop again
        if last_raw_line is None:
            time.sleep(0.1)
            continue

        raw_line = last_raw_line
        # -------------------------------------
        
        if True: # Keep indentation compatible
            try:
                # DEBUG PRINT for USER
                print(f"DEBUG RAW: {raw_line}")
                
                parts = raw_line.split(',')
                
                # Default variables
                temp = 25.0
                ac_power = 0
                oven = 0
                light = 0
                total = 0
                valid_parse = False

                if len(parts) == 5:
                    # Original Format: Temp, AC, Oven, Light, Total
                    try:
                        temp = float(parts[0])
                        ac_power = int(parts[1])
                        oven = int(parts[2])
                        light = int(parts[3])
                        total = int(parts[4])
                        valid_parse = True
                    except ValueError: pass
                        
                elif len(parts) == 4:
                    # Fallback Format: Temp, AC, Oven, Light (Calculate Total)
                    try:
                        temp = float(parts[0])
                        ac_power = int(parts[1])
                        oven = int(parts[2])
                        light = int(parts[3])
                        
                        # ESTIMATE TOTAL LOAD based on status
                        # Oven default ~2000W if ON, Light ~100W
                        oven_load = 2000 if oven == 1 else 0
                        light_load = 100 if light == 1 else 0
                        total = ac_power + oven_load + light_load
                        print(f"⚠️ Warning: Received 4 values. Calculated Total: {total}W")
                        valid_parse = True
                    except ValueError: pass

                if valid_parse:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # --- SAVE TO CSV ---
                    csv_writer.writerow([timestamp, temp, ac_power, oven, light, total])
                    csv_file.flush() # Force save immediately
                    
                    # --- PUSH TO FIREBASE ---
                    url = f"https://{FIREBASE_HOST}/SmartHome/Sensors.json?auth={FIREBASE_SECRET}"
                    data = {
                        "Temperature": temp,
                        "AC_Power": ac_power,
                        "Oven_Status": oven,
                        "Lights_Status": light,
                        "Total_Load": total,
                        "Last_Updated": timestamp
                    }
                    try:
                        requests.patch(url, json=data, timeout=1) # 1s timeout for speed
                    except Exception as e:
                        # minimal print to avoid spamming console
                        pass

                    print(f"[{timestamp}] Saved & Uploaded: {total}W")

            except ValueError:
                pass

except KeyboardInterrupt:
    print("\nStopping...")
    csv_file.close()
    arduino.close()