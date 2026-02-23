import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from app import app, read_latest_data, map_csv_to_features

def test_live_api():
    print("Testing Live Data API...")
    
    # 1. Test CSV Reading
    try:
        row = read_latest_data()
        if row is None:
            print("❌ CSV Reading Failed: No data returned.")
            return
        print(f"✅ CSV Read Success. Last Timestamp: {row.get('Timestamp', 'N/A')}")
        print(f"   Values: AC={row.get('AC_Power')}, Oven={row.get('Oven_Status')}")
    except Exception as e:
        print(f"❌ Read CSV Exception: {e}")
        return

    # 2. Test Feature Mapping
    try:
        feats = map_csv_to_features(row)
        print(f"✅ Feature Mapping Success. Count: {len(feats)}")
    except Exception as e:
        print(f"❌ Mapping Exception: {e}")

    # 3. Test Endpoint Response
    with app.test_client() as client:
        try:
            resp = client.get('/api/live-data')
            if resp.status_code == 200:
                print("✅ API Endpoint /api/live-data returned 200 OK")
                data = resp.get_json()
                print(f"   Predicted Units: {data.get('predicted_units')}")
                print(f"   Suggestions Count: {len(data.get('suggestions'))}")
                print(f"   DP Status: {data.get('dp_status')}")
                
                # Check for AC warning logic if applicable
                ac_power = row.get('AC_Power', 0)
                suggestions = data.get('suggestions', [])
                found_ac_warn = any('Air Conditioning' in s['appliance'] for s in suggestions)
                
                if ac_power > 1500 and found_ac_warn:
                    print("   ✅ AC High Usage Warning verified found.")
                elif ac_power > 1500 and not found_ac_warn:
                    print("   ⚠️ AC Power is high but no warning found?")
                
            else:
                print(f"❌ API Failed with status: {resp.status_code}")
                print(resp.data)
        except Exception as e:
            print(f"❌ API Exception: {e}")

if __name__ == "__main__":
    test_live_api()
