import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    exit()

print(f"✅ Found API Key: {api_key[:5]}...{api_key[-5:]}")

# 2. Configure
genai.configure(api_key=api_key)

# 3. Test Generation
try:
    print("\n--- Sending Test Prompt ---")
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Hello! Give me one short, fun fact about electricity.")
    
    print("\n✅ API Response Received:")
    print("--------------------------------------------------")
    print(response.text)
    print("--------------------------------------------------")
    print("\nSUCCESS: The API Key is working perfectly!")
    
except Exception as e:
    print(f"\n❌ API TEST FAILED: {e}")