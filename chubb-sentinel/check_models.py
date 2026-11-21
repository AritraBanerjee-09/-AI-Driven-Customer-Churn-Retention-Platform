# check_models.py
import google.generativeai as genai
import os

# --- IMPORTANT ---
# Replace this with the API key you got from Google AI Studio.
# For this simple check, it's okay to paste it here.
# Do NOT commit this file to GitHub with your key in it.
API_KEY = "AIzaSyANPYoNvSxIKFYbE1y-GHd0l4sDrCSKKlw"

try:
    genai.configure(api_key=API_KEY)

    print("✅ Successfully connected to Google AI. Available models for 'generateContent':")
    print("-" * 60)

    # List all models and filter for the ones you can use for your app
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)

    print("-" * 60)
    print("💡 Use one of the names above (e.g., 'gemini-1.5-flash-latest') in your app.")

except Exception as e:
    print(f"❌ An error occurred: {e}")
    print("   Please check if your API key is correct and has been pasted properly.")