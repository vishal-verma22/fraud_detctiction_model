import os
from google.cloud import speech

# Check credentials
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
print(f"📁 Credentials path: {cred_path}")

if cred_path and os.path.exists(cred_path):
    print("✅ Credentials file found!")
    
    try:
        client = speech.SpeechClient()
        print("✅ Google Speech-to-Text client created!")
        print("🎉 Setup complete! Your app can now use Google STT.")
        
        # Optional: List available models
        print("\n📊 Available features:")
        print("   - Indian English (en-IN) supported")
        print("   - 16kHz sample rate")
        print("   - Real-time transcription")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Credentials file NOT found")
    print(f"   Looking at: {cred_path}")