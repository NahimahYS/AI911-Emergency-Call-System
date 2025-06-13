import azure.cognitiveservices.speech as speech

# Your Azure credentials
AZURE_KEY = "2AjtcKASybFagZKRTfXk3EciDWNNPEpqYS9rs5Dm3U4uCg4RO2BLJQQJ99BFACREanaXJ3w3AAAYACOGOH7j"  # Same key from line 41
AZURE_REGION = "canadaeast"  # Same region from line 42

# Test
print(f"Testing Azure with key: {AZURE_KEY[:8]}...")
print(f"Region: {AZURE_REGION}")

try:
    speech_config = speech.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    print("✅ Azure connection successful!")
except Exception as e:
    print(f"❌ Azure error: {e}")