import os
from dotenv import load_dotenv

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "canadaeast")

print(f"API Key length: {len(AZURE_SPEECH_KEY)}")
print(f"First 10 chars: {AZURE_SPEECH_KEY[:10]}...")
print(f"Last 10 chars: ...{AZURE_SPEECH_KEY[-10:]}")

try:
    import azure.cognitiveservices.speech as speechsdk
    
    # Simple connection test
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    
    # Create a synthesizer to test the connection
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    
    # Try to synthesize (this will validate the key)
    result = synthesizer.speak_text_async("Test").get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ Azure connection successful! API key is valid.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"❌ Azure connection failed: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
            
except Exception as e:
    print(f"Error: {e}")