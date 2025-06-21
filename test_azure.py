import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test 1: Check if API key is loaded
print("=== Testing Azure Configuration ===")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "canadaeast")

print(f"✓ API Key loaded: {bool(AZURE_SPEECH_KEY)}")
print(f"✓ API Key length: {len(AZURE_SPEECH_KEY)}")
print(f"✓ Region: {AZURE_SPEECH_REGION}")

# Test 2: Check if Azure SDK is installed
print("\n=== Testing Azure SDK ===")
try:
    import azure.cognitiveservices.speech as speechsdk
    print("✓ Azure Speech SDK is installed")
except ImportError:
    print("✗ Azure Speech SDK NOT installed")
    print("Run: pip install azure-cognitiveservices-speech")
    exit()

# Test 3: Test actual transcription
print("\n=== Testing Transcription ===")
try:
    # Create a simple test audio file
    import wave
    import struct
    
    # Generate test audio (1 second of silence)
    test_filename = "test_audio.wav"
    with wave.open(test_filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(16000)  # 16kHz
        # Write 1 second of silence
        for _ in range(16000):
            wav_file.writeframes(struct.pack('<h', 0))
    
    print(f"✓ Created test audio file: {test_filename}")
    
    # Configure Azure
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = "en-CA"
    
    audio_config = speechsdk.audio.AudioConfig(filename=test_filename)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        audio_config=audio_config
    )
    
    print("✓ Azure Speech configured")
    print("⏳ Testing transcription...")
    
    result = recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"✓ Transcription successful: '{result.text}'")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("✓ Azure working but no speech detected (test was silence)")
    else:
        print(f"✗ Transcription failed: {result.reason}")
        
    # Cleanup
    os.remove(test_filename)
    
except Exception as e:
    print(f"✗ Error during transcription test: {e}")

print("\n=== Test Complete ===")