import streamlit as st
import io
import wave
import struct
import numpy as np

st.set_page_config(page_title="Audio Recording Test", page_icon="🎤")

st.title("🎤 Audio Recording Test")
st.markdown("Let's test if audio recording is working properly")

# Check Streamlit version
st.info(f"Your Streamlit version: {st.__version__}")

# Initialize session state
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

st.header("Test 1: File Upload")
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'm4a', 'ogg'])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    audio_bytes = uploaded_file.read()
    st.write(f"File size: {len(audio_bytes):,} bytes")
    st.audio(audio_bytes, format=f'audio/{uploaded_file.type.split("/")[-1]}')
    
    # Reset file pointer for reuse
    uploaded_file.seek(0)

st.header("Test 2: Direct Recording")

# Method 1: Try native audio_input (Streamlit 1.31.0+)
try:
    st.markdown("### Method 1: Native Audio Input (Streamlit 1.31.0+)")
    audio_input = st.audio_input("Click the microphone icon to record")
    
    if audio_input:
        st.session_state.recorded_audio = audio_input.read()
        st.success(f"✅ Audio recorded! Size: {len(st.session_state.recorded_audio):,} bytes")
        
        # Display the audio player
        st.audio(st.session_state.recorded_audio, format='audio/wav')
        
        # Save button
        if st.button("Save Recording to File"):
            with open("test_recording.wav", "wb") as f:
                f.write(st.session_state.recorded_audio)
            st.success("Saved as test_recording.wav")
        
        # Analyze the audio
        if st.button("Analyze Audio"):
            try:
                # Try to open as WAV
                audio_io = io.BytesIO(st.session_state.recorded_audio)
                with wave.open(audio_io, 'rb') as wav_file:
                    st.write("**Audio Properties:**")
                    st.write(f"- Channels: {wav_file.getnchannels()}")
                    st.write(f"- Sample width: {wav_file.getsampwidth()} bytes")
                    st.write(f"- Frame rate: {wav_file.getframerate()} Hz")
                    st.write(f"- Number of frames: {wav_file.getnframes()}")
                    st.write(f"- Duration: {wav_file.getnframes() / wav_file.getframerate():.2f} seconds")
            except Exception as e:
                st.error(f"Error analyzing audio: {str(e)}")
                st.info("The audio might be in a different format. Try converting it to WAV.")

except AttributeError:
    st.warning("""
    ⚠️ `audio_input` not available in your Streamlit version.
    
    **To enable native recording:**
    ```bash
    pip install streamlit --upgrade
    ```
    
    Current version requirement: Streamlit 1.31.0 or higher
    """)

# Method 2: Alternative with JavaScript
st.markdown("### Method 2: JavaScript Recorder with Download")

recording_interface = """
<div style="border: 2px solid #ddd; padding: 20px; border-radius: 10px; background: #f9f9f9;">
    <h4 style="margin-top: 0;">Alternative Recording Method</h4>
    <button id="startBtn" onclick="startRecording()" style="
        background: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin-right: 10px;
    ">🎤 Start Recording</button>
    
    <button id="stopBtn" onclick="stopRecording()" disabled style="
        background: #f44336;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        opacity: 0.5;
    ">⏹️ Stop Recording</button>
    
    <div id="status" style="margin-top: 15px; font-weight: bold;"></div>
    <div id="audioContainer" style="margin-top: 15px;"></div>
</div>

<script>
let mediaRecorder;
let audioChunks = [];

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        document.getElementById('startBtn').disabled = true;
        document.getElementById('startBtn').style.opacity = '0.5';
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('stopBtn').style.opacity = '1';
        document.getElementById('status').innerHTML = '🔴 Recording in progress...';
        document.getElementById('audioContainer').innerHTML = '';
        
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Create audio player
            const audio = document.createElement('audio');
            audio.controls = true;
            audio.src = audioUrl;
            
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = audioUrl;
            downloadLink.download = 'recording_' + Date.now() + '.webm';
            downloadLink.innerHTML = '💾 Download Recording';
            downloadLink.style.display = 'block';
            downloadLink.style.marginTop = '10px';
            downloadLink.style.color = '#1976d2';
            
            const container = document.getElementById('audioContainer');
            container.innerHTML = '<p style="margin: 10px 0;"><strong>Your Recording:</strong></p>';
            container.appendChild(audio);
            container.appendChild(downloadLink);
            
            document.getElementById('status').innerHTML = '✅ Recording complete! You can play it below or download it.';
        };
        
        mediaRecorder.start();
        
    } catch (err) {
        console.error('Error:', err);
        document.getElementById('status').innerHTML = '❌ Error: ' + err.message + '<br>Please ensure you have granted microphone permissions.';
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').style.opacity = '1';
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').style.opacity = '1';
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('stopBtn').style.opacity = '0.5';
    }
}
</script>
"""

import streamlit.components.v1 as components
components.html(recording_interface, height=300)

st.info("""
**Instructions for Method 2:**
1. Click "Start Recording" and allow microphone access
2. Speak your message
3. Click "Stop Recording"
4. Play the recording to verify it worked
5. Download the recording
6. Use the file uploader above to process it
""")

# Test 3: Create a test audio file
st.header("Test 3: Generate Test Audio")
if st.button("Generate Test WAV File"):
    # Create a simple sine wave
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # A4 note
    
    # Generate samples
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    # Create WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    wav_data = wav_buffer.getvalue()
    st.session_state.recorded_audio = wav_data
    
    st.success(f"Generated {duration}-second test tone (440 Hz)")
    st.audio(wav_data, format='audio/wav')
    
    # Save option
    if st.button("Save Test Audio"):
        with open("test_tone.wav", "wb") as f:
            f.write(wav_data)
        st.success("Saved as test_tone.wav")

# Display session state
with st.expander("Debug Info"):
    st.write("**Session State:**")
    st.write(f"- recorded_audio exists: {'recorded_audio' in st.session_state}")
    if 'recorded_audio' in st.session_state and st.session_state.recorded_audio:
        st.write(f"- recorded_audio size: {len(st.session_state.recorded_audio):,} bytes")
    st.write(f"- Streamlit version: {st.__version__}")
    
    # Check for required packages
    st.write("\n**Package Status:**")
    packages = ['numpy', 'wave', 'io']
    for pkg in packages:
        try:
            __import__(pkg)
            st.write(f"✅ {pkg}")
        except ImportError:
            st.write(f"❌ {pkg}")