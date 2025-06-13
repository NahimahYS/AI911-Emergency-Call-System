import streamlit as st
import base64
import time

st.set_page_config(page_title="Audio Debug", page_icon="🔊")

st.title("🔊 Audio Playback Debug")

# Test 1: Basic HTML5 Audio
st.header("Test 1: Direct HTML5 Audio")

# Create a simple beep sound in base64
beep_base64 = "UklGRoYIAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YWIIAAD//////////v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v/+//7//v8="

# Method 1: Using HTML component
html_audio = f"""
<audio controls autoplay>
    <source src="data:audio/wav;base64,{beep_base64}" type="audio/wav">
    Your browser does not support the audio element.
</audio>
<p>If you can't hear anything, check your browser's autoplay settings.</p>
"""

st.markdown("**Method 1: HTML Audio Element**")
st.components.v1.html(html_audio, height=100)

# Method 2: Using st.audio with base64
st.markdown("**Method 2: Streamlit Audio Widget**")
audio_bytes = base64.b64decode(beep_base64)
st.audio(audio_bytes, format='audio/wav')

# Test 2: File upload
st.header("Test 2: File Upload")
uploaded_file = st.file_uploader("Upload any audio file", type=['wav', 'mp3', 'ogg', 'webm', 'm4a'])

if uploaded_file:
    bytes_data = uploaded_file.read()
    st.success(f"File uploaded: {uploaded_file.name} ({len(bytes_data):,} bytes)")
    
    # Try different methods to play
    st.markdown("**Trying different playback methods:**")
    
    # Method 1: Direct
    st.markdown("1. Direct playback:")
    st.audio(bytes_data)
    
    # Method 2: With explicit format
    st.markdown("2. With format specified:")
    file_extension = uploaded_file.name.split('.')[-1].lower()
    st.audio(bytes_data, format=f'audio/{file_extension}')
    
    # Method 3: As base64 in HTML
    st.markdown("3. As HTML5 audio:")
    b64 = base64.b64encode(bytes_data).decode()
    html = f'<audio controls src="data:audio/{file_extension};base64,{b64}"></audio>'
    st.components.v1.html(html, height=100)

# Test 3: Browser and environment info
st.header("Test 3: Environment Check")

# JavaScript to check browser capabilities
browser_check = """
<script>
function checkAudioSupport() {
    const audio = document.createElement('audio');
    const results = {
        'Audio Element Support': !!audio.canPlayType,
        'WAV Support': audio.canPlayType('audio/wav'),
        'MP3 Support': audio.canPlayType('audio/mp3'),
        'OGG Support': audio.canPlayType('audio/ogg'),
        'WebM Support': audio.canPlayType('audio/webm'),
        'User Agent': navigator.userAgent
    };
    
    let output = '<h4>Browser Audio Capabilities:</h4><ul>';
    for (let [key, value] of Object.entries(results)) {
        output += `<li><b>${key}:</b> ${value || 'Not supported'}</li>`;
    }
    output += '</ul>';
    
    document.getElementById('browser-info').innerHTML = output;
}

// Check autoplay policy
async function checkAutoplay() {
    try {
        const audio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQQAAAAA');
        await audio.play();
        document.getElementById('autoplay-info').innerHTML = '<p style="color: green;">✓ Autoplay is allowed</p>';
    } catch (err) {
        document.getElementById('autoplay-info').innerHTML = '<p style="color: red;">✗ Autoplay is blocked. You may need to interact with the page first.</p>';
    }
}

checkAudioSupport();
checkAutoplay();
</script>
<div id="browser-info"></div>
<div id="autoplay-info"></div>
"""

st.components.v1.html(browser_check, height=300)

# Python environment info
st.markdown("**Python Environment:**")
col1, col2 = st.columns(2)

with col1:
    st.write(f"- Streamlit version: {st.__version__}")
    st.write(f"- Python version: {st.sys.version.split()[0]}")

with col2:
    # Check if running locally or deployed
    st.write(f"- Running on: {'Local' if 'localhost' in st.get_option('server.address') else 'Deployed'}")

# Test 4: Simple recording test
st.header("Test 4: Simple Recording")

try:
    audio_input = st.audio_input("Click to record (Streamlit 1.31.0+ only)")
    if audio_input:
        audio_data = audio_input.read()
        st.success(f"Recorded {len(audio_data):,} bytes")
        st.audio(audio_data, format='audio/wav')
except:
    st.info("Native recording not available. Using alternative method...")
    
    record_html = """
    <button onclick="testRecording()">Test Microphone Access</button>
    <div id="status"></div>
    <script>
    async function testRecording() {
        const status = document.getElementById('status');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            status.innerHTML = '<p style="color: green;">✓ Microphone access granted!</p>';
            stream.getTracks().forEach(track => track.stop());
        } catch (err) {
            status.innerHTML = '<p style="color: red;">✗ Microphone access denied: ' + err.message + '</p>';
        }
    }
    </script>
    """
    st.components.v1.html(record_html, height=100)

# Troubleshooting tips
st.header("Troubleshooting Tips")
st.info("""
**If audio isn't playing:**

1. **Check browser settings:**
   - Chrome: Settings → Privacy → Site Settings → Sound
   - Firefox: Settings → Privacy → Permissions → Autoplay
   - Edge: Settings → Cookies and site permissions → Media autoplay

2. **Check system volume:**
   - Make sure system volume is not muted
   - Check if browser tab is muted (look for speaker icon in tab)

3. **Try a different browser:**
   - Chrome, Firefox, and Edge typically work best
   - Safari may have stricter autoplay policies

4. **Check firewall/antivirus:**
   - Some security software may block local audio playback

5. **For recording issues:**
   - Make sure you've granted microphone permissions
   - Check if another app is using the microphone
   - Try closing and reopening the browser

**Still not working?** 
- Open browser console (F12) and check for errors
- Try running on a different port: `streamlit run test.py --server.port 8502`
""")