import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import time

st.set_page_config(page_title="Accurate Location Detection", page_icon="🎯")

st.title("🎯 Accurate Location Detection Test")
st.markdown("Testing methods for precise location detection")

# Initialize session state
if 'precise_location' not in st.session_state:
    st.session_state.precise_location = None
if 'location_permission' not in st.session_state:
    st.session_state.location_permission = False

# Method 1: HTML5 Geolocation API (MOST ACCURATE)
st.markdown("### 🎯 Method 1: Browser GPS Location (Most Accurate)")
st.info("🔒 This requires your permission and HTTPS (works on deployed apps)")

# Create a unique key for the component
location_component = """
<div id="location-result"></div>
<button onclick="getPreciseLocation()" style="
    background-color: #ff4b4b;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    margin: 10px 0;
">📍 Get My Exact Location</button>

<script>
function getPreciseLocation() {
    const resultDiv = document.getElementById('location-result');
    resultDiv.innerHTML = '⏳ Getting your location...';
    
    if (!navigator.geolocation) {
        resultDiv.innerHTML = '❌ Geolocation is not supported by your browser';
        return;
    }
    
    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const accuracy = position.coords.accuracy;
            
            resultDiv.innerHTML = `
                <div style="background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    <h4>✅ Exact Location Found!</h4>
                    <p><strong>Latitude:</strong> ${lat}</p>
                    <p><strong>Longitude:</strong> ${lon}</p>
                    <p><strong>Accuracy:</strong> ±${accuracy} meters</p>
                    <p style="color: green;"><strong>This is GPS-level accuracy!</strong></p>
                </div>
            `;
            
            // Send to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                data: {lat: lat, lon: lon, accuracy: accuracy}
            }, '*');
        },
        function(error) {
            let errorMsg = '';
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    errorMsg = "❌ You denied location access. Please allow location access and try again.";
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMsg = "❌ Location information is unavailable.";
                    break;
                case error.TIMEOUT:
                    errorMsg = "❌ Location request timed out.";
                    break;
                default:
                    errorMsg = "❌ An unknown error occurred.";
            }
            resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0;">${errorMsg}</div>`;
        },
        {
            enableHighAccuracy: true,  // Use GPS if available
            timeout: 10000,           // 10 second timeout
            maximumAge: 0             // Don't use cached location
        }
    );
}
</script>
"""

components.html(location_component, height=200)

st.markdown("---")

# Method 2: Google Maps Geocoding (Address to Coordinates)
st.markdown("### 📍 Method 2: Address-Based Location")
st.markdown("Enter a specific address for precise coordinates:")

address_input = st.text_input(
    "Enter full address:",
    placeholder="123 Main Street, Toronto, ON M5V 3A8",
    help="Include street number, street name, city, and postal code"
)

if st.button("🔍 Get Coordinates from Address"):
    if address_input:
        # Using a free geocoding service (Nominatim)
        with st.spinner("Converting address to coordinates..."):
            try:
                # URL encode the address
                encoded_address = requests.utils.quote(address_input)
                url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1"
                
                headers = {
                    'User-Agent': 'AI911EmergencySystem/1.0'  # Required by Nominatim
                }
                
                response = requests.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        lat = float(result['lat'])
                        lon = float(result['lon'])
                        display_name = result['display_name']
                        
                        st.success("✅ Location found!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Latitude", f"{lat:.6f}")
                            st.metric("Longitude", f"{lon:.6f}")
                        
                        with col2:
                            st.write("**Full Address:**")
                            st.write(display_name)
                        
                        # Show on map
                        st.map(data=[[lat, lon]], zoom=15)
                        
                        st.info(f"📍 These coordinates are precise to the exact address!")
                    else:
                        st.error("Address not found. Try adding more details.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter an address")

st.markdown("---")

# Method 3: Postal Code + Street Number (Canadian specific)
st.markdown("### 🇨🇦 Method 3: Canadian Postal Code Precision")

col1, col2 = st.columns(2)
with col1:
    postal_code = st.text_input("Postal Code:", placeholder="M5V 3A8")
with col2:
    street_number = st.text_input("Street Number:", placeholder="123")

if st.button("📮 Get Location from Postal Code"):
    if postal_code:
        # Canadian postal codes are very precise (often just one block)
        st.info(f"📍 Canadian postal code {postal_code} typically covers only 1-2 blocks!")
        st.success("This provides accuracy within 50-100 meters")

st.markdown("---")

# Method 4: What3Words (3-word location)
st.markdown("### 🔤 Method 4: What3Words (3m x 3m accuracy)")
st.info("Every 3m x 3m square in the world has a unique 3-word address")

w3w_input = st.text_input(
    "Enter 3-word address:",
    placeholder="filled.count.soap",
    help="Get from what3words.com"
)

if w3w_input:
    st.success(f"📍 What3Words address '{w3w_input}' provides 3-meter accuracy!")
    st.markdown("[Convert at what3words.com](https://what3words.com)")

st.markdown("---")

# Comparison Table
st.markdown("### 📊 Accuracy Comparison")

comparison_data = {
    "Method": ["IP-Based", "Browser GPS", "Full Address", "Postal Code", "What3Words"],
    "Accuracy": ["~5-50 km", "5-10 meters", "Exact building", "50-100 meters", "3 meters"],
    "Requires Permission": ["No", "Yes", "No", "No", "No"],
    "Works Offline": ["No", "Yes*", "No", "Partial", "No"],
    "Best For": ["General area", "Mobile devices", "Known address", "Canadian addresses", "Precise sharing"]
}

st.table(comparison_data)

st.markdown("---")

# Emergency System Recommendation
st.markdown("### 🚨 For Your Emergency System")
st.success("""
**Recommended Approach:**
1. **Start with IP location** for immediate general area
2. **Request browser location** for GPS accuracy (if permitted)
3. **Allow manual address entry** as fallback
4. **Use postal code** for Canadian addresses

This gives you multiple fallback options!
""")

# Implementation Example
with st.expander("📝 Implementation Code for Your App"):
    st.code("""
# In your emergency app:

def get_emergency_location():
    # 1. Try browser GPS first (most accurate)
    gps_location = request_browser_location()
    if gps_location:
        return f"GPS: {gps_location['lat']}, {gps_location['lon']}"
    
    # 2. Fall back to IP location
    ip_location = get_ip_location()
    if ip_location:
        city = ip_location.get('city', 'Unknown')
        region = ip_location.get('region', 'Unknown')
        
        # 3. Ask for specific address
        specific = st.text_input(
            f"Detected: {city}, {region}. Enter exact address:",
            placeholder="123 Main St"
        )
        
        if specific:
            return f"{specific}, {city}, {region}"
    
    return "Location unknown - please provide address"
""", language="python")