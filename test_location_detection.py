# AI Classification imports
from transformers import pipeline
import torch

import streamlit as st
import os
import datetime
import json
import tempfile
import time
import re
from typing import Dict, List, Tuple, Optional
import base64
import io
import wave
import struct
import uuid
import logging
from pathlib import Path
import requests

# Geolocation imports
try:
    from streamlit_geolocation import streamlit_geolocation
    GEOLOCATION_AVAILABLE = True
except ImportError:
    GEOLOCATION_AVAILABLE = False
    
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI911 Emergency Call System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state with better defaults
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'audio_buffer': None,
        'current_location': "",
        'detected_coordinates': None,
        'location_accuracy': None,
        'show_location_map': False,
        'analysis_result': None,
        'show_results': False,
        'processed_transcript': "",
        'is_recording': False,
        'call_counter': 0,
        'system_stats': {
            'total_calls': 0,
            'avg_response_time': 1.2,
            'system_status': 'OPERATIONAL'
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Ensure directories exist
def setup_directories():
    """Create necessary directories"""
    directories = ["audio_files", "transcripts", "analysis_results", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

setup_directories()

# Azure configuration
AZURE_SPEECH_KEY = "2AjtcKASybFagZKRTfXk3EciDWNNPEpqYS9rs5Dm3U4uCg4RO2BLJQQJ99BFACREanaXJ3w3AAAYACOGOH7j"
AZURE_SPEECH_REGION = "canadaeast"

# ===================================
# GEOLOCATION FUNCTIONS
# ===================================

def reverse_geocode(lat, lon):
    """Convert coordinates to street address"""
    try:
        # Using Nominatim (OpenStreetMap) - free, no API key needed
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'AI911-Emergency-System/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Build proper address format for emergency response
            addr = data.get('address', {})
            address_parts = []
            
            # Street number and name (most important for emergency)
            if addr.get('house_number'):
                address_parts.append(addr['house_number'])
            if addr.get('road'):
                address_parts.append(addr['road'])
            elif addr.get('street'):
                address_parts.append(addr['street'])
            
            # City
            city = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('municipality', 'Toronto')
            if city:
                address_parts.append(city)
            
            # Province/State
            state = addr.get('state', 'ON')
            if state:
                address_parts.append(state)
            
            # Postal code
            if addr.get('postcode'):
                address_parts.append(addr['postcode'])
            
            return ', '.join(address_parts) if address_parts else data.get('display_name', f"{lat:.6f}, {lon:.6f}")
            
    except Exception as e:
        logger.error(f"Reverse geocoding error: {e}")
        return f"Coordinates: {lat:.6f}, {lon:.6f}"

def show_enhanced_location_section():
    """Enhanced location input with real geolocation"""
    st.markdown("### 📍 Emergency Location")
    
    # Check if geolocation is available
    if not GEOLOCATION_AVAILABLE:
        st.warning("⚠️ Geolocation not available. Install with: `pip install streamlit-geolocation`")
    
    # Main location input
    location_col1, location_col2 = st.columns([3, 1])
    
    with location_col1:
        location_input = st.text_input(
            "Address or Location:",
            value=st.session_state.current_location,
            placeholder="123 Main Street, Toronto, ON",
            help="Enter address manually or use the location detector"
        )
    
    with location_col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if GEOLOCATION_AVAILABLE:
            detect_location = st.button(
                "📍 Detect My Location", 
                use_container_width=True,
                type="secondary",
                help="Uses your device's GPS for accurate location"
            )
        else:
            st.button(
                "📍 Install Geolocation", 
                use_container_width=True,
                type="secondary",
                disabled=True,
                help="Geolocation package not installed"
            )
            detect_location = False
    
    # Geolocation detection
    if detect_location and GEOLOCATION_AVAILABLE:
        with st.spinner("🔍 Detecting your location... (Please allow location access if prompted)"):
            try:
                # Get location using streamlit-geolocation
                location_data = streamlit_geolocation(key="geolocation")
                
                if location_data and location_data['latitude'] is not None:
                    lat = location_data['latitude']
                    lon = location_data['longitude']
                    accuracy = location_data.get('accuracy', 'Unknown')
                    
                    # Convert coordinates to address
                    address = reverse_geocode(lat, lon)
                    
                    # Update session state
                    st.session_state.current_location = address
                    st.session_state.detected_coordinates = (lat, lon)
                    st.session_state.location_accuracy = accuracy
                    st.session_state.show_location_map = True
                    
                    # Show success message with accuracy
                    st.success(f"✅ Location detected successfully!")
                    
                    # Display location info
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.info(f"📍 **Address:** {address}")
                    with info_col2:
                        if isinstance(accuracy, (int, float)):
                            if accuracy < 50:
                                accuracy_emoji = "🟢"
                                accuracy_text = "High"
                            elif accuracy < 100:
                                accuracy_emoji = "🟡"
                                accuracy_text = "Medium"
                            else:
                                accuracy_emoji = "🔴"
                                accuracy_text = "Low"
                            st.info(f"{accuracy_emoji} **Accuracy:** ±{accuracy:.0f}m ({accuracy_text})")
                    
                    st.rerun()
                else:
                    st.error("❌ Could not get location. Please ensure location services are enabled and try again.")
                    
            except Exception as e:
                st.error(f"❌ Location detection failed: {str(e)}")
                st.info("💡 Tip: Make sure you're using HTTPS (not HTTP) and have location services enabled")
    
    # Show map if location was detected and folium is available
    if st.session_state.show_location_map and st.session_state.detected_coordinates and FOLIUM_AVAILABLE:
        with st.expander("🗺️ Location Map", expanded=True):
            lat, lon = st.session_state.detected_coordinates
            
            # Create map
            m = folium.Map(location=[lat, lon], zoom_start=17)
            
            # Add marker with accuracy circle
            folium.Marker(
                [lat, lon],
                popup=f"📍 Emergency Location<br>{st.session_state.current_location}",
                tooltip="Emergency Location",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add accuracy circle
            if st.session_state.location_accuracy and isinstance(st.session_state.location_accuracy, (int, float)):
                folium.Circle(
                    location=[lat, lon],
                    radius=st.session_state.location_accuracy,
                    popup=f"Accuracy: ±{st.session_state.location_accuracy}m",
                    color='blue',
                    fill=True,
                    fillOpacity=0.2
                ).add_to(m)
            
            # Display map
            st_folium(m, height=300, width=None, returned_objects=[])
            
            # Additional options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Update Location"):
                    st.session_state.show_location_map = False
                    st.rerun()
            with col2:
                maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                st.markdown(f"[🗺️ Open in Google Maps]({maps_url})")
            with col3:
                st.markdown(f"**GPS:** {lat:.6f}, {lon:.6f}")
    
    # Additional location details (optional)
    with st.expander("➕ Additional Location Details", expanded=False):
        add_col1, add_col2 = st.columns(2)
        
        with add_col1:
            floor_apt = st.text_input(
                "Floor/Apartment/Unit:",
                placeholder="e.g., 3rd Floor, Apt 301",
                help="Specific unit or floor information"
            )
            
            landmark = st.text_input(
                "Nearby Landmark:",
                placeholder="e.g., Across from Tim Hortons",
                help="Visible landmarks for easier identification"
            )
        
        with add_col2:
            entrance = st.text_input(
                "Entrance/Access:",
                placeholder="e.g., Back entrance, Blue door",
                help="Special entrance or access instructions"
            )
            
            special_instructions = st.text_area(
                "Special Instructions:",
                placeholder="e.g., Gate code 1234, Ring buzzer 301",
                height=70,
                help="Any special access instructions for emergency responders"
            )
    
    # Compile full location information
    full_location = location_input or st.session_state.current_location
    
    # Add additional details if provided
    additional_details = []
    if 'floor_apt' in locals() and floor_apt:
        additional_details.append(floor_apt)
    if 'landmark' in locals() and landmark:
        additional_details.append(f"Near {landmark}")
    if 'entrance' in locals() and entrance:
        additional_details.append(f"Access: {entrance}")
    if 'special_instructions' in locals() and special_instructions:
        additional_details.append(special_instructions)
    
    if additional_details:
        full_location += f" | {' | '.join(additional_details)}"
    
    # Add GPS coordinates if available
    if st.session_state.detected_coordinates:
        lat, lon = st.session_state.detected_coordinates
        full_location += f" | GPS: {lat:.6f}, {lon:.6f}"
    
    return full_location

# ===================================
# ENHANCED KEYWORD CLASSIFIER
# ===================================

class AdvancedEmergencyClassifier:
    """Enhanced emergency classification with improved accuracy"""
    
    def __init__(self):
        self.keywords = self._build_keyword_database()
        self.location_patterns = self._build_location_patterns()
        self.context_analyzers = self._build_context_analyzers()
        
    def _build_keyword_database(self) -> Dict[str, List]:
        """Build comprehensive keyword database"""
        keywords_db = {
            "Medical": [
                # Critical medical keywords
                {"keyword": "can't breathe", "severity": "critical", "weight": 10},
                {"keyword": "cannot breathe", "severity": "critical", "weight": 10},
                {"keyword": "not breathing", "severity": "critical", "weight": 10},
                {"keyword": "unconscious", "severity": "critical", "weight": 10},
                {"keyword": "unresponsive", "severity": "critical", "weight": 10},
                {"keyword": "heart attack", "severity": "critical", "weight": 10},
                {"keyword": "cardiac arrest", "severity": "critical", "weight": 10},
                {"keyword": "chest pain", "severity": "critical", "weight": 9},
                {"keyword": "stroke", "severity": "critical", "weight": 10},
                {"keyword": "seizure", "severity": "critical", "weight": 9},
                {"keyword": "choking", "severity": "critical", "weight": 10},
                {"keyword": "severe bleeding", "severity": "critical", "weight": 9},
                {"keyword": "overdose", "severity": "critical", "weight": 9},
                {"keyword": "allergic reaction", "severity": "critical", "weight": 8},
                
                # High priority medical
                {"keyword": "bleeding", "severity": "high", "weight": 7},
                {"keyword": "broken bone", "severity": "high", "weight": 6},
                {"keyword": "fracture", "severity": "high", "weight": 6},
                {"keyword": "burn", "severity": "high", "weight": 7},
                {"keyword": "head injury", "severity": "high", "weight": 8},
                {"keyword": "pregnant", "severity": "high", "weight": 6},
                {"keyword": "labor", "severity": "high", "weight": 7},
                
                # Medium priority medical
                {"keyword": "fever", "severity": "medium", "weight": 4},
                {"keyword": "vomiting", "severity": "medium", "weight": 4},
                {"keyword": "pain", "severity": "medium", "weight": 3},
                {"keyword": "dizzy", "severity": "medium", "weight": 4},
            ],
            
            "Fire": [
                # Critical fire keywords
                {"keyword": "building on fire", "severity": "critical", "weight": 10},
                {"keyword": "house fire", "severity": "critical", "weight": 10},
                {"keyword": "apartment fire", "severity": "critical", "weight": 10},
                {"keyword": "explosion", "severity": "critical", "weight": 10},
                {"keyword": "people trapped", "severity": "critical", "weight": 10},
                
                # High priority fire
                {"keyword": "fire", "severity": "high", "weight": 8},
                {"keyword": "smoke", "severity": "high", "weight": 6},
                {"keyword": "gas leak", "severity": "high", "weight": 8},
                {"keyword": "electrical fire", "severity": "high", "weight": 7},
                {"keyword": "kitchen fire", "severity": "high", "weight": 6},
                
                # Medium priority fire
                {"keyword": "smoke alarm", "severity": "medium", "weight": 5},
                {"keyword": "burning smell", "severity": "medium", "weight": 4},
            ],
            
            "Police": [
                # Critical police keywords
                {"keyword": "gun", "severity": "critical", "weight": 10},
                {"keyword": "shooting", "severity": "critical", "weight": 10},
                {"keyword": "shots fired", "severity": "critical", "weight": 10},
                {"keyword": "armed robbery", "severity": "critical", "weight": 10},
                {"keyword": "hostage", "severity": "critical", "weight": 10},
                {"keyword": "kidnapping", "severity": "critical", "weight": 10},
                {"keyword": "home invasion", "severity": "critical", "weight": 9},
                {"keyword": "domestic violence", "severity": "critical", "weight": 9},
                
                # High priority police
                {"keyword": "break in", "severity": "high", "weight": 7},
                {"keyword": "breaking and entering", "severity": "high", "weight": 7},
                {"keyword": "burglary", "severity": "high", "weight": 7},
                {"keyword": "robbery", "severity": "high", "weight": 8},
                {"keyword": "assault", "severity": "high", "weight": 7},
                {"keyword": "threatening", "severity": "high", "weight": 6},
                
                # Medium priority police
                {"keyword": "suspicious person", "severity": "medium", "weight": 4},
                {"keyword": "vandalism", "severity": "medium", "weight": 4},
                {"keyword": "theft", "severity": "medium", "weight": 5},
            ],
            
            "Traffic": [
                # Critical traffic keywords
                {"keyword": "major accident", "severity": "critical", "weight": 9},
                {"keyword": "head-on collision", "severity": "critical", "weight": 10},
                {"keyword": "rollover", "severity": "critical", "weight": 9},
                {"keyword": "car flipped", "severity": "critical", "weight": 9},
                {"keyword": "multiple vehicle", "severity": "critical", "weight": 8},
                {"keyword": "pedestrian struck", "severity": "critical", "weight": 10},
                {"keyword": "hit and run", "severity": "critical", "weight": 8},
                
                # High priority traffic
                {"keyword": "car accident", "severity": "high", "weight": 7},
                {"keyword": "vehicle crash", "severity": "high", "weight": 7},
                {"keyword": "motorcycle accident", "severity": "high", "weight": 8},
                {"keyword": "drunk driver", "severity": "high", "weight": 7},
                
                # Medium priority traffic
                {"keyword": "fender bender", "severity": "medium", "weight": 4},
                {"keyword": "minor accident", "severity": "medium", "weight": 4},
            ]
        }
        return keywords_db
    
    def _build_location_patterns(self) -> List[str]:
        """Build location extraction patterns"""
        return [
            r'\d+\s+\w+\s+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?)',
            r'(?:intersection|corner)\s+of\s+\w+\s+and\s+\w+',
            r'(?:near|by|at)\s+(?:the\s+)?\w+\s+(?:building|store|mall|school|park)',
            r'[A-Z]\d[A-Z]\s*\d[A-Z]\d',  # Postal codes
            r'(?:Highway|Hwy)\s*(?:401|404|427|QEW)',
            r'\d{3,5}\s+\w+\s+(?:Street|Avenue|Road)',
        ]
    
    def _build_context_analyzers(self) -> Dict:
        """Build context analysis patterns"""
        return {
            "urgency_indicators": [
                r"right now", r"immediately", r"urgent", r"emergency",
                r"hurry", r"quickly", r"fast"
            ],
            "severity_modifiers": {
                "increase": [
                    "multiple", "several", "many", "child", "children", 
                    "baby", "elderly", "severe", "serious", "critical"
                ],
                "decrease": [
                    "minor", "small", "slight", "controlled", "stable"
                ]
            },
            "victim_count_patterns": [
                r'(\d+)\s*(?:people|persons?|victims?|injured)',
                r'(?:two|three|four|five|several|multiple)\s*(?:people|persons?)'
            ]
        }
    
    def analyze_emergency(self, transcript: str) -> Dict:
        """Analyze emergency call and return results as dictionary"""
        start_time = time.time()
        
        # Preprocess transcript
        transcript_clean = transcript.lower().strip()
        
        # KEYWORD ANALYSIS
        # Score emergency types
        type_scores = self._score_emergency_types(transcript_clean)
        
        # Determine primary emergency type and confidence
        keyword_emergency_type, keyword_confidence = self._determine_emergency_type(type_scores)
        
        # Determine severity
        keyword_severity = self._determine_severity(transcript_clean, keyword_emergency_type, type_scores)
        
        # AI ANALYSIS (if available)
        ai_emergency_type = None
        ai_confidence = None
        ai_severity = None
        
        if ai_classifier:
            try:
                # Define labels for emergency types
                emergency_labels = [
                    "medical emergency requiring ambulance",
                    "fire emergency requiring fire department",
                    "police emergency requiring law enforcement",
                    "traffic accident requiring emergency response"
                ]
                
                # Classify emergency type with AI
                ai_type_result = ai_classifier(
                    transcript,
                    candidate_labels=emergency_labels,
                    multi_label=False
                )
                
                # Map AI results to our emergency types
                top_label = ai_type_result['labels'][0]
                if "medical" in top_label:
                    ai_emergency_type = "Medical"
                elif "fire" in top_label:
                    ai_emergency_type = "Fire"
                elif "police" in top_label:
                    ai_emergency_type = "Police"
                else:
                    ai_emergency_type = "Traffic"
                
                ai_confidence = ai_type_result['scores'][0]
                
                # Classify severity with AI
                severity_labels = [
                    "life-threatening critical emergency",
                    "high priority urgent situation",
                    "medium priority emergency",
                    "low priority non-urgent"
                ]
                
                ai_severity_result = ai_classifier(
                    transcript,
                    candidate_labels=severity_labels,
                    multi_label=False
                )
                
                # Map severity
                severity_label = ai_severity_result['labels'][0]
                if "critical" in severity_label:
                    ai_severity = "Critical"
                elif "high" in severity_label:
                    ai_severity = "High"
                elif "medium" in severity_label:
                    ai_severity = "Medium"
                else:
                    ai_severity = "Low"
                    
            except Exception as e:
                logger.warning(f"AI classification failed: {e}")
                # AI failed, we'll use keyword results only
        
        # COMBINE RESULTS - Use AI if confident, otherwise keyword
        if ai_emergency_type and ai_confidence > 0.7:
            emergency_type = ai_emergency_type
            confidence = ai_confidence
            severity = ai_severity
            method = "ai_classification"
        else:
            emergency_type = keyword_emergency_type
            confidence = keyword_confidence
            severity = keyword_severity
            method = "keyword_classification"
        
        # Extract information
        extracted_info = self._extract_comprehensive_info(transcript_clean)
        
        # Get matched keywords
        matched_keywords = self._get_matched_keywords(transcript_clean, emergency_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(emergency_type, severity, extracted_info)
        
        # Create call ID
        call_id = f"CALL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds using {method}")
        
        # Build comprehensive result
        result = {
            "call_id": call_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "emergency_type": emergency_type,
            "severity": severity,
            "confidence": confidence,
            "transcript": transcript,
            "location": extracted_info.get('primary_location'),
            "matched_keywords": matched_keywords,
            "extracted_info": extracted_info,
            "recommendations": recommendations,
            "method": method,
            # Include both analyses for comparison
            "keyword_analysis": {
                "type": keyword_emergency_type,
                "severity": keyword_severity,
                "confidence": keyword_confidence
            }
        }
        
        # Add AI analysis if available
        if ai_emergency_type:
            result["ai_analysis"] = {
                "type": ai_emergency_type,
                "severity": ai_severity,
                "confidence": ai_confidence
            }
        
        return result
    
    def _score_emergency_types(self, transcript: str) -> Dict[str, float]:
        """Score each emergency type based on keywords"""
        scores = {etype: 0.0 for etype in self.keywords.keys()}
        
        for etype, keywords in self.keywords.items():
            for keyword_obj in keywords:
                if keyword_obj["keyword"] in transcript:
                    scores[etype] += keyword_obj["weight"]
        
        return scores
    
    def _determine_emergency_type(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """Determine primary emergency type and confidence"""
        if not any(scores.values()):
            return "Unknown", 0.3
        
        max_score = max(scores.values())
        total_score = sum(scores.values())
        
        emergency_type = max(scores, key=scores.get)
        confidence = max_score / total_score if total_score > 0 else 0.5
        
        return emergency_type, min(confidence, 0.99)
    
    def _determine_severity(self, transcript: str, emergency_type: str, scores: Dict[str, float]) -> str:
        """Determine severity with enhanced logic"""
        base_severity = "Medium"
        
        # Get severity from matched keywords
        if emergency_type in self.keywords:
            critical_keywords = [k for k in self.keywords[emergency_type] 
                               if k["severity"] == "critical" and k["keyword"] in transcript]
            high_keywords = [k for k in self.keywords[emergency_type] 
                           if k["severity"] == "high" and k["keyword"] in transcript]
            
            if critical_keywords:
                base_severity = "Critical"
            elif high_keywords:
                base_severity = "High"
        
        # Apply modifiers
        severity_levels = ["Low", "Medium", "High", "Critical"]
        current_index = severity_levels.index(base_severity)
        
        modifiers = self.context_analyzers["severity_modifiers"]
        for modifier in modifiers["increase"]:
            if modifier in transcript:
                current_index = min(current_index + 1, len(severity_levels) - 1)
                break
        
        for modifier in modifiers["decrease"]:
            if modifier in transcript:
                current_index = max(current_index - 1, 0)
                break
        
        return severity_levels[current_index]
    
    def _extract_comprehensive_info(self, transcript: str) -> Dict:
        """Extract comprehensive information from transcript"""
        info = {
            'locations': [],
            'primary_location': None,
            'victim_count': None,
            'urgency_level': 'standard',
            'special_circumstances': []
        }
        
        # Extract locations
        for pattern in self.location_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            info['locations'].extend(matches)
        
        # Set primary location as dictionary
        if info['locations']:
            info['primary_location'] = {
                'address': info['locations'][0],
                'confidence': 0.8,
                'source': 'extracted'
            }
        
        # Extract victim count
        for pattern in self.context_analyzers["victim_count_patterns"]:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                info['victim_count'] = match.group(0)
                break
        
        # Detect urgency
        urgency_indicators = self.context_analyzers["urgency_indicators"]
        for indicator in urgency_indicators:
            if re.search(indicator, transcript):
                info['urgency_level'] = 'high'
                break
        
        return info
    
    def _get_matched_keywords(self, transcript: str, emergency_type: str) -> List[Tuple[str, str]]:
        """Get list of matched keywords for the emergency type"""
        matched = []
        
        if emergency_type in self.keywords:
            for keyword_obj in self.keywords[emergency_type]:
                if keyword_obj["keyword"] in transcript:
                    matched.append((keyword_obj["keyword"], keyword_obj["severity"]))
        
        return matched[:10]
    
    def _generate_recommendations(self, emergency_type: str, severity: str, extracted_info: Dict) -> List[str]:
        """Generate contextual recommendations"""
        base_recs = {
            "Medical": {
                "Critical": [
                    "🚑 Dispatch ALS ambulance immediately - Code 3",
                    "🏥 Alert nearest trauma center",
                    "📞 Keep caller on line for medical instructions",
                    "🚁 Consider air ambulance if needed"
                ],
                "High": [
                    "🚑 Dispatch BLS ambulance with priority",
                    "📋 Obtain patient vitals and history",
                    "🏥 Pre-notify receiving hospital"
                ]
            },
            "Fire": {
                "Critical": [
                    "🚒 Full alarm response - multiple units",
                    "🚑 Stage EMS at safe distance",
                    "👮 Request police for scene control",
                    "🚁 Alert aerial/ladder company"
                ],
                "High": [
                    "🚒 Dispatch fire suppression unit",
                    "🚑 EMS on standby",
                    "⚠️ Check for hazardous materials"
                ]
            },
            "Police": {
                "Critical": [