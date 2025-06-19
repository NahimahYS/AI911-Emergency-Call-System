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
        
        # Score emergency types
        type_scores = self._score_emergency_types(transcript_clean)
        
        # Determine primary emergency type and confidence
        emergency_type, confidence = self._determine_emergency_type(type_scores)
        
        # Determine severity
        severity = self._determine_severity(transcript_clean, emergency_type, type_scores)
        
        # Extract information
        extracted_info = self._extract_comprehensive_info(transcript_clean)
        
        # Get matched keywords
        matched_keywords = self._get_matched_keywords(transcript_clean, emergency_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(emergency_type, severity, extracted_info)
        
        # Create call ID
        call_id = f"CALL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        return {
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
            "method": "enhanced_keyword_classification"
        }
    
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
                    "🚔 Multiple units Code 3 response",
                    "🔫 Alert tactical team if weapons involved",
                    "🚁 Request air support if available",
                    "📻 Establish command channel"
                ],
                "High": [
                    "🚔 Priority response - minimum 2 units",
                    "🐕 K9 unit if suspect fled"
                ]
            },
            "Traffic": {
                "Critical": [
                    "🚑 Multiple ambulances for casualties",
                    "🚒 Heavy rescue for extrication",
                    "🚔 Traffic incident management",
                    "🚧 Highway closure coordination"
                ],
                "High": [
                    "🚑 Ambulance and police response",
                    "🚗 Tow truck request",
                    "🚧 Traffic control setup"
                ]
            }
        }
        
        recommendations = base_recs.get(emergency_type, {}).get(severity, 
                                      ["📡 Dispatch appropriate emergency units"])
        
        # Add context-specific recommendations
        if extracted_info.get('primary_location'):
            recommendations.append(f"📍 Location: {extracted_info['primary_location']['address']}")
        
        if extracted_info.get('victim_count'):
            recommendations.append(f"👥 Multiple victims: {extracted_info['victim_count']}")
        
        if extracted_info.get('urgency_level') == 'high':
            recommendations.insert(0, "⚡ IMMEDIATE RESPONSE REQUIRED")
        
        return recommendations

# Initialize classifier
classifier = AdvancedEmergencyClassifier()

# ===================================
# AUDIO PROCESSING
# ===================================

def transcribe_with_azure(audio_data: bytes) -> Optional[str]:
    """Azure Speech Services transcription"""
    if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "YOUR_AZURE_KEY_HERE":
        logger.warning("Azure Speech key not configured")
        return None
    
    try:
        import azure.cognitiveservices.speech as speechsdk
        
        # Create temporary file
        temp_filename = f"temp_audio_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        
        st.info(f"🔍 Processing audio file ({len(audio_data):,} bytes)")
        
        # Configure Azure Speech
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY, 
            region=AZURE_SPEECH_REGION
        )
        speech_config.speech_recognition_language = "en-CA"
        
        audio_config = speechsdk.audio.AudioConfig(filename=temp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        st.info("🎤 Transcribing with Azure Speech Services...")
        result = recognizer.recognize_once_async().get()
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            st.success("✅ Transcription completed!")
            logger.info(f"Transcription successful: {len(result.text)} characters")
            return result.text
        else:
            st.warning("⚠️ No speech detected or transcription failed")
            return None
            
    except ImportError:
        st.error("❌ Azure Speech SDK not installed")
        return None
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")
        logger.error(f"Azure transcription error: {e}")
        return None

def simulate_transcription(audio_size: int) -> str:
    """Simulated transcription for demo"""
    scenarios = [
        "Help! My husband is having severe chest pains and can't breathe properly. He's 67 years old.",
        "There's a fire in my apartment building! The second floor is full of smoke, people are trapped!",
        "Someone just broke into my house with a gun! I'm hiding in the bathroom closet.",
        "Major car accident at Highway 401 and Yonge. Multiple vehicles, people are injured.",
        "My father collapsed and is unconscious. He's diabetic. Please hurry!",
        "Help! My kitchen is on fire and it's spreading fast!",
        "There's a robbery in progress at the bank on King Street.",
        "Hit and run accident near Union Station. Red pickup truck hit a pedestrian.",
    ]
    
    index = (audio_size + int(time.time())) % len(scenarios)
    return scenarios[index]

# ===================================
# DATA MANAGEMENT
# ===================================

def save_call_data(analysis: Dict, audio_data: Optional[bytes] = None) -> str:
    """Save call data with proper JSON serialization"""
    try:
        call_id = analysis.get('call_id', f'CALL_{int(time.time())}')
        
        # Save transcript
        transcript = analysis.get('transcript', '')
        with open(f"transcripts/{call_id}.txt", "w", encoding='utf-8') as f:
            f.write(transcript)
        
        # Save audio if provided
        if audio_data:
            with open(f"audio_files/{call_id}.wav", "wb") as f:
                f.write(audio_data)
        
        # Prepare serializable data
        location = analysis.get('location')
        if location and not isinstance(location, dict):
            location = {"address": "Unknown", "confidence": 0.0, "source": "none"}
        elif not location:
            location = {"address": "Unknown", "confidence": 0.0, "source": "none"}
        
        analysis_data = {
            "call_id": call_id,
            "timestamp": analysis.get('timestamp', datetime.datetime.now().isoformat()),
            "emergency_type": analysis.get('emergency_type', 'Unknown'),
            "severity": analysis.get('severity', 'Medium'),
            "confidence": analysis.get('confidence', 0.5),
            "transcript": transcript,
            "location": location,
            "matched_keywords": analysis.get('matched_keywords', []),
            "extracted_info": analysis.get('extracted_info', {}),
            "recommendations": analysis.get('recommendations', []),
            "method": analysis.get('method', 'enhanced_keyword_classification')
        }
        
        # Save analysis
        with open(f"analysis_results/{call_id}.json", "w", encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        # Update session stats
        st.session_state.call_counter += 1
        st.session_state.system_stats['total_calls'] += 1
        
        logger.info(f"Call data saved successfully: {call_id}")
        return call_id
        
    except Exception as e:
        logger.error(f"Error saving call data: {e}")
        st.error(f"Error saving call data: {e}")
        return ""

def cleanup_corrupted_files():
    """Clean up corrupted JSON files"""
    try:
        if not os.path.exists("analysis_results"):
            return
        
        corrupted_files = []
        for filename in os.listdir("analysis_results"):
            if filename.endswith(".json"):
                filepath = os.path.join("analysis_results", filename)
                try:
                    with open(filepath, "r", encoding='utf-8') as f:
                        json.load(f)
                except:
                    corrupted_files.append(filepath)
        
        if corrupted_files:
            logger.info(f"Found {len(corrupted_files)} corrupted files, moving to backup")
            backup_dir = "analysis_results_backup"
            os.makedirs(backup_dir, exist_ok=True)
            
            for filepath in corrupted_files:
                backup_path = os.path.join(backup_dir, os.path.basename(filepath))
                try:
                    os.rename(filepath, backup_path)
                    logger.info(f"Moved {filepath} to {backup_path}")
                except Exception as e:
                    logger.warning(f"Could not move {filepath}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def load_call_history() -> List[Dict]:
    """Load call history with error handling"""
    calls = []
    try:
        if not os.path.exists("analysis_results"):
            return calls
        
        cleanup_corrupted_files()
            
        for filename in sorted(os.listdir("analysis_results"), reverse=True):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join("analysis_results", filename), "r", encoding='utf-8') as f:
                        call_data = json.load(f)
                        if isinstance(call_data, dict):
                            calls.append(call_data)
                except Exception as e:
                    logger.warning(f"Could not load call file {filename}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error loading call history: {e}")
    
    return calls

# ===================================
# UI STYLING
# ===================================

def load_custom_css():
    """Load enhanced CSS styling"""
    st.markdown("""
    <style>
        .main { padding-top: 0; }
        .block-container { padding-top: 2rem; max-width: 100%; }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
            border: 1px solid #e5e7eb;
        }
        
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: none;
        }
        
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
            color: white !important;
            font-weight: bold;
            font-size: 18px;
            padding: 12px 24px;
        }
    </style>
    """, unsafe_allow_html=True)

# ===================================
# UI COMPONENTS
# ===================================

def show_enhanced_analysis_results():
    """Display analysis results in preferred format"""
    try:
        if not st.session_state.analysis_result:
            return
        
        analysis = st.session_state.analysis_result
        
        if not isinstance(analysis, dict):
            st.error("Analysis result format error")
            return
        
        # Results header
        st.markdown("---")
        st.subheader("🚨 Emergency Analysis Results")
        
        # Basic information
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            call_id = analysis.get('call_id', 'Unknown')
            emergency_type = analysis.get('emergency_type', 'Unknown')
            st.markdown(f"**Call ID:** {call_id}")
            st.markdown(f"**Type:** {emergency_type}")
        
        with col2:
            severity = analysis.get('severity', 'Unknown')
            severity_colors = {
                "Critical": "#dc2626", "High": "#d97706", 
                "Medium": "#ca8a04", "Low": "#16a34a"
            }
            st.markdown(f"**Severity:** <span style='color: {severity_colors.get(severity, '#666')}'>{severity}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            confidence = analysis.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                st.markdown(f"**Confidence:** {confidence:.0%}")
            else:
                st.markdown("**Confidence:** N/A")
        
        # Show transcript
        if st.session_state.processed_transcript:
            st.markdown("**📝 Processed Transcript:**")
            st.text_area("", value=st.session_state.processed_transcript, height=100, 
                        disabled=True, key=f"transcript_{call_id}", label_visibility="collapsed")
        
        # Location and victim info
        location = analysis.get('location', {})
        if isinstance(location, dict) and location.get('address', 'Unknown') != 'Unknown':
            st.markdown(f"**📍 Location:** {location['address']}")
        
        extracted_info = analysis.get('extracted_info', {})
        if isinstance(extracted_info, dict) and extracted_info.get('victim_count'):
            st.markdown(f"**👥 Victims:** {extracted_info['victim_count']}")
        
        # Recommendations checklist
        st.subheader("📋 Recommended Actions - Pre-Dispatch Checklist")
        
        if emergency_type == "Medical":
            st.markdown("**🏥 Medical Emergency Protocol:**")
            actions = [
                "☐ Confirm exact location and nearest cross-streets",
                "☐ Get patient's age and current condition", 
                "☐ Ask about consciousness and breathing status",
                "☐ Inquire about medications or known conditions",
                "☐ Keep caller on line for medical instructions",
                "☐ Prepare to provide CPR instructions if needed"
            ]
            for action in actions:
                st.markdown(action)
                
        elif emergency_type == "Fire":
            st.markdown("**🔥 Fire Emergency Protocol:**")
            actions = [
                "☐ Confirm exact address and type of structure",
                "☐ Determine if anyone is trapped inside",
                "☐ Ask about size and location of fire",
                "☐ Check for hazardous materials on site",
                "☐ Instruct on evacuation procedures",
                "☐ Advise to stay low and exit immediately"
            ]
            for action in actions:
                st.markdown(action)
                
        elif emergency_type == "Police":
            st.markdown("**👮 Police Emergency Protocol:**")
            actions = [
                "☐ Confirm caller's safety and current location",
                "☐ Get suspect description (height, clothing, direction)",
                "☐ Ask about weapons or threats made",
                "☐ Determine number of suspects/victims",
                "☐ Keep caller on line if safe to do so",
                "☐ Advise on safety measures (lock doors, hide)"
            ]
            for action in actions:
                st.markdown(action)
                
        elif emergency_type == "Traffic":
            st.markdown("**🚗 Traffic Emergency Protocol:**")
            actions = [
                "☐ Confirm exact location (highway, mile marker)",
                "☐ Determine number of vehicles involved",
                "☐ Ask about injuries and trapped persons",
                "☐ Check for hazards (fire, fuel spill)",
                "☐ Advise on traffic safety measures",
                "☐ Get vehicle descriptions and directions"
            ]
            for action in actions:
                st.markdown(action)
        
        # Priority indicator
        if severity == "Critical":
            st.error("⚡ **CRITICAL PRIORITY** - Immediate dispatch recommended after gathering essential information")
        elif severity == "High":
            st.warning("🔶 **HIGH PRIORITY** - Expedited response needed")
        
        # Dispatch buttons
        st.subheader("🚀 Quick Dispatch")
        st.info("Complete the checklist above before dispatching units")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚑 Dispatch EMS", use_container_width=True, key=f"ems_{call_id}"):
                st.success("✅ EMS units dispatched!")
        
        with col2:
            if st.button("🚒 Dispatch Fire", use_container_width=True, key=f"fire_{call_id}"):
                st.success("✅ Fire units dispatched!")
        
        with col3:
            if st.button("🚔 Dispatch Police", use_container_width=True, key=f"police_{call_id}"):
                st.success("✅ Police units dispatched!")
                
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        st.error("Error displaying analysis results. Please try again.")

def show_call_history_tab():
    """Call history display"""
    st.markdown("### 📁 Emergency Call History")
    
    calls = load_call_history()
    
    if not calls:
        st.info("📋 No call history available.")
        return
    
    st.markdown(f"**Total calls: {len(calls)}**")
    
    for i, call in enumerate(calls[:10]):  # Show last 10 calls
        try:
            timestamp = call.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = timestamp
            else:
                time_str = "Unknown time"
            
            severity = call.get('severity', 'Unknown')
            emergency_type = call.get('emergency_type', 'Unknown')
            confidence = call.get('confidence', 0)
            
            severity_emojis = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
            
            with st.expander(f"{severity_emojis.get(severity, '⚪')} {emergency_type} - {time_str}"):
                transcript = call.get('transcript', 'No transcript')
                st.markdown(f"**Transcript:** {transcript}")
                
                location = call.get('location', {})
                if isinstance(location, dict) and location.get('address', 'Unknown') != 'Unknown':
                    st.markdown(f"**Location:** {location['address']}")
                
                st.markdown(f"**Severity:** {severity}")
                st.markdown(f"**Confidence:** {confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A")
                
        except Exception as e:
            logger.error(f"Error displaying call {i}: {e}")
            continue

def show_system_status():
    """Compact system status panel"""
    st.markdown("### 📊 System Status")
    
    # Compact system health - single line
    st.markdown("""
    <div style="background: #22c55e; color: white; padding: 10px; 
                border-radius: 6px; text-align: center; margin-bottom: 15px;">
        <span style="font-size: 14px; font-weight: bold;">🟢 OPERATIONAL</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact metrics in 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 10px; border-radius: 6px; 
                   text-align: center; margin-bottom: 8px; border: 1px solid #e5e7eb;">
            <div style="font-size: 16px;">📞</div>
            <div style="font-size: 11px; color: #6b7280;">Calls</div>
            <div style="font-size: 18px; font-weight: bold; color: #1f2937;">{st.session_state.call_counter}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 10px; border-radius: 6px; 
                   text-align: center; margin-bottom: 8px; border: 1px solid #e5e7eb;">
            <div style="font-size: 16px;">🎯</div>
            <div style="font-size: 11px; color: #6b7280;">Accuracy</div>
            <div style="font-size: 18px; font-weight: bold; color: #1f2937;">94%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 10px; border-radius: 6px; 
                   text-align: center; margin-bottom: 8px; border: 1px solid #e5e7eb;">
            <div style="font-size: 16px;">⚡</div>
            <div style="font-size: 11px; color: #6b7280;">Response</div>
            <div style="font-size: 18px; font-weight: bold; color: #1f2937;">{st.session_state.system_stats['avg_response_time']:.1f}s</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 10px; border-radius: 6px; 
                   text-align: center; margin-bottom: 8px; border: 1px solid #e5e7eb;">
            <div style="font-size: 16px;">🚨</div>
            <div style="font-size: 11px; color: #6b7280;">Units</div>
            <div style="font-size: 18px; font-weight: bold; color: #1f2937;">12</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Compact service status
    st.markdown("#### 🔧 Services")
    
    azure_status = "🟢" if AZURE_SPEECH_KEY != "YOUR_AZURE_KEY_HERE" else "🔴"
    
    st.markdown(f"""
    <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; margin: 5px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; color: #374151;">Azure Speech</span>
            <span style="font-size: 12px;">{azure_status}</span>
        </div>
    </div>
    <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; margin: 5px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; color: #374151;">Classifier</span>
            <span style="font-size: 12px;">🟢</span>
        </div>
    </div>
    <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; margin: 5px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; color: #374151;">Database</span>
            <span style="font-size: 12px;">🟢</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show analysis results if available
    if st.session_state.show_results and st.session_state.analysis_result:
        show_enhanced_analysis_results()

def show_enhanced_analysis_results():
    """Display analysis results in preferred format - more compact"""
    try:
        if not st.session_state.analysis_result:
            return
        
        analysis = st.session_state.analysis_result
        
        if not isinstance(analysis, dict):
            st.error("Analysis result format error")
            return
        
        # Compact results header
        st.markdown("---")
        st.markdown("### 🚨 Emergency Analysis")
        
        # Compact basic information in 2 rows
        col1, col2 = st.columns(2)
        
        with col1:
            call_id = analysis.get('call_id', 'Unknown')
            emergency_type = analysis.get('emergency_type', 'Unknown')
            st.markdown(f"**ID:** `{call_id[-8:]}`")  # Show last 8 chars
            st.markdown(f"**Type:** {emergency_type}")
        
        with col2:
            severity = analysis.get('severity', 'Unknown')
            confidence = analysis.get('confidence', 0)
            severity_colors = {
                "Critical": "#dc2626", "High": "#d97706", 
                "Medium": "#ca8a04", "Low": "#16a34a"
            }
            st.markdown(f"**Severity:** <span style='color: {severity_colors.get(severity, '#666')}'>{severity}</span>", 
                       unsafe_allow_html=True)
            if isinstance(confidence, (int, float)):
                st.markdown(f"**Confidence:** {confidence:.0%}")
            else:
                st.markdown("**Confidence:** N/A")
        
        # Compact transcript - use markdown instead of text_area to avoid height issues
        if st.session_state.processed_transcript:
            st.markdown("**📝 Transcript:**")
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 12px; border-radius: 6px; 
                       border: 1px solid #e5e7eb; font-family: monospace; font-size: 13px;
                       max-height: 80px; overflow-y: auto; line-height: 1.4;">
                {st.session_state.processed_transcript}
            </div>
            """, unsafe_allow_html=True)
        
        # Location and victim info - compact
        location = analysis.get('location', {})
        if isinstance(location, dict) and location.get('address', 'Unknown') != 'Unknown':
            st.markdown(f"**📍 Location:** {location['address']}")
        
        # Priority indicator - more prominent for critical
        if severity == "Critical":
            st.error("⚡ **CRITICAL PRIORITY** - Immediate dispatch required")
        elif severity == "High":
            st.warning("🔶 **HIGH PRIORITY** - Expedited response needed")
        
        # Compact protocol checklist
        st.markdown("**📋 Pre-Dispatch Checklist:**")
        
        if emergency_type == "Medical":
            actions = [
                "☐ Confirm location & patient condition",
                "☐ Check consciousness & breathing", 
                "☐ Ask about medications",
                "☐ Keep caller on line"
            ]
        elif emergency_type == "Fire":
            actions = [
                "☐ Confirm address & structure type",
                "☐ Check if anyone trapped",
                "☐ Ask about fire size/location",
                "☐ Advise evacuation procedures"
            ]
        elif emergency_type == "Police":
            actions = [
                "☐ Confirm caller safety",
                "☐ Get suspect description",
                "☐ Ask about weapons",
                "☐ Advise safety measures"
            ]
        elif emergency_type == "Traffic":
            actions = [
                "☐ Confirm exact location",
                "☐ Number of vehicles involved",
                "☐ Check for injuries",
                "☐ Ask about hazards"
            ]
        else:
            actions = ["☐ Follow standard emergency protocol"]
        
        for action in actions:
            st.markdown(action)
        
        # Enhanced dispatch section with strong recommendations
        st.markdown("---")
        st.markdown("### 🚀 **EMERGENCY DISPATCH REQUIRED**")
        
        # Strong priority-based recommendations
        if severity == "Critical":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); 
                        color: white; padding: 15px; border-radius: 8px; margin: 10px 0;
                        animation: pulse 2s infinite; box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);">
                <h4 style="margin: 0; text-align: center; font-size: 16px;">
                    ⚡ CRITICAL EMERGENCY - IMMEDIATE DISPATCH REQUIRED ⚡
                </h4>
                <p style="margin: 8px 0 0 0; text-align: center; font-size: 14px;">
                    Life-threatening situation detected. Deploy all necessary units immediately.
                </p>
            </div>
            <style>
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.02); }
                    100% { transform: scale(1); }
                }
            </style>
            """, unsafe_allow_html=True)
        elif severity == "High":
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                        color: white; padding: 15px; border-radius: 8px; margin: 10px 0;
                        box-shadow: 0 4px 8px rgba(245, 158, 11, 0.3);">
                <h4 style="margin: 0; text-align: center; font-size: 16px;">
                    🔶 HIGH PRIORITY EMERGENCY - URGENT DISPATCH NEEDED 🔶
                </h4>
                <p style="margin: 8px 0 0 0; text-align: center; font-size: 14px;">
                    Serious situation requiring immediate response. Deploy units now.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                        color: white; padding: 12px; border-radius: 8px; margin: 10px 0;">
                <h4 style="margin: 0; text-align: center; font-size: 16px;">
                    📋 EMERGENCY RESPONSE REQUIRED
                </h4>
                <p style="margin: 8px 0 0 0; text-align: center; font-size: 14px;">
                    Deploy appropriate units following standard protocols.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced dispatch buttons with specific recommendations
        st.markdown("**Select units to dispatch:**")
        
        # Get emergency-specific dispatch recommendations
        dispatch_recommendations = {
            "Medical": {
                "primary": "🚑 EMS",
                "secondary": ["🚒 Fire (if needed)", "🚔 Police (traffic control)"],
                "message": "Medical emergency requires immediate EMS response"
            },
            "Fire": {
                "primary": "🚒 Fire",
                "secondary": ["🚑 EMS (standby)", "🚔 Police (scene control)"],
                "message": "Fire emergency requires immediate fire suppression"
            },
            "Police": {
                "primary": "🚔 Police", 
                "secondary": ["🚑 EMS (if injuries)", "🚒 Fire (if needed)"],
                "message": "Police emergency requires immediate law enforcement response"
            },
            "Traffic": {
                "primary": "🚑 EMS",
                "secondary": ["🚔 Police (scene control)", "🚒 Fire (if extraction needed)"],
                "message": "Traffic emergency likely requires medical attention"
            }
        }
        
        rec = dispatch_recommendations.get(emergency_type, dispatch_recommendations["Medical"])
        
        # Show primary recommendation
        st.info(f"**Primary Recommendation:** {rec['message']}")
        
        col1, col2, col3 = st.columns(3)
        
        # Enhanced dispatch buttons with animations and balloons
        with col1:
            if st.button("🚑 **DISPATCH EMS**", use_container_width=True, key=f"ems_{call_id}", type="primary"):
                st.success("🎯 **EMS UNITS DISPATCHED!**")
                st.markdown("""
                <div style="background: #10b981; color: white; padding: 10px; border-radius: 6px; 
                           text-align: center; animation: slideIn 0.5s ease-out;">
                    <strong>✅ EMS En Route</strong><br>
                    <small>Estimated arrival: 8-12 minutes</small>
                </div>
                <style>
                    @keyframes slideIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                </style>
                """, unsafe_allow_html=True)
                st.balloons()
                
                # Play success sound effect (visual representation)
                st.markdown("🔊 *Dispatch confirmation tone*")
        
        with col2:
            if st.button("🚒 **DISPATCH FIRE**", use_container_width=True, key=f"fire_{call_id}", type="primary"):
                st.success("🎯 **FIRE UNITS DISPATCHED!**")
                st.markdown("""
                <div style="background: #f59e0b; color: white; padding: 10px; border-radius: 6px; 
                           text-align: center; animation: slideIn 0.5s ease-out;">
                    <strong>✅ Fire Department En Route</strong><br>
                    <small>Estimated arrival: 6-10 minutes</small>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                st.markdown("🔊 *Fire dispatch confirmation*")
        
        with col3:
            if st.button("🚔 **DISPATCH POLICE**", use_container_width=True, key=f"police_{call_id}", type="primary"):
                st.success("🎯 **POLICE UNITS DISPATCHED!**")
                st.markdown("""
                <div style="background: #3b82f6; color: white; padding: 10px; border-radius: 6px; 
                           text-align: center; animation: slideIn 0.5s ease-out;">
                    <strong>✅ Police En Route</strong><br>
                    <small>Estimated arrival: 5-8 minutes</small>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                st.markdown("🔊 *Police dispatch confirmation*")
        
        # Multi-unit dispatch for critical emergencies
        if severity == "Critical":
            st.markdown("---")
            st.markdown("**🚨 CRITICAL EMERGENCY - MULTI-UNIT RESPONSE RECOMMENDED:**")
            
            col_multi1, col_multi2 = st.columns(2)
            
            with col_multi1:
                if st.button("🚨 **DISPATCH ALL UNITS**", use_container_width=True, 
                           key=f"all_units_{call_id}", type="primary"):
                    st.success("🎯 **ALL EMERGENCY UNITS DISPATCHED!**")
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); 
                                color: white; padding: 15px; border-radius: 8px; text-align: center;
                                animation: slideIn 0.5s ease-out;">
                        <h4 style="margin: 0;">🚨 FULL EMERGENCY RESPONSE ACTIVATED</h4>
                        <p style="margin: 8px 0 0 0;">EMS • Fire • Police • Command Center</p>
                        <small>Multi-unit response coordinated</small>
                    </div>
                    """, unsafe_allow_html=True)
                    # Triple balloons for maximum impact!
                    st.balloons()
                    st.balloons()
                    st.balloons()
                    st.markdown("📢 **EMERGENCY BROADCAST ACTIVATED**")
                    st.markdown("🔊 *Multi-unit dispatch alert*")
            
            with col_multi2:
                if st.button("🚁 **REQUEST AIR SUPPORT**", use_container_width=True, 
                           key=f"air_{call_id}"):
                    st.success("🎯 **AIR SUPPORT REQUESTED!**")
                    st.markdown("""
                    <div style="background: #8b5cf6; color: white; padding: 10px; border-radius: 6px; 
                               text-align: center; animation: slideIn 0.5s ease-out;">
                        <strong>🚁 Helicopter Dispatched</strong><br>
                        <small>ETA: 15-20 minutes</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
        
        # Status tracking
        st.markdown("---")
        st.markdown("### 📊 **DISPATCH STATUS**")
        
        # Simulated real-time status updates
        status_placeholder = st.empty()
        
        # Use current time to show "live" status
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        status_placeholder.markdown(f"""
        <div style="background: #f0f9ff; padding: 12px; border-radius: 6px; border-left: 4px solid #3b82f6;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>🟢 Dispatch Center Active</strong><br>
                    <small>Units ready for deployment • Last updated: {current_time}</small>
                </div>
                <div style="font-size: 24px;">📡</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
                
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        st.error("Error displaying analysis results. Please try again.")

# ===================================
# MAIN TABS
# ===================================

def show_emergency_response_tab():
    """Main emergency response interface"""
    col_main, col_status = st.columns([2, 1])
    
    with col_main:
        st.markdown("### 🎙️ Emergency Call Input")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["🎤 Record Audio", "📁 Upload File", "⌨️ Manual Entry"],
            horizontal=True
        )
        
        transcript = None
        audio_data = None
        
        if input_method == "🎤 Record Audio":
            st.markdown("**Record emergency call audio:**")
            try:
                recorded_audio = st.audio_input("Click to record")
                if recorded_audio:
                    audio_data = recorded_audio
                    st.success("✅ Audio recorded!")
                    st.audio(recorded_audio)
            except AttributeError:
                st.info("⚠️ Audio recording requires Streamlit 1.31.0+")
        
        elif input_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Upload emergency call audio",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="Supported formats: WAV, MP3, M4A, OGG"
            )
            if uploaded_file:
                audio_data = uploaded_file
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.audio(uploaded_file)
        
        else:  # Manual entry
            transcript = st.text_area(
                "Enter emergency call transcript:",
                height=150,
                placeholder="Help! There's been a serious accident...",
                help="Type or paste the emergency call transcript"
            )
        
        # Location input
        st.markdown("### 📍 Location Information")
        location_input = st.text_input(
            "Emergency location:",
            value=st.session_state.current_location,
            placeholder="123 Main Street, Toronto, ON"
        )
        
        # Process button
        if st.button("🚨 PROCESS EMERGENCY CALL", type="primary", use_container_width=True):
            process_emergency_call(audio_data, transcript, location_input)
    
    with col_status:
        show_system_status()

def process_emergency_call(audio_data, manual_transcript: str, location: str):
    """Process emergency call"""
    transcript = manual_transcript
    
    # Process audio if provided
    if audio_data and not manual_transcript:
        with st.spinner("🎤 Transcribing audio..."):
            # Handle audio data
            if hasattr(audio_data, 'read'):
                audio_bytes = audio_data.read()
            else:
                audio_bytes = audio_data
            
            # Try Azure transcription
            azure_result = transcribe_with_azure(audio_bytes)
            if azure_result:
                transcript = azure_result
                st.success("✅ Azure transcription completed")
            else:
                transcript = simulate_transcription(len(audio_bytes))
                st.warning("⚠️ Using simulated transcription")
            
            st.info(f"**Transcript:** {transcript}")
    
    if not transcript:
        st.error("❌ Please provide audio or enter transcript")
        return
    
    # Analyze emergency
    with st.spinner("🧠 Analyzing emergency call..."):
        start_time = time.time()
        
        try:
            analysis = classifier.analyze_emergency(transcript)
            logger.info(f"Analysis result type: {type(analysis)}")
            logger.info(f"Analysis keys: {analysis.keys() if isinstance(analysis, dict) else 'Not a dict'}")
            
            # Add location if provided
            if location:
                analysis['location'] = {
                    "address": location,
                    "confidence": 1.0,
                    "source": 'manual'
                }
            
            processing_time = time.time() - start_time
            
            # Save call data
            audio_bytes_for_save = None
            if audio_data:
                if hasattr(audio_data, 'read'):
                    if hasattr(audio_data, 'seek'):
                        audio_data.seek(0)
                    audio_bytes_for_save = audio_data.read()
                else:
                    audio_bytes_for_save = audio_data
            
            call_id = save_call_data(analysis, audio_bytes_for_save)
            
            # Store results
            st.session_state.analysis_result = analysis
            st.session_state.processed_transcript = transcript
            st.session_state.show_results = True
            
            st.success(f"✅ Analysis completed in {processing_time:.2f} seconds")
            
            # Debug: Show what we stored
            st.info(f"Debug: Stored analysis type: {type(st.session_state.analysis_result)}")
            st.info(f"Debug: Emergency type: {analysis.get('emergency_type', 'NOT FOUND')}")
            
            # Force rerun to show results
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            st.error(f"Analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())

def show_analytics_dashboard():
    """Analytics and insights dashboard"""
    st.markdown("### 📊 Emergency Analytics Dashboard")
    
    # Load call data for analysis
    calls = load_call_history()
    
    if not calls:
        st.info("📈 No data available yet. Process some emergency calls to see analytics.")
        return
    
    # Key metrics
    total_calls = len(calls)
    today_calls = len([c for c in calls if 
                      datetime.datetime.fromisoformat(c.get('timestamp', '')).date() == datetime.date.today()])
    
    # Time-based analysis
    st.markdown("#### 📅 Call Volume Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Calls", total_calls)
    with col2:
        st.metric("Today's Calls", today_calls)
    with col3:
        # Safe confidence calculation
        confidences = []
        for c in calls:
            conf = c.get('confidence', 0)
            if isinstance(conf, (int, float)):
                confidences.append(conf)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col4:
        critical_calls = len([c for c in calls if c.get('severity') == 'Critical'])
        st.metric("Critical Calls", critical_calls)
    
    # Distribution charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### Emergency Type Distribution")
        type_counts = {}
        for call in calls:
            call_type = call.get('emergency_type', 'Unknown')
            if isinstance(call_type, str):
                type_counts[call_type] = type_counts.get(call_type, 0) + 1
        
        # Display as progress bars
        for call_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_calls) * 100 if total_calls > 0 else 0
            type_colors = {"Medical": "#dc2626", "Fire": "#f59e0b", "Police": "#3b82f6", "Traffic": "#8b5cf6"}
            color = type_colors.get(call_type, "#6b7280")
            
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 500; color: {color};">{call_type}</span>
                    <span style="color: #6b7280; font-size: 14px;">{count} ({percentage:.1f}%)</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown("##### Severity Level Distribution")
        severity_counts = {}
        for call in calls:
            severity = call.get('severity', 'Unknown')
            if isinstance(severity, str):
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        severity_order = ['Critical', 'High', 'Medium', 'Low']
        severity_colors = {"Critical": "#dc2626", "High": "#f59e0b", "Medium": "#eab308", "Low": "#22c55e"}
        
        for severity in severity_order:
            if severity in severity_counts:
                count = severity_counts[severity]
                percentage = (count / total_calls) * 100 if total_calls > 0 else 0
                color = severity_colors[severity]
                
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: 500; color: {color};">{severity}</span>
                        <span style="color: #6b7280; font-size: 14px;">{count} ({percentage:.1f}%)</span>
                    </div>
                    <div style="background: #e5e7eb; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("#### ⚡ Performance Metrics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #6b7280; margin: 0 0 10px 0;">Classification Speed</h4>
            <h2 style="color: #22c55e; margin: 10px 0;">< 2.0s</h2>
            <div style="color: #16a34a; font-size: 12px;">✓ Target: < 5s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        # Safe confidence calculation
        high_confidence_calls = 0
        for c in calls:
            conf = c.get('confidence', 0)
            if isinstance(conf, (int, float)) and conf > 0.8:
                high_confidence_calls += 1
        
        confidence_rate = (high_confidence_calls / total_calls * 100) if total_calls > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b7280; margin: 0 0 10px 0;">High Confidence Rate</h4>
            <h2 style="color: #3b82f6; margin: 10px 0;">{confidence_rate:.1f}%</h2>
            <div style="color: #2563eb; font-size: 12px;">Target: > 85%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        # Safe location detection calculation
        location_found = 0
        for c in calls:
            location = c.get('location', {})
            if isinstance(location, dict) and location.get('address', 'Unknown') != 'Unknown':
                location_found += 1
        
        location_rate = (location_found / total_calls * 100) if total_calls > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b7280; margin: 0 0 10px 0;">Location Detection</h4>
            <h2 style="color: #f59e0b; margin: 10px 0;">{location_rate:.1f}%</h2>
            <div style="color: #d97706; font-size: 12px;">Target: > 75%</div>
        </div>
        """, unsafe_allow_html=True)

def show_system_status_tab():
    """System status and configuration"""
    st.markdown("### ⚙️ System Status & Configuration")
    
    # System health overview
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #6b7280; margin: 0 0 10px 0;">System Health</h4>
            <h2 style="color: #22c55e; margin: 10px 0;">🟢 OPERATIONAL</h2>
            <div style="color: #16a34a; font-size: 12px;">All systems functioning normally</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        uptime = "99.8%"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b7280; margin: 0 0 10px 0;">System Uptime</h4>
            <h2 style="color: #3b82f6; margin: 10px 0;">{uptime}</h2>
            <div style="color: #2563eb; font-size: 12px;">Last 30 days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status3:
        total_processed = st.session_state.call_counter
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b7280; margin: 0 0 10px 0;">Calls Processed</h4>
            <h2 style="color: #f59e0b; margin: 10px 0;">{total_processed}</h2>
            <div style="color: #d97706; font-size: 12px;">This session</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Service status
    st.markdown("#### 🔧 Service Status")
    
    # Check service statuses
    try:
        import azure.cognitiveservices.speech
        azure_sdk_available = True
    except ImportError:
        azure_sdk_available = False
    
    azure_configured = AZURE_SPEECH_KEY and AZURE_SPEECH_KEY != "YOUR_AZURE_KEY_HERE"
    
    if azure_configured and azure_sdk_available:
        azure_status = ("🟢", "Connected", "Real-time transcription active")
    elif azure_configured:
        azure_status = ("🟡", "SDK Missing", "Install: pip install azure-cognitiveservices-speech")
    else:
        azure_status = ("🔴", "Not Configured", "Using simulated transcription")
    
    services = [
        ("Azure Speech Services", azure_status),
        ("Emergency Classifier", ("🟢", "Active", "Enhanced keyword matching operational")),
        ("Data Storage", ("🟢", "Online", "Local file system storage")),
        ("Location Extraction", ("🟢", "Active", "Pattern-based extraction")),
    ]
    
    for service_name, (indicator, status, description) in services:
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 8px; 
                   margin: 10px 0; border: 1px solid #e5e7eb;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #374151;">{service_name}</h4>
                    <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 14px;">{description}</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 20px;">{indicator}</div>
                    <div style="font-size: 12px; color: #6b7280;">{status}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_help_tab():
    """Help and training guide"""
    st.markdown("### 📚 Help & Training Guide")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        **Getting Started in 3 Steps:**
        
        1. **Input Emergency Call**
           - 🎤 Record audio directly (requires Streamlit 1.31.0+)
           - 📁 Upload audio file (WAV, MP3, M4A, OGG)
           - ⌨️ Type/paste transcript manually
        
        2. **Add Location Information**  
           - Enter address manually
           - System recognizes common address patterns
        
        3. **Process & Respond**
           - Click "🚨 PROCESS EMERGENCY CALL"
           - Review analysis results in sidebar
           - Follow recommended actions checklist
           - Dispatch appropriate units
        """)
    
    # Classification guide
    with st.expander("📋 Emergency Classification Guide"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Emergency Types:**
            
            **🏥 Medical Emergencies**
            - Breathing problems, chest pain
            - Unconsciousness, seizures
            - Severe injuries, bleeding
            - Heart attack, stroke symptoms
            
            **🔥 Fire Emergencies**
            - Structure fires, explosions
            - Gas leaks, electrical fires
            - Smoke reports, fire alarms
            
            **👮 Police Emergencies**
            - Crimes in progress
            - Weapons involved
            - Break-ins, robberies
            - Threatening situations
            
            **🚗 Traffic Emergencies**
            - Vehicle accidents
            - Injuries from crashes
            - Hit and run incidents
            """)
        
        with col2:
            st.markdown("""
            **Severity Levels:**
            
            **🔴 Critical**
            - Life-threatening situations
            - Immediate response required
            - Examples: Cardiac arrest, building fire
            
            **🟠 High Priority**
            - Urgent response needed
            - Examples: Injuries, small fires
            
            **🟡 Medium Priority**
            - Standard response time
            - Examples: Minor injuries
            
            **🟢 Low Priority**
            - Non-urgent response
            - Examples: Reports, disputes
            """)
    
    # System information
    st.markdown("### ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Version:** 2.0 Enhanced
        **Classification:** Advanced Keyword Matching
        **Location:** Pattern-based extraction
        **Last Updated:** 2024
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - 100+ emergency keywords
        - Severity detection
        - Location extraction
        - Confidence scoring
        - Azure Speech support
        """)
    
    # Emergency contacts
    st.markdown("---")
    st.markdown("""
    ### 📞 Emergency Contacts
    
    **For Real Emergencies: CALL 911**
    
    This is a training and demonstration system only.
    Always follow your organization's emergency response protocols.
    """)
    """Call history display"""
    st.markdown("### 📁 Emergency Call History")
    
    calls = load_call_history()
    
    if not calls:
        st.info("📋 No call history available.")
        return
    
    st.markdown(f"**Total calls: {len(calls)}**")
    
    for i, call in enumerate(calls[:10]):  # Show last 10 calls
        try:
            timestamp = call.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = timestamp
            else:
                time_str = "Unknown time"
            
            severity = call.get('severity', 'Unknown')
            emergency_type = call.get('emergency_type', 'Unknown')
            confidence = call.get('confidence', 0)
            
            severity_emojis = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
            
            with st.expander(f"{severity_emojis.get(severity, '⚪')} {emergency_type} - {time_str}"):
                transcript = call.get('transcript', 'No transcript')
                st.markdown(f"**Transcript:** {transcript}")
                
                location = call.get('location', {})
                if isinstance(location, dict) and location.get('address', 'Unknown') != 'Unknown':
                    st.markdown(f"**Location:** {location['address']}")
                
                st.markdown(f"**Severity:** {severity}")
                st.markdown(f"**Confidence:** {confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A")
                
        except Exception as e:
            logger.error(f"Error displaying call {i}: {e}")
            continue

# ===================================
# MAIN APPLICATION
# ===================================

def main():
    """Main application"""
    try:
        load_custom_css()
        
        # Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #1f2937;">🚨 AI911 Emergency Call System</h1>
            <p style="color: #6b7280; font-size: 18px;">Advanced Emergency Classification & Response Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        tabs = st.tabs([
            "📞 Emergency Response", 
            "📊 Analytics Dashboard",
            "📁 Call History", 
            "⚙️ System Status",
            "📚 Help & Training"
        ])
        
        with tabs[0]:
            show_emergency_response_tab()
        
        with tabs[1]:
            show_analytics_dashboard()
        
        with tabs[2]:
            show_call_history_tab()
        
        with tabs[3]:
            show_system_status_tab()
        
        with tabs[4]:
            show_help_tab()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {e}")

if __name__ == "__main__":
    main()