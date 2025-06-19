import streamlit as st
from transformers import pipeline
import time
import torch

st.set_page_config(page_title="LLM Emergency Classification Test", page_icon="🧠")

st.title("🧠 Hugging Face Emergency Classification Test")
st.markdown("Testing FREE AI models for better emergency classification")

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'results' not in st.session_state:
    st.session_state.results = []

# Load model (only once)
@st.cache_resource
def load_classifier():
    """Load the zero-shot classification model"""
    with st.spinner("🤖 Loading AI model (this may take a minute)..."):
        try:
            # Use a smaller, faster model
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # Use CPU
            )
            return classifier
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# Initialize classifier
if st.session_state.classifier is None:
    st.session_state.classifier = load_classifier()

st.success("✅ AI Model loaded and ready!")

# Test scenarios
st.markdown("### 🧪 Test Emergency Scenarios")

test_scenarios = {
    "Clear Medical": "Help! My husband is having chest pains and can't breathe!",
    "Unclear Medical": "Something's wrong with grandma, she won't wake up",
    "Fire Emergency": "There's smoke coming from my neighbor's apartment",
    "Subtle Fire": "I smell something burning and it's getting stronger",
    "Police Emergency": "Someone is trying to break into my house right now",
    "Ambiguous Police": "There's a suspicious person following me",
    "Traffic Accident": "There's been a collision on Highway 401, people are hurt",
    "Complex Scenario": "My ex is here threatening me and my child, he's drunk",
    "Panic Call": "OH GOD PLEASE HELP SOMETHING TERRIBLE HAPPENED",
    "Multi-Emergency": "Car crash, the vehicle is on fire and driver is unconscious"
}

# Emergency type labels
emergency_labels = [
    "medical emergency requiring ambulance",
    "fire emergency requiring fire department",
    "police emergency requiring law enforcement",
    "traffic accident requiring emergency response"
]

# Severity labels
severity_labels = [
    "life-threatening critical emergency",
    "high priority urgent situation",
    "medium priority emergency",
    "low priority non-urgent"
]

# Test individual scenario
st.markdown("### 📝 Test Custom Scenario")

custom_text = st.text_area(
    "Enter emergency call transcript:",
    placeholder="Help! There's been an accident...",
    height=100
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🧠 Classify with AI", type="primary", disabled=not custom_text):
        if st.session_state.classifier and custom_text:
            with st.spinner("🤖 AI is analyzing..."):
                start_time = time.time()
                
                # Classify emergency type
                type_result = st.session_state.classifier(
                    custom_text,
                    candidate_labels=emergency_labels,
                    multi_label=False
                )
                
                # Classify severity
                severity_result = st.session_state.classifier(
                    custom_text,
                    candidate_labels=severity_labels,
                    multi_label=False
                )
                
                elapsed_time = time.time() - start_time
                
                # Display results
                st.markdown("### 🎯 AI Classification Results")
                
                # Emergency type
                st.markdown("**Emergency Type:**")
                for i, (label, score) in enumerate(zip(type_result['labels'], type_result['scores'])):
                    # Extract key type
                    if "medical" in label:
                        type_key = "🏥 Medical"
                    elif "fire" in label:
                        type_key = "🔥 Fire"
                    elif "police" in label:
                        type_key = "👮 Police"
                    else:
                        type_key = "🚗 Traffic"
                    
                    # Show top result prominently
                    if i == 0:
                        st.success(f"{type_key} - Confidence: {score:.1%}")
                    else:
                        st.caption(f"{type_key} - Confidence: {score:.1%}")
                
                # Severity
                st.markdown("**Severity Level:**")
                severity = severity_result['labels'][0]
                severity_score = severity_result['scores'][0]
                
                if "critical" in severity:
                    st.error(f"🔴 CRITICAL - Confidence: {severity_score:.1%}")
                elif "high" in severity:
                    st.warning(f"🟠 HIGH PRIORITY - Confidence: {severity_score:.1%}")
                elif "medium" in severity:
                    st.info(f"🟡 MEDIUM PRIORITY - Confidence: {severity_score:.1%}")
                else:
                    st.success(f"🟢 LOW PRIORITY - Confidence: {severity_score:.1%}")
                
                st.caption(f"⏱️ Analysis time: {elapsed_time:.2f} seconds")

with col2:
    if st.button("📊 Test All Scenarios"):
        progress_bar = st.progress(0)
        results_container = st.container()
        
        all_results = []
        
        for i, (scenario_name, scenario_text) in enumerate(test_scenarios.items()):
            progress_bar.progress((i + 1) / len(test_scenarios))
            
            # Classify
            type_result = st.session_state.classifier(
                scenario_text,
                candidate_labels=emergency_labels,
                multi_label=False
            )
            
            # Get top classification
            top_label = type_result['labels'][0]
            top_score = type_result['scores'][0]
            
            # Determine type
            if "medical" in top_label:
                emergency_type = "Medical"
            elif "fire" in top_label:
                emergency_type = "Fire"
            elif "police" in top_label:
                emergency_type = "Police"
            else:
                emergency_type = "Traffic"
            
            all_results.append({
                "Scenario": scenario_name,
                "Type": emergency_type,
                "Confidence": f"{top_score:.1%}"
            })
        
        # Display results table
        results_container.markdown("### 📊 Batch Test Results")
        results_container.table(all_results)

# Comparison with keyword system
st.markdown("---")
st.markdown("### 🔄 Keyword vs AI Comparison")

comparison_examples = {
    "My chest hurts": {
        "keyword": "❌ Might miss (no clear keyword)",
        "ai": "✅ Medical Emergency (understands context)"
    },
    "I think someone broke in": {
        "keyword": "⚠️ Might detect 'break in'",
        "ai": "✅ Police Emergency (high confidence)"
    },
    "Something doesn't smell right": {
        "keyword": "❌ Would miss completely",
        "ai": "✅ Possible Fire (from context)"
    }
}

for example, results in comparison_examples.items():
    with st.expander(f"Example: '{example}'"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Keyword System:**")
            st.markdown(results['keyword'])
        with col2:
            st.markdown("**AI System:**")
            st.markdown(results['ai'])

# Implementation guide
st.markdown("---")
st.markdown("### 🚀 How to Add to Your App")

with st.expander("📝 Implementation Code"):
    st.code("""
# Add to requirements.txt:
transformers>=4.30.0
torch>=2.0.0

# In your ai911_app.py:
from transformers import pipeline

@st.cache_resource
def load_ai_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

def classify_with_ai(transcript):
    classifier = load_ai_classifier()
    
    # Define labels
    emergency_types = [
        "medical emergency",
        "fire emergency",
        "police emergency",
        "traffic accident"
    ]
    
    # Classify
    result = classifier(
        transcript,
        candidate_labels=emergency_types
    )
    
    # Return top result
    return {
        "type": result['labels'][0],
        "confidence": result['scores'][0]
    }

# Use both systems:
keyword_result = classify_with_keywords(transcript)
ai_result = classify_with_ai(transcript)

# Show both to dispatcher!
""", language="python")

st.markdown("---")
st.info("""
**🎯 Advantages of Hugging Face:**
- ✅ Completely FREE (no API keys!)
- ✅ Understands context and nuance
- ✅ Works offline after download
- ✅ No data sent to external servers
- ✅ Great for academic projects!

**⚠️ Considerations:**
- First load takes ~30 seconds
- Needs ~2GB disk space for model
- ~2-3 seconds per classification
- Requires more RAM than keywords
""")