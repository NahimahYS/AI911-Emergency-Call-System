@st.cache_resource
def load_ai_classifier():
    """Load the AI model once and cache it"""
    try:
        # Check if we're on Streamlit Cloud with limited memory
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        if available_memory < 2.0:  # Less than 2GB available
            st.info("Running in memory-optimized mode")
            return None
            
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
    except Exception as e:
        st.warning(f"AI model disabled to conserve memory")
        return None