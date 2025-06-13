import streamlit as st
import os

def get_azure_config():
    """Get Azure Speech configuration securely"""
    try:
        # Try Streamlit secrets first (for deployment)
        if hasattr(st, 'secrets') and "AZURE_SPEECH_KEY" in st.secrets:
            return {
                "key": st.secrets["AZURE_SPEECH_KEY"],
                "region": st.secrets.get("AZURE_SPEECH_REGION", "canadaeast")
            }
    except:
        pass
    
    # Fall back to hardcoded for demo
    return {
        "key": "2AjtcKASybFagZKRTfXk3EciDWNNPEpqYS9rs5Dm3U4uCg4RO2BLJQQJ99BFACREanaXJ3w3AAAYACOGOH7j",
        "region": "canadaeast"
    }
