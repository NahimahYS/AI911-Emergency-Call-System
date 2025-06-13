import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_azure_config():
    """Get Azure configuration from environment variables"""
    return {
        "key": os.getenv("AZURE_SPEECH_KEY", ""),
        "region": os.getenv("AZURE_SPEECH_REGION", "canadaeast")
    }