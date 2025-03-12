# utils/local_tts.py
import requests
import os
import json

# Constants for the local TTS endpoint (kept for backward compatibility)
LOCAL_TTS_URL = "http://127.0.0.1:8000/generate_speech"

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_fd0fd11e16525e19aa5c5952b9be54be53c0d879718447ee")  # Set your API key in environment variables
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (default), change to your preferred voice
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Flag to control whether to attempt local TTS first or just use fallback
USE_LOCAL_TTS_FIRST = False  # Set to False to use ElevenLabs directly

def call_local_tts(text):
    """
    Calls the local TTS Flask endpoint to generate speech, or uses ElevenLabs.
    Returns the audio bytes if successful, otherwise returns None.
    """
    if USE_LOCAL_TTS_FIRST:
        # Try local TTS first
        payload = {"text": text}
        try:
            response = requests.post(LOCAL_TTS_URL, json=payload)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error calling local TTS: {e}")
            print("Falling back to ElevenLabs...")
            # Fall through to ElevenLabs
    
    # Use ElevenLabs for TTS
    return call_elevenlabs_tts(text)

def call_elevenlabs_tts(text):
    """
    Uses ElevenLabs API to generate speech from text.
    Returns audio bytes or None if there was an error.
    """
    if not ELEVENLABS_API_KEY:
        print("Error: ElevenLabs API key is not set. Please set the ELEVENLABS_API_KEY environment variable.")
        return None
    
    # Prepare the API endpoint with voice ID
    url = f"{ELEVENLABS_API_URL}/{ELEVENLABS_VOICE_ID}"
    
    # Request headers
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    
    # Request payload
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"ElevenLabs TTS generated {len(response.content)} bytes of audio")
        return response.content
    except Exception as e:
        print(f"Error calling ElevenLabs TTS API: {e}")
        return None

# Optional function to list available voices (useful for setup)
def list_elevenlabs_voices():
    if not ELEVENLABS_API_KEY:
        print("Error: ElevenLabs API key is not set")
        return None
    
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        voices = response.json().get("voices", [])
        
        print("Available ElevenLabs voices:")
        for voice in voices:
            print(f"- {voice['name']}: {voice['voice_id']}")
        
        return voices
    except Exception as e:
        print(f"Error listing ElevenLabs voices: {e}")
        return None
