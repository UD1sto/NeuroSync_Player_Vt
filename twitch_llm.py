# twitch_llm.py
import os
import time
from threading import Thread, Lock
from queue import Queue, Empty
import pygame
import warnings
warnings.filterwarnings(
    "ignore", 
    message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)

from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.tts.tts_bridge import tts_worker
from utils.files.file_utils import initialize_directories
from utils.llm.chat_utils import load_chat_history, save_chat_log
from utils.llm.llm_utils import stream_llm_chunks 
from utils.audio_face_workers import audio_face_queue_worker
from utils.llm.livepeer_llm_handler import get_livepeer_response


from utils.streamer_utils.twitch_utils import run_twitch_bot, twitch_input_worker

# Configuration for LLM and audio
USE_LOCAL_LLM = False    
USE_STREAMING = True   
LLM_API_URL = "http://127.0.0.1:5050/generate_llama"
LLM_STREAM_URL = "http://127.0.0.1:5050/generate_stream"
VOICE_NAME = 'Lily'

USE_LOCAL_AUDIO = True 

llm_config = {
    "USE_LOCAL_LLM": USE_LOCAL_LLM,
    "USE_STREAMING": USE_STREAMING,
    "USE_LIVEPEER": True,
    "LLM_API_URL": LLM_API_URL,
    "LLM_STREAM_URL": LLM_STREAM_URL,
    "OPENAI_API_KEY": '',
    "max_chunk_length": 500,
    "flush_token_count": 300  
}

def flush_queue(q):
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass

# Twitch configuration from environment variables
TWITCH_NICK = os.getenv("TWITCH_NICK", "serial_hustla")
TWITCH_TOKEN = os.getenv("TWITCH_TOKEN", "oauth:vfplwhmunrocsrdamvcigc3un1s6e9")
TWITCH_CHANNEL = os.getenv("TWITCH_CHANNEL", "serial_hustla")

def main():
    # Initialize directories and connections
    initialize_directories()
    py_face = initialize_py_face()
    socket_connection = create_socket_connection()
    chat_history = load_chat_history()
    
    # Start default face animation
    default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
    default_animation_thread.start()
    
    # Create queues for TTS and audio
    chunk_queue = Queue()
    audio_queue = Queue()
    tts_worker_thread = Thread(target=tts_worker, args=(chunk_queue, audio_queue, USE_LOCAL_AUDIO, VOICE_NAME))
    tts_worker_thread.start()
    audio_worker_thread = Thread(target=audio_face_queue_worker, args=(audio_queue, py_face, socket_connection, default_animation_thread))
    audio_worker_thread.start()
    
    # --- Start Twitch Chat Worker Threads ---
    twitch_queue = Queue()  # Queue for Twitch messages
    llm_lock = Lock()       # Lock to ensure only one LLM call is processed at a time
    
    # Start the Twitch chat bot in a separate thread.
    twitch_bot_thread = Thread(target=run_twitch_bot, args=(twitch_queue, TWITCH_NICK, TWITCH_TOKEN, TWITCH_CHANNEL))
    twitch_bot_thread.daemon = True
    twitch_bot_thread.start()
    
    # Pass the llm_config to the twitch_input_worker (already done above)
    twitch_worker_thread = Thread(
        target=twitch_input_worker,
        args=(twitch_queue, chat_history, chunk_queue, llm_lock, llm_config)
    )
    twitch_worker_thread.daemon = True
    twitch_worker_thread.start()
    
    # --- Text Input Mode ---
    try:
        while True:
            user_input = input("Enter text (or 'q' to quit): ").strip()
            if user_input.lower() == 'q':
                break

            with llm_lock:
                flush_queue(chunk_queue)
                flush_queue(audio_queue)
                if pygame.mixer.get_init():
                    pygame.mixer.stop()

                messages = []
                
                messages.append({"role": "user", "content": user_input})
                
                # Call Livepeer with chunk_queue to enable streaming
                full_response = get_livepeer_response(messages, chunk_queue=chunk_queue, max_tokens=256, temperature=0.7)
                if full_response:
                    chat_history.append({"input": user_input, "response": full_response})
                    # Don't directly queue the full response again since we're streaming chunks
                    save_chat_log(chat_history)

    finally:
        chunk_queue.join()
        chunk_queue.put(None)
        tts_worker_thread.join()
        audio_queue.join()
        audio_queue.put(None)
        audio_worker_thread.join()
        stop_default_animation.set()
        default_animation_thread.join()
        pygame.quit()
        socket_connection.close()

if __name__ == "__main__":
    main()