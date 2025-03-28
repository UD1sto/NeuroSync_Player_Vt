# --- utils/twitch_utils.py ---
import time
import asyncio
from queue import Empty
from twitchio.ext import commands

class TwitchChatBot(commands.Bot):
    """
    TwitchChatBot connects to Twitch using twitchio and enqueues every incoming
    chat message into the provided message_queue.
    """
    def __init__(self, message_queue, nick, token, channel):
        super().__init__(token=token, nick=nick, prefix="!", initial_channels=[channel])
        self.message_queue = message_queue

    async def event_ready(self):
        print(f"Connected to Twitch chat as {self.nick}")

    async def event_message(self, message):
        if message.echo:
            return
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] {message.author.name}: {message.content}"
        self.message_queue.put(formatted)

def run_twitch_bot(message_queue, nick, token, channel):
    """
    Runs the Twitch chat bot on a new asyncio event loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot = TwitchChatBot(message_queue, nick, token, channel)
    bot.run()

def twitch_input_worker(twitch_queue, chat_history, chunk_queue, llm_lock, config):
    """
    Worker that checks the Twitch queue every 0.5 seconds.
    When a new chat message is found, it processes it with the LLM endpoint.
    """
    from utils.llm.llm_utils import stream_llm_chunks
    from utils.llm.chat_utils import save_chat_log
    # Import the Livepeer handler
    from utils.llm.livepeer_llm_handler import get_livepeer_response
    import pygame

    def flush_queue(q):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    while True:
        try:
            twitch_message = twitch_queue.get(timeout=0.5)
            if twitch_message:
                print("\nTwitch chat input detected:")
                print(twitch_message)
                with llm_lock:
                    flush_queue(chunk_queue)
                    if pygame.mixer.get_init():
                        pygame.mixer.stop()
                    
                    # Check if Livepeer should be used
                    if config.get("USE_LIVEPEER", False):
                        # Format message for Livepeer
                        messages = [{"role": "user", "content": twitch_message}]
                        # Use Livepeer handler directly, similar to main.py
                        full_response = get_livepeer_response(
                            messages, 
                            chunk_queue=chunk_queue, 
                            max_tokens=256, 
                            temperature=0.7
                        )
                    else:
                        # Use the original LLM handler
                        full_response = stream_llm_chunks(twitch_message, chat_history, chunk_queue, config=config)
                    
                    chat_history.append({"input": twitch_message, "response": full_response})
                    save_chat_log(chat_history)
        except Empty:
            continue
        except Exception as e:
            print(f"Error in Twitch worker: {str(e)}")
            continue  # Continue processing messages even if one fails 