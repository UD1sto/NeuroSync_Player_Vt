import os
import requests
import json
import time
from livepeer_ai import Livepeer
from queue import Queue

LIVEPEER_BEARER_TOKEN = os.getenv("LIVEPEER_BEARER_TOKEN", "eliza-app-llm")
LIVEPEER_GATEWAY_URL = os.getenv("LIVEPEER_GATEWAY_URL", "https://gateway.livepeer-eliza.com")
LIVEPEER_MODEL_NAME = os.getenv("LIVEPEER_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
USE_STREAMING = True  # Changed to True to enable streaming

def get_livepeer_response(messages, chunk_queue=None, max_tokens=256, temperature=0.7):
    """
    Sends the specified 'messages' to Livepeer's LLM endpoint.
    If chunk_queue is provided, chunks will be streamed to it.
    Returns the final text completion.
    """
    # Optionally insert a system message if none present
    system_found = any(msg.get("role") == "system" for msg in messages)
    if not system_found:
        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant"
        })

    request_body = {
        "model": LIVEPEER_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": USE_STREAMING,
        "temperature": temperature
    }

    full_response = ""
    
    # Connect using the Python SDK
    with Livepeer(http_bearer=LIVEPEER_BEARER_TOKEN) as livepeer:
        if USE_STREAMING and chunk_queue:
            # Use direct requests for streaming to have more control
            with requests.post(
                f"{LIVEPEER_GATEWAY_URL}/llm",
                headers={
                    "accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {LIVEPEER_BEARER_TOKEN}"
                },
                json=request_body,
                stream=True
            ) as response:
                if not response.ok:
                    print(f"Error: Livepeer request failed ({response.status_code})")
                    return ""
                
                # Stream chunks as they come in
                buffer = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_response += content
                                        buffer += content
                                        
                                        # Send chunks to TTS when we have a reasonable size
                                        # or when we hit sentence endings
                                        if len(buffer) > 50 or any(x in buffer for x in ['.', '!', '?', '\n']):
                                            print(f"Sending chunk: {buffer}")
                                            chunk_queue.put(buffer)
                                            buffer = ""
                            except json.JSONDecodeError:
                                print(f"Error parsing JSON: {data}")
                
                # Send any remaining text
                if buffer:
                    chunk_queue.put(buffer)
        else:
            # Non-streaming approach (fallback)
            res = livepeer.generate.llm(request=request_body)
            
            # Extract the actual text content from the LLMResponse object
            if hasattr(res, 'choices') and res.choices and len(res.choices) > 0:
                # If response has choices property with message content structure
                message_content = res.choices[0].message.content
                # Remove header if present
                full_response = message_content.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
            else:
                # Fall back to llm_response string if available
                full_response = str(res.llm_response or "")
            
            # If we have a queue but weren't streaming, chunk the response manually
            if chunk_queue:
                # Break response into sentence-like chunks
                chunks = []
                temp = ""
                for char in full_response:
                    temp += char
                    if char in ['.', '!', '?'] and len(temp) > 30:
                        chunks.append(temp)
                        temp = ""
                if temp:
                    chunks.append(temp)
                
                # Send each chunk to the queue with a small delay
                for chunk in chunks:
                    chunk_queue.put(chunk)
                    time.sleep(0.1)  # Small delay between chunks
        
        print(f"Full response: {full_response}")
        return full_response 