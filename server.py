import os
import asyncio
import json
import torch
import torchaudio
import base64
import io
import websockets
import logging
import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import CSM components
from generator import load_csm_1b, Segment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Triton compilation for CSM
os.environ["NO_TORCH_COMPILE"] = "1"

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
WHISPER_SERVICE_URL = os.getenv("WHISPER_SERVICE_URL", "ws://localhost:8000/ws/transcribe")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Global variables
connected_clients = {}
csm_generator = None
conversation_contexts = {}  # Store contexts per client
openai_client = None

async def connect_to_whisper(audio_data):
    """Connect to external Whisper service and get transcription"""
    try:
        logger.info(f"Connecting to Whisper service at {WHISPER_SERVICE_URL}")
        async with websockets.connect(WHISPER_SERVICE_URL) as whisper_ws:
            # Use send() instead of send_bytes()
            await whisper_ws.send(audio_data)
            
            # Receive the transcription response
            response = await whisper_ws.recv()
            logger.info(f"Received response from Whisper: {response}")
            
            # Parse JSON response if it's a string
            if isinstance(response, str):
                try:
                    result = json.loads(response)
                    return result.get("text", "")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response: {response}")
                    return response if response else ""
            else:
                # If it's binary data, just return empty string
                logger.error("Received binary response from Whisper")
                return ""
            
    except Exception as e:
        logger.error(f"Error connecting to Whisper service: {str(e)}")
        return ""

async def generate_llm_response(text, client_id):
    """Generate response using OpenAI API"""
    global openai_client
    
    # Build conversation history
    history = conversation_contexts.get(client_id, [])
    messages = []
    
    # Add conversation history (last few exchanges)
    for segment in history[-6:]:
        role = "user" if segment.speaker == 0 else "assistant"
        messages.append({"role": role, "content": segment.text})
    
    # Add current user message
    messages.append({"role": "user", "content": text})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or gpt-4 depending on needs
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "I'm having trouble connecting. Could you try again?"

def encode_audio(audio_tensor, sample_rate=24000):
    """Convert audio tensor to base64 encoded WAV"""
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), sample_rate, format="wav")
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def load_models():
    """Initialize CSM and other models"""
    global csm_generator, openai_client
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CSM model on {device}...")
    
    # Load the CSM model
    csm_generator = load_csm_1b(device=device)
    logger.info("CSM model loaded successfully")
    
    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    # Log environment configuration
    logger.info(f"Whisper Service URL: {WHISPER_SERVICE_URL}")
    logger.info(f"OpenAI API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")
    
    # Load models
    await asyncio.to_thread(load_models)

@app.websocket("/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connected_clients[client_id] = websocket
    
    # Initialize conversation context for this client
    if client_id not in conversation_contexts:
        conversation_contexts[client_id] = []
    
    try:
        logger.info(f"Client {client_id} connected")
        while True:
            # Receive audio data from client
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio":
                # Decode base64 audio
                audio_bytes = base64.b64decode(data.get("data", ""))
                logger.info(f"Received {len(audio_bytes)} bytes of audio from client")
                
                # If the length is zero or very small, there's a problem with the client recording
                if len(audio_bytes) < 1000:
                    logger.error(f"Audio data too small: {len(audio_bytes)} bytes")
                    await websocket.send_json({"type": "error", "message": "Audio data too small"})
                    continue
                
                # Get transcription from Whisper service
                transcription = await connect_to_whisper(audio_bytes)
                logger.info(f"User said: {transcription}")
                
                # Skip processing if no transcription
                if not transcription:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Could not transcribe audio"
                    })
                    continue
                
                # Get LLM response from OpenAI
                llm_response = await generate_llm_response(transcription, client_id)
                logger.info(f"Bot response: {llm_response}")
                
                # Generate audio with CSM
                context = conversation_contexts[client_id]
                audio = await asyncio.to_thread(
                    csm_generator.generate,
                    text=llm_response,
                    speaker=1,
                    context=context,
                    max_audio_length_ms=10_000,
                )
                
                # Create user segment (without audio for simplicity)
                user_segment = Segment(
                    text=transcription,
                    speaker=0,
                    audio=torch.zeros(1000)  # placeholder, ideally extract from input
                )
                
                # Create bot segment
                bot_segment = Segment(
                    text=llm_response,
                    speaker=1,
                    audio=audio
                )
                
                # Update context (maintain last 10 exchanges)
                conversation_contexts[client_id].append(user_segment)
                conversation_contexts[client_id].append(bot_segment)
                if len(conversation_contexts[client_id]) > 20:
                    conversation_contexts[client_id] = conversation_contexts[client_id][-20:]
                
                # Encode and send audio response
                encoded_audio = encode_audio(audio)
                await websocket.send_json({
                    "type": "response",
                    "text": llm_response,
                    "audio": encoded_audio
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        if client_id in connected_clients:
            del connected_clients[client_id]
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        if client_id in connected_clients:
            del connected_clients[client_id]

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, log_level="info")