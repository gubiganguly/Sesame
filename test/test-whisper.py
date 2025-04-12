import asyncio
import websockets
import json
import wave
import sys

async def test_whisper():
    uri = "ws://localhost:8000/ws/transcribe"
    
    # Use a test audio file
    with open("test.wav", "rb") as f:
        audio_data = f.read()
    
    print(f"Connecting to {uri} with {len(audio_data)} bytes of audio")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected, sending audio data...")
            
            # Send as raw binary data
            await websocket.send(audio_data)
            print("Audio sent, waiting for response...")
            
            # Add a longer timeout
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            print(f"Received response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Create a simple test wave file if it doesn't exist
    try:
        with open("test.wav", "rb") as f:
            pass
    except FileNotFoundError:
        print("Creating test audio file...")
        with wave.open("test.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # Create 1 second of silence followed by a sine wave
            for i in range(16000):
                value = 0 if i < 8000 else int(32767 * 0.5 * (i % 100) / 100)
                wf.writeframes(value.to_bytes(2, byteorder='little', signed=True))
    
    asyncio.run(test_whisper())