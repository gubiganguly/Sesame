import asyncio
import websockets
import json
import wave
import sys

async def test_whisper():
    uri = "ws://localhost:8000/ws/transcribe"
    
    # Use the provided speech sample
    audio_file = "test-speech.wav"
    
    try:
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        print(f"Connecting to {uri} with {len(audio_data)} bytes of audio from {audio_file}")
        
        async with websockets.connect(uri) as websocket:
            print("Connected, sending audio data...")
            
            # Send as raw binary data
            await websocket.send(audio_data)
            print("Audio sent, waiting for response...")
            
            # Add a longer timeout for processing longer audio
            response = await asyncio.wait_for(websocket.recv(), timeout=60.0)
            
            # Parse and display the response
            try:
                result = json.loads(response)
                print(f"Response type: {result.get('type', 'unknown')}")
                
                if result.get('text'):
                    print(f"\nTranscription result: \"{result['text']}\"\n")
                elif result.get('error'):
                    print(f"Error: {result['error']}")
                else:
                    print(f"Full response: {response}")
            except json.JSONDecodeError:
                print(f"Raw response (not JSON): {response}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_whisper())