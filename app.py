import asyncio
import websockets
import sounddevice as sd
import numpy as np
import threading

import ssl
import base64
import json
import queue
import os

from dotenv import load_dotenv

# ==== グローバル設定 ====
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "OpenAI-Beta": "realtime=v1"
}
ssl_context = ssl._create_unverified_context()

audio_send_queue = queue.Queue(maxsize=10)
audio_receive_queue = queue.Queue(maxsize=10)

SAMPLE_RATE = 16000

def base64_to_pcm16(base64_audio: str) -> np.ndarray:
    audio_data = base64.b64decode(base64_audio)
    return np.frombuffer(audio_data, dtype=np.int16)

def pcm16_to_base64(pcm16_audio: np.ndarray) -> str:
    audio_data = pcm16_audio.astype(np.int16).tobytes()
    return base64.b64encode(audio_data).decode("utf-8")

async def send_audio_from_queue(websocket: websockets.WebSocketClientProtocol) -> None:
    while True:
        try:
            audio_data = audio_send_queue.get(timeout=0.1)
            base64_audio = pcm16_to_base64(audio_data)
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            await websocket.send(json.dumps(audio_event))
        except queue.Empty:
            await asyncio.sleep(0.01)

async def receive_audio_to_queue(websocket: websockets.WebSocketClientProtocol) -> None:
    while True:
        response = await websocket.recv()
        response_data = json.loads(response)
        if response_data.get("type") == "response.audio.delta":
            base64_audio_response = response_data.get("delta")
            if base64_audio_response:
                pcm16_audio = base64_to_pcm16(base64_audio_response)
                audio_receive_queue.put(pcm16_audio)
        await asyncio.sleep(0.01)

def record_audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    if status:
        print(f"録音エラー: {status}")
    audio_send_queue.put(indata.copy())

def audio_playback_worker():
    def callback(outdata, frames, time, status):
        if status:
            print(f"再生エラー: {status}")
        try:
            data = audio_receive_queue.get_nowait()
        except queue.Empty:
            outdata.fill(0)
        else:
            outdata[:] = data[:frames] if len(data) >= frames else np.pad(data, (0, frames - len(data)))

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        blocksize=512,
        latency='low',
        callback=callback
    ):
        while True:
            pass

async def stream_audio_and_receive_response() -> None:
    async with websockets.connect(WS_URL, additional_headers=HEADERS, ssl=ssl_context) as websocket:
        print("WebSocketに接続しました。")
        update_request = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": "日本語かつ関西弁で回答してください。",
                "voice": "alloy",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                },
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }
        await websocket.send(json.dumps(update_request))

        play_thread = threading.Thread(target=audio_playback_worker, daemon=True)
        play_thread.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=record_audio_callback,
            blocksize=512,
            device=0,
            latency='low'
        ):
            try:
                send_task = asyncio.create_task(send_audio_from_queue(websocket))
                receive_task = asyncio.create_task(receive_audio_to_queue(websocket))
                await asyncio.gather(send_task, receive_task)
            except KeyboardInterrupt:
                print("KeyboardInterruptによる終了...")
            except Exception as e:
                print("ストリーム開始エラー:", e)

if __name__ == "__main__":
    print("=== デバイス一覧 ===")
    print(sd.query_devices())
    print("=================")

    asyncio.run(stream_audio_and_receive_response())
