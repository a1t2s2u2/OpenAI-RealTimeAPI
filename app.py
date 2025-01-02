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

# 環境変数(.env)からAPIキー読み込み
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

# WebSocketのURL・ヘッダー・SSL
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "OpenAI-Beta": "realtime=v1"
}
ssl_context = ssl._create_unverified_context()

# オーディオ送受信キュー
audio_send_queue = queue.Queue()
audio_receive_queue = queue.Queue()

# ==== パラメータ設定 ====
SAMPLE_RATE = 16000

# ==== PCM <-> Base64 変換ユーティリティ ====

def base64_to_pcm16(base64_audio: str) -> np.ndarray:
    """
    Base64エンコードされた音声データをPCM16形式(ndarray)に変換する。
    """
    audio_data = base64.b64decode(base64_audio)
    return np.frombuffer(audio_data, dtype=np.int16)

def pcm16_to_base64(pcm16_audio: np.ndarray) -> str:
    """
    PCM16形式(ndarray)の音声データをBase64文字列に変換する。
    """
    audio_data = pcm16_audio.astype(np.int16).tobytes()
    return base64.b64encode(audio_data).decode("utf-8")

# ==== 非同期送受信処理 ====

async def send_audio_from_queue(websocket: websockets.WebSocketClientProtocol) -> None:
    """
    audio_send_queueから音声データを取り出してBase64に変換し、
    WebSocketに送信し続ける非同期タスク。
    """
    loop = asyncio.get_event_loop()
    while True:
        audio_data = await loop.run_in_executor(None, audio_send_queue.get)
        if audio_data is None:
            continue
        
        base64_audio = pcm16_to_base64(audio_data)
        audio_event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        await websocket.send(json.dumps(audio_event))
        await asyncio.sleep()

async def receive_audio_to_queue(websocket: websockets.WebSocketClientProtocol) -> None:
    """
    WebSocketからサーバー送信される音声データを受信し、PCM16形式に変換して
    audio_receive_queueに格納し続ける非同期タスク。
    """
    while True:
        response = await websocket.recv()
        if response:
            response_data = json.loads(response)
            if response_data.get("type") == "response.audio.delta":
                base64_audio_response = response_data.get("delta")
                if base64_audio_response:
                    pcm16_audio = base64_to_pcm16(base64_audio_response)
                    audio_receive_queue.put(pcm16_audio)
        await asyncio.sleep(0)

# ==== 音声入出力(録音・再生)処理 ====

def record_audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    """
    マイク入力ストリームのコールバック関数。
    取得した音声データをaudio_send_queueに格納する。
    """
    if status:
        print(f"録音エラー: {status}")
    audio_send_queue.put(indata.copy())

def audio_playback_worker():
    """
    OutputStreamを使ってqueueに送られてくる音声を連続再生するワーカー。
    """
    def callback(outdata, frames, time, status):
        """
        OutputStreamの書き込みコールバック。
        queueが空であれば無音（ゼロ埋め）を書き込み、音声データがあればそこから書き込む。
        """
        if status:
            print(f"再生エラー: {status}")
        
        try:
            data = audio_receive_queue.get_nowait()
        except queue.Empty:
            # キューが空なら無音
            outdata.fill(0)
        else:
            # キューのデータが指定の frames より短い可能性があるので、サイズを合わせる
            length = len(data)
            needed = frames
            if length < needed:
                # dataをコピーして足りない分を0埋め
                outdata[:length, 0] = data
                outdata[length:, 0] = 0
            else:
                # frames分だけ書き込む
                outdata[:, 0] = data[:needed]

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        callback=callback
    ):
        # OutputStream開始後、終了まで待機し続けるだけのループ
        while True:
            # 必要に応じて終了条件を設定
            # ここではずっと再生し続ける例
            pass

# ==== メイン処理 ====

async def stream_audio_and_receive_response() -> None:
    """
    WebSocketに接続し、録音・送信・受信・再生を並行して行う。
    """
    async with websockets.connect(WS_URL, additional_headers=HEADERS, ssl=ssl_context) as websocket:
        print("WebSocketに接続しました。")

        # セッション初期化(指示や音声モデルの指定)
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

        # 再生スレッド(ストリーミング再生)を起動
        play_thread = threading.Thread(target=audio_playback_worker, daemon=True)
        play_thread.start()

        # 録音ストリームを開始
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=record_audio_callback,
            device=1  # 環境に応じて変更
        ):
            try:
                # 送信・受信を並行実行
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
