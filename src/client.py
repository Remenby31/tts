#!/usr/bin/env python3
"""
TTS Client - Communicates with daemon, plays/outputs audio
"""

import argparse
import io
import json
import os
import socket
import struct
import sys
import threading
import queue
from pathlib import Path
from typing import Optional

import numpy as np

SOCKET_PATH = os.environ.get("TTS_SOCKET", "/tmp/tts.sock")
CHUNK_SIZE = 4096


class TTSClient:
    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.audio_queue = queue.Queue()
        self.sample_rate = None

    def connect(self) -> socket.socket:
        """Connect to daemon"""
        if not os.path.exists(self.socket_path):
            raise ConnectionError(f"Daemon not running (socket not found: {self.socket_path})")

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        return sock

    def list_voices(self) -> list:
        """List available voices"""
        sock = self.connect()

        try:
            request = {"command": "list_voices", "text": ""}
            data = json.dumps(request).encode()
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Read response
            header_len_data = self._recv_exact(sock, 4)
            header_len = struct.unpack(">I", header_len_data)[0]
            header_data = self._recv_exact(sock, header_len)
            header = json.loads(header_data.decode())

            return header.get("voices", [])
        finally:
            sock.close()

    def generate(
        self,
        text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        voice: str = None,
    ):
        """Generate audio from text, yields audio chunks"""
        sock = self.connect()

        try:
            # Send request
            request = {
                "text": text,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            }
            if voice:
                request["voice"] = voice
            data = json.dumps(request).encode()
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)

            # Receive responses
            while True:
                # Read header length
                header_len_data = self._recv_exact(sock, 4)
                if not header_len_data:
                    break

                header_len = struct.unpack(">I", header_len_data)[0]
                header_data = self._recv_exact(sock, header_len)
                header = json.loads(header_data.decode())

                if header.get("type") == "audio":
                    sample_rate = header["sample_rate"]
                    samples = header["samples"]
                    audio_bytes = self._recv_exact(sock, samples * 4)  # float32
                    audio = np.frombuffer(audio_bytes, dtype=np.float32)

                    yield {
                        "type": "audio",
                        "data": audio,
                        "sample_rate": sample_rate,
                    }

                elif header.get("type") == "done":
                    yield {"type": "done"}
                    break

                elif header.get("error"):
                    raise RuntimeError(header["error"])

        finally:
            sock.close()

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes"""
        data = b""
        while len(data) < n:
            chunk = sock.recv(min(n - len(data), CHUNK_SIZE))
            if not chunk:
                return None
            data += chunk
        return data


def play_audio_stream(client: TTSClient, text: str, **kwargs):
    """Play audio as it's generated using sounddevice"""
    import sounddevice as sd

    audio_queue = queue.Queue()
    sample_rate = None
    done = threading.Event()

    def audio_callback(outdata, frames, time_info, status):
        """Callback for sounddevice stream"""
        try:
            data = audio_queue.get_nowait()
            if data is None:
                raise sd.CallbackStop()

            if len(data) < frames:
                outdata[:len(data), 0] = data
                outdata[len(data):] = 0
            else:
                outdata[:, 0] = data[:frames]
                # Put back remaining
                if len(data) > frames:
                    audio_queue.put(data[frames:])
        except queue.Empty:
            outdata.fill(0)

    def generator_thread():
        nonlocal sample_rate
        buffer = []
        buffer_size = 0
        min_buffer = 4096  # Min samples before starting playback

        for chunk in client.generate(text, **kwargs):
            if chunk["type"] == "audio":
                if sample_rate is None:
                    sample_rate = chunk["sample_rate"]

                audio = chunk["data"]
                buffer.append(audio)
                buffer_size += len(audio)

                # Once we have enough, start feeding queue
                if buffer_size >= min_buffer:
                    combined = np.concatenate(buffer)
                    # Feed in chunks
                    chunk_size = 1024
                    for i in range(0, len(combined), chunk_size):
                        audio_queue.put(combined[i:i+chunk_size])
                    buffer = []
                    buffer_size = 0

            elif chunk["type"] == "done":
                # Flush remaining buffer
                if buffer:
                    combined = np.concatenate(buffer)
                    chunk_size = 1024
                    for i in range(0, len(combined), chunk_size):
                        audio_queue.put(combined[i:i+chunk_size])

                audio_queue.put(None)  # Signal end
                done.set()
                break

    # Start generator in thread
    gen_thread = threading.Thread(target=generator_thread)
    gen_thread.start()

    # Wait for sample rate
    while sample_rate is None and not done.is_set():
        import time
        time.sleep(0.01)

    if sample_rate is None:
        gen_thread.join()
        return

    # Play audio
    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=1024,
        ):
            done.wait()
            # Wait for queue to drain
            while not audio_queue.empty():
                import time
                time.sleep(0.1)
            import time
            time.sleep(0.5)  # Extra buffer
    except Exception as e:
        print(f"Audio error: {e}", file=sys.stderr)

    gen_thread.join()


def save_audio(client: TTSClient, text: str, output_path: str, format: str = "wav", **kwargs):
    """Generate and save audio to file"""
    import soundfile as sf

    all_audio = []
    sample_rate = None

    for chunk in client.generate(text, **kwargs):
        if chunk["type"] == "audio":
            all_audio.append(chunk["data"])
            sample_rate = chunk["sample_rate"]
        elif chunk["type"] == "done":
            break

    if not all_audio:
        print("No audio generated", file=sys.stderr)
        return

    audio = np.concatenate(all_audio)

    sf.write(output_path, audio, sample_rate, format=format)
    print(f"Saved to {output_path}", file=sys.stderr)


def pipe_audio(client: TTSClient, text: str, format: str = "wav", **kwargs):
    """Generate and pipe audio to stdout"""
    import soundfile as sf

    all_audio = []
    sample_rate = None

    for chunk in client.generate(text, **kwargs):
        if chunk["type"] == "audio":
            all_audio.append(chunk["data"])
            sample_rate = chunk["sample_rate"]
        elif chunk["type"] == "done":
            break

    if not all_audio:
        return

    audio = np.concatenate(all_audio)

    # Write to stdout as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=format)
    sys.stdout.buffer.write(buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(
        description="TTS Client - Text to Speech via Fish Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tts "Hello world"                    # Play audio
  tts "Hello world" -v jarvis          # Use Jarvis voice
  tts "Hello world" -o output.wav      # Save to file
  tts "Hello world" --pipe | mpv -     # Pipe to player
  echo "Hello" | tts --stdin           # Read from stdin
  tts --list-voices                    # List available voices
"""
    )

    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-v", "--voice", help="Voice to use (e.g., jarvis)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--pipe", action="store_true", help="Pipe audio to stdout")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--stream", action="store_true", help="Stream stdin (process line by line)")
    parser.add_argument("--format", default="wav", choices=["wav", "flac", "ogg"], help="Output format")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Daemon socket path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")

    args = parser.parse_args()

    client = TTSClient(args.socket)

    # Handle list-voices command
    if args.list_voices:
        try:
            voices = client.list_voices()
            if voices:
                print("Available voices:")
                for v in voices:
                    print(f"  - {v}")
            else:
                print("No voices available")
        except ConnectionError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Get text
    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    if not text:
        print("No text provided", file=sys.stderr)
        sys.exit(1)

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "voice": args.voice,
    }

    try:
        if args.output:
            save_audio(client, text, args.output, args.format, **gen_kwargs)
        elif args.pipe:
            pipe_audio(client, text, args.format, **gen_kwargs)
        else:
            play_audio_stream(client, text, **gen_kwargs)

    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Is the daemon running? Start it with: tts-daemon", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
