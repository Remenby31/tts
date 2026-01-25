#!/usr/bin/env python3
"""
TTS Daemon - Fish Speech inference server
Listens on Unix socket, streams audio back to clients
"""

import asyncio
import json
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import soundfile as sf
from loguru import logger

# Add fish-speech to path
FISH_SPEECH_PATH = Path(__file__).parent.parent / "fish-speech-upstream"
sys.path.insert(0, str(FISH_SPEECH_PATH))
os.chdir(FISH_SPEECH_PATH)  # Needed for hydra config path

from fish_speech.models.text2semantic.inference import (
    GenerateResponse,
    init_model,
    generate_long,
)
from fish_speech.models.dac.inference import load_model as load_decoder_model


SOCKET_PATH = os.environ.get("TTS_SOCKET", "/tmp/tts.sock")
MODEL_PATH = os.environ.get("TTS_MODEL", str(Path(__file__).parent.parent / "models/openaudio-s1-mini"))
DEVICE = os.environ.get("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PRECISION = torch.bfloat16


REFERENCES_PATH = FISH_SPEECH_PATH / "references"


class TTSDaemon:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.decoder = None
        self.decode_one_token = None
        self.running = False
        self.voice_cache: Dict[str, Tuple[List[torch.Tensor], List[str]]] = {}

    def load_models(self):
        """Load TTS and decoder models"""
        logger.info(f"Loading model from {self.model_path}")
        t0 = time.time()

        # Load text-to-semantic model
        self.model, self.decode_one_token = init_model(
            self.model_path,
            self.device,
            PRECISION,
            compile=False  # Disable torch.compile for faster startup
        )

        # Setup KV cache
        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=PRECISION,
            )

        # Load decoder (DAC) using hydra config
        codec_path = self.model_path / "codec.pth"
        self.decoder = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(codec_path),
            device=self.device
        )

        logger.info(f"Models loaded in {time.time() - t0:.2f}s")
        logger.info(f"Sample rate: {self.decoder.sample_rate}")

        # Load voice references
        self._load_voice_references()

        # Warmup
        logger.info("Warming up...")
        self._warmup()
        logger.info("Ready!")

    def _load_voice_references(self):
        """Load and encode all voice references from the references folder"""
        if not REFERENCES_PATH.exists():
            logger.info("No references folder found")
            return

        for voice_dir in REFERENCES_PATH.iterdir():
            if not voice_dir.is_dir():
                continue

            voice_id = voice_dir.name
            try:
                prompt_tokens, prompt_texts = self._encode_voice_reference(voice_dir)
                if prompt_tokens:
                    self.voice_cache[voice_id] = (prompt_tokens, prompt_texts)
                    logger.info(f"Loaded voice reference: {voice_id} ({len(prompt_tokens)} samples)")
            except Exception as e:
                logger.warning(f"Failed to load voice reference {voice_id}: {e}")

        logger.info(f"Loaded {len(self.voice_cache)} voice references")

    def _encode_voice_reference(self, voice_dir: Path) -> Tuple[List[torch.Tensor], List[str]]:
        """Encode audio files in a voice reference directory"""
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
        prompt_tokens = []
        prompt_texts = []

        for audio_file in sorted(voice_dir.iterdir()):
            if audio_file.suffix.lower() not in audio_extensions:
                continue

            lab_file = audio_file.with_suffix(".lab")
            if not lab_file.exists():
                continue

            # Load transcript
            transcript = lab_file.read_text().strip()

            # Load audio using soundfile
            audio_data, sr = sf.read(str(audio_file))
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # Convert stereo to mono

            # Resample if needed
            if sr != self.decoder.sample_rate:
                import scipy.signal
                num_samples = int(len(audio_data) * self.decoder.sample_rate / sr)
                audio_data = scipy.signal.resample(audio_data, num_samples)

            waveform = torch.from_numpy(audio_data.astype(np.float32))

            # Encode to VQ tokens
            audio = waveform.to(self.device)[None, None, :]  # [1, 1, samples]
            audio_lengths = torch.tensor([audio.shape[2]], device=self.device, dtype=torch.long)

            with torch.no_grad():
                tokens = self.decoder.encode(audio, audio_lengths)[0][0]

            prompt_tokens.append(tokens)
            prompt_texts.append(transcript)

        return prompt_tokens, prompt_texts

    def get_voice(self, voice_id: str) -> Tuple[Optional[List[torch.Tensor]], Optional[List[str]]]:
        """Get cached voice reference by ID"""
        if voice_id in self.voice_cache:
            return self.voice_cache[voice_id]
        return None, None

    def list_voices(self) -> List[str]:
        """List available voice IDs"""
        return list(self.voice_cache.keys())

    def _warmup(self):
        """Warmup inference to compile kernels"""
        try:
            for _ in self.generate("Hello", max_new_tokens=50):
                pass
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        voice: Optional[str] = None,
    ):
        """Generate audio from text, yields audio chunks"""

        prompt_tokens_list = None
        prompt_text_list = None

        # Load voice reference if specified
        if voice:
            prompt_tokens_list, prompt_text_list = self.get_voice(voice)
            if prompt_tokens_list is None:
                logger.warning(f"Voice '{voice}' not found, using default")

        # Generate semantic tokens
        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=False,
            chunk_length=0,
            prompt_text=prompt_text_list,
            prompt_tokens=prompt_tokens_list,
        )

        for response in generator:
            if response.action == "sample":
                # Decode codes to audio
                codes = response.codes
                if codes is not None:
                    # codes shape: [num_codebooks, seq_len]
                    feature_lengths = torch.tensor(
                        [codes.shape[1]], device=self.device
                    )

                    with torch.no_grad():
                        audio, _ = self.decoder.decode(
                            indices=codes[None],
                            feature_lengths=feature_lengths,
                        )
                        audio = audio.squeeze().cpu().numpy()

                    yield {
                        "type": "audio",
                        "data": audio,
                        "sample_rate": self.decoder.sample_rate,
                    }
            elif response.action == "next":
                yield {"type": "done"}

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a client connection"""
        addr = writer.get_extra_info('peername') or "unix"
        logger.debug(f"Client connected: {addr}")

        try:
            # Read request header (4 bytes length + json)
            header = await reader.readexactly(4)
            length = struct.unpack(">I", header)[0]

            data = await reader.readexactly(length)
            request = json.loads(data.decode())

            # Handle special commands first
            command = request.get("command")
            if command == "list_voices":
                voices = self.list_voices()
                response = {"type": "voices", "voices": voices}
                header = json.dumps(response).encode()
                writer.write(struct.pack(">I", len(header)))
                writer.write(header)
                await writer.drain()
                return

            text = request.get("text", "")
            if not text:
                error = json.dumps({"error": "No text provided"}).encode()
                writer.write(struct.pack(">I", len(error)))
                writer.write(error)
                await writer.drain()
                return

            voice = request.get("voice")
            if voice:
                logger.info(f"Generating with voice '{voice}': {text[:50]}...")
            else:
                logger.info(f"Generating: {text[:50]}...")

            # Generate and stream audio
            for chunk in self.generate(
                text=text,
                max_new_tokens=request.get("max_new_tokens", 1024),
                temperature=request.get("temperature", 0.7),
                top_p=request.get("top_p", 0.8),
                repetition_penalty=request.get("repetition_penalty", 1.1),
                voice=voice,
            ):
                if chunk["type"] == "audio":
                    audio = chunk["data"]
                    sample_rate = chunk["sample_rate"]

                    # Send audio chunk
                    response = {
                        "type": "audio",
                        "sample_rate": sample_rate,
                        "samples": len(audio),
                    }
                    header = json.dumps(response).encode()
                    writer.write(struct.pack(">I", len(header)))
                    writer.write(header)
                    writer.write(audio.astype(np.float32).tobytes())
                    await writer.drain()

                elif chunk["type"] == "done":
                    response = {"type": "done"}
                    header = json.dumps(response).encode()
                    writer.write(struct.pack(">I", len(header)))
                    writer.write(header)
                    await writer.drain()

        except asyncio.IncompleteReadError:
            logger.debug("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            import traceback
            traceback.print_exc()
            try:
                error = json.dumps({"error": str(e)}).encode()
                writer.write(struct.pack(">I", len(error)))
                writer.write(error)
                await writer.drain()
            except:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def run(self):
        """Run the daemon"""
        # Remove stale socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        self.running = True

        # Load models
        self.load_models()

        # Start server
        server = await asyncio.start_unix_server(
            self.handle_client,
            path=SOCKET_PATH
        )

        # Set socket permissions
        os.chmod(SOCKET_PATH, 0o666)

        logger.info(f"Listening on {SOCKET_PATH}")

        async with server:
            await server.serve_forever()

    def stop(self):
        """Stop the daemon"""
        self.running = False
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


def main():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )

    daemon = TTSDaemon(MODEL_PATH, DEVICE)

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
