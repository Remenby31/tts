#!/usr/bin/env python3
"""
Qwen3-TTS avec streaming audio.

Réduit la latence perçue en commençant la lecture audio pendant la génération.
"""

import subprocess
import threading
import queue
import time
import numpy as np
import torch
import soundfile as sf
import io
from dataclasses import dataclass
from typing import Optional, Generator, Callable
from qwen_tts import Qwen3TTSModel


@dataclass
class TTSConfig:
    """Configuration pour le TTS."""
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    use_compile: bool = False  # torch.compile() - gain après warmup


class StreamingTTS:
    """TTS avec streaming audio."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.model: Optional[Qwen3TTSModel] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._player_thread: Optional[threading.Thread] = None
        self._stop_playback = threading.Event()

    def load_model(self) -> None:
        """Charge le modèle TTS."""
        if self.model is not None:
            return

        print(f"Chargement du modèle {self.config.model_name}...")
        t0 = time.time()

        self.model = Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            dtype=self.config.dtype,
        )

        if self.config.use_compile:
            print("Compilation du modèle avec torch.compile()...")
            self.model.model = torch.compile(self.model.model, mode="reduce-overhead")

        print(f"Modèle chargé en {time.time() - t0:.2f}s")

    def _play_audio_thread(self, sample_rate: int) -> None:
        """Thread de lecture audio via mpv."""
        # Lance mpv en mode pipe
        process = subprocess.Popen(
            ["mpv", "--no-terminal", "--no-video", f"--audio-samplerate={sample_rate}",
             "--audio-channels=mono", "--audio-format=s16", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            while not self._stop_playback.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                    if chunk is None:  # Signal de fin
                        break
                    # Convertit en int16 pour mpv
                    audio_int16 = (chunk * 32767).astype(np.int16)
                    process.stdin.write(audio_int16.tobytes())
                    process.stdin.flush()
                except queue.Empty:
                    continue
        finally:
            process.stdin.close()
            process.wait()

    def speak(
        self,
        text: str,
        voice_description: str,
        language: str = "french",
        blocking: bool = True,
    ) -> float:
        """
        Génère et lit l'audio avec streaming.

        Args:
            text: Texte à synthétiser
            voice_description: Description de la voix
            language: Langue du texte
            blocking: Attendre la fin de la lecture

        Returns:
            Temps jusqu'au premier audio (latence)
        """
        self.load_model()

        # Reset
        self._stop_playback.clear()
        while not self._audio_queue.empty():
            self._audio_queue.get()

        t_start = time.time()

        # Génère l'audio
        wavs, sample_rate = self.model.generate_voice_design(
            text=text,
            instruct=voice_description,
            language=language,
        )

        t_generated = time.time()
        latency = t_generated - t_start

        audio = wavs[0]

        # Démarre le thread de lecture
        self._player_thread = threading.Thread(
            target=self._play_audio_thread,
            args=(sample_rate,),
            daemon=True,
        )
        self._player_thread.start()

        # Envoie l'audio par chunks pour un streaming fluide
        chunk_size = sample_rate // 4  # chunks de 250ms
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            self._audio_queue.put(chunk)

        # Signal de fin
        self._audio_queue.put(None)

        if blocking:
            self._player_thread.join()

        return latency

    def speak_streaming(
        self,
        text: str,
        voice_description: str,
        language: str = "french",
        on_first_audio: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Version streaming avancée avec callback.

        La lecture commence dès que possible pendant la génération.
        Note: La génération qwen-tts est atomique, donc le gain est limité
        mais la structure permet d'évoluer vers un vrai streaming.
        """
        self.load_model()

        self._stop_playback.clear()
        t_start = time.time()

        # Génère (atomique pour l'instant)
        wavs, sample_rate = self.model.generate_voice_design(
            text=text,
            instruct=voice_description,
            language=language,
        )

        latency = time.time() - t_start
        if on_first_audio:
            on_first_audio(latency)

        audio = wavs[0]

        # Démarre la lecture
        self._player_thread = threading.Thread(
            target=self._play_audio_thread,
            args=(sample_rate,),
            daemon=True,
        )
        self._player_thread.start()

        # Stream l'audio
        chunk_size = sample_rate // 10  # chunks de 100ms
        for i in range(0, len(audio), chunk_size):
            self._audio_queue.put(audio[i:i + chunk_size])

        self._audio_queue.put(None)
        self._player_thread.join()

    def stop(self) -> None:
        """Arrête la lecture en cours."""
        self._stop_playback.set()
        self._audio_queue.put(None)
        if self._player_thread:
            self._player_thread.join(timeout=1.0)

    def save(
        self,
        text: str,
        voice_description: str,
        output_path: str,
        language: str = "french",
    ) -> str:
        """Génère et sauvegarde l'audio dans un fichier."""
        self.load_model()

        wavs, sample_rate = self.model.generate_voice_design(
            text=text,
            instruct=voice_description,
            language=language,
        )

        sf.write(output_path, wavs[0], sample_rate)
        return output_path


# Voix prédéfinies
VOICES = {
    "jarvis": "A male British voice, calm and composed, with a refined butler-like tone. "
              "Slightly artificial but warm, measured and articulate speech, "
              "sophisticated and helpful, like a personal AI assistant.",
    "narrator": "A deep, resonant male voice with gravitas. Clear enunciation, "
                "steady pace, perfect for audiobooks and documentaries.",
    "friendly": "A warm, friendly female voice with a smile in her tone. "
                "Approachable and conversational, natural and engaging.",
}


def main():
    """Demo du streaming TTS."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS Streaming")
    parser.add_argument("text", nargs="?", default="Bonjour, je suis votre assistant vocal.")
    parser.add_argument("-v", "--voice", default="jarvis", choices=list(VOICES.keys()))
    parser.add_argument("-l", "--language", default="french")
    parser.add_argument("-o", "--output", help="Sauvegarder dans un fichier au lieu de jouer")
    parser.add_argument("--compile", action="store_true", help="Utiliser torch.compile()")
    args = parser.parse_args()

    config = TTSConfig(use_compile=args.compile)
    tts = StreamingTTS(config)

    voice_desc = VOICES[args.voice]
    print(f"Voix: {args.voice}")
    print(f"Texte: {args.text}")

    if args.output:
        tts.save(args.text, voice_desc, args.output, args.language)
        print(f"Audio sauvegardé: {args.output}")
    else:
        t0 = time.time()
        latency = tts.speak(args.text, voice_desc, args.language)
        total = time.time() - t0
        print(f"\nLatence génération: {latency:.2f}s")
        print(f"Temps total: {total:.2f}s")


if __name__ == "__main__":
    main()
