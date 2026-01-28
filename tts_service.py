#!/usr/bin/env python3
"""
Service TTS Qwen3 avec streaming audio temps réel.

Architecture:
- TTSEngine: Gestion du modèle et génération
- AudioPlayer: Lecture audio non-bloquante
- TTSService: Interface haut niveau

Usage:
    from tts_service import TTSService

    tts = TTSService()
    tts.say("Bonjour!")  # Bloquant
    tts.say_async("Au revoir!")  # Non-bloquant
"""

import subprocess
import threading
import queue
import time
import atexit
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


@dataclass
class Voice:
    """Définition d'une voix."""
    name: str
    description: str
    language: str = "french"

    def __str__(self) -> str:
        return f"Voice({self.name})"


# Voix prédéfinies
VOICES: Dict[str, Voice] = {
    "jarvis": Voice(
        name="jarvis",
        description=(
            "A male British voice, calm and composed, with a refined butler-like tone. "
            "Slightly artificial but warm, measured and articulate speech, "
            "sophisticated and helpful, like a personal AI assistant."
        ),
    ),
    "narrator": Voice(
        name="narrator",
        description=(
            "A deep, resonant male voice with gravitas and authority. "
            "Clear enunciation, steady measured pace, rich timbre. "
            "Perfect for audiobooks, documentaries, and storytelling."
        ),
    ),
    "friendly": Voice(
        name="friendly",
        description=(
            "A warm, friendly female voice with natural enthusiasm. "
            "Conversational tone, approachable and engaging. "
            "Speaks with a genuine smile."
        ),
    ),
    "news": Voice(
        name="news",
        description=(
            "A professional male news anchor voice. Clear, authoritative, "
            "neutral tone with perfect diction. Confident and trustworthy."
        ),
    ),
}


class AudioPlayer:
    """Lecteur audio non-bloquant avec queue."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._playing = threading.Event()

    def _player_loop(self) -> None:
        """Boucle de lecture audio."""
        process = subprocess.Popen(
            [
                "mpv",
                "--no-terminal",
                "--no-video",
                f"--audio-samplerate={self.sample_rate}",
                "--audio-channels=mono",
                "--audio-format=s16",
                "--cache=no",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.05)
                    if item is None:
                        break

                    audio_data, is_last = item
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    process.stdin.write(audio_int16.tobytes())
                    process.stdin.flush()

                    if is_last:
                        self._playing.clear()

                except queue.Empty:
                    continue
        except BrokenPipeError:
            pass
        finally:
            try:
                process.stdin.close()
            except:
                pass
            process.wait()

    def start(self) -> None:
        """Démarre le thread de lecture."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._player_loop, daemon=True)
        self._thread.start()

    def play(self, audio: np.ndarray, chunk_ms: int = 100) -> None:
        """
        Envoie l'audio à la queue de lecture.

        Args:
            audio: Données audio (float32, -1 à 1)
            chunk_ms: Taille des chunks en millisecondes
        """
        self.start()
        self._playing.set()

        chunk_size = int(self.sample_rate * chunk_ms / 1000)
        n_chunks = (len(audio) + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]
            is_last = (i == n_chunks - 1)
            self._queue.put((chunk, is_last))

    def wait(self) -> None:
        """Attend la fin de la lecture en cours."""
        while self._playing.is_set() and not self._queue.empty():
            time.sleep(0.05)
        # Attend que le dernier chunk soit joué
        time.sleep(0.1)

    def stop(self) -> None:
        """Arrête la lecture."""
        self._stop_event.set()
        # Vide la queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=1.0)
        self._playing.clear()

    def clear(self) -> None:
        """Vide la queue sans arrêter le player."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class TTSEngine:
    """Moteur TTS Qwen3."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._sample_rate: int = 24000

    @property
    def model(self):
        """Lazy loading du modèle."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _load_model(self) -> None:
        """Charge le modèle."""
        from qwen_tts import Qwen3TTSModel

        print(f"[TTSEngine] Chargement de {self.model_name}...")
        t0 = time.time()

        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=self.dtype,
        )

        print(f"[TTSEngine] Modèle chargé en {time.time() - t0:.2f}s")

    def generate(
        self,
        text: str,
        voice: Voice,
    ) -> tuple[np.ndarray, int]:
        """
        Génère l'audio pour un texte.

        Returns:
            Tuple (audio_array, sample_rate)
        """
        wavs, sr = self.model.generate_voice_design(
            text=text,
            instruct=voice.description,
            language=voice.language,
        )
        self._sample_rate = sr
        return wavs[0], sr

    def warmup(self, voice: Voice) -> None:
        """Warmup du modèle avec une génération courte."""
        print("[TTSEngine] Warmup...")
        self.generate("Test.", voice)
        print("[TTSEngine] Warmup terminé")


class TTSService:
    """
    Service TTS haut niveau.

    Exemple:
        tts = TTSService()
        tts.say("Bonjour!")
        tts.say("Comment allez-vous?", voice="friendly")
    """

    def __init__(
        self,
        default_voice: str = "jarvis",
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        preload: bool = False,
    ):
        self.engine = TTSEngine(model_name=model_name)
        self.player = AudioPlayer()
        self.default_voice = default_voice
        self._voices = VOICES.copy()

        if preload:
            self.warmup()

        atexit.register(self.shutdown)

    def add_voice(self, name: str, description: str, language: str = "french") -> None:
        """Ajoute une voix personnalisée."""
        self._voices[name] = Voice(name=name, description=description, language=language)

    def get_voice(self, name: Optional[str] = None) -> Voice:
        """Récupère une voix par son nom."""
        name = name or self.default_voice
        if name not in self._voices:
            raise ValueError(f"Voix inconnue: {name}. Disponibles: {list(self._voices.keys())}")
        return self._voices[name]

    def list_voices(self) -> list[str]:
        """Liste les voix disponibles."""
        return list(self._voices.keys())

    def warmup(self) -> None:
        """Préchauffe le modèle."""
        self.engine.warmup(self.get_voice())

    def say(
        self,
        text: str,
        voice: Optional[str] = None,
        wait: bool = True,
    ) -> float:
        """
        Dit un texte.

        Args:
            text: Texte à dire
            voice: Nom de la voix (défaut: default_voice)
            wait: Attendre la fin de la lecture

        Returns:
            Latence de génération en secondes
        """
        voice_obj = self.get_voice(voice)

        t0 = time.time()
        audio, sr = self.engine.generate(text, voice_obj)
        latency = time.time() - t0

        self.player.sample_rate = sr
        self.player.play(audio)

        if wait:
            self.player.wait()

        return latency

    def say_async(
        self,
        text: str,
        voice: Optional[str] = None,
        callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Dit un texte de manière asynchrone.

        Args:
            text: Texte à dire
            voice: Nom de la voix
            callback: Appelé avec la latence quand l'audio commence
        """
        def _generate_and_play():
            latency = self.say(text, voice, wait=False)
            if callback:
                callback(latency)

        thread = threading.Thread(target=_generate_and_play, daemon=True)
        thread.start()

    def save(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
    ) -> str:
        """
        Génère et sauvegarde l'audio.

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie
            voice: Nom de la voix

        Returns:
            Chemin du fichier créé
        """
        voice_obj = self.get_voice(voice)
        audio, sr = self.engine.generate(text, voice_obj)
        sf.write(output_path, audio, sr)
        return output_path

    def stop(self) -> None:
        """Arrête la lecture en cours."""
        self.player.stop()

    def shutdown(self) -> None:
        """Arrête proprement le service."""
        self.player.stop()


# Singleton global pour usage simple
_default_service: Optional[TTSService] = None


def get_tts() -> TTSService:
    """Récupère le service TTS global."""
    global _default_service
    if _default_service is None:
        _default_service = TTSService()
    return _default_service


def say(text: str, voice: Optional[str] = None, wait: bool = True) -> float:
    """Raccourci pour get_tts().say()"""
    return get_tts().say(text, voice, wait)


def say_async(text: str, voice: Optional[str] = None) -> None:
    """Raccourci pour get_tts().say_async()"""
    get_tts().say_async(text, voice)


# CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS Service")
    parser.add_argument("text", nargs="?", help="Texte à dire")
    parser.add_argument("-v", "--voice", default="jarvis", help="Voix à utiliser")
    parser.add_argument("-l", "--list-voices", action="store_true", help="Liste les voix")
    parser.add_argument("-o", "--output", help="Sauvegarder dans un fichier")
    parser.add_argument("--bench", action="store_true", help="Benchmark de latence")
    args = parser.parse_args()

    tts = TTSService()

    if args.list_voices:
        print("Voix disponibles:")
        for name, voice in VOICES.items():
            print(f"  {name}: {voice.description[:60]}...")
        return

    if args.bench:
        print("Benchmark de latence...")
        texts = [
            "Bonjour.",
            "Comment allez-vous aujourd'hui?",
            "Je suis votre assistant vocal, prêt à vous aider dans toutes vos tâches quotidiennes.",
        ]
        for text in texts:
            latency = tts.say(text, args.voice)
            print(f"  [{len(text):3d} chars] Latence: {latency:.2f}s")
        return

    if not args.text:
        args.text = "Bonjour, je suis votre assistant vocal Jarvis."

    if args.output:
        tts.save(args.text, args.output, args.voice)
        print(f"Audio sauvegardé: {args.output}")
    else:
        print(f"Voix: {args.voice}")
        print(f"Texte: {args.text}")
        latency = tts.say(args.text, args.voice)
        print(f"Latence: {latency:.2f}s")


if __name__ == "__main__":
    main()
