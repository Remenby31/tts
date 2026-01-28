#!/usr/bin/env python3
"""Test de latence Qwen3-TTS VoiceDesign."""

import torch
import time
from qwen_tts import Qwen3TTSModel

VOICE_DESCRIPTION = """A male British voice, calm and composed, with a refined butler-like tone. Slightly artificial but warm, measured and articulate speech, sophisticated and helpful, like a personal AI assistant."""

# Textes de différentes longueurs
TESTS = [
    ("Court", "Bonjour Monsieur."),
    ("Moyen", "Bonjour Monsieur. Tous les systèmes sont opérationnels."),
    ("Long", "Bonjour Monsieur. Tous les systèmes sont opérationnels. Comment puis-je vous assister aujourd'hui ? J'ai préparé votre agenda et vos messages prioritaires."),
]

print("Chargement du modèle...")
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16
)
print(f"Modèle chargé en {time.time() - t0:.2f}s\n")

print("=" * 60)
print(f"{'Test':<10} {'Chars':<8} {'Latence':<12} {'Audio':<10} {'RTF':<8}")
print("=" * 60)

for name, text in TESTS:
    # Warmup pour le premier
    if name == "Court":
        _ = model.generate_voice_design(text=text, instruct=VOICE_DESCRIPTION, language="french")

    # Mesure
    t0 = time.time()
    wavs, sr = model.generate_voice_design(
        text=text,
        instruct=VOICE_DESCRIPTION,
        language="french",
    )
    latency = time.time() - t0

    audio_duration = len(wavs[0]) / sr
    rtf = latency / audio_duration  # Real-Time Factor (<1 = plus rapide que temps réel)

    print(f"{name:<10} {len(text):<8} {latency:.2f}s{'':<6} {audio_duration:.2f}s{'':<5} {rtf:.2f}x")

print("=" * 60)
print("\nRTF = Real-Time Factor (< 1.0 = plus rapide que temps réel)")
