#!/usr/bin/env python3
"""Test Qwen3-TTS VoiceDesign avec une voix style Jarvis."""

import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf

# Description de la voix style Jarvis
VOICE_DESCRIPTION = """A male British voice, calm and composed, with a refined butler-like tone. Slightly artificial but warm, measured and articulate speech, sophisticated and helpful, like a personal AI assistant."""

# Texte à synthétiser
TEXT = "Bonjour Monsieur. Tous les systèmes sont opérationnels. Comment puis-je vous assister aujourd'hui ?"

print("Chargement du modèle VoiceDesign 1.7B...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16
)

print(f"Génération avec description:\n{VOICE_DESCRIPTION.strip()}\n")
print(f"Texte: {TEXT}\n")

# Génération avec l'API correcte
wavs, sample_rate = model.generate_voice_design(
    text=TEXT,
    instruct=VOICE_DESCRIPTION,
    language="french",
)

# Sauvegarde
output_file = "jarvis_test.wav"
sf.write(output_file, wavs[0], sample_rate)
print(f"Audio sauvegardé: {output_file}")
