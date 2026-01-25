# TTS - Fish Speech Daemon

Local text-to-speech daemon using [Fish Speech](https://github.com/fishaudio/fish-speech) (OpenAudio S1-mini model). Runs as a Unix socket server with voice cloning support.

## Features

- Fast local TTS (~10 tokens/sec on RTX 4070)
- Voice cloning with reference audio
- Streaming audio output
- Unix socket daemon/client architecture
- Multilingual support (English, French, etc.)
- ~5GB VRAM usage

## Requirements

- Python 3.12
- CUDA-capable GPU (8GB+ VRAM recommended)
- PipeWire/PulseAudio for audio playback

## Installation

```bash
# Clone
git clone https://github.com/rmusic/tts.git
cd tts

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Clone Fish Speech
git clone https://github.com/fishaudio/fish-speech.git fish-speech-upstream

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install -e fish-speech-upstream
pip install sounddevice soundfile scipy loguru

# Download model
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir models/openaudio-s1-mini
```

## Usage

### Start the daemon

```bash
./tts-daemon
# Or via systemd
systemctl --user start tts-daemon
```

### CLI

```bash
# Play audio
tts "Hello world"

# Use a cloned voice
tts "Good morning sir" -v jarvis

# Save to file
tts "Hello" -o output.wav

# Pipe to player
tts "Hello" --pipe | mpv -

# Read from stdin
echo "Hello world" | tts --stdin

# List available voices
tts --list-voices
```

### Options

```
-v, --voice      Voice to use (e.g., jarvis)
-o, --output     Output file path
--pipe           Pipe audio to stdout
--stdin          Read text from stdin
--format         Output format: wav, flac, ogg
--temperature    Sampling temperature (default: 0.7)
--top-p          Top-p sampling (default: 0.8)
--max-tokens     Max tokens to generate (default: 1024)
--list-voices    List available voices
```

## Voice Cloning

To add a custom voice, create a folder in `fish-speech-upstream/references/<voice_name>/` with:

- Audio files (`.wav`, `.mp3`, `.flac`)
- Corresponding transcript files (`.lab`) with the same name

Example structure:
```
fish-speech-upstream/references/jarvis/
├── sample1.wav
├── sample1.lab    # "It appears that the Iron Man suit is accelerating your condition."
├── sample2.wav
└── sample2.lab    # "Sir, the reactor has accepted the modified core."
```

Voices are loaded automatically when the daemon starts.

## Systemd Service

```bash
# Install service
mkdir -p ~/.config/systemd/user
cp tts-daemon.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable tts-daemon
systemctl --user start tts-daemon

# Check status
systemctl --user status tts-daemon
```

## Environment Variables

- `TTS_SOCKET` - Socket path (default: `/tmp/tts.sock`)
- `TTS_MODEL` - Model path (default: `./models/openaudio-s1-mini`)
- `TTS_DEVICE` - Device: `cuda` or `cpu` (default: `cuda`)

## Architecture

```
┌─────────┐     Unix Socket     ┌────────────┐
│   CLI   │ ◄─────────────────► │   Daemon   │
│ (client)│    /tmp/tts.sock    │  (server)  │
└─────────┘                     └────────────┘
                                      │
                                      ▼
                               ┌────────────┐
                               │ Fish Speech│
                               │   Model    │
                               └────────────┘
```

## License

MIT
