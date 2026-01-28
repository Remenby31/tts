#!/usr/bin/env python3
"""
Daemon TTS avec socket Unix pour latence minimale.

Le modèle reste chargé en mémoire, seule la génération est effectuée à chaque requête.

Usage serveur:
    python tts_daemon.py serve

Usage client:
    python tts_daemon.py say "Bonjour!"
    python tts_daemon.py say "Hello!" --voice narrator --language english

Ou via API Python:
    from tts_daemon import TTSClient
    client = TTSClient()
    client.say("Bonjour!")
"""

import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional

SOCKET_PATH = "/tmp/qwen_tts.sock"
PID_FILE = "/tmp/qwen_tts.pid"


class TTSDaemon:
    """Daemon TTS avec socket Unix."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.tts = None

    def _init_tts(self):
        """Initialise le service TTS."""
        from tts_service import TTSService
        print("[Daemon] Initialisation du TTS...")
        self.tts = TTSService(preload=False)
        print("[Daemon] Warmup...")
        self.tts.warmup()
        print("[Daemon] Prêt!")

    def _handle_client(self, conn: socket.socket):
        """Gère une connexion client."""
        try:
            # Lit la taille du message (4 bytes, big endian)
            size_data = conn.recv(4)
            if not size_data:
                return

            msg_size = struct.unpack(">I", size_data)[0]
            data = conn.recv(msg_size).decode("utf-8")
            request = json.loads(data)

            command = request.get("command", "say")
            text = request.get("text", "")
            voice = request.get("voice")
            language = request.get("language", "french")

            response = {"status": "ok"}

            if command == "say":
                t0 = time.time()

                # Met à jour la langue de la voix si spécifiée
                voice_obj = self.tts.get_voice(voice)
                if language != voice_obj.language:
                    # Crée une copie temporaire avec la nouvelle langue
                    from tts_service import Voice
                    voice_obj = Voice(
                        name=voice_obj.name,
                        description=voice_obj.description,
                        language=language,
                    )
                    self.tts._voices[f"_temp_{voice_obj.name}"] = voice_obj
                    voice = f"_temp_{voice_obj.name}"

                latency = self.tts.say(text, voice=voice, wait=True)
                response["latency"] = latency
                response["total_time"] = time.time() - t0

            elif command == "ping":
                response["message"] = "pong"

            elif command == "voices":
                response["voices"] = self.tts.list_voices()

            elif command == "stop":
                self.tts.stop()

            elif command == "shutdown":
                response["message"] = "shutting down"
                self.running = False

            else:
                response = {"status": "error", "message": f"Unknown command: {command}"}

            # Envoie la réponse
            response_data = json.dumps(response).encode("utf-8")
            conn.sendall(struct.pack(">I", len(response_data)) + response_data)

        except Exception as e:
            error_response = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
            try:
                conn.sendall(struct.pack(">I", len(error_response)) + error_response)
            except:
                pass
        finally:
            conn.close()

    def serve(self):
        """Lance le serveur daemon."""
        # Supprime le socket existant
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Écrit le PID
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        # Initialise le TTS
        self._init_tts()

        # Crée le socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)

        self.running = True
        print(f"[Daemon] Écoute sur {self.socket_path}")

        def signal_handler(signum, frame):
            print("\n[Daemon] Arrêt demandé...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                try:
                    conn, _ = self.server_socket.accept()
                    # Gère chaque client dans un thread
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(conn,),
                        daemon=True,
                    )
                    thread.start()
                except socket.timeout:
                    continue
        finally:
            self.server_socket.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(PID_FILE):
                os.unlink(PID_FILE)
            print("[Daemon] Arrêté")


class TTSClient:
    """Client pour le daemon TTS."""

    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 60.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def _send_request(self, request: dict) -> dict:
        """Envoie une requête au daemon."""
        if not os.path.exists(self.socket_path):
            raise ConnectionError(
                f"Daemon non disponible. Lancez: python tts_daemon.py serve"
            )

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect(self.socket_path)

            # Envoie la requête
            data = json.dumps(request).encode("utf-8")
            sock.sendall(struct.pack(">I", len(data)) + data)

            # Lit la réponse
            size_data = sock.recv(4)
            if not size_data:
                raise ConnectionError("Connexion fermée par le daemon")

            msg_size = struct.unpack(">I", size_data)[0]
            response_data = sock.recv(msg_size).decode("utf-8")
            return json.loads(response_data)

        finally:
            sock.close()

    def say(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "french",
    ) -> dict:
        """Dit un texte via le daemon."""
        return self._send_request({
            "command": "say",
            "text": text,
            "voice": voice,
            "language": language,
        })

    def ping(self) -> bool:
        """Vérifie si le daemon est actif."""
        try:
            response = self._send_request({"command": "ping"})
            return response.get("status") == "ok"
        except:
            return False

    def list_voices(self) -> list[str]:
        """Liste les voix disponibles."""
        response = self._send_request({"command": "voices"})
        return response.get("voices", [])

    def stop(self) -> None:
        """Arrête la lecture en cours."""
        self._send_request({"command": "stop"})

    def shutdown(self) -> None:
        """Arrête le daemon."""
        try:
            self._send_request({"command": "shutdown"})
        except:
            pass


def is_daemon_running() -> bool:
    """Vérifie si le daemon est en cours d'exécution."""
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Vérifie si le processus existe
        return True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="TTS Daemon")
    subparsers = parser.add_subparsers(dest="command", help="Commandes")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Lance le daemon")

    # say
    say_parser = subparsers.add_parser("say", help="Dit un texte")
    say_parser.add_argument("text", help="Texte à dire")
    say_parser.add_argument("-v", "--voice", help="Voix à utiliser")
    say_parser.add_argument("-l", "--language", default="french", help="Langue")

    # status
    status_parser = subparsers.add_parser("status", help="Vérifie le statut du daemon")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Arrête la lecture")

    # shutdown
    shutdown_parser = subparsers.add_parser("shutdown", help="Arrête le daemon")

    # voices
    voices_parser = subparsers.add_parser("voices", help="Liste les voix")

    args = parser.parse_args()

    if args.command == "serve":
        daemon = TTSDaemon()
        daemon.serve()

    elif args.command == "say":
        client = TTSClient()
        t0 = time.time()
        response = client.say(args.text, voice=args.voice, language=args.language)
        if response.get("status") == "ok":
            print(f"Latence génération: {response.get('latency', 0):.2f}s")
            print(f"Temps total: {response.get('total_time', 0):.2f}s")
        else:
            print(f"Erreur: {response.get('message')}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        if is_daemon_running():
            client = TTSClient()
            if client.ping():
                print("Daemon actif et opérationnel")
            else:
                print("Daemon en cours de démarrage...")
        else:
            print("Daemon non actif")

    elif args.command == "stop":
        client = TTSClient()
        client.stop()
        print("Lecture arrêtée")

    elif args.command == "shutdown":
        client = TTSClient()
        client.shutdown()
        print("Daemon arrêté")

    elif args.command == "voices":
        client = TTSClient()
        voices = client.list_voices()
        print("Voix disponibles:")
        for v in voices:
            print(f"  - {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
