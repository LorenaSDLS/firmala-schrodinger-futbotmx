from __future__ import annotations
import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(description="Lista voces disponibles para la narración.")
    parser.add_argument("--engine", choices=["edge", "windows"], default="edge")
    args = parser.parse_args()
    if args.engine == "edge":
        try:
            import edge_tts
        except ImportError as exc:
            raise SystemExit("Instala edge-tts con: pip install edge-tts") from exc
        voices = asyncio.run(edge_tts.list_voices())
        for voice in voices:
            if str(voice.get("Locale", "")).lower().startswith("es"):
                print(f"{voice['ShortName']} - {voice.get('Gender', '')} - {voice.get('Locale', '')}")
    else:
        import pyttsx3
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices"):
            print(f"{voice.id} - {voice.name}")

if __name__ == "__main__":
    main()
