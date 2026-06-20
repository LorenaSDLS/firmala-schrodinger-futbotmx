from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import tempfile


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Falló la conversión de audio.")


def convert_to_wav(source: str | Path, destination: str | Path, sample_rate: int = 48000) -> Path:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(destination),
    ])
    return destination


def synthesize_edge(text: str, output_wav: Path, voice: str, rate: str = "+0%", volume: str = "+0%") -> Path:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Falta edge-tts. Instala con: pip install edge-tts") from exc
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "speech.mp3"
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
        asyncio.run(communicate.save(str(temporary)))
        return convert_to_wav(temporary, output_wav)


def synthesize_gtts(text: str, output_wav: Path, language: str = "es", tld: str = "com.mx") -> Path:
    from gtts import gTTS
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "speech.mp3"
        gTTS(text=text, lang=language, tld=tld).save(str(temporary))
        return convert_to_wav(temporary, output_wav)


def synthesize_pyttsx3(text: str, output_wav: Path, voice: str | None = None, rate: int = 185) -> Path:
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("Falta pyttsx3. Instala con: pip install pyttsx3") from exc
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    if voice:
        voices = engine.getProperty("voices")
        selected = next((item for item in voices if voice.lower() in (item.id + " " + item.name).lower()), None)
        if selected is None:
            raise ValueError(f"No se encontró una voz de Windows que coincida con: {voice}")
        engine.setProperty("voice", selected.id)
    engine.setProperty("rate", int(rate))
    engine.save_to_file(text, str(output_wav))
    engine.runAndWait()
    if not output_wav.exists():
        raise RuntimeError("pyttsx3 no produjo el WAV esperado.")
    return output_wav


def synthesize(
    text: str,
    output_wav: str | Path,
    engine: str = "edge",
    voice: str | None = None,
    rate: str | int = "+0%",
    volume: str = "+0%",
) -> Path:
    output_wav = Path(output_wav)
    engine = engine.lower()
    if engine == "edge":
        return synthesize_edge(text, output_wav, voice or "es-MX-JorgeNeural", str(rate), volume)
    if engine == "gtts":
        return synthesize_gtts(text, output_wav)
    if engine in {"windows", "pyttsx3"}:
        return synthesize_pyttsx3(text, output_wav, voice=voice, rate=int(rate) if str(rate).lstrip("+-").isdigit() else 185)
    raise ValueError("Motor de voz inválido. Usa edge, gtts o windows.")
