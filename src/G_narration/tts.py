from __future__ import annotations
import asyncio, os, shutil, subprocess, tempfile, time
from pathlib import Path

def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0: raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Falló la conversión de audio.")

def convert_to_wav(source: str | Path, destination: str | Path, sample_rate: int = 48000) -> Path:
    source, destination = Path(source), Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg","-y","-loglevel","error","-i",str(source),"-ac","1","-ar",str(sample_rate),"-c:a","pcm_s16le",str(destination)])
    return destination

EDGE_VOICE_ALIASES = {
    # This voice was used by early FutBot builds but is not available in current
    # Edge voice catalogs. Keep old commands/configs working.
    "es-US-PedroNeural": "es-US-AlonsoNeural",
}

def _edge_percent(value, default="+0%"):
    raw = str(value if value is not None else default).strip()
    if raw.endswith("%") and raw[:1] in {"+", "-"}:
        return raw
    if raw.endswith("%"):
        raw = raw[:-1]
    try:
        number = int(float(raw))
    except (TypeError, ValueError):
        return default
    return f"{number:+d}%"

def synthesize_edge(text, output_wav, voice, rate="+0%", volume="+0%"):
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Falta edge-tts. Instala con: pip install edge-tts") from exc

    requested = str(voice or "es-MX-JorgeNeural")
    primary = EDGE_VOICE_ALIASES.get(requested, requested)
    candidates = [primary]
    if primary != "es-MX-JorgeNeural":
        candidates.append("es-MX-JorgeNeural")
    errors = []
    for candidate in candidates:
        for attempt in range(2):
            try:
                with tempfile.TemporaryDirectory() as directory:
                    temp = Path(directory) / "speech.mp3"
                    asyncio.run(
                        edge_tts.Communicate(
                            text=text,
                            voice=candidate,
                            rate=_edge_percent(rate),
                            volume=_edge_percent(volume),
                        ).save(str(temp))
                    )
                    if not temp.exists() or temp.stat().st_size < 128:
                        raise RuntimeError("Edge TTS no produjo datos de audio.")
                    return convert_to_wav(temp, output_wav)
            except Exception as exc:
                errors.append(f"{candidate} intento {attempt + 1}: {exc}")
                if attempt == 0:
                    time.sleep(0.35)
    raise RuntimeError("Edge TTS no pudo generar audio. " + " | ".join(errors))

def synthesize_gtts(text, output_wav):
    try: from gtts import gTTS
    except ImportError as exc: raise RuntimeError("Falta gTTS. Instala con: pip install gTTS") from exc
    with tempfile.TemporaryDirectory() as directory:
        temp=Path(directory)/"speech.mp3"; gTTS(text=text,lang="es",tld="com.mx").save(str(temp)); return convert_to_wav(temp, output_wav)

def synthesize_pyttsx3(text, output_wav, voice=None, rate=185):
    try: import pyttsx3
    except ImportError as exc: raise RuntimeError("Falta pyttsx3. Instala con: pip install pyttsx3") from exc
    output_wav=Path(output_wav); output_wav.parent.mkdir(parents=True,exist_ok=True); engine=pyttsx3.init()
    if voice:
        selected=next((v for v in engine.getProperty("voices") if voice.lower() in (v.id+" "+v.name).lower()),None)
        if selected: engine.setProperty("voice",selected.id)
        else: print(f"[narración] No se encontró la voz '{voice}'; se usará la predeterminada.")
    engine.setProperty("rate",int(rate)); engine.save_to_file(text,str(output_wav)); engine.runAndWait()
    if not output_wav.exists() or output_wav.stat().st_size<128: raise RuntimeError("pyttsx3 no produjo el WAV esperado.")
    return output_wav

def synthesize_espeak(text, output_wav, voice="es", rate=175, pitch=50):
    executable=shutil.which("espeak-ng") or shutil.which("espeak")
    if not executable: raise RuntimeError("No se encontró espeak/espeak-ng en PATH.")
    output_wav=Path(output_wav); output_wav.parent.mkdir(parents=True,exist_ok=True); _run([executable,"-v",voice or "es","-s",str(int(rate)),"-p",str(int(pitch)),"-w",str(output_wav),text]); return output_wav

def _ps_quote(value): return str(value).replace("'","''")
def synthesize_loquendo(text, output_wav, voice="Jorge", rate=3):
    if os.name!="nt": raise RuntimeError("El motor loquendo solo está disponible en Windows.")
    root=Path(os.environ.get("SystemRoot",r"C:\Windows")); candidates=[root/"SysWOW64"/"WindowsPowerShell"/"v1.0"/"powershell.exe",root/"System32"/"WindowsPowerShell"/"v1.0"/"powershell.exe"]
    powershell=next((p for p in candidates if p.exists()),None)
    if powershell is None: raise RuntimeError("No se encontró PowerShell para ejecutar la voz SAPI.")
    output_wav=Path(output_wav).resolve(); output_wav.parent.mkdir(parents=True,exist_ok=True)
    script=f"""Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$pattern = '{_ps_quote(voice or '')}'
if ($pattern) {{ $selected = $synth.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Name -match [regex]::Escape($pattern) }} | Select-Object -First 1; if ($selected) {{ $synth.SelectVoice($selected.VoiceInfo.Name) }} }}
$synth.Rate = {max(-10,min(10,int(rate)))}
$synth.SetOutputToWaveFile('{_ps_quote(output_wav)}')
$synth.Speak('{_ps_quote(text)}')
$synth.Dispose()
"""
    with tempfile.TemporaryDirectory() as directory:
        ps=Path(directory)/"futbot_tts.ps1"; ps.write_text(script,encoding="utf-8-sig"); _run([str(powershell),"-NoProfile","-ExecutionPolicy","Bypass","-File",str(ps)])
    if not output_wav.exists() or output_wav.stat().st_size<128: raise RuntimeError("La voz SAPI/Loquendo no produjo el WAV esperado.")
    return output_wav

def synthesize_silence(text, output_wav):
    duration=max(0.8,min(8.0,len(text)/14.0)); output_wav=Path(output_wav); output_wav.parent.mkdir(parents=True,exist_ok=True)
    _run(["ffmpeg","-y","-loglevel","error","-f","lavfi","-i","anullsrc=r=48000:cl=mono","-t",f"{duration:.3f}","-c:a","pcm_s16le",str(output_wav)]); return output_wav

def synthesize(text, output_wav, engine="edge", voice=None, rate="+0%", volume="+0%", *, pitch=50):
    output_wav=Path(output_wav); engine=str(engine).lower()
    if engine=="edge": return synthesize_edge(text,output_wav,voice or "es-MX-JorgeNeural",str(rate),volume)
    if engine=="gtts": return synthesize_gtts(text,output_wav)
    if engine in {"windows","pyttsx3"}: return synthesize_pyttsx3(text,output_wav,voice,int(rate) if str(rate).lstrip("+-").isdigit() else 185)
    if engine in {"loquendo","sapi32"}: return synthesize_loquendo(text,output_wav,voice or "Jorge",int(rate) if str(rate).lstrip("+-").isdigit() else 3)
    if engine in {"espeak","offline"}: return synthesize_espeak(text,output_wav,voice or "es",int(rate) if str(rate).lstrip("+-").isdigit() else 175,pitch)
    if engine in {"silent","silence","test"}: return synthesize_silence(text,output_wav)
    raise ValueError("Motor desconocido: edge, gtts, windows, loquendo, espeak o silent.")
