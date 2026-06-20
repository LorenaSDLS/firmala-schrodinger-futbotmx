from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import subprocess

from pydub import AudioSegment

from src.G_narration.event_editor import (
    load_events,
    save_editorial_manifest,
    select_editorial_events,
)
from src.G_narration.tts import synthesize


def _probe_duration(path: str | Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return float(result.stdout.strip())


def _escape_srt(text: str) -> str:
    return text.replace("\n", " ").strip()


def _format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _write_srt(editorial_events, path: Path) -> Path:
    lines = []
    for index, event in enumerate(editorial_events, start=1):
        if isinstance(event, dict):
            start = max(0.0, float(event.get("audio_start_seconds", event.get("timestamp_seconds", 0.0))))
            end = start + max(1.8, float(event.get("audio_duration_seconds", event.get("estimated_duration_seconds", 2.0))))
            text = str(event.get("narration_text", ""))
        else:
            start = max(0.0, event.timestamp_seconds)
            end = start + max(1.8, event.estimated_duration_seconds)
            text = event.narration_text
        lines.extend([
            str(index),
            f"{_format_srt_time(start)} --> {_format_srt_time(end)}",
            _escape_srt(text),
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _video_has_audio(path: Path) -> bool:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True)
    return bool(result.stdout.strip())


def _burn_sample_video(video_path: Path, narration_wav: Path, srt_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
    subtitle_filter = (
        f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=20,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00111111,BorderStyle=3,"
        "BackColour=&H99000000,Outline=1,Shadow=0,MarginV=32'"
    )
    if _video_has_audio(video_path):
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(narration_wav),
            "-filter_complex", "[0:a]volume=0.28[base];[base][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-vf", subtitle_filter, "-map", "0:v:0", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path),
        ]
    else:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(narration_wav),
            "-vf", subtitle_filter, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path),
        ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return output_path


def run_narration(
    events_path: str | Path,
    video_path: str | Path,
    output_directory: str | Path,
    preview_video_path: str | Path | None = None,
    engine: str = "edge",
    voice: str | None = "es-MX-JorgeNeural",
    rate: str | int = "+0%",
    volume: str = "+0%",
    max_events: int = 12,
    maximum_coverage_ratio: float = 0.45,
    minimum_silence_seconds: float = 2.3,
    generate_sample_video: bool = True,
) -> dict:
    output_directory = Path(output_directory)
    narration_directory = output_directory / "narration"
    clips_directory = narration_directory / "events"
    clips_directory.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video_path)
    events = load_events(events_path)
    selected = select_editorial_events(
        events,
        video_duration_seconds=duration,
        max_events=max_events,
        maximum_coverage_ratio=maximum_coverage_ratio,
        minimum_silence_seconds=minimum_silence_seconds,
    )
    manifest_path = save_editorial_manifest(selected, narration_directory / "narration_manifest.json")

    timeline = AudioSegment.silent(duration=int(round(duration * 1000)), frame_rate=48000).set_channels(1)
    clip_paths: list[str] = []
    actual_manifest = []
    cursor_ms = 0
    for index, event in enumerate(selected):
        nominal_ms = int(round(event.timestamp_seconds * 1000))
        start_ms = max(nominal_ms, cursor_ms)
        output_wav = clips_directory / f"{index:03d}_{event.timestamp_seconds:08.2f}_{event.event_type}.wav"
        synthesize(event.narration_text, output_wav, engine=engine, voice=voice, rate=rate, volume=volume)
        clip = AudioSegment.from_wav(output_wav).set_frame_rate(48000).set_channels(1)
        if start_ms + len(clip) > len(timeline):
            # Important narration may start slightly earlier to fit.
            start_ms = max(0, len(timeline) - len(clip))
        timeline = timeline.overlay(clip, position=start_ms)
        cursor_ms = start_ms + len(clip) + int(1000 * minimum_silence_seconds)
        clip_paths.append(str(output_wav))
        item = asdict(event)
        item.update({"audio_start_seconds": round(start_ms / 1000, 3), "audio_duration_seconds": round(len(clip) / 1000, 3)})
        actual_manifest.append(item)

    complete_wav = narration_directory / "narracion_completa.wav"
    timeline.set_frame_rate(48000).set_sample_width(2).export(complete_wav, format="wav", parameters=["-acodec", "pcm_s16le"])
    manifest_path.write_text(json.dumps(actual_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_video_path = None
    subtitles_path = narration_directory / "narracion_eventos.srt"
    _write_srt(actual_manifest, subtitles_path)
    if generate_sample_video and preview_video_path:
        sample_video_path = narration_directory / "video_muestra_narrado.mp4"
        _burn_sample_video(Path(preview_video_path), complete_wav, subtitles_path, sample_video_path)

    return {
        "complete_wav_path": str(complete_wav),
        "event_audio_paths": clip_paths,
        "manifest_path": str(manifest_path),
        "subtitles_path": str(subtitles_path),
        "sample_video_path": str(sample_video_path) if sample_video_path else None,
        "selected_event_count": len(selected),
    }
