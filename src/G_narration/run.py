from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import subprocess

from pydub import AudioSegment

from src.G_narration.dialogue import DialogueLine, generate_dialogue
from src.G_narration.event_editor import load_events, select_editorial_events
from src.G_narration.scheduler import (
    PlannedNarration,
    plan_narration,
    prepare_narration,
    verify_no_overlap,
)
from src.G_narration.tts import synthesize


def _probe_duration(path: str | Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return float(result.stdout.strip())


def _format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _write_srt(items: list[dict], path: Path) -> Path:
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        start = float(item.get("audio_start_seconds", item.get("timestamp_seconds", 0.0)))
        end = start + max(0.8, float(item.get("audio_duration_seconds", 2.0)))
        dialogue = item.get("dialogue") or []
        text = (
            "  ".join(f"{line['speaker']}: {line['text']}" for line in dialogue)
            if dialogue
            else item.get("narration_text", "")
        )
        lines.extend(
            [
                str(index),
                f"{_format_srt_time(start)} --> {_format_srt_time(end)}",
                str(text).replace("\n", " "),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _video_has_audio(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _subtitle_filter(srt: Path) -> str:
    escaped = str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
    return (
        f"subtitles='{escaped}':"
        "force_style='FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00111111,BorderStyle=3,BackColour=&H99000000,"
        "Outline=1,Shadow=0,MarginV=32'"
    )


def _burn_sample_video(video: Path, narration: Path, srt: Path, output: Path) -> Path:
    """Mix narration without cutting late commentary or drowning source audio.

    The last frame is held when a goal/card happens near the end. Original audio
    is ducked only while speech is present instead of being permanently reduced.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    video_duration = _probe_duration(video)
    narration_duration = _probe_duration(narration)
    target_duration = max(video_duration, narration_duration)
    pad_seconds = max(0.0, target_duration - video_duration)
    video_chain = "[0:v]"
    if pad_seconds > 0.01:
        video_chain += f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f},"
    video_chain += _subtitle_filter(srt) + "[v]"

    if _video_has_audio(video):
        audio_pad = pad_seconds + 0.5
        filter_complex = (
            video_chain
            + ";"
            + f"[0:a]volume=0.78,apad=pad_dur={audio_pad:.3f}[base];"
            + "[1:a]asplit=2[voice_sc][voice_mix];"
            + "[base][voice_sc]sidechaincompress="
              "threshold=0.018:ratio=9:attack=18:release=360[ducked];"
            + "[voice_mix]volume=1.08[voice];"
            + "[ducked][voice]amix=inputs=2:duration=longest:dropout_transition=0[a]"
        )
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video), "-i", str(narration),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{target_duration:.3f}",
            str(output),
        ]
    else:
        filter_complex = video_chain
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video), "-i", str(narration),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{target_duration:.3f}",
            str(output),
        ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return output


def _render_dialogue(
    dialogue: tuple[DialogueLine, ...] | list[DialogueLine],
    output: Path,
    *,
    engine: str,
    voice1: str | None,
    voice2: str | None,
    rate: str,
    volume: str,
    inter_speaker_pause_ms: int = 260,
) -> Path:
    combined = AudioSegment.silent(duration=0, frame_rate=48_000).set_channels(1)
    temporary_paths: list[Path] = []
    try:
        for index, line in enumerate(dialogue):
            temporary = output.with_name(f"{output.stem}_line_{index:02d}.wav")
            temporary_paths.append(temporary)
            synthesize(
                line.text,
                temporary,
                engine=engine,
                voice=voice1 if line.speaker == "MARTINOLI" else voice2,
                rate=rate,
                volume=volume,
                pitch=58 if line.speaker == "MARTINOLI" else 38,
            )
            segment = (
                AudioSegment.from_file(temporary)
                .set_frame_rate(48_000)
                .set_sample_width(2)
                .set_channels(1)
                .fade_in(25)
                .fade_out(70)
            )
            if len(combined):
                combined += AudioSegment.silent(
                    duration=max(0, int(inter_speaker_pause_ms)),
                    frame_rate=48_000,
                ).set_channels(1)
            combined += segment
        output.parent.mkdir(parents=True, exist_ok=True)
        combined.export(output, format="wav", parameters=["-acodec", "pcm_s16le"])
        return output
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink()
            except OSError:
                pass


def _prepare_dialogues(selected, mode, script_engine, api_config, groq_model):
    prepared = []
    for event in selected:
        if mode == "single":
            dialogue, source = [DialogueLine("MARTINOLI", event.narration_text)], "event_editor"
        else:
            dialogue, source = generate_dialogue(
                event,
                script_engine=script_engine,
                api_config_path=api_config,
                groq_model=groq_model,
            )
        prepared.append(prepare_narration(event, dialogue, source))
    return prepared


def run_narration(
    events_path,
    video_path,
    output_directory,
    preview_video_path=None,
    engine="edge",
    voice="es-MX-JorgeNeural",
    secondary_voice="es-US-AlonsoNeural",
    rate="+0%",
    volume="+0%",
    max_events=10,
    maximum_coverage_ratio=0.42,
    minimum_silence_seconds=1.35,
    generate_sample_video=True,
    narration_mode="duo",
    script_engine="template",
    api_config_path=None,
    groq_model="llama-3.3-70b-versatile",
    maximum_start_delay_seconds=2.5,
    maximum_tail_extension_seconds=4.5,
):
    output_directory = Path(output_directory)
    narration_directory = output_directory / "narration"
    clips_directory = narration_directory / "events"
    clips_directory.mkdir(parents=True, exist_ok=True)
    for stale in clips_directory.glob("*.wav"):
        try:
            stale.unlink()
        except OSError:
            pass

    duration = _probe_duration(video_path)
    selected = select_editorial_events(
        load_events(events_path),
        video_duration_seconds=duration,
        max_events=max_events,
        maximum_coverage_ratio=maximum_coverage_ratio,
        minimum_silence_seconds=minimum_silence_seconds,
    )
    prepared = _prepare_dialogues(
        selected,
        narration_mode,
        script_engine,
        api_config_path,
        groq_model,
    )
    planned, skipped = plan_narration(
        prepared,
        video_duration_seconds=duration,
        minimum_silence_seconds=minimum_silence_seconds,
        maximum_start_delay_seconds=maximum_start_delay_seconds,
        maximum_tail_extension_seconds=maximum_tail_extension_seconds,
        maximum_coverage_ratio=maximum_coverage_ratio,
    )

    rendered: list[tuple[PlannedNarration, Path, AudioSegment]] = []
    for index, plan in enumerate(planned):
        event = plan.prepared.event
        output = clips_directory / (
            f"{index:03d}_{event.timestamp_seconds:08.2f}_{event.event_type}_{plan.variant}.wav"
        )
        _render_dialogue(
            plan.dialogue,
            output,
            engine=engine,
            voice1=voice,
            voice2=secondary_voice,
            rate=rate,
            volume=volume,
        )
        clip = AudioSegment.from_wav(output).set_frame_rate(48_000).set_sample_width(2).set_channels(1)
        rendered.append((plan, output, clip))

    # Conservative estimates should be longer than the produced clips. Validate
    # that invariant and drop, never overlap, any pathological outlier.
    accepted: list[tuple[PlannedNarration, Path, AudioSegment, float]] = []
    cursor = 0.0
    end_limit = duration + max(0.0, float(maximum_tail_extension_seconds))
    for plan, output, clip in sorted(rendered, key=lambda value: value[0].start_seconds):
        actual_seconds = len(clip) / 1000.0
        start = max(plan.start_seconds, cursor)
        event_time = float(plan.prepared.event.timestamp_seconds)
        if start > event_time + maximum_start_delay_seconds + 1e-6:
            skipped.append({
                "frame_index": int(plan.prepared.event.frame_index),
                "timestamp_seconds": event_time,
                "event_type": str(plan.prepared.event.event_type),
                "priority": float(plan.prepared.event.priority),
                "reason": "actual_audio_delay",
            })
            continue
        if start + actual_seconds > end_limit + 1e-6:
            skipped.append({
                "frame_index": int(plan.prepared.event.frame_index),
                "timestamp_seconds": event_time,
                "event_type": str(plan.prepared.event.event_type),
                "priority": float(plan.prepared.event.priority),
                "reason": "actual_audio_exceeds_tail",
            })
            continue
        accepted.append((plan, output, clip, start))
        cursor = start + actual_seconds + minimum_silence_seconds

    slots = [(start, start + len(clip) / 1000.0) for _, _, clip, start in accepted]
    if not verify_no_overlap(slots, minimum_silence_seconds=minimum_silence_seconds):
        raise RuntimeError("El planificador produjo clips de narración superpuestos.")

    final_duration = max(
        duration,
        max((start + len(clip) / 1000.0 for _, _, clip, start in accepted), default=duration),
    )
    timeline = AudioSegment.silent(duration=int(round(final_duration * 1000)), frame_rate=48_000).set_channels(1)
    manifest_items: list[dict] = []
    audio_paths: list[str] = []

    for plan, output, clip, start in accepted:
        timeline = timeline.overlay(clip, position=int(round(start * 1000)))
        event = plan.prepared.event
        item = asdict(event)
        item.update(
            audio_start_seconds=round(start, 3),
            audio_duration_seconds=round(len(clip) / 1000.0, 3),
            audio_end_seconds=round(start + len(clip) / 1000.0, 3),
            scheduled_delay_seconds=round(start - float(event.timestamp_seconds), 3),
            narration_mode=narration_mode,
            narration_variant=plan.variant,
            dialogue_source=plan.prepared.dialogue_source,
            dialogue=[asdict(line) for line in plan.dialogue],
        )
        manifest_items.append(item)
        audio_paths.append(str(output))

    complete = narration_directory / "narracion_completa.wav"
    timeline.set_frame_rate(48_000).set_sample_width(2).export(
        complete,
        format="wav",
        parameters=["-acodec", "pcm_s16le"],
    )
    manifest = narration_directory / "narration_manifest.json"
    manifest.write_text(json.dumps(manifest_items, ensure_ascii=False, indent=2), encoding="utf-8")
    schedule_path = narration_directory / "narration_schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "video_duration_seconds": round(duration, 3),
                "narration_duration_seconds": round(final_duration, 3),
                "minimum_silence_seconds": float(minimum_silence_seconds),
                "maximum_start_delay_seconds": float(maximum_start_delay_seconds),
                "maximum_tail_extension_seconds": float(maximum_tail_extension_seconds),
                "selected_count_before_scheduling": len(selected),
                "narrated_count": len(manifest_items),
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    subtitles = _write_srt(manifest_items, narration_directory / "narracion_eventos.srt")

    sample = None
    if generate_sample_video and preview_video_path and Path(preview_video_path).exists():
        sample = _burn_sample_video(
            Path(preview_video_path),
            complete,
            subtitles,
            narration_directory / "video_muestra_narrado.mp4",
        )

    return {
        "complete_wav_path": str(complete),
        "event_audio_paths": audio_paths,
        "manifest_path": str(manifest),
        "schedule_path": str(schedule_path),
        "subtitles_path": str(subtitles),
        "sample_video_path": str(sample) if sample else None,
        "selected_event_count": len(manifest_items),
        "skipped_event_count": len(skipped),
        "narration_mode": narration_mode,
        "script_engine": script_engine,
        "tts_engine": engine,
    }
