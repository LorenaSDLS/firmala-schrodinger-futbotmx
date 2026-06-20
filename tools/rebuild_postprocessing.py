"""Rebuild events, Mesa Replay, report and narration without rerunning YOLO.

Useful after a post-processing hotfix or after manually editing/refining events.
The output directory must already contain ``quick_detections.jsonl`` and
``video_metadata.json`` from a completed FutBotMX analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# When this file is executed directly (``python tools/rebuild_postprocessing.py``),
# Python places ``tools/`` on sys.path instead of the project root. Add the
# repository root explicitly so the sibling ``src`` package is importable on
# Windows, Linux and macOS. Running with ``python -m tools.rebuild_postprocessing``
# continues to work as well.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if not (PROJECT_ROOT / "src").is_dir():
    raise RuntimeError(
        "No se encontro la carpeta 'src' junto a 'tools'. "
        f"Raiz detectada: {PROJECT_ROOT}"
    )


def rebuild(
    output_directory: str | Path,
    *,
    video_path: str | Path | None = None,
    field_spec_path: str | Path | None = None,
    replay_frame_stride: int = 1,
    generate_pdf: bool = False,
    generate_narration: bool = False,
    narration_engine: str = "edge",
    narration_voice: str | None = "es-MX-JorgeNeural",
    narration_secondary_voice: str | None = "es-US-AlonsoNeural",
    narration_mode: str = "duo",
    narration_script_engine: str = "template",
    narration_api_config: str | Path | None = None,
    narration_groq_model: str = "llama-3.3-70b-versatile",
    narration_max_events: int = 10,
    narration_coverage_ratio: float = 0.42,
    narration_minimum_silence: float = 1.35,
    narration_maximum_start_delay: float = 2.5,
    narration_maximum_tail_extension: float = 4.5,
    generate_sample_video: bool = True,
) -> dict[str, str | None]:
    output_directory = Path(output_directory).expanduser().resolve()
    if not (output_directory / "quick_detections.jsonl").exists():
        raise FileNotFoundError(
            f"No se encontro quick_detections.jsonl en {output_directory}"
        )

    from src.A_pipeline.step_03_extract_events.run import run_step_03
    from src.E_events.unity_exporter import export_unity_mesa_json
    from src.F_simulation.mesa_replay_exporter import export_mesa_replay_video

    step_03 = run_step_03(output_directory)
    unity_path = export_unity_mesa_json(output_directory)
    replay_path = export_mesa_replay_video(
        unity_path,
        output_directory / "mesa_replay_events.mp4",
        frame_stride=max(1, int(replay_frame_stride)),
        field_spec_path=field_spec_path,
    )

    events_path = output_directory / "match_events_with_referee.json"
    if not events_path.exists():
        events_path = Path(step_03["events_path"])

    report_path: str | None = None
    if generate_pdf:
        from src.H_report.run import run_report

        result = run_report(
            output_directory=output_directory,
            events_path=events_path,
            summary_path=step_03["summary_path"],
            tracks_path=step_03["tracks_path"],
        )
        report_path = str(result["pdf_path"])

    narration_path: str | None = None
    if generate_narration:
        if video_path is None:
            raise ValueError("--video es obligatorio para regenerar narracion.")
        from src.G_narration.run import run_narration

        result = run_narration(
            events_path=events_path,
            video_path=video_path,
            output_directory=output_directory,
            preview_video_path=output_directory / "quick_preview.mp4",
            engine=narration_engine,
            voice=narration_voice,
            secondary_voice=narration_secondary_voice,
            narration_mode=narration_mode,
            script_engine=narration_script_engine,
            api_config_path=narration_api_config,
            groq_model=narration_groq_model,
            max_events=narration_max_events,
            maximum_coverage_ratio=narration_coverage_ratio,
            minimum_silence_seconds=narration_minimum_silence,
            maximum_start_delay_seconds=narration_maximum_start_delay,
            maximum_tail_extension_seconds=narration_maximum_tail_extension,
            generate_sample_video=generate_sample_video,
        )
        narration_path = str(result["complete_wav_path"])

    return {
        "tracks": str(step_03["tracks_path"]),
        "events": str(events_path),
        "unity_mesa": str(unity_path),
        "mesa_replay": str(replay_path),
        "report": report_path,
        "narration": narration_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenera eventos y salidas sin volver a ejecutar YOLO/SAM."
    )
    parser.add_argument("output_directory")
    parser.add_argument("--video", default=None)
    parser.add_argument("--field-spec", default=None)
    parser.add_argument("--replay-frame-stride", type=int, default=1)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--narration", "--narracion", action="store_true")
    parser.add_argument("--narration-engine", choices=["edge", "gtts", "windows", "loquendo", "espeak", "silent"], default="edge")
    parser.add_argument("--narration-voice", default="es-MX-JorgeNeural")
    parser.add_argument("--narration-secondary-voice", default="es-US-AlonsoNeural")
    parser.add_argument("--narration-mode", choices=["single", "duo"], default="duo")
    parser.add_argument("--narration-script-engine", choices=["template", "groq", "auto"], default="template")
    parser.add_argument("--narration-api-config", default=None)
    parser.add_argument("--narration-groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--narration-max-events", type=int, default=10)
    parser.add_argument("--narration-coverage-ratio", type=float, default=0.42)
    parser.add_argument("--narration-min-silence", type=float, default=1.35)
    parser.add_argument("--narration-max-start-delay", type=float, default=2.5)
    parser.add_argument("--narration-max-tail-extension", type=float, default=4.5)
    parser.add_argument("--no-sample-video", action="store_true", help="No remezcla el video; genera WAV, SRT y manifiesto.")
    args = parser.parse_args()
    result = rebuild(
        args.output_directory,
        video_path=args.video,
        field_spec_path=args.field_spec,
        replay_frame_stride=args.replay_frame_stride,
        generate_pdf=args.pdf,
        generate_narration=args.narration,
        narration_engine=args.narration_engine,
        narration_voice=args.narration_voice,
        narration_secondary_voice=args.narration_secondary_voice,
        narration_mode=args.narration_mode,
        narration_script_engine=args.narration_script_engine,
        narration_api_config=args.narration_api_config,
        narration_groq_model=args.narration_groq_model,
        narration_max_events=args.narration_max_events,
        narration_coverage_ratio=args.narration_coverage_ratio,
        narration_minimum_silence=args.narration_min_silence,
        narration_maximum_start_delay=args.narration_max_start_delay,
        narration_maximum_tail_extension=args.narration_max_tail_extension,
        generate_sample_video=not args.no_sample_video,
    )
    print("\nSalidas regeneradas:")
    for name, path in result.items():
        if path:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
