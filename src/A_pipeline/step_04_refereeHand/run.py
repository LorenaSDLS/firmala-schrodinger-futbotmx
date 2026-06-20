import argparse
from pathlib import Path

from src.E_events.referee_hand_detector import analyze_referee_hand_candidates


def run_step_04(
    video_path: str | Path,
    output_directory: str | Path,
    sam_mode: str = "LoHa",
    sam_confidence: float = 0.25,
    frame_window: int = 20,
    max_candidates: int | None = None,
) -> dict:
    result = analyze_referee_hand_candidates(
        video_path=video_path,
        output_directory=output_directory,
        sam_mode=sam_mode,
        sam_confidence=sam_confidence,
        frame_window=frame_window,
        max_candidates=max_candidates,
    )

    print("\n" + "=" * 55)
    print(" PASO 04 - MANO DEL ARBITRO")
    print("=" * 55)

    

    print(f"Candidatos JSON:      {result['candidates_path']}")
    print(f"Eventos refinados:    {result['updated_events_path']}")
    print(f"Imagenes debug:       {result['debug_directory']}")
    print(f"Candidatos detectados:{result['candidates_detected']}")
    print(f"Eventos nuevos:       {result['new_events']}")

    print("=" * 55 + "\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca mano del arbitro en frames sospechosos."
    )

    parser.add_argument("video_path")
    parser.add_argument("output_directory")

    parser.add_argument(
        "--sam-mode",
        choices=["LoHa", "DoRa"],
        default="LoHa",
    )

    parser.add_argument(
        "--sam-conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--frame-window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    run_step_04(
        video_path=arguments.video_path,
        output_directory=arguments.output_directory,
        sam_mode=arguments.sam_mode,
        sam_confidence=arguments.sam_conf,
        frame_window=arguments.frame_window,
        max_candidates=arguments.max_candidates,
    )


if __name__ == "__main__":
    main()