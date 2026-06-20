from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.G_narration.dialogue import DialogueLine


@dataclass(frozen=True)
class PreparedNarration:
    event: Any
    dialogue: tuple[DialogueLine, ...]
    compact_dialogue: tuple[DialogueLine, ...]
    dialogue_source: str
    full_estimated_seconds: float
    compact_estimated_seconds: float


@dataclass(frozen=True)
class PlannedNarration:
    prepared: PreparedNarration
    dialogue: tuple[DialogueLine, ...]
    variant: str
    start_seconds: float
    estimated_duration_seconds: float

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.estimated_duration_seconds


def estimate_dialogue_seconds(
    dialogue: Iterable[DialogueLine],
    *,
    words_per_second: float = 1.80,
    inter_line_pause_seconds: float = 0.24,
) -> float:
    """Conservative TTS estimate used before audio is synthesized.

    Edge voices used in this project normally land near 2.1--2.3 words/s.
    Using 1.8 words/s deliberately over-estimates clips so the scheduler leaves
    breathing room instead of creating overlaps after synthesis.
    """
    lines = tuple(dialogue)
    if not lines:
        return 0.0
    words = sum(max(1, len(line.text.split())) for line in lines)
    punctuation_pause = sum(line.text.count(mark) for line in lines for mark in ",;:") * 0.045
    return max(
        1.2,
        words / max(1.0, words_per_second)
        + max(0, len(lines) - 1) * inter_line_pause_seconds
        + punctuation_pause
        + 0.30,
    )


def make_compact_dialogue(event: Any, dialogue: Iterable[DialogueLine]) -> tuple[DialogueLine, ...]:
    """Return one concise line for crowded timelines.

    The editorial event already contains a short, uncertainty-aware sentence.
    Prefer it when it is shorter than the first generated line. This keeps goals,
    collisions and cards narratable without forcing two long commentators into a
    small gap.
    """
    full = tuple(dialogue)
    event_text = str(getattr(event, "narration_text", "") or "").strip()
    first = full[0] if full else DialogueLine("MARTINOLI", event_text or "Evento registrado.")
    if event_text and len(event_text.split()) <= len(first.text.split()):
        return (DialogueLine("MARTINOLI", event_text),)
    return (first,)


def prepare_narration(event: Any, dialogue: Iterable[DialogueLine], source: str) -> PreparedNarration:
    full = tuple(dialogue)
    compact = make_compact_dialogue(event, full)
    return PreparedNarration(
        event=event,
        dialogue=full,
        compact_dialogue=compact,
        dialogue_source=str(source),
        full_estimated_seconds=estimate_dialogue_seconds(full),
        compact_estimated_seconds=estimate_dialogue_seconds(compact),
    )


def _find_slot(
    preferred_start: float,
    duration: float,
    reservations: list[tuple[float, float]],
    *,
    minimum_silence_seconds: float,
    maximum_start_delay_seconds: float,
    end_limit_seconds: float,
) -> float | None:
    start = max(0.0, float(preferred_start))
    latest_start = start + max(0.0, float(maximum_start_delay_seconds))
    silence = max(0.0, float(minimum_silence_seconds))

    for old_start, old_end in sorted(reservations):
        # Enough room before the existing reservation.
        if start + duration + silence <= old_start:
            break
        # Otherwise move after it when the windows overlap or are too close.
        if start < old_end + silence:
            start = old_end + silence
        if start > latest_start:
            return None

    if start > latest_start or start + duration > end_limit_seconds:
        return None
    return start


def plan_narration(
    prepared_items: Iterable[PreparedNarration],
    *,
    video_duration_seconds: float,
    minimum_silence_seconds: float = 1.35,
    reaction_delay_seconds: float = 0.12,
    maximum_start_delay_seconds: float = 2.5,
    maximum_tail_extension_seconds: float = 4.5,
    maximum_coverage_ratio: float = 0.42,
) -> tuple[list[PlannedNarration], list[dict[str, Any]]]:
    """Reserve non-overlapping commentary slots, highest priority first.

    Core guarantees:
      * narration never starts before its event;
      * clips never overlap;
      * a natural silence is reserved between clips;
      * crowded events are compacted to one line;
      * low-priority events are omitted instead of moved far away;
      * late goals/cards may extend the sample video by a small, bounded tail.
    """
    items = list(prepared_items)
    end_limit = max(0.0, float(video_duration_seconds)) + max(0.0, float(maximum_tail_extension_seconds))
    coverage_budget = max(4.0, float(video_duration_seconds) * max(0.05, float(maximum_coverage_ratio)))
    reservations: list[tuple[float, float]] = []
    planned: list[PlannedNarration] = []
    skipped: list[dict[str, Any]] = []
    used_speech = 0.0

    # High-priority events reserve their slots first. Chronological ordering is
    # restored at the end for rendering and subtitles.
    ranked = sorted(
        items,
        key=lambda item: (
            -float(getattr(item.event, "priority", 0.0)),
            float(getattr(item.event, "timestamp_seconds", 0.0)),
        ),
    )

    for item in ranked:
        event = item.event
        event_time = max(0.0, float(getattr(event, "timestamp_seconds", 0.0)))
        preferred = event_time + max(0.0, float(reaction_delay_seconds))
        priority = float(getattr(event, "priority", 0.0))
        event_type = str(getattr(event, "event_type", "unknown"))

        variants: list[tuple[str, tuple[DialogueLine, ...], float]] = []
        if item.dialogue:
            variants.append(("full", item.dialogue, item.full_estimated_seconds))
        if item.compact_dialogue and item.compact_dialogue != item.dialogue:
            variants.append(("compact", item.compact_dialogue, item.compact_estimated_seconds))

        accepted: PlannedNarration | None = None
        reject_reason = "no_time_slot"
        for variant, dialogue, duration in variants:
            # Keep the overall program from becoming wall-to-wall commentary.
            # Major events may exceed the soft budget, but only in compact form.
            over_budget = used_speech + duration > coverage_budget
            if over_budget and not (priority >= 75.0 and variant == "compact"):
                reject_reason = "coverage_budget"
                continue

            start = _find_slot(
                preferred,
                duration,
                reservations,
                minimum_silence_seconds=minimum_silence_seconds,
                maximum_start_delay_seconds=maximum_start_delay_seconds,
                end_limit_seconds=end_limit,
            )
            if start is None:
                reject_reason = "no_time_slot"
                continue

            accepted = PlannedNarration(
                prepared=item,
                dialogue=dialogue,
                variant=variant,
                start_seconds=start,
                estimated_duration_seconds=duration,
            )
            break

        if accepted is None:
            skipped.append(
                {
                    "frame_index": int(getattr(event, "frame_index", 0)),
                    "timestamp_seconds": event_time,
                    "event_type": event_type,
                    "priority": priority,
                    "reason": reject_reason,
                }
            )
            continue

        planned.append(accepted)
        reservations.append((accepted.start_seconds, accepted.end_seconds))
        used_speech += accepted.estimated_duration_seconds

    planned.sort(key=lambda item: item.start_seconds)
    return planned, skipped


def verify_no_overlap(
    slots: Iterable[tuple[float, float]],
    *,
    minimum_silence_seconds: float = 0.0,
) -> bool:
    ordered = sorted((float(start), float(end)) for start, end in slots)
    gap = max(0.0, float(minimum_silence_seconds))
    return all(left_end + gap <= right_start + 1e-6 for (_, left_end), (right_start, _) in zip(ordered, ordered[1:]))
