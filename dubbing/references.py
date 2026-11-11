from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .segments import SpeechSegment


MIN_REFERENCE_SECONDS = 1.0
MIN_FALLBACK_REFERENCE_SECONDS = 0.5
MAX_REFERENCE_SECONDS = 20.0
PREFERRED_REFERENCE_MIN_SECONDS = 3.0
PREFERRED_REFERENCE_MAX_SECONDS = 12.0
MIN_REFERENCE_RMS = 0.008
MIN_REFERENCE_PEAK = 0.02


@dataclass(frozen=True)
class AudioQuality:
    rms: float
    peak: float
    clipped_fraction: float

    @property
    def below_reference_floor(self) -> bool:
        return self.rms < MIN_REFERENCE_RMS or self.peak < MIN_REFERENCE_PEAK


class NoSafeVoiceReference(ValueError):
    """The available audio cannot safely condition synthesis for this segment."""


class QuietVoiceReference(NoSafeVoiceReference):
    """Speech was found, but its measured level is below the reference floor."""


@dataclass(frozen=True)
class SpeakerReference:
    speaker: Optional[str]
    wav_path: Path
    transcript: str
    source_segment_index: int
    fallback: bool = False


def reference_score(segment: SpeechSegment, quality: AudioQuality) -> Optional[tuple[float, ...]]:
    """Return a deterministic quality ordering, or None for an unsafe reference."""

    if segment.overlap or segment.speaker_ambiguous:
        return None
    if (
        segment.duration < MIN_REFERENCE_SECONDS
        or segment.duration > MAX_REFERENCE_SECONDS
    ):
        return None
    if quality.below_reference_floor:
        return None
    if quality.peak >= 0.999 or quality.clipped_fraction > 0.01:
        return None
    ideal_duration_distance = abs(segment.duration - 6.0)
    preferred_duration = float(
        PREFERRED_REFERENCE_MIN_SECONDS
        <= segment.duration
        <= PREFERRED_REFERENCE_MAX_SECONDS
    )
    return (
        preferred_duration,
        -quality.clipped_fraction,
        quality.rms,
        -ideal_duration_distance,
        min(len(segment.text.strip()), 240),
        -segment.start,
    )


def select_reference_segments(
    segments: Sequence[SpeechSegment],
    qualities: Mapping[int, AudioQuality],
) -> dict[str, SpeechSegment]:
    selected: dict[str, tuple[tuple[float, ...], SpeechSegment]] = {}
    for segment in segments:
        if not segment.speaker or segment.index not in qualities:
            continue
        score = reference_score(segment, qualities[segment.index])
        if score is None:
            continue
        current = selected.get(segment.speaker)
        if current is None or score > current[0]:
            selected[segment.speaker] = (score, segment)
    return {speaker: value[1] for speaker, value in selected.items()}


class SpeakerReferenceManager:
    """Selects one clean reference per speaker and preserves its exact transcript."""

    def __init__(
        self,
        segments: Sequence[SpeechSegment],
        qualities: Mapping[int, AudioQuality],
        clip_writer: Callable[[SpeechSegment, Path], None],
        output_dir: Path,
    ) -> None:
        self.segments = list(segments)
        self.qualities = qualities
        self.clip_writer = clip_writer
        self.output_dir = output_dir
        self._speaker_references: dict[str, SpeakerReference] = {}
        self._segment_fallbacks: dict[int, SpeakerReference] = {}

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for speaker, segment in select_reference_segments(self.segments, self.qualities).items():
            safe_speaker = "".join(c for c in speaker if c.isalnum() or c in "_-")
            path = self.output_dir / f"{safe_speaker}.wav"
            self.clip_writer(segment, path)
            self._speaker_references[speaker] = SpeakerReference(
                speaker=speaker,
                wav_path=path,
                transcript=segment.text.strip(),
                source_segment_index=segment.index,
            )
        self._write_manifest()

    def _write_manifest(self) -> None:
        references = list(self._speaker_references.values()) + list(
            self._segment_fallbacks.values()
        )
        payload = [
            {
                "speaker": reference.speaker,
                "wav_path": str(reference.wav_path.resolve()),
                "transcript": reference.transcript,
                "source_segment_index": reference.source_segment_index,
                "fallback": reference.fallback,
            }
            for reference in sorted(
                references,
                key=lambda item: (item.fallback, item.source_segment_index),
            )
        ]
        (self.output_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def reference_for_group(self, segments: Sequence[SpeechSegment]) -> SpeakerReference:
        """Resolve against the measured audio of an actual group member."""
        failure = None
        for segment in segments:
            try:
                return self.reference_for(segment)
            except NoSafeVoiceReference as exc:
                if failure is None or isinstance(exc, QuietVoiceReference):
                    failure = exc
        if failure is not None:
            raise failure
        raise ValueError("A timing group must contain at least one source segment.")

    def reference_for(self, segment: SpeechSegment) -> SpeakerReference:
        if (
            segment.speaker
            and not segment.speaker_ambiguous
            and segment.speaker in self._speaker_references
        ):
            return self._speaker_references[segment.speaker]

        if segment.speaker_ambiguous:
            raise NoSafeVoiceReference(
                "Cannot synthesize transcript segment "
                f"{segment.index + 1} ({segment.start:.2f}–{segment.end:.2f}s): "
                "speaker identity is ambiguous. Review the diarization or provide "
                "a stable speaker assignment; no voice reference was borrowed."
            )

        if segment.index not in self._segment_fallbacks:
            quality = self.qualities.get(segment.index)
            eligible_interval = (
                MIN_FALLBACK_REFERENCE_SECONDS <= segment.duration <= MAX_REFERENCE_SECONDS
                and not segment.overlap
                and quality is not None
                and quality.peak < 0.999
                and quality.clipped_fraction <= 0.01
            )
            if eligible_interval and quality.peak > 0 and quality.below_reference_floor:
                raise QuietVoiceReference(
                    f"Speech audio is too quiet for a voice reference at "
                    f"{segment.start:.2f}–{segment.end:.2f}s "
                    f"(RMS {quality.rms:.4f}, peak {quality.peak:.4f}). "
                    "Raise the source speech level without clipping and try again."
                )
            if not eligible_interval or quality.below_reference_floor:
                raise NoSafeVoiceReference(
                    "No safe voice reference is available for transcript segment "
                    f"{segment.index + 1} ({segment.start:.2f}–{segment.end:.2f}s). "
                    "No stable same-speaker reference was found, and this segment "
                    "is too short, overlapping, silent, clipped, or long to use as "
                    "its own voice reference."
                )
            path = self.output_dir / f"segment_{segment.index + 1:04d}.wav"
            self.clip_writer(segment, path)
            self._segment_fallbacks[segment.index] = SpeakerReference(
                speaker=segment.speaker,
                wav_path=path,
                transcript=segment.text.strip(),
                source_segment_index=segment.index,
                fallback=True,
            )
            self._write_manifest()
        return self._segment_fallbacks[segment.index]
