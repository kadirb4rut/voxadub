"""Offline regressions from the November 11 independent test report."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import Video_Translator as pipeline
from diarization.assignment import DiarizationResult, DiarizationTurn, assign_speakers
from dubbing.references import AudioQuality, NoSafeVoiceReference, SpeakerReferenceManager
from dubbing.segments import SpeechSegment
from dubbing.timing import TimingGroupConfig, build_timing_groups
from engine.output_paths import available_mp4_path, publish_mp4


def test_short_name_uses_only_corroborated_local_speaker_evidence():
    phrases = [
        SpeechSegment(0, .3, 2.23, "Well, Mr."),
        SpeechSegment(1, 2.25, 2.55, "Hector."),
        SpeechSegment(2, 2.59, 2.80, "Yes."),
        SpeechSegment(3, 2.87, 3.17, "Then."),
        SpeechSegment(4, 3.2, 5.8, "A different speaker."),
        SpeechSegment(5, 6.0, 6.3, "Alone."),
    ]
    turns = (
        DiarizationTurn(.2, 2.74, "a"),
        DiarizationTurn(2.74, 5.85, "b"),
        DiarizationTurn(6.0, 6.3, "isolated"),
    )
    assigned = assign_speakers(phrases, DiarizationResult(turns, turns))
    assert [segment.speaker for segment in assigned] == [
        "SPEAKER_00", "SPEAKER_00", None, "SPEAKER_01", "SPEAKER_01", None,
    ]
    assert assigned[2].speaker_ambiguous and assigned[5].speaker_ambiguous
    groups = build_timing_groups(assigned)
    assert [group.original_segment_indices for group in groups] == [(0, 1), (2,), (3, 4), (5,)]
    assert groups[0].internal_pauses == pytest.approx((0.02,))


def test_short_assignment_rejects_gaps_disagreement_and_duplicate_evidence():
    short = SpeechSegment(0, 2.25, 2.55, "Hector.")
    regular = (DiarizationTurn(0, 6, "a"),)
    duplicate_fragments = (DiarizationTurn(2.25, 2.4, "a"),) * 2
    wrong_regular = (DiarizationTurn(0, 6, "b"),)
    for result in (
        DiarizationResult(regular, duplicate_fragments),
        DiarizationResult(wrong_regular, regular),
        DiarizationResult((DiarizationTurn(0, 2.24, "a"), DiarizationTurn(2.56, 6, "a"))),
    ):
        resolved = assign_speakers([short], result)[0]
        assert resolved.speaker is None and resolved.speaker_ambiguous


def test_timing_cap_applies_to_the_entire_joined_group():
    words = [SpeechSegment(i, i * .5, (i + 1) * .5, "Yes.", "SPEAKER_00") for i in range(5)]
    groups = build_timing_groups(words, config=TimingGroupConfig(max_group_duration_seconds=1.0))
    assert [group.original_segment_indices for group in groups] == [(0, 1), (2, 3), (4,)]


def test_mixed_segment_never_becomes_a_reference(tmp_path):
    regular = (DiarizationTurn(0, 5.5, "a"), DiarizationTurn(2.5, 5.5, "b"))
    source = [
        SpeechSegment(0, .2, 1.8, "Safe."),
        SpeechSegment(1, 2.5, 4.0, "Cross-talk."),
    ]
    writes = []
    quality = {i: AudioQuality(.1, .4, 0) for i in range(2)}
    ambiguous = assign_speakers(source, DiarizationResult(regular))
    manager = SpeakerReferenceManager(ambiguous, quality, lambda segment, _: writes.append(segment.index), tmp_path)
    manager.prepare()
    with pytest.raises(NoSafeVoiceReference, match="ambiguous"):
        manager.reference_for(ambiguous[1])
    assert writes == [0]

    exclusive = (DiarizationTurn(0, 5.5, "a"),)
    resolved = assign_speakers(source, DiarizationResult(regular, exclusive))
    manager = SpeakerReferenceManager(resolved, quality, lambda segment, _: writes.append(segment.index), tmp_path)
    manager.prepare()
    assert resolved[1].overlap
    assert manager.reference_for(resolved[1]).source_segment_index == 0
    assert writes == [0, 0]


def test_exclusive_speaker_change_during_overlap_preserves_uncertain_audio():
    segment = SpeechSegment(0, 0, 4, "A mixed sentence.")
    regular = (DiarizationTurn(0, 3.4, "a"), DiarizationTurn(2.7, 4, "b"))
    exclusive = (DiarizationTurn(0, 3, "a"), DiarizationTurn(3, 4, "b"))
    result = assign_speakers([segment], DiarizationResult(regular, exclusive))[0]
    assert result.overlap and result.speaker_ambiguous and result.speaker is None


def test_aligner_failure_names_language_and_does_not_translate_music():
    def unavailable(**kwargs):
        assert kwargs["language_code"] == "cy"
        raise ValueError("No default align-model for language: cy")

    def align(*args, **kwargs):
        pytest.fail("Unaligned guesses must not reach the word timing pipeline")

    whisperx = SimpleNamespace(load_align_model=unavailable, align=align)
    with pytest.raises(pipeline.TranscriptionAlignmentError, match=r"'cy'.*--source-language.*music"):
        pipeline.align_transcript_words(whisperx, {"language": "cy", "segments": [{"text": "guess"}]}, [], "cpu")
    assert pipeline.align_transcript_words(whisperx, {"language": "cy", "segments": []}, [], "cpu") == []

    def device_error(**kwargs):
        raise ValueError("CUDA device unavailable")

    whisperx.load_align_model = device_error
    with pytest.raises(ValueError, match="CUDA device unavailable"):
        pipeline.align_transcript_words(whisperx, {"language": "cy", "segments": [{}]}, [], "cuda")


def test_aligned_words_keep_the_existing_word_timing_contract():
    first, second = {"word": "Mr.", "start": 2.11, "end": 2.23}, {"word": "Hector.", "start": 2.25, "end": 2.55}
    whisperx = SimpleNamespace(
        load_align_model=lambda **kwargs: ("model", {"language": "en"}),
        align=lambda *args, **kwargs: {"word_segments": [first, {"word": "missing"}, second]},
    )
    words = pipeline.align_transcript_words(whisperx, {"language": "en", "segments": [{}]}, [], "cpu")
    assert words == [first, second]
    assert pipeline.create_TimeStamps(words, ".!?") == ([2.11, 2.25], [2.23, 2.55])
    assert pipeline.create_Sentences(words, ".!?") == ["Mr.", "Hector."]


def test_output_publication_keeps_existing_files_even_on_late_collision(tmp_path):
    existing = tmp_path / "clip.v1.mp4"
    existing.write_bytes(b"original")
    destination = available_mp4_path(tmp_path, "clip.v1.webm")
    assert destination.name == "clip.v1-2.mp4"
    with pytest.raises(FileExistsError):
        with publish_mp4(destination) as staging:
            staging.write_bytes(b"new")
            destination.write_bytes(b"other job")
    assert existing.read_bytes() == b"original"
    assert destination.read_bytes() == b"other job"
    assert not list(tmp_path.glob(".voxadub-*"))


@pytest.fixture
def media_tools():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):
        pytest.skip("FFmpeg and FFprobe required")
    return ffmpeg, ffprobe


def checked(*args):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True)


def make_media(ffmpeg, path, duration, *, layout="normal", video_codec="libx264", audio_codec="aac"):
    size, fps = ("80x128", 30) if layout == "portrait" else ("128x80", 25)
    video = f"testsrc2=size={size}:rate={fps}:duration={duration}"
    audio = f"sine=frequency=431:sample_rate=48000:duration={duration}"
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if layout != "audio_only":
        command += ["-f", "lavfi", "-i", video]
    if layout != "no_audio":
        command += ["-f", "lavfi", "-i", audio, "-c:a", "pcm_s16le" if layout == "audio_only" else audio_codec]
    if layout != "audio_only":
        command += ["-c:v", video_codec, "-threads", "2"]
        if video_codec == "libx265":
            command += ["-x265-params", "pools=1:frame-threads=1:log-level=error"]
    checked(*command, path)


class Translator:
    def __init__(self, *_): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def translate(self, *_args, **_kwargs): return "Olá."


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(pipeline, "BASE_DIR", root)
    monkeypatch.setattr(pipeline, "VOCAL_REMOVER_DIR", root / "vocal-remover")
    monkeypatch.setattr(pipeline, "HyMT2Translator", Translator)
    monkeypatch.setattr(pipeline, "load_model", lambda: object())
    tts_calls = []

    def separate(audio_path, directory):
        signal, sr = sf.read(audio_path)
        name = Path(audio_path).stem
        sf.write(directory / f"{name}_Vocals.wav", signal, sr)
        sf.write(directory / f"{name}_Instruments.wav", np.zeros_like(signal), sr)

    def synthesize(*, output_wav_path, reference_wav_path, seed, **kwargs):
        tts_calls.append((Path(reference_wav_path).name, seed))
        frames = 36_000
        sf.write(output_wav_path, .13 * np.sin(np.arange(frames) * 2 * np.pi * 713 / 48_000), 48_000)

    monkeypatch.setattr(pipeline, "seperate_Speaker_and_Background_Sound_from_audio", separate)
    monkeypatch.setattr(pipeline, "synthesize_cloned_speech", synthesize)

    def run(path, words, turns, *, output=None, arguments=()):
        monkeypatch.setattr(pipeline, "transcript_Text", lambda *_: words)
        monkeypatch.setattr(pipeline, "diarize", lambda *_args, **_kwargs: turns)
        args = ["Video_Translator.py", str(path), "--source-language", "en",
                "--target-language", "pt", "--max-refinement-passes", "0", *arguments]
        if output:
            args.extend(["--output", str(output)])
        monkeypatch.setattr(sys, "argv", args)
        pipeline.main()
        return root, tts_calls

    return run


def word(start, end, text="Words."):
    return {"word": text, "start": start, "end": end}


def assert_mp4(ffmpeg, ffprobe, output, seconds):
    assert output.is_file()
    probe = json.loads(checked(
        ffprobe, "-v", "error", "-show_entries", "stream=codec_name,codec_type:format=duration,format_name",
        "-of", "json", output,
    ).stdout)
    assert {stream["codec_type"]: stream["codec_name"] for stream in probe["streams"]} == {
        "video": "h264", "audio": "aac",
    }
    assert "mp4" in probe["format"]["format_name"]
    assert float(probe["format"]["duration"]) == pytest.approx(seconds, abs=.16)
    checked(ffmpeg, "-v", "error", "-xerror", "-i", output, "-f", "null", "-")


@pytest.mark.parametrize("case,seconds,layout", [
    ("T1", 15, "normal"), ("T2", 60, "normal"), ("T3", 2.2, "normal"),
    ("T4", 75, "normal"), ("T5", 15, "no_audio"), ("T6", 15, "audio_only"),
    ("T7", 68, "normal"), ("T8", 15, "portrait"), ("T9", .6, "normal"),
])
def test_nine_case_matrix_with_real_media_and_model_doubles(
    media_tools, offline_pipeline, tmp_path, case, seconds, layout,
):
    ffmpeg, ffprobe = media_tools
    video = tmp_path / (case + (".wav" if layout == "audio_only" else ".mp4"))
    make_media(ffmpeg, video, seconds, layout=layout)
    output = tmp_path / "out" / f"{case}.mp4"
    invalid = {"T5": "no audio stream", "T6": "no video stream", "T9": "shorter than 1.0 second"}
    if case in invalid:
        with pytest.raises(SystemExit, match=invalid[case]):
            offline_pipeline(video, (), None, output=output)
        assert not output.exists()
        return

    words = [word(.2, 1.8)]
    turns = [DiarizationTurn(.15, 1.9, "a")]
    if case in ("T2", "T4", "T7"):
        words.extend((word(2.11, 2.23, "Mr."), word(2.25, 2.55, "Hector."), word(2.6, 4.1)))
        turns = [DiarizationTurn(.15, 4.2, "a")]
    if case == "T4":
        words.extend((word(35, 36.5), word(36.6, 36.8, "Yes.")))
        turns.append(DiarizationTurn(34.9, 37, "b"))
    options = ("--num-speakers", "1") if case in ("T2", "T7") else ()
    _, calls = offline_pipeline(video, words, DiarizationResult(tuple(turns), tuple(turns)),
                                output=output, arguments=options)
    assert calls
    assert_mp4(ffmpeg, ffprobe, output, seconds)


@pytest.mark.parametrize("suffix,video_codec,audio_codec", [
    ("mp4", "libx264", "aac"), ("webm", "libvpx-vp9", "libopus"),
    ("mov", "libx264", "aac"), ("mkv", "libx264", "aac"), ("mp4", "libx265", "aac"),
])
def test_export_normalizes_each_supported_input_to_mp4(
    media_tools, offline_pipeline, tmp_path, suffix, video_codec, audio_codec,
):
    ffmpeg, ffprobe = media_tools
    source = tmp_path / f"example.v1.{suffix}"
    make_media(ffmpeg, source, 2.2, video_codec=video_codec, audio_codec=audio_codec)
    turn = DiarizationTurn(.1, 1.9, "a")
    root, _ = offline_pipeline(source, [word(.2, 1.8)], DiarizationResult((turn,), (turn,)))
    output = root / "vocal-remover" / "final_video" / "final" / "example.v1.mp4"
    assert_mp4(ffmpeg, ffprobe, output, 2.2)


def test_mixed_tracks_keep_original_only_inside_unresolved_windows(
    media_tools, offline_pipeline, tmp_path,
):
    ffmpeg, ffprobe = media_tools
    video = tmp_path / "overlap.mp4"
    checked(
        ffmpeg, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x80:rate=25:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=431:sample_rate=48000:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=1.2",
        "-filter_complex", "[2:a]adelay=2000[b];[1:a][b]amix=inputs=2:duration=first:normalize=0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", video,
    )
    words = [word(.1, 1.65), word(2, 3.2, "Overlap."), word(3.5, 4.65)]
    regular = (DiarizationTurn(0, 5, "a"), DiarizationTurn(2, 3.2, "b"))
    output = tmp_path / "dub.mp4"
    root, calls = offline_pipeline(video, words, DiarizationResult(regular), output=output)
    assert {name for name, _ in calls} == {"SPEAKER_00.wav"}
    assert not any((seed - 42) % 1000 == 1 for _, seed in calls)
    assert_mp4(ffmpeg, ffprobe, output, 5)
    report = json.loads(next((root / "temp").rglob("quality_report.json")).read_text())
    assert report["summary"]["source_preserved"] == 1
    assert report["groups"][1]["audio_mode"] == "source_preserved"
    assert report["groups"][2]["original_group_index"] == 3

    def signal(path):
        result = checked(ffmpeg, "-v", "error", "-i", path, "-vn", "-ar", 48000, "-ac", 1, "-f", "f32le", "-")
        return np.frombuffer(result.stdout, dtype="<f4")

    def tone(window, hz):
        return abs(np.dot(window, np.exp(2j * np.pi * hz * np.arange(len(window)) / 48_000))) / len(window)

    original, dubbed = signal(video), signal(output)
    outside = slice(15_000, 27_000)
    inside = slice(108_000, 132_000)
    assert tone(dubbed[outside], 713) > 5 * tone(dubbed[outside], 431)
    assert tone(dubbed[inside], 713) < .01
    for frequency in (431, 997):
        assert tone(dubbed[inside], frequency) == pytest.approx(tone(original[inside], frequency), rel=.12)


def test_very_quiet_speech_gives_measured_diagnosis_before_synthesis(
    media_tools, offline_pipeline, tmp_path,
):
    ffmpeg, _ = media_tools
    source, quiet = tmp_path / "source.mp4", tmp_path / "quiet.mp4"
    make_media(ffmpeg, source, 3)
    checked(ffmpeg, "-v", "error", "-i", source, "-c:v", "copy", "-af", "volume=0.03", quiet)
    turn = DiarizationTurn(.1, 2.1, "a")
    output = tmp_path / "dub.mp4"
    with pytest.raises(SystemExit, match=r"too quiet.*RMS.*peak.*without clipping"):
        offline_pipeline(quiet, [word(.2, 2)], DiarizationResult((turn,), (turn,)), output=output)
    assert not output.exists()


def test_all_unresolved_overlap_exports_original_audio_without_loading_tts(
    media_tools, offline_pipeline, monkeypatch, tmp_path,
):
    ffmpeg, ffprobe = media_tools
    video = tmp_path / "all-mixed.mp4"
    checked(
        ffmpeg, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x80:rate=25:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=431:sample_rate=48000:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=997:sample_rate=48000:duration=3",
        "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", video,
    )
    monkeypatch.setattr(pipeline, "load_model", lambda: pytest.fail("No safe group needs TTS"))
    regular = (DiarizationTurn(0, 3, "a"), DiarizationTurn(0, 3, "b"))
    output = tmp_path / "kept-source.mp4"
    root, calls = offline_pipeline(video, [word(.2, 2.8)], DiarizationResult(regular), output=output)
    assert not calls
    assert_mp4(ffmpeg, ffprobe, output, 3)
    report = json.loads(next((root / "temp").rglob("quality_report.json")).read_text())
    assert report["summary"]["source_preserved"] == 1
    assert report["summary"]["passed"] == 0

    def samples(path):
        result = checked(ffmpeg, "-v", "error", "-i", path, "-vn", "-ac", 1, "-f", "f32le", "-")
        return np.frombuffer(result.stdout, dtype="<f4")

    # Check both the transcribed interval and its surroundings.
    original, output_audio = samples(video), samples(output)
    for window in (slice(2_400, 8_400), slice(48_000, 72_000), slice(137_000, 142_000)):
        assert np.corrcoef(original[window], output_audio[window])[0, 1] > .99
        assert np.linalg.norm(output_audio[window]) == pytest.approx(
            np.linalg.norm(original[window]), rel=.07
        )
