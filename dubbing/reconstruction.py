"""Rebuild the audio on the source timeline when a group keeps its original sound."""

from pathlib import Path


SAMPLE_RATE = 48000


def _write_constant(output, frames, value):
    import numpy as np

    block = np.full(SAMPLE_RATE, value, dtype="float32")
    while frames > 0:
        count = min(frames, len(block))
        output.write(block[:count])
        frames -= count


def _source_intervals(groups, fallback_indices, total_frames):
    if len(fallback_indices) == len(groups):
        return [(0, total_frames)]
    intervals = []
    for group in groups:
        if group.index not in fallback_indices:
            continue
        start = max(0, min(total_frames, round(group.start * SAMPLE_RATE)))
        end = max(0, min(total_frames, round(group.end * SAMPLE_RATE)))
        if end <= start:
            continue
        if intervals and start <= intervals[-1][1]:
            previous_start, previous_end = intervals[-1]
            intervals[-1] = previous_start, max(previous_end, end)
        else:
            intervals.append((start, end))
    return intervals


def render_with_source_audio(
    video, background, groups, fallback_indices, duration, wav_dir, workspace,
    destination, *, ffmpeg, run, probe_duration, min_tempo, max_tempo,
):
    """Mux dubbed speech and source windows without moving any picture or onset."""
    import soundfile as sf

    workspace = Path(workspace)
    speech = workspace / "timed-speech.wav"
    mask = workspace / "source-mask.wav"
    adjusted = workspace / "adjusted-speech.wav"
    total_frames = round(duration * SAMPLE_RATE)
    with sf.SoundFile(speech, "w", SAMPLE_RATE, 1, subtype="PCM_24") as out:
        for number, group in enumerate(groups):
            if group.index in fallback_indices:
                continue
            source = Path(wav_dir) / f"{number + 1}.wav"
            start = min(total_frames, round(group.start * SAMPLE_RATE))
            next_start = groups[number + 1].start if number + 1 < len(groups) else duration
            end = min(total_frames, round(next_start * SAMPLE_RATE))
            if end <= start:
                continue
            preferred = group.budget.preferred_duration if group.budget else group.duration
            tempo = min(max(probe_duration(source) / preferred, min_tempo), max_tempo)
            run([
                ffmpeg, "-y", "-i", source, "-af", f"atempo={tempo:.8f}",
                "-ar", SAMPLE_RATE, "-ac", "1", "-c:a", "pcm_s24le", adjusted,
            ])
            _write_constant(out, start - out.tell(), 0.0)
            with sf.SoundFile(adjusted) as dub:
                for block in dub.blocks(blocksize=SAMPLE_RATE, frames=end - start, dtype="float32"):
                    out.write(block)
        _write_constant(out, total_frames - out.tell(), 0.0)

    with sf.SoundFile(mask, "w", SAMPLE_RATE, 1, subtype="FLOAT") as out:
        for start, end in _source_intervals(groups, fallback_indices, total_frames):
            _write_constant(out, start - out.tell(), 0.0)
            # PCM16's largest positive sample is just below 1. Use float WAV
            # for the gain mask so source intervals completely suppress the dub.
            _write_constant(out, end - start, 1.0)
        _write_constant(out, total_frames - out.tell(), 0.0)

    stereo = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    run([
        ffmpeg, "-y", "-i", video, "-i", speech, "-i", background, "-i", mask,
        "-filter_complex",
        f"[1:a]{stereo},loudnorm=I=-20:TP=-2:LRA=11,aresample=48000,apad[dub];"
        f"[2:a]{stereo},volume=0.35,apad[bg];"
        "[dub][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix];"
        f"[0:a:0]aresample=48000:async=1:first_pts=0,{stereo},apad[original];"
        f"[3:a]pan=stereo|c0=c0|c1=c0,{stereo},asplit[keep][inverse];"
        "[inverse]aeval=1-val(0)|1-val(1)[replace];"
        "[mix][replace]amultiply[dub_window];[original][keep]amultiply[source_window];"
        "[dub_window][source_window]amix=inputs=2:duration=first:normalize=0[a]",
        "-map", "0:v:0", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.8f}", "-movflags", "+faststart",
        destination,
    ])
