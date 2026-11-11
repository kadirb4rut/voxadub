# Offline dubbing regression verification, November 11, 2026

The independent report tested an unmodified `63d21bf` checkout on Ubuntu 22.04 with an RTX 4090. It confirmed the earlier fixes from PR #9 and identified five offline regressions. This fix started from the same commit, which was the current `main`.

| Report item | Cause | Change |
| --- | --- | --- |
| R1, short or ambiguous phrases | A segment shorter than the 0.35 s evidence minimum could never pass it. Ambiguity then raised from reference lookup, even for a normal name inside a longer turn. | A short phrase needs at least 90% timestamp coverage from one identity and corroboration by a sustained regular turn. A competing speaker or overlap prevents the short-phrase override. Only confirmed adjacent same-speaker text can merge. The duration cap applies to the whole group. |
| R2, overlapping speech | A group with no clean same-speaker reference raised before synthesis. The generic reference error hid the distinction between overlap and a missing reference. | Known speakers can reuse an eligible clean same-speaker reference. Mixed audio never becomes a reference. If attribution or reference selection is unsafe, the job logs the interval and keeps the original audio there. Dubbed groups keep their source onsets and duration budgets. |
| D1, WebM export | The local writer used H.264 and AAC with the original file extension. Some wrappers predicted different paths. Input-derived intermediate files also shared the source basename across jobs. | Local, cloud, and lip-sync destinations use MP4 and keep the source stem. Default destinations with an existing file receive a numeric suffix. An explicit destination must be an unused MP4 path. Workers publish finished outputs without replacing an existing file and isolate input-derived artifacts per run. |
| D2, unsupported alignment language | The error arose in `whisperx.load_align_model`, before alignment. The sentence splitter depends on aligned `word_segments`. | The CLI names the detected language and explains how to correct the detection or provide a speech track. Empty transcriptions bypass alignment. Unsupported explicit language selections report the supported choices before inference. No fabricated word timings or replacement language enter the translation stage. |
| D3, quiet speech | The reference floor rejected otherwise eligible samples with the same message as overlap, clipping or short duration. | The reference error reports measured RMS and peak and advises increasing the source speech level without clipping. A job with only unusable references and a measured quiet reference exits with this diagnosis. |

Reconstruction uses the original picture timeline for inputs converted to MP4 from another container and for any job with source-audio windows. This also avoids a MoviePy end-of-clip failure observed in the Ubuntu MKV fixture. The extraction retains FFmpeg's existing audio stream selection and preserves the stream's offset against the picture. Outside a source-audio window, the renderer mixes dubbed speech and separated background using the existing levels. Inside the window, the original mix replaces both streams. If no group has a safe reference and none yielded a quiet-speech diagnosis, the export keeps the complete source mix and logs that no groups were dubbed. The quality report marks source-preserved groups and excludes them from synthesized-audio pass counts.

## Deterministic verification

`tests/test_offline_regressions.py` runs the nine reported durations/layouts through real FFmpeg and the offline pipeline with deterministic ASR, diarization, translation and TTS doubles. It covers six completed MP4 outputs, three validation rejections, the 0.30 s “Hector.” fragment, true speaker boundaries, overlap with and without a clean reference, and the exact timing window where a mixed source is retained. The test suite also exercises MP4, VP9/Opus WebM, MOV, MKV and HEVC-in-MP4, late output collisions, missing Welsh alignment and quiet audio. The original PR #9 tests are part of the same suite.

The two-track integration test uses separable frequencies to verify that both source tracks survive in the overlap, that synthesized speech is absent in that interval, and that neighboring dubbed speech remains in place. Additional cases verify that complete source preservation uses the same stereo stream as speech analysis and retains a 450 ms audio offset. These tests measure routing and timing with synthetic signals.

The full suite passed 391 Python tests on macOS ARM64 and in Ubuntu CI, including headless GUI HTTP endpoints and the earlier PR #9 safeguards. The 16 optional Live Dubbing skips require Silero or a spaCy English model. All three browser extension test files passed in both environments.

## End-to-end GPU verification

The GPU runs used one Colab G4 NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), Python 3.10, WhisperX 3.3.1, Hy-MT2 1.8B, VoxCPM2 revision `32279ef` and pyannote Community-1. The CLI ran real separation, transcription, alignment, diarization, translation, synthesis and muxing. Portuguese was the target language. The fixtures were recreated with the repository's `scripts/create_stability_e2e_fixtures.py` generator, trimming its trailing-silence video to the reported 68 s. Each successful output decoded through FFmpeg with error checking enabled.

| Case | Input | End-to-end result |
| --- | --- | --- |
| T1 | 15 s, one speaker | Passed, 15.083 s H.264/AAC MP4 |
| T2 | 60 s, one speaker | Passed, 60.125 s H.264/AAC MP4 |
| T3 | 2.2 s short clip | Passed, 2.250 s H.264/AAC MP4 |
| T4 | 75 s, two speakers | Passed, 75.000 s H.264/AAC MP4. Two speaker identities were assigned and the original audio was kept at 57.76–61.82 s. |
| T5 | Video without audio | Rejected before inference with a missing audio stream message |
| T6 | Audio-only WAV | Rejected before inference with a missing video stream message |
| T7 | 68 s, trailing silence | Passed, 68.083 s H.264/AAC MP4 |
| T8 | 15 s, portrait at 30 fps | Passed, 15.067 s H.264/AAC MP4 with portrait dimensions |
| T9 | 0.6 s video | Rejected before inference with the one-second minimum-duration message |

On the separate R2 mixed-voice fixture, Community-1 detected overlap in two transcript segments. The pipeline synthesized the resolved groups and kept the source mix at 1.13–3.14 s and 3.20–6.52 s. The 12.000 s MP4 decoded fully. Decoded source and output audio had a waveform correlation of approximately 0.99995 in windows within both retained intervals, with at most 0.02 dB level difference. A window after the retained intervals contained dubbed audio. The two-speaker T4 fixture independently showed approximately 0.99996 waveform correlation in its retained interval.

T2 also completed with `--num-speakers 1`. The VP9/Opus fixture produced a 15.000 s H.264/AAC MP4 through the CLI. A second GPU run used the headless GUI's actual upload, run, status, authorization and download endpoints. It preserved a preexisting same-stem file, published `D1_vp9_opus-2.mp4`, refused playback before authorization, served an authorized range with HTTP 206 and returned a fully decodable download. The same GUI rejected the missing-audio fixture with an actionable message.

The 330 Hz tone fixture was detected as `cy` by Whisper and yielded an empty transcript. The pipeline reported that it found no usable speech and exited before synthesis. A separate controlled nonempty Welsh transcript reached the installed WhisperX `load_align_model()` and verified the actionable missing-aligner message. The synthetic Welsh audio by itself yielded no usable Whisper transcript. Explicit `--source-language cy` reported the supported choices before extraction. The quiet speech fixture reported RMS 0.0024 and peak 0.0211 and exited before synthesis, advising the operator to raise the source speech level without clipping.

An initial Tesla T4 runtime was terminated after the VoxCPM2 loading process received SIGKILL. The completed inference evidence comes from the RTX PRO 6000 G4. The GPU runtimes were released after evidence was saved.

## Remaining limitations

The tester's media and RTX 4090 environment were unavailable. The recreated fixtures match the reported durations, speaker layouts and invalid input types. Model confidence depends on the input, so some intervals can keep their original language. The log and quality report give the affected timestamps. Some translations reached the existing tempo bounds on the recreated fixtures. The GPU runs verify media decoding, waveform timing and model execution. Subjective speech quality and optional LatentSync inference were outside these GPU checks. With diarization disabled, the legacy segment-reference mode has no overlap detection. Offline dubbing requires a supported WhisperX word aligner. Quiet source speech requires operator adjustment.
