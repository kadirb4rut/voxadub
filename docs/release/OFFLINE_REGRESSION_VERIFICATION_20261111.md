# Offline dubbing regression verification, November 11, 2026

The independent report tested an unmodified `63d21bf` checkout on Ubuntu 22.04 with an RTX 4090. It confirmed the earlier fixes from PR #9 and identified five offline regressions. This fix started from the same commit, which was the current `main`.

| Report item | Cause | Change |
| --- | --- | --- |
| R1, short or ambiguous phrases | A segment shorter than the 0.35 s evidence minimum could never pass it. Ambiguity then raised from reference lookup, even for a normal name inside a longer turn. | A short phrase needs at least 90% timestamp coverage from one identity and corroboration by a sustained regular turn. A competing speaker or overlap prevents the short-phrase override. Only confirmed adjacent same-speaker text can merge. The duration cap applies to the whole group. |
| R2, overlapping speech | A group with no clean same-speaker reference raised before synthesis. The generic reference error hid the distinction between overlap and a missing reference. | Known speakers can reuse an eligible clean same-speaker reference. Mixed audio never becomes a reference. If attribution or reference selection is unsafe, the job logs the interval and keeps the original audio there. Dubbed groups keep their source onsets and duration budgets. |
| D1, WebM export | The local writer used H.264 and AAC with the original file extension. Some wrappers predicted different paths. Input-derived intermediate files also shared the source basename across jobs. | Local, cloud, and lip-sync destinations use MP4 and keep the source stem. Collisions receive a numeric suffix for default destinations. An explicit destination must be an unused MP4 path. Workers publish finished outputs without replacing an existing file and isolate input-derived artifacts per run. |
| D2, unsupported alignment language | The error arose in `whisperx.load_align_model`, before alignment. The sentence splitter depends on aligned `word_segments`. | The CLI names the detected language and explains how to correct the detection or provide a speech track. Empty transcriptions bypass alignment. No fabricated word timings or replacement language enter the translation stage. |
| D3, quiet speech | The reference floor rejected otherwise eligible samples with the same message as overlap, clipping or short duration. | The reference error reports measured RMS and peak and advises increasing the source speech level without clipping. A job with only unusable references and a measured quiet reference exits with this diagnosis. |

When any group needs source audio, reconstruction uses the original video timeline. It mixes dubbed speech and separated background as usual outside the affected intervals. Inside them, the original mix replaces both streams, so background audio does not play twice. If no group has a safe reference and none yielded a quiet-speech diagnosis, the export preserves the complete source mix and logs that no groups were dubbed. The quality report identifies source-preserved groups and excludes them from synthesized-audio pass counts.

## Deterministic verification

`tests/test_offline_regressions.py` runs the nine reported durations/layouts through real FFmpeg and the offline pipeline with deterministic ASR, diarization, translation and TTS doubles. It covers six completed MP4 outputs, three validation rejections, the 0.30 s “Hector.” fragment, true speaker boundaries, overlap with and without a clean reference, and the exact timing window where a mixed source is retained. The test suite also exercises MP4, VP9/Opus WebM, MOV, MKV and HEVC-in-MP4, late output collisions, missing Welsh alignment and quiet audio. The original PR #9 tests are part of the same suite.

The two-track integration test uses separable frequencies to verify that both source tracks survive in the overlap, that synthesized speech is absent in that interval, and that neighboring dubbed speech remains in place. The test measures routing and timing with synthetic signals.

On macOS ARM64 and Python 3.10, the full Python suite reported 388 passed and 16 skipped. The skips were existing optional Live Dubbing tests requiring Silero or a spaCy English model. All three browser extension test files passed. The MKV fixture produced a playable MP4 within the duration tolerance. MoviePy repeated its last valid frame once when the intermediate clip ended a frame before its reported duration.

## Remaining limitations

The tester's original media was unavailable. Recreated fixtures match the reported durations, speaker layouts and invalid input types. Model confidence depends on the input, so some intervals can keep their original language. The log and quality report give the affected timestamps. With diarization disabled, the legacy segment-reference mode has no overlap detection. VoxaDub currently requires a supported WhisperX word aligner for offline dubbing. The application does not apply automatic gain to quiet source speech.
