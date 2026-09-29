"""
Real speech-feature extraction from an uploaded audio file. No pretrained
model needed for these three signals — they're computed directly from
the waveform, so this works fully offline.

- pitch_variance: how much the fundamental frequency (pitch) wobbles.
  Higher = more erratic/strained voice.
- pause_ratio: fraction of the clip that's silence/long pauses.
- speech_rate_delta: rough proxy for unusually fast or slow speech,
  based on voiced-segment density.

These feed straight into scoring.score_text(text, speech_stress=...).
A trained speech-emotion classifier is the natural upgrade path here
(swap this function's output for the model's), but that needs labeled
audio data and a model host this environment doesn't have — the 0-100
interface stays the same either way.

IMPORTANT (root cause of the "voice gives no response" bug): browsers
record audio as webm/opus via MediaRecorder, and decoding that requires
ffmpeg to be installed and on PATH — librosa can't read it otherwise.
If ffmpeg is missing (common on a fresh Windows install — it doesn't
ship with Python or with any of the pip packages here), every single
voice message would previously crash the whole request with an
unhandled exception, which is exactly why the chat went silent instead
of replying: the crash happened before a reply was ever generated.
Every step below is now wrapped so a decoding/analysis failure degrades
to a neutral 0 signal instead of killing the request — the transcript
still gets scored and the conversation still gets a reply either way.
Install ffmpeg for the richer voice signal (see README).
"""
import io
import numpy as np
import librosa

_NEUTRAL = {"pitch_variance": 0.0, "pause_ratio": 0.0, "speech_rate_delta": 0.0, "audio_analysis_ok": False}


def warm_up():
    """Runs pitch detection once on a tiny synthetic clip so the
    one-time compilation cost (see main.py's startup hook) happens now
    instead of on the first real voice message."""
    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
    librosa.pyin(silence, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=16000)


def extract_speech_features(audio_bytes: bytes, channel=None, max_seconds=None) -> dict:
    """channel / max_seconds are used only for 14566 call recordings
    (calls.py): a stereo call recording usually has the caller on one
    channel and the agent on the other, so only the caller's channel is
    analysed when it is known; and a long call is capped so pitch
    tracking stays fast. Chat voice notes use the defaults, unchanged."""
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000,
                             mono=channel is None, duration=max_seconds)
        if channel is not None and y.ndim == 2:
            y = y[min(int(channel), y.shape[0] - 1)]
        elif y.ndim == 2:
            y = librosa.to_mono(y)
    except Exception:
        return dict(_NEUTRAL)

    if len(y) < sr * 0.3:  # too short to say anything meaningful
        return dict(_NEUTRAL, audio_analysis_ok=True)

    pitch_variance = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])
        if len(voiced_f0) > 1:
            pitch_variance = float(np.clip(np.std(voiced_f0) / 80.0, 0, 1))
    except Exception:
        pass

    pause_ratio = 0.0
    try:
        intervals = librosa.effects.split(y, top_db=30)
        voiced_samples = sum(e - s for s, e in intervals)
        pause_ratio = float(np.clip(1 - (voiced_samples / len(y)), 0, 1))
    except Exception:
        pass

    speech_rate_delta = 0.0
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration_s = len(y) / sr
        onset_rate = len(onsets) / max(duration_s, 0.1)
        speech_rate_delta = float(np.clip(abs(onset_rate - 3.0) / 5.0, 0, 1))  # ~3/s = calm baseline
    except Exception:
        pass

    return {
        "pitch_variance": round(pitch_variance, 3),
        "pause_ratio": round(pause_ratio, 3),
        "speech_rate_delta": round(speech_rate_delta, 3),
        "audio_analysis_ok": True,
    }


def speech_stress_from(features: dict) -> float:
    """Same 0-100 blend main.py uses for chat voice notes."""
    return round(100 * (0.4 * features["pitch_variance"] + 0.35 * features["pause_ratio"]
                        + 0.25 * features["speech_rate_delta"]), 1)
