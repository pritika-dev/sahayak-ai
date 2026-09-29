"""
Speech-to-text for call recordings that officers upload (Round 21).

A recording uploaded from the dashboard has no transcript, and the SVI
needs both the words and the voice. This module turns the audio into
text, on the server, when an engine is available.

ENGINES (set SAHAYK_STT in the host's environment)
  auto     default — use Whisper if faster-whisper is installed, else none
  whisper  faster-whisper (open-source Whisper, runs on CPU, multilingual:
           Hindi, Gujarati, Marathi, Tamil, Bengali, English…). Install with
           `pip install -r requirements-stt.txt`. The model downloads from
           Hugging Face on first use (tiny ≈ 75 MB, base ≈ 145 MB); pick the
           size with SAHAYK_STT_MODEL. Needs roughly 300–600 MB of extra RAM,
           so it does NOT fit Render's free 512 MB plan together with the
           rest of the app — use a laptop or a paid instance for it.
  off      no automatic transcription

With no engine, nothing breaks: the officer can type or paste what was
said, and if they leave it empty the recording is still analysed for
voice stress and filed as "words not analysed yet — add a transcript",
never as low risk.

Bhashini (the government's Indian-language speech API) is the intended
production engine; it needs MeitY credentials and slots in here as a
third engine with the same transcribe() signature.
"""
import os
import tempfile
import threading

ENGINE = os.environ.get("SAHAYK_STT", "auto").lower()
MODEL_SIZE = os.environ.get("SAHAYK_STT_MODEL", "tiny")

_model = None
_load_error = None
_load_failed_at = 0.0
_RETRY_AFTER_S = 600          # a failed model download is retried after 10 minutes
_lock = threading.Lock()


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def engine_name() -> str:
    if ENGINE == "off":
        return "off"
    if ENGINE in ("auto", "whisper") and _whisper_available():
        return "whisper"
    return "off"


def _load_whisper():
    global _model, _load_error, _load_failed_at
    import time
    with _lock:
        if _model is not None:
            return _model
        if _load_error is not None and time.time() - _load_failed_at < _RETRY_AFTER_S:
            return None
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            _load_error = None
        except Exception as exc:          # download blocked, out of memory, bad size name…
            _load_error = f"{type(exc).__name__}: {exc}"[:200]
            _load_failed_at = time.time()
        return _model


def status() -> dict:
    name = engine_name()
    return {"engine": name, "model": MODEL_SIZE if name == "whisper" else None,
            "error": _load_error}


def transcribe(audio_bytes: bytes, language: str = None):
    """Returns (text, None) or (None, reason). Never raises."""
    if engine_name() != "whisper":
        return None, "no speech-to-text engine installed on this server"
    model = _load_whisper()
    if model is None:
        return None, f"speech-to-text model could not load ({_load_error})"
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as fh:
            fh.write(audio_bytes)
            path = fh.name
        lang = (language or "").split("-")[0] or None
        segments, _info = model.transcribe(path, language=lang, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return (text, None) if text else (None, "no speech found in the recording")
    except Exception as exc:
        return None, f"transcription failed ({type(exc).__name__})"
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
