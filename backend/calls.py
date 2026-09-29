"""
14566 call ingestion — turns every finished helpline call into a scored
case on the authority dashboard, automatically.

HOW IT FITS
-----------
A voice-agent / IVR platform answers the call, records and transcribes
it, and when the call ends it POSTs a "call finished" webhook to
`/webhook/calls?provider=<name>`. This module:

  1. stores the raw payload verbatim in `call_events` (append-only), so a
     case can always be re-scored later if the scoring rules change;
  2. normalises the payload into one flat shape, whatever the provider
     (see ADAPTERS below);
  3. keeps only what the CALLER said — the bot/agent side of the
     conversation is never scored, or every call would pick up the
     agent's own "are you safe right now?" as a disclosure;
  4. scores it with exactly the same `scoring.score_text` the chat uses
     (rules + trained model + safety floors + routing tags);
  5. creates the case — or, if the same call_id arrives again (providers
     retry webhooks), updates the existing case instead of duplicating it.

The caller never sees a score. Nothing here triggers police action;
routing tags are suggestions for a human, exactly as in the chat.

PROVIDERS
---------
Adapters exist for OmniDimension, Vapi, Retell, Bland, ElevenLabs
Conversational AI, Twilio and a documented generic JSON shape. They
work by sniffing the fields each platform is known to send. The
Vapi / Retell / Bland / ElevenLabs / Twilio shapes follow those
platforms' public webhook formats; OmniDimension's is best-effort. If a
platform changes its payload, the generic sniffing fallback still finds
the common field names — and the raw payload is kept either way, so
nothing is lost while an adapter is fixed.

PRIVACY
-------
- The caller's number is masked to its last 4 digits on the case.
- Raw payloads (which can contain the full number) are stored for
  re-processing only; no endpoint exposes them.
- If a payload says consent was explicitly REFUSED, the call is logged
  but not analysed and no case is created.
"""
import base64
import ipaddress
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import config
import database as db
from audio_features import extract_speech_features, speech_stress_from
from scoring import score_text

DEMO_AUDIO_DIR = Path(__file__).parent / "demo_audio"

CALLER_ROLES = {"user", "customer", "caller", "human", "client", "patient", "victim", "complainant"}
AGENT_ROLES = {"assistant", "agent", "bot", "ai", "system", "tool", "function", "operator"}

MAX_PAYLOAD_BYTES = 1_000_000


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _get(d, *path, default=None):
    """Safe nested lookup: _get(p, "call", "customer", "number")."""
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _first(*values):
    for v in values:
        if v not in (None, "", [], {}):
            return v
    return None


def _to_epoch(value):
    """Accepts ISO-8601 strings, epoch seconds or epoch milliseconds."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return value / 1000.0 if value > 10_000_000_000 else float(value)
    try:
        s = str(value).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None


def mask_number(number):
    if not number:
        return None
    digits = re.sub(r"\D", "", str(number))
    if len(digits) <= 4:
        return "••••"
    return "•" * (len(digits) - 4) + digits[-4:]


_SPEAKER_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]{0,20})\s*:\s*(.*)$")


def caller_text_from_turns(turns):
    """turns: list of dicts with a role/speaker key and a text key, in
    whatever naming the provider uses. Keeps caller turns only; if no
    roles are present at all, keeps everything (better to score an
    unlabelled transcript than drop it)."""
    kept, labelled = [], False
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        role = str(_first(t.get("role"), t.get("speaker"), t.get("user"), t.get("from"), "") or "").lower()
        text = _first(t.get("message"), t.get("content"), t.get("text"), t.get("transcript"), "")
        if not isinstance(text, str) or not text.strip():
            continue
        if role:
            labelled = True
        if role in AGENT_ROLES:
            continue
        kept.append(text.strip())
    return " ".join(kept) if (kept or labelled) else ""


def caller_text_from_string(transcript: str):
    """Plain-text transcripts usually look like 'AI: ...\\nUser: ...'.
    Keep the caller's lines; if nothing is labelled, keep all of it."""
    if not transcript:
        return ""
    lines = [l for l in str(transcript).splitlines() if l.strip()]
    labelled, kept = False, []
    for line in lines:
        m = _SPEAKER_LINE.match(line)
        if m:
            labelled = True
            speaker = m.group(1).strip().lower()
            if speaker in AGENT_ROLES or speaker.startswith(("agent", "assistant", "bot", "ai")):
                continue
            kept.append(m.group(2).strip())
        else:
            kept.append(line.strip())
    return " ".join(kept) if labelled else " ".join(l.strip() for l in lines)


def _normalised(provider, call_id, caller, transcript_text, started=None, ended=None,
                duration=None, language=None, recording_url=None, consent=None, recording_b64=None):
    started_ts, ended_ts = _to_epoch(started), _to_epoch(ended)
    if duration in (None, "") and started_ts and ended_ts:
        duration = max(0.0, ended_ts - started_ts)
    try:
        duration = round(float(duration), 1) if duration not in (None, "") else None
    except (TypeError, ValueError):
        duration = None
    return {
        "provider": provider,
        "call_id": str(call_id) if call_id not in (None, "") else None,
        "caller_masked": mask_number(caller),
        "caller_text": (transcript_text or "").strip(),
        "started_at": started_ts,
        "ended_at": ended_ts,
        "duration_s": duration,
        "language": language,
        "recording_url": recording_url,
        "recording_b64": recording_b64,
        "consent": consent,
    }


# --------------------------------------------------------------------------
# Provider adapters
# --------------------------------------------------------------------------

def adapt_vapi(p):
    msg = p.get("message", p)
    call = msg.get("call", {}) or {}
    artifact = msg.get("artifact", {}) or {}
    turns = _first(artifact.get("messages"), msg.get("messages"))
    text = caller_text_from_turns(turns) if turns else caller_text_from_string(
        _first(artifact.get("transcript"), msg.get("transcript"), ""))
    return _normalised(
        "vapi", _first(call.get("id"), msg.get("callId")),
        _first(_get(call, "customer", "number"), _get(msg, "customer", "number")),
        text, _first(msg.get("startedAt"), call.get("startedAt")),
        _first(msg.get("endedAt"), call.get("endedAt")), msg.get("durationSeconds"),
        recording_url=_first(
            artifact.get("stereoRecordingUrl") if config.CALL_AUDIO_CALLER_CHANNEL else None,
            msg.get("recordingUrl"), artifact.get("recordingUrl")),
    )


def adapt_retell(p):
    call = p.get("call", p)
    turns = call.get("transcript_object")
    text = caller_text_from_turns(turns) if turns else caller_text_from_string(call.get("transcript", ""))
    return _normalised(
        "retell", call.get("call_id"), call.get("from_number"), text,
        call.get("start_timestamp"), call.get("end_timestamp"),
        recording_url=call.get("recording_url"),
    )


def adapt_bland(p):
    turns = p.get("transcripts")
    text = caller_text_from_turns(turns) if turns else caller_text_from_string(
        _first(p.get("concatenated_transcript"), p.get("transcript"), ""))
    minutes = p.get("call_length")
    return _normalised(
        "bland", _first(p.get("call_id"), p.get("c_id")), _first(p.get("from"), p.get("from_number")),
        text, p.get("started_at"), _first(p.get("end_at"), p.get("ended_at")),
        (float(minutes) * 60) if isinstance(minutes, (int, float)) else None,
        recording_url=p.get("recording_url"),
    )


def adapt_elevenlabs(p):
    data = p.get("data", p)
    meta = data.get("metadata", {}) or {}
    return _normalised(
        "elevenlabs", _first(data.get("conversation_id"), data.get("call_id")),
        _first(_get(meta, "phone_call", "external_number"), meta.get("caller_id")),
        caller_text_from_turns(data.get("transcript")),
        meta.get("start_time_unix_secs"), None, meta.get("call_duration_secs"),
        recording_b64=data.get("full_audio"),   # ElevenLabs "post_call_audio" webhook
    )


def adapt_twilio(p):
    # Twilio posts form fields (CallSid, From, TranscriptionText…). Its
    # RecordingUrl has no extension; adding .wav asks Twilio for WAV.
    rec = p.get("RecordingUrl")
    if rec and not re.search(r"\.(wav|mp3)$", rec):
        rec = rec + ".wav"
    return _normalised(
        "twilio", _first(p.get("CallSid"), p.get("TranscriptionSid")), p.get("From"),
        caller_text_from_string(_first(p.get("TranscriptionText"), p.get("SpeechResult"), "")),
        duration=_first(p.get("RecordingDuration"), p.get("CallDuration")),
        recording_url=rec,
    )


def adapt_omnidimension(p):
    # Best-effort: OmniDimension post-call payloads carry a call report
    # with the full conversation. Falls back to generic sniffing.
    report = _first(p.get("call_report"), p.get("report"), p.get("data"), {}) or {}
    convo = _first(report.get("full_conversation"), report.get("transcript"), p.get("full_conversation"))
    text = caller_text_from_turns(convo) if isinstance(convo, list) else caller_text_from_string(convo or "")
    return _normalised(
        "omnidimension", _first(p.get("call_id"), report.get("call_id"), p.get("id")),
        _first(p.get("phone_number"), p.get("from_number"), report.get("phone_number")), text,
        duration=_first(report.get("call_duration_in_seconds"), p.get("duration")),
        recording_url=_first(report.get("recording_url"), p.get("recording_url")),
    )


def adapt_generic(p):
    """The documented shape for any other platform, and the fallback:
        {"call_id": "...", "caller": "+91...", "language": "hi",
         "started_at": "2026-09-29T10:00:00Z", "ended_at": "...",
         "duration_seconds": 312, "consent": true,
         "transcript": "User: ...\\nAgent: ..."   OR
         "transcript": [{"role": "user", "text": "..."}, ...]}
    Also sniffs the common alternatives each field goes by."""
    transcript = _first(p.get("transcript"), p.get("messages"), p.get("conversation"),
                        p.get("full_conversation"), p.get("text"), "")
    text = caller_text_from_turns(transcript) if isinstance(transcript, list) \
        else caller_text_from_string(transcript)
    return _normalised(
        p.get("_provider", "generic"),
        _first(p.get("call_id"), p.get("id"), p.get("callId"), p.get("sid")),
        _first(p.get("caller"), p.get("from"), p.get("from_number"), p.get("phone_number"), p.get("caller_number")),
        text, _first(p.get("started_at"), p.get("start_time")), _first(p.get("ended_at"), p.get("end_time")),
        _first(p.get("duration_seconds"), p.get("duration")),
        language=p.get("language"), recording_url=p.get("recording_url"),
        consent=p.get("consent"), recording_b64=p.get("recording_base64"),
    )


ADAPTERS = {
    "vapi": adapt_vapi,
    "retell": adapt_retell,
    "bland": adapt_bland,
    "elevenlabs": adapt_elevenlabs,
    "twilio": adapt_twilio,
    "omnidimension": adapt_omnidimension,
    "generic": adapt_generic,
}


def detect_provider(p: dict) -> str:
    """Used when ?provider= is missing."""
    if "message" in p and isinstance(p["message"], dict) and (
            "call" in p["message"] or p["message"].get("type") == "end-of-call-report"):
        return "vapi"
    if isinstance(p.get("call"), dict) and ("transcript_object" in p["call"] or "call_id" in p["call"]):
        return "retell"
    if "concatenated_transcript" in p or "transcripts" in p:
        return "bland"
    if p.get("type") == "post_call_transcription" or "conversation_id" in (p.get("data") or {}):
        return "elevenlabs"
    if "CallSid" in p:
        return "twilio"
    if "call_report" in p or "full_conversation" in p:
        return "omnidimension"
    return "generic"


def normalise(payload: dict, provider: str = None) -> dict:
    provider = (provider or "").lower().strip() or detect_provider(payload)
    adapter = ADAPTERS.get(provider, adapt_generic)
    try:
        call = adapter(payload)
    except Exception:           # a malformed payload for one adapter -> try generic sniffing
        call = adapt_generic(payload)
    if adapter is not adapt_generic:
        fallback = adapt_generic(payload)
        for key in ("caller_text", "recording_url", "recording_b64"):
            if not call.get(key) and fallback.get(key):
                call[key] = fallback[key]
    call["provider"] = provider if provider in ADAPTERS else "generic"
    if "consent" in payload:
        call["consent"] = payload["consent"]
    return call


def _guess_language(text: str) -> str:
    for ch in text:
        name = ord(ch)
        if 0x0900 <= name <= 0x097F:
            return "hi"
        if 0x0A80 <= name <= 0x0AFF:
            return "gu"
        if 0x0B80 <= name <= 0x0BFF:
            return "ta"
        if 0x0980 <= name <= 0x09FF:
            return "bn"
    return "en"


# --------------------------------------------------------------------------
# Call recording: download + voice analysis
# --------------------------------------------------------------------------
#
# The recording link comes from outside, so fetching it is guarded like
# any server-side request to an untrusted URL (SSRF): http/https only,
# the host must resolve ONLY to public internet addresses (no localhost,
# 10.x, 192.168.x, cloud metadata 169.254.169.254…), every redirect is
# re-checked, and the download is capped in size and time. demo://<name>
# serves the bundled synthetic clips in backend/demo_audio/ for the
# dashboard's "Simulate a 14566 call" button.

class _Blocked(Exception):
    pass


def _check_public(url: str):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise _Blocked("only http/https recording links are fetched")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        raise _Blocked("recording host does not resolve")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise _Blocked("recording host is not a public address")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_recording(url: str):
    """Returns (audio_bytes, None) or (None, reason)."""
    if url.startswith("demo://"):
        name = url[len("demo://"):]
        if not re.fullmatch(r"[a-z0-9_]+\.wav", name) or not (DEMO_AUDIO_DIR / name).is_file():
            return None, "unknown demo recording"
        return (DEMO_AUDIO_DIR / name).read_bytes(), None
    try:
        _check_public(url)
        opener = urllib.request.build_opener(_CheckedRedirects)
        req = urllib.request.Request(url, headers={"User-Agent": "Sahayak-14566/1.0"})
        if config.RECORDING_AUTH_USER:
            token = base64.b64encode(
                f"{config.RECORDING_AUTH_USER}:{config.RECORDING_AUTH_PASS}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        limit = int(config.CALL_AUDIO_MAX_MB * 1024 * 1024)
        with opener.open(req, timeout=config.CALL_AUDIO_TIMEOUT_S) as resp:
            if int(resp.headers.get("Content-Length") or 0) > limit:
                return None, "recording larger than the size limit"
            data = resp.read(limit + 1)
        if len(data) > limit:
            return None, "recording larger than the size limit"
        return data, None
    except _Blocked as exc:
        return None, str(exc)
    except Exception as exc:                      # network error, 4xx/5xx, timeout
        return None, f"download failed ({type(exc).__name__})"


def analyse_audio_bytes(audio: bytes, channel=None) -> dict:
    """Voice features for one recording. channel: None = use the
    configured caller channel; "" = mixed; "0"/"1" = that channel."""
    if channel is None:
        channel = config.CALL_AUDIO_CALLER_CHANNEL
    channel = str(channel or "")
    features = extract_speech_features(
        audio, channel=int(channel) if channel.strip().isdigit() else None,
        max_seconds=config.CALL_AUDIO_MAX_SECONDS)
    if not features.get("audio_analysis_ok"):
        return {"status": "not analysed: audio format not readable"}
    return {"status": "analysed", "speech_stress": speech_stress_from(features),
            "pitch_variance": features["pitch_variance"], "pause_ratio": features["pause_ratio"],
            "speech_rate_delta": features["speech_rate_delta"],
            "caller_channel": channel or "mixed"}


def analyse_call_audio(call: dict) -> dict:
    """Downloads (or decodes) the call recording and measures the caller's
    voice. Never raises: any failure is recorded and the call is scored
    on its words alone, exactly as before this feature existed."""
    if not config.CALL_AUDIO_ENABLED:
        return {"status": "voice analysis switched off"}
    audio, reason = None, None
    if call.get("recording_b64"):
        try:
            audio = base64.b64decode(call["recording_b64"], validate=False)
        except (ValueError, TypeError):
            reason = "recording data could not be decoded"
    elif call.get("recording_url"):
        audio, reason = fetch_recording(call["recording_url"])
    else:
        return {"status": "no recording sent by the platform"}
    if audio is None:
        return {"status": f"not analysed: {reason}"}
    return analyse_audio_bytes(audio)


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

def ingest(payload: dict, provider: str = None) -> dict:
    """Stores, normalises, scores and files one finished call. Returns a
    small receipt for the calling platform (never the transcript)."""
    raw_id = db.insert_call_event(provider or "", json.dumps(payload, ensure_ascii=False))
    call = normalise(payload, provider)
    if not call["call_id"]:
        call["call_id"] = f"{call['provider']}-noid-{raw_id}"

    if call["consent"] is False or str(call["consent"]).lower() in ("false", "no", "0", "refused"):
        db.upsert_call(call, case_id=None, status="consent_refused")
        return {"ok": True, "call_id": call["call_id"], "status": "consent_refused", "case_id": None}

    existing = db.get_call(call["call_id"])
    if not call["caller_text"]:
        # Some platforms send the audio in a second webhook after the
        # transcript (e.g. ElevenLabs post_call_audio): re-score the case
        # the transcript already created, now with the voice added.
        if existing and existing["case_id"] and (call.get("recording_url") or call.get("recording_b64")):
            case = db.get_case(existing["case_id"])
            call["caller_text"] = (case or {}).get("transcript") or ""
        if not call["caller_text"]:
            db.upsert_call(call, case_id=None, status="empty_transcript")
            return {"ok": True, "call_id": call["call_id"], "status": "empty_transcript", "case_id": None}

    language = call["language"] or _guess_language(call["caller_text"])
    voice = analyse_call_audio(call)
    result = score_text(call["caller_text"],
                        speech_stress=voice.get("speech_stress", 0.0),
                        has_speech_data=voice["status"] == "analysed")
    result["channel"] = "call"
    result["call"] = {
        "call_id": call["call_id"], "provider": call["provider"],
        "caller_masked": call["caller_masked"], "duration_s": call["duration_s"],
        "started_at": call["started_at"], "recording_url": call["recording_url"],
        "voice": voice,
        "consent": call["consent"] if call["consent"] is not None
        else "not sent by platform — IVR consent notice assumed",
    }

    if existing and existing["case_id"]:
        db.update_case(existing["case_id"], call["caller_text"], result)
        db.upsert_call(call, case_id=existing["case_id"], status="rescored")
        return {"ok": True, "call_id": call["call_id"], "status": "updated", "case_id": existing["case_id"]}

    case_id = db.insert_case(True, language, call["caller_text"], result)
    db.upsert_call(call, case_id=case_id, status="scored")
    return {"ok": True, "call_id": call["call_id"], "status": "created", "case_id": case_id}


# --------------------------------------------------------------------------
# Demo: sample calls in different provider formats
# --------------------------------------------------------------------------

def sample_payloads():
    """Realistic finished-call payloads, one per provider format, used by
    the dashboard's "Simulate a 14566 call" button and tools/demo_calls.py.
    Synthetic — no real caller data."""
    now = int(time.time())
    uid = f"{now}{int(time.time() * 1000) % 1000:03d}"
    calm = DEMO_AUDIO_DIR / "calm_caller.wav"
    calm_b64 = base64.b64encode(calm.read_bytes()).decode() if calm.is_file() else None
    return [
        ("vapi", {"message": {
            "type": "end-of-call-report",
            "call": {"id": f"vapi-{uid}", "customer": {"number": "+919812345678"}},
            "startedAt": datetime.fromtimestamp(now - 240, timezone.utc).isoformat(),
            "endedAt": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "artifact": {"messages": [
                {"role": "assistant", "message": "Namaste, this is the 14566 helpline. Are you safe right now?"},
                {"role": "user", "message": "Upper caste men from our village beat my husband with sticks yesterday."},
                {"role": "assistant", "message": "I am so sorry. Is anyone hurt badly?"},
                {"role": "user", "message": "He is in hospital. Now they are telling us to take back the case or they will burn our house."},
            ], "recordingUrl": "demo://distressed_caller.wav"},
        }}),
        ("retell", {"event": "call_ended", "call": {
            "call_id": f"retell-{uid}", "from_number": "+917012345678",
            "start_timestamp": (now - 300) * 1000, "end_timestamp": now * 1000,
            "transcript_object": [
                {"role": "agent", "content": "14566 helpline, how can I help you?"},
                {"role": "user", "content": "मेरी बेटी के साथ गांव के लड़कों ने गलत काम किया है, पुलिस शिकायत नहीं लिख रही"},
            ],
            "recording_url": "demo://distressed_caller.wav",
        }}),
        ("bland", {
            "call_id": f"bland-{uid}", "from": "+918812345678", "call_length": 3.5,
            "transcripts": [
                {"user": "assistant", "text": "Hello, you have reached 14566."},
                {"user": "user", "text": "gaon walon ne hamara hukka paani band kar diya hai, koi hume kaam nahi deta"},
            ],
            "recording_url": "demo://distressed_caller.wav",
        }),
        ("elevenlabs", {"type": "post_call_transcription", "data": {
            "conversation_id": f"el-{uid}",
            "metadata": {"call_duration_secs": 180, "phone_call": {"external_number": "+919912345678"}},
            "transcript": [
                {"role": "agent", "message": "Aap surakshit hain?"},
                {"role": "user", "message": "I just wanted to know the status of my relief compensation application."},
            ],
            "full_audio": calm_b64,
        }}),
        ("generic", {
            "call_id": f"ivr-{uid}", "caller": "+919712345678", "language": "gu", "consent": True,
            "recording_url": "demo://distressed_caller.wav",
            "transcript": "Agent: 14566 માં આપનું સ્વાગત છે\nCaller: મારે મરી જવું છે, કોઈ મારી વાત સાંભળતું નથી",
        }),
    ]


# --------------------------------------------------------------------------
# Officer upload: a counsellor / admin uploads a call recording by hand
# --------------------------------------------------------------------------
#
# For helpline staff whose phone system has no webhook, or for older
# recordings. Same analysis as a webhook call: the caller's voice is
# measured, the words come from the officer's typed transcript or from
# automatic speech-to-text (stt.py), and the case goes into the same
# queue. Every upload is written to the audit log with the officer's name.

UPLOAD_PLACEHOLDER = "[Call recording uploaded — words not analysed yet. Add a transcript.]"


class UploadError(ValueError):
    pass


def _audio_duration(audio: bytes):
    try:
        import io
        import soundfile as sf
        info = sf.info(io.BytesIO(audio))
        return round(info.frames / float(info.samplerate), 1)
    except Exception:
        return None


def voice_only_result(voice: dict) -> dict:
    """The words are unknown, so the case can never be filed as Low: it
    is held at Moderate or above (like an unreadable-language message)
    and marked for review until someone adds a transcript."""
    from scoring import _band_for
    speech = voice.get("speech_stress", 0.0) if voice.get("status") == "analysed" else 0.0
    svi = round(max(40.0, speech), 1)
    return {
        "svi": svi, "risk_category": _band_for(svi),
        "breakdown": {"text_sentiment": None, "keyword_severity": 0.0,
                      "speech_stress": speech if voice.get("status") == "analysed" else None,
                      "isolation_indicator": 0.0},
        "keyword_hits": {}, "category_confidences": {},
        "recommended_actions": ["Listen to the recording and add a transcript", "Counselling referral"],
        "language_confidence": "low", "human_review_required": True, "floor_applied": None,
        "detection": None,
    }


def ingest_upload(audio: bytes, filename: str, officer: dict, transcript: str = "",
                  language: str = "", channel: str = "", caller: str = "", notes: str = "") -> dict:
    import stt
    if not audio:
        raise UploadError("The recording file is empty.")
    if len(audio) > config.CALL_AUDIO_MAX_MB * 1024 * 1024:
        raise UploadError(f"Recording is larger than {config.CALL_AUDIO_MAX_MB:g} MB.")

    voice = analyse_audio_bytes(audio, channel=channel)
    text = (transcript or "").strip()
    source = "typed by officer" if text else None
    if not text:
        text, reason = stt.transcribe(audio, language)
        if text:
            source = f"auto-transcribed ({stt.status()['engine']} {stt.status()['model'] or ''})".strip()
        else:
            source = f"not available — {reason}"
            text = ""
    if voice["status"] != "analysed" and not text:
        raise UploadError("Could not read this audio file. Upload a WAV file (MP3/M4A need ffmpeg "
                          "on the server), or type what was said in the transcript box.")

    if text:
        result = score_text(text, speech_stress=voice.get("speech_stress", 0.0),
                            has_speech_data=voice["status"] == "analysed")
    else:
        result = voice_only_result(voice)

    language = language or (_guess_language(text) if text else "")
    result["channel"] = "upload"
    result["call"] = {
        "provider": "uploaded by officer", "call_id": None,
        "caller_masked": mask_number(caller), "duration_s": _audio_duration(audio),
        "filename": (filename or "")[:120], "voice": voice, "transcript_source": source,
        "uploaded_by": {"name": officer.get("name"), "email": officer.get("email"),
                        "role": officer.get("role_type")},
        "notes": (notes or "")[:1000],
        "consent": f"confirmed by {officer.get('name')} at upload",
    }
    case_id = db.insert_case(True, language or "unknown", text or UPLOAD_PLACEHOLDER, result)
    db.log_action(case_id, officer, "call recording uploaded")
    return {
        "case_id": case_id, "svi": result["svi"], "risk_category": result["risk_category"],
        "voice": voice, "transcript_source": source, "words_analysed": bool(text),
        "categories": sorted(result.get("keyword_hits", {}).keys()),
        "recommended_actions": result.get("recommended_actions", []),
        "floor_applied": result.get("floor_applied"),
    }


def add_transcript(case: dict, text: str, officer: dict) -> dict:
    """Re-scores an uploaded or call case once its words are known, keeping
    the voice measurement that was taken from the recording."""
    text = (text or "").strip()
    if not text:
        raise UploadError("Transcript is empty.")
    meta = json.loads(case.get("meta_json") or "{}")
    call_info = meta.get("call") or {}
    voice = call_info.get("voice") or {}
    result = score_text(text, speech_stress=voice.get("speech_stress", 0.0),
                        has_speech_data=voice.get("status") == "analysed")
    result["channel"] = meta.get("channel", "upload")
    call_info["transcript_source"] = f"added by {officer.get('name')}"
    result["call"] = call_info
    result["counselor_pref"] = meta.get("counselor_pref") or result.get("counselor_pref")
    db.update_case(case["id"], text, result)
    db.log_action(case["id"], officer, "transcript added and case re-scored")
    return {"case_id": case["id"], "svi": result["svi"], "risk_category": result["risk_category"]}
