"""
Shared API key for the victim-facing endpoints (chat, analyze). This is
a simple app-level shared secret — it stops random/automated traffic
from hitting the API, not a specific person from misusing it once
they have your frontend's source (any key embedded in client-side JS
is visible to anyone who opens dev tools/view-source).

For real protection beyond a hackathon demo, move this server-side
(a backend-for-frontend proxy that holds the key, never shipped to the
browser) or replace it with real per-user auth. Rotate this value
before any real deployment — generate a new one with:
    python -c "import secrets; print(secrets.token_hex(24))"
"""
API_KEY = "e59d8c99f3b6b9b992b991988d458d679df9e7e475555dad"

# Key the 14566 voice/IVR platform sends with its "call finished"
# webhook (see calls.py). Server-to-server only — it never reaches a
# browser, so unlike API_KEY above it can stay secret. Set
# SAHAYK_WEBHOOK_KEY in the host's environment (e.g. Render →
# Environment) to override the demo value below.
import os
WEBHOOK_KEY = os.environ.get("SAHAYK_WEBHOOK_KEY", "demo-14566-webhook-key-change-me")


# ---------- 14566 call recordings (calls.py) ----------
# When a finished-call webhook includes a recording link, the recording
# is downloaded and the caller's voice is analysed (pitch, pauses, pace)
# alongside the transcript. All overridable from the host's environment.
CALL_AUDIO_ENABLED = os.environ.get("SAHAYK_CALL_AUDIO", "on").lower() not in ("off", "0", "false")
CALL_AUDIO_MAX_MB = float(os.environ.get("SAHAYK_CALL_AUDIO_MAX_MB", "25"))
CALL_AUDIO_MAX_SECONDS = float(os.environ.get("SAHAYK_CALL_AUDIO_MAX_SECONDS", "120"))
CALL_AUDIO_TIMEOUT_S = float(os.environ.get("SAHAYK_CALL_AUDIO_TIMEOUT_S", "15"))
# Which channel of a STEREO recording is the caller: "0" (left), "1"
# (right), or "" to analyse the mixed-down audio. Platforms differ — check
# yours with one test call. Mono recordings are always analysed whole.
CALL_AUDIO_CALLER_CHANNEL = os.environ.get("SAHAYK_CALL_AUDIO_CALLER_CHANNEL", "")
# Some platforms protect recording links (e.g. Twilio: Account SID +
# Auth Token). Leave empty if links are public or pre-signed.
RECORDING_AUTH_USER = os.environ.get("SAHAYK_RECORDING_AUTH_USER", "")
RECORDING_AUTH_PASS = os.environ.get("SAHAYK_RECORDING_AUTH_PASS", "")
