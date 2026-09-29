import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from scoring import score_text
from audio_features import extract_speech_features, warm_up
import database as db
import auth
import chat as chatmod
import calls

app = FastAPI(title="Sahayk — Stress Vulnerability Index API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.on_event("startup")
def _startup_warmup():
    """The pitch-detection library (numba, under librosa) has a one-time
    compilation cost the very first time it runs on a machine — around
    8-10 seconds. Without this, that delay lands on whoever sends the
    FIRST voice message after the server starts, which is exactly what
    "voice is loading so much" was: not a real per-message cost, a
    one-time startup cost landing in the wrong place. Running it here
    means it happens once, during startup, before anyone is waiting on
    it — every real voice message after this is ~1-2 seconds."""
    # Runs in a background thread so the server opens its port at once.
    # On a small cloud instance (Render free) this warm-up took ~85 s; done
    # inline it kept the port closed that long, and a cold-start visitor got
    # "Not Found" from Render instead of the app.
    import threading

    def _bg():
        try:
            import ml_model
            ml_model._load()   # load the trained model before the first message
        except Exception:
            pass
        try:
            warm_up()
        except Exception:
            pass  # never fatal — worst case, the first real voice message pays the cost

    threading.Thread(target=_bg, daemon=True).start()


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    """Root cause of the "voice gives no response" bug: an unhandled
    exception anywhere used to return a bare, non-JSON 500 page, which
    breaks the frontend's res.json() call and aborts the script before
    it ever displays a reply — so the person just sees silence with no
    indication anything went wrong. This guarantees every response is
    valid JSON with a real reply the frontend can always show, on top
    of the specific fixes already made in the audio pipeline itself."""
    return JSONResponse(
        status_code=200,
        content={
            "reply": "Sorry, something went wrong processing that. Please try again, or type your message instead of using voice.",
            "done": False,
        },
    )


# ---------- public pages ----------
# no-store: these are actively-changing local dev pages; without this,
# browsers can serve a stale cached copy after you unzip an updated
# version, showing old UI/behavior that was already fixed server-side.

@app.api_route("/", methods=["GET", "HEAD"])
def victim_chat_page():
    return FileResponse(FRONTEND_DIR / "chat.html", headers={"Cache-Control": "no-store"})


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Cheap check for uptime pingers and the demo landing page."""
    return {"ok": True}


@app.api_route("/authority", methods=["GET", "HEAD"])
def authority_page():
    return FileResponse(FRONTEND_DIR / "authority.html", headers={"Cache-Control": "no-store"})


# ---------- auth ----------

class LoginRequest(BaseModel):
    username: str  # registered email, or the demo "officer" account
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    role_type: str  # counsellor | legal | police | rehab | admin
    gender: str      # female | male | other
    age: int
    password: str


@app.post("/auth/login")
def login(payload: LoginRequest):
    return auth.login(payload.username, payload.password)


@app.post("/auth/register")
def register(payload: RegisterRequest):
    """Registers a new authority account and logs them straight in —
    whatever email/password they register with here is exactly what
    they use on /auth/login afterwards (saved to the `authorities`
    table in database.py, password hashed, never stored in plain
    text)."""
    return auth.register(payload.name, payload.email, payload.role_type, payload.gender, payload.age, payload.password)


# ---------- victim-facing: chat intake (no auth, never returns case data) ----------

class ChatStart(BaseModel):
    language: str = "en"


class ChatMessage(BaseModel):
    session_id: str
    text: str


@app.post("/chat/start")
def chat_start(payload: ChatStart, _key=Depends(auth.require_api_key)):
    return chatmod.start_session(payload.language)


@app.get("/chat/resume/{session_id}")
def chat_resume(session_id: str, _key=Depends(auth.require_api_key)):
    """Called by chat.html on page load when it finds a session_id saved
    in sessionStorage from before an accidental reload. Returns the
    full bot/user turn log so the frontend can repaint the conversation
    exactly as it was, instead of the person having to start over from
    the consent screen. 404 means the session genuinely isn't on file
    (or never existed) — the frontend falls back to a fresh consent
    screen in that case."""
    data = chatmod.get_session_log(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return {"session_id": session_id, **data}


@app.post("/chat/message")
def chat_message(payload: ChatMessage, _key=Depends(auth.require_api_key)):
    return chatmod.send_message(payload.session_id, payload.text)


@app.post("/chat/voice_message")
async def chat_voice_message(
    session_id: str = Form(...),
    transcript: str = Form(...),
    file: UploadFile = File(...),
    _key=Depends(auth.require_api_key),
):
    """Voice turn within an active chat session: extracts real pitch/pause/
    rate features from the recording and folds them into the SAME case
    record as the rest of the conversation (previously this went to a
    separate, disconnected case via /analyze/voice/audio, so the voice
    signal never actually reached the chat's own score — fixed here).

    The transcript (already produced client-side by the browser) is what
    actually drives the conversation — the audio is only for the extra
    tone/pace signal. So even if reading the audio fails outright for
    any reason, this still scores the transcript and returns a real
    reply rather than raising and leaving the chat silent."""
    try:
        audio_bytes = await file.read()
        features = extract_speech_features(audio_bytes)
    except Exception:
        features = {"pitch_variance": 0.0, "pause_ratio": 0.0, "speech_rate_delta": 0.0, "audio_analysis_ok": False}

    speech_stress = round(
        100 * (0.4 * features["pitch_variance"] + 0.35 * features["pause_ratio"]
               + 0.25 * features["speech_rate_delta"]), 1
    )
    result = chatmod.send_message(session_id, transcript, speech_stress=speech_stress, audio_ok=features["audio_analysis_ok"])
    result["speech_features"] = features
    return result


# ---------- victim-facing: single-shot text/voice analysis (kept for direct API testing) ----------

class TextIntake(BaseModel):
    transcript: str
    consent_given: bool
    language: str = "en"


@app.post("/analyze/text")
def analyze_text(payload: TextIntake, _key=Depends(auth.require_api_key)):
    if not payload.consent_given:
        raise HTTPException(status_code=403, detail="Consent required before analysis.")
    result = score_text(payload.transcript)
    case_id = db.insert_case(payload.consent_given, payload.language, payload.transcript, result)
    return {"case_id": case_id, "status": "recorded"}  # score intentionally not returned to the submitter


@app.post("/analyze/voice/audio")
async def analyze_voice_audio(
    file: UploadFile = File(...),
    transcript: str = Form(...),
    consent_given: bool = Form(...),
    language: str = Form("en"),
    _key=Depends(auth.require_api_key),
):
    if not consent_given:
        raise HTTPException(status_code=403, detail="Consent required before analysis.")
    audio_bytes = await file.read()
    features = extract_speech_features(audio_bytes)
    speech_stress = round(
        100 * (0.4 * features["pitch_variance"] + 0.35 * features["pause_ratio"]
               + 0.25 * features["speech_rate_delta"]), 1
    )
    result = score_text(transcript, speech_stress=speech_stress, has_speech_data=features["audio_analysis_ok"])
    case_id = db.insert_case(consent_given, language, transcript, result)
    return {"case_id": case_id, "status": "recorded", "speech_features": features}


# ---------- authority-only: review queue (requires officer login) ----------
#
# Role-based visibility: a case's `recommended_actions` (from scoring.py's
# SERVICE_TAGS) already say who needs to act on it — "Legal aid referral",
# "Police intervention consideration...", etc. Legal Aid Officer and Law
# Enforcement accounts only see cases whose actions actually route to them,
# so a legal-aid login isn't scrolling through unrelated cases. Counsellor
# and District Admin stay unrestricted: counsellors are the front-line
# reviewer for every case regardless of what else it needs, and admins get
# full oversight across every case and every other role's queue — the same
# broad visibility the original hardcoded demo account always had (it logs
# in with role_type "admin", so this is a strict continuation of that,
# never a narrowing of what already worked).
ROLE_ACTION_FILTERS = {
    "legal": ("Legal aid referral", "Legal aid referral / cybercrime cell"),
    "police": (
        "Police intervention consideration — human decision only",
        "Consider police intervention / witness protection — human decision only",
        "Child protection referral",
        "Witness protection referral — human decision only",
    ),
    # Rehabilitation & welfare authorities (a named SIH26093 stakeholder):
    # relief/compensation and rehabilitation cases — displacement,
    # boycott, killings, sexual violence, caste atrocities.
    "rehab": ("Relief & rehabilitation referral",),
}


def _closest_counselor(target_age: int, candidates: list):
    """Finds the registered counsellor account whose age is closest to
    a victim's stated exact-age preference (see chat.py's
    _extract_age_number). Ties are broken toward the GREATER age —
    e.g. a preference of 25 with both a 23- and a 27-year-old
    counsellor registered resolves to the 27-year-old. Returns None if
    no counsellor account has an age on file yet."""
    best = None
    best_diff = None
    for c in candidates:
        diff = abs(c["age"] - target_age)
        if best is None or diff < best_diff or (diff == best_diff and c["age"] > best["age"]):
            best = c
            best_diff = diff
    return best


def _serialize(row, counsellor_ages=None):
    row["breakdown"] = json.loads(row.pop("breakdown_json"))
    row["keyword_hits"] = json.loads(row.pop("keyword_hits_json"))
    row["recommended_actions"] = json.loads(row.pop("recommended_actions_json"))
    row.update(json.loads(row.pop("meta_json") or "{}"))

    # Matched LIVE against currently-registered counsellors, not frozen
    # at chat time — if a closer-aged counsellor registers after this
    # case was created, the dashboard reflects that immediately rather
    # than showing a stale match from intake.
    pref = row.get("counselor_pref") or {}
    if pref.get("age") == "exact" and pref.get("age_value") is not None:
        candidates = db.list_counsellor_ages() if counsellor_ages is None else counsellor_ages
        pref["matched_counselor"] = _closest_counselor(pref["age_value"], candidates)
    return row


def _visible_to(case: dict, role_type: str) -> bool:
    allowed_actions = ROLE_ACTION_FILTERS.get(role_type)
    if not allowed_actions:  # counsellor, admin, or any unrecognized role: unrestricted
        return True
    return any(a in case["recommended_actions"] for a in allowed_actions)


@app.get("/cases")
def get_cases(officer=Depends(auth.require_officer)):
    counsellor_ages = db.list_counsellor_ages()  # fetched once, not once per row
    cases = [_serialize(r, counsellor_ages) for r in db.list_cases()]
    return [c for c in cases if _visible_to(c, officer.get("role_type"))]


@app.get("/cases/{case_id}")
def get_case(case_id: int, officer=Depends(auth.require_officer)):
    row = db.get_case(case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    case = _serialize(row)
    if not _visible_to(case, officer.get("role_type")):
        raise HTTPException(status_code=403, detail="This case is outside your authority type's scope.")
    return case


class ReviewAction(BaseModel):
    action_taken: str


@app.patch("/cases/{case_id}/review")
def review_case(case_id: int, payload: ReviewAction, officer=Depends(auth.require_officer)):
    row = db.get_case(case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    if not _visible_to(_serialize(dict(row)), officer.get("role_type")):
        raise HTTPException(status_code=403, detail="This case is outside your authority type's scope.")
    db.update_review(case_id, payload.action_taken, officer_email=officer.get("email"),
                      officer_name=officer.get("name"), role_type=officer.get("role_type"))
    return {"status": "reviewed", "action_taken": payload.action_taken}


# ---------- 14566 call ingestion (server-to-server webhook) ----------
#
# The voice/IVR platform that answers 14566 posts here when a call ends.
# See calls.py for what happens next. Returns only a small receipt —
# never the transcript, never the score.

@app.post("/webhook/calls")
async def call_webhook(request: Request, provider: str = "", _key=Depends(auth.require_webhook_key)):
    body = await request.body()
    if len(body) > calls.MAX_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")
    ctype = request.headers.get("content-type", "")
    try:
        if "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
            payload = dict(await request.form())          # Twilio posts form fields
        else:
            payload = json.loads(body or b"{}")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Body must be JSON or form data")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    # Run in a worker thread: downloading and analysing a call recording
    # takes a few seconds and must not block other requests meanwhile.
    return await run_in_threadpool(calls.ingest, payload, provider or None)


@app.post("/demo/simulate_call")
def simulate_call(officer=Depends(auth.require_officer)):
    """Demo helper for the dashboard: pushes one realistic finished call,
    in a different provider format each time, through the SAME ingest
    path a real 14566 webhook uses. Counsellor and District Admin only."""
    if officer.get("role_type") not in ("admin", "counsellor"):
        raise HTTPException(status_code=403, detail="Only counsellors and district admins can run the demo")
    samples = calls.sample_payloads()
    idx = sum(db.call_stats().values()) % len(samples)
    provider, payload = samples[idx]
    receipt = calls.ingest(payload, provider)
    receipt["provider"] = provider
    return receipt


# ---------- officer upload of a call recording (Round 21) ----------
#
# Helpline staff (counsellors, district admins) upload a recording from
# the dashboard: the caller's voice is analysed, the words come from the
# transcript box or automatic speech-to-text, and a scored case goes into
# the queue. Consent must be confirmed by the uploader. See calls.py.

UPLOAD_ROLES = ("counsellor", "admin")


@app.get("/stt/status")
def stt_status(officer=Depends(auth.require_officer)):
    import stt
    return stt.status()


@app.post("/cases/upload_call")
async def upload_call(
    file: UploadFile = File(...),
    consent_confirmed: bool = Form(False),
    transcript: str = Form(""),
    language: str = Form(""),
    caller_channel: str = Form(""),
    caller_number: str = Form(""),
    notes: str = Form(""),
    officer=Depends(auth.require_officer),
):
    if officer.get("role_type") not in UPLOAD_ROLES:
        raise HTTPException(status_code=403, detail="Only counsellors and district admins can upload recordings")
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="Confirm that the caller was told about and agreed to recording analysis")
    if caller_channel not in ("", "0", "1"):
        raise HTTPException(status_code=400, detail="caller_channel must be empty, 0 or 1")
    audio = await file.read(int(calls.config.CALL_AUDIO_MAX_MB * 1024 * 1024) + 1)
    try:
        return await run_in_threadpool(
            calls.ingest_upload, audio, file.filename, officer, transcript, language,
            caller_channel, caller_number, notes)
    except calls.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class TranscriptIn(BaseModel):
    transcript: str


@app.post("/cases/{case_id}/transcript")
def add_case_transcript(case_id: int, payload: TranscriptIn, officer=Depends(auth.require_officer)):
    if officer.get("role_type") not in UPLOAD_ROLES:
        raise HTTPException(status_code=403, detail="Only counsellors and district admins can add transcripts")
    row = db.get_case(case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    meta = json.loads(row.get("meta_json") or "{}")
    if meta.get("channel") not in ("upload", "call"):
        raise HTTPException(status_code=400, detail="Transcripts can only be added to call or uploaded-recording cases")
    try:
        return calls.add_transcript(row, payload.transcript, officer)
    except calls.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
