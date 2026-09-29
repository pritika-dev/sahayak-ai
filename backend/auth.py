"""
Minimal token auth so the review queue (transcripts + SVI scores) is
never reachable by the person who submitted the case — only by a
logged-in authority/officer account.

Officer accounts are real, DB-backed records now: `/auth/register`
saves a new authority (name, email, role, password) to the `authorities`
table in database.py, hashed with PBKDF2-HMAC-SHA256 + a random
per-account salt (stdlib `hashlib`, no extra dependency). `/auth/login`
then checks the email + password straight against that same table —
whatever you register with is exactly what you log in with afterwards.

The original demo account (officer / changeme123) is kept working too,
as a fallback that doesn't require registering first, so nothing that
depended on it breaks.

Tokens are cached in memory for speed but persisted to the
`officer_tokens` table (database.py) as the source of truth, so a
server restart no longer signs every officer out — require_officer()
falls back to the DB on a cache miss and re-warms the in-memory cache.
They still don't expire in this prototype (no TTL/rotation) — real
deployment should add that, plus a logout that actually deletes the
row, same caveat as the plaintext demo API key in config.py.
"""
import hashlib
import os
import secrets
import config
import database as db
from fastapi import Header, HTTPException

DEMO_OFFICER_USERNAME = "officer"
DEMO_OFFICER_PASSWORD = "changeme123"  # change before any real deployment

VALID_ROLE_TYPES = {"counsellor", "legal", "police", "rehab", "admin"}
VALID_GENDERS = {"female", "male", "other"}
MIN_AGE, MAX_AGE = 18, 100

# token -> {"name": ..., "email": ..., "role_type": ...}
_active_tokens = {}


def _hash_password(password: str, salt: bytes = None) -> str:
    """PBKDF2-HMAC-SHA256, 100k iterations, random 16-byte salt.
    Stored as 'salt_hex$hash_hex' so verification never needs a second
    lookup for the salt."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    return secrets.compare_digest(_hash_password(password, salt), stored)


def _issue_token(name: str, email: str, role_type: str) -> str:
    token = secrets.token_hex(24)
    _active_tokens[token] = {"name": name, "email": email, "role_type": role_type}
    db.save_officer_token(token, name, email, role_type)
    return token


def register(name: str, email: str, role_type: str, gender: str, age: int, password: str) -> dict:
    """Creates a new authority account and immediately logs them in
    (same UX as the round-2 React prototype this was ported from:
    register once, then just sign in with those same credentials from
    then on — no separate approval step in this demo)."""
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if not name or not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid name and email are required.")
    if role_type not in VALID_ROLE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid authority type.")
    if gender not in VALID_GENDERS:
        raise HTTPException(status_code=400, detail="Invalid gender.")
    if not isinstance(age, int) or not (MIN_AGE <= age <= MAX_AGE):
        raise HTTPException(status_code=400, detail=f"Age must be between {MIN_AGE} and {MAX_AGE}.")
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if email == DEMO_OFFICER_USERNAME.lower() or db.get_authority_by_email(email):
        raise HTTPException(status_code=409, detail="An authority with this email is already registered.")

    password_hash = _hash_password(password)
    db.insert_authority(name, email, role_type, gender, age, password_hash)
    token = _issue_token(name, email, role_type)
    return {"token": token, "name": name, "email": email, "role_type": role_type}


def login(username: str, password: str) -> dict:
    identifier = (username or "").strip()

    # Backward-compatible demo account — no registration needed.
    if identifier.lower() == DEMO_OFFICER_USERNAME.lower() and password == DEMO_OFFICER_PASSWORD:
        token = _issue_token("Demo Officer", DEMO_OFFICER_USERNAME, "admin")
        return {"token": token, "name": "Demo Officer", "email": DEMO_OFFICER_USERNAME, "role_type": "admin"}

    # Real, registered authority — looked up by email, verified against
    # the PBKDF2 hash saved at registration time.
    record = db.get_authority_by_email(identifier)
    if record and _verify_password(password, record["password_hash"]):
        token = _issue_token(record["name"], record["email"], record["role_type"])
        return {"token": token, "name": record["name"], "email": record["email"], "role_type": record["role_type"]}

    raise HTTPException(status_code=401, detail="Invalid credentials")


def require_officer(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Officer login required")
    if token in _active_tokens:
        return _active_tokens[token]
    # Cache miss — most likely the server restarted since this officer
    # logged in and the in-memory _active_tokens dict was rebuilt empty.
    # Check the persisted record before rejecting a token that's
    # actually still valid, and re-warm the cache so this only costs a
    # DB lookup once per token per process lifetime.
    record = db.get_officer_token(token)
    if not record:
        raise HTTPException(status_code=401, detail="Officer login required")
    officer = {"name": record["name"], "email": record["email"], "role_type": record["role_type"]}
    _active_tokens[token] = officer
    return officer


def require_webhook_key(x_api_key: str = Header(default=""), authorization: str = Header(default=""),
                        api_key: str = ""):
    """Gate for /webhook/calls. Platforms differ in how they can send a
    secret, so it is accepted as an X-API-Key header, a Bearer token, or
    an ?api_key= query parameter (some IVR consoles only allow a URL)."""
    bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if config.WEBHOOK_KEY not in (x_api_key, bearer, api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid webhook key")


def require_api_key(x_api_key: str = Header(default="")):
    """Gate for the victim-facing endpoints — any frontend calling this
    API (this project's own chat.html, or a separate one you build)
    must send this key on every request. See config.py for the value
    and its limits."""
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
