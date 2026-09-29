import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "sahayk.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------- schema migrations (idempotent — safe to run on every startup) ----------
#
# Every column added to a table after its ORIGINAL release goes here,
# never edited into the CREATE TABLE below. ALTER TABLE ADD COLUMN only
# runs for a column that isn't already present, so an existing
# sahayk.db from an older build of this prototype picks up new columns
# automatically the next time the server starts — this is what used to
# force deleting the database file (and every case/account on it) just
# to add a field like `gender` or `reviewed_by`. See README Round 16.
_MIGRATIONS = [
    ("authorities", {"gender": "TEXT", "age": "INTEGER"}),
    ("cases", {"reviewed_by_email": "TEXT", "reviewed_by_name": "TEXT", "reviewed_at": "REAL"}),
]


def _table_columns(conn, table: str) -> set:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_columns(conn, table: str, columns: dict):
    existing = _table_columns(conn, table)
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db():
    conn = get_conn()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL,
            consent_given INTEGER,
            language TEXT,
            transcript TEXT,
            svi REAL,
            risk_category TEXT,
            breakdown_json TEXT,
            keyword_hits_json TEXT,
            recommended_actions_json TEXT,
            meta_json TEXT,
            status TEXT DEFAULT 'pending_review',
            action_taken TEXT
        )
        """
    )
    # Registered authority (officer) accounts. Any officer who signs up here
    # can log in with that same email + password afterwards — see auth.py.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS authorities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL,
            name TEXT,
            email TEXT UNIQUE,
            role_type TEXT,
            password_hash TEXT
        )
        """
    )
    # An in-progress (or finished) victim chat session — the FULL state
    # needed to resume it verbatim: conversation stage, the cumulative
    # transcript used for scoring, the visual bot/user turn log used to
    # repaint the chat window, counsellor-preference answers, etc.,
    # serialized as one JSON blob. This is what fixes two related
    # problems that used to exist here: (1) an accidental page reload
    # wiped the in-progress conversation from the browser even though
    # nothing was actually lost server-side, forcing a restart from the
    # consent screen; (2) a server restart invalidated every session_id
    # outright ("this session has expired"), since sessions used to
    # live only in a plain Python dict in chat.py that's gone the
    # instant the process restarts. Both now read through this table
    # first — see chat.py's _load_session / _persist_session.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            case_id INTEGER,
            state_json TEXT,
            updated_at REAL
        )
        """
    )
    # Officer login tokens. Previously a bare in-memory dict in auth.py
    # (_active_tokens) — wiped on every server restart, forcing every
    # signed-in officer to log in again mid-review. Persisted here now
    # so a restart (including a routine --reload during development)
    # doesn't sign anyone out. These don't expire in this prototype —
    # real deployment should add token TTL/rotation and a logout that
    # deletes the row, same caveat as the plaintext demo API key.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS officer_tokens (
            token TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            role_type TEXT,
            created_at REAL
        )
        """
    )
    # Append-only audit trail — one row per review action taken, kept
    # separately from `cases.reviewed_by_*` (which only holds the
    # LATEST review of a case). This is the fix for "no audit logging
    # of who reviewed what": previously nothing recorded which officer
    # took which action, or when.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER,
            officer_email TEXT,
            officer_name TEXT,
            role_type TEXT,
            action_taken TEXT,
            reviewed_at REAL
        )
        """
    )

    # 14566 call ingestion (calls.py). call_events keeps every webhook
    # body exactly as received — append-only — so a call can be re-scored
    # later if the rules or the model change, without asking the
    # platform to resend it. calls holds one row per call_id and links
    # it to the case it produced; a retried webhook for the same call
    # updates that case instead of creating a duplicate.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS call_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at REAL,
            provider TEXT,
            payload_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            call_id TEXT PRIMARY KEY,
            provider TEXT,
            case_id INTEGER,
            caller_masked TEXT,
            duration_s REAL,
            started_at REAL,
            status TEXT,
            first_received_at REAL,
            last_received_at REAL
        )
        """
    )

    for table, columns in _MIGRATIONS:
        _ensure_columns(conn, table, columns)

    conn.commit()
    conn.close()


# ---------- authority accounts ----------

def insert_authority(name: str, email: str, role_type: str, gender: str, age: int, password_hash: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO authorities (created_at, name, email, role_type, gender, age, password_hash) VALUES (?,?,?,?,?,?,?)",
        (time.time(), name, email.lower(), role_type, gender, age, password_hash),
    )
    conn.commit()
    authority_id = cur.lastrowid
    conn.close()
    return authority_id


def get_authority_by_email(email: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM authorities WHERE email=?", (email.lower(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_authorities():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, created_at, name, email, role_type, gender, age FROM authorities ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_counsellor_ages():
    """id/name/age for every registered counsellor account that has an
    age on file — used to find the closest match to a victim's stated
    age preference (see main.py's _closest_counselor). Restricted to
    role_type='counsellor' since the preference question is specifically
    about a counsellor, not any authority type."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, age FROM authorities WHERE role_type='counsellor' AND age IS NOT NULL ORDER BY age"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- officer tokens (persisted so a server restart doesn't sign anyone out) ----------

def save_officer_token(token: str, name: str, email: str, role_type: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO officer_tokens (token, name, email, role_type, created_at) VALUES (?,?,?,?,?)",
        (token, name, email, role_type, time.time()),
    )
    conn.commit()
    conn.close()


def get_officer_token(token: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM officer_tokens WHERE token=?", (token,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- chat sessions (persisted so a reload or restart doesn't lose the conversation) ----------

def save_chat_session(session_id: str, case_id: int, state_json: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO chat_sessions (session_id, case_id, state_json, updated_at) VALUES (?,?,?,?)",
        (session_id, case_id, state_json, time.time()),
    )
    conn.commit()
    conn.close()


def get_chat_session(session_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM chat_sessions WHERE session_id=?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


_DEFAULT_COUNSELOR_PREF = {"gender": None, "gender_raw": None, "age": None, "age_raw": None, "age_value": None}


def _meta_json(result: dict) -> str:
    return json.dumps({
        "language_confidence": result.get("language_confidence", "ok"),
        "human_review_required": result.get("human_review_required", False),
        "floor_applied": result.get("floor_applied"),
        "category_confidences": result.get("category_confidences", {}),
        # Stated preference for a female/male counsellor and an age
        # (exact number, younger/older, or no preference), asked right
        # after consent — see chat.py. A preference to honour when
        # assigning a reviewer, not something that auto-assigns.
        "counselor_pref": result.get("counselor_pref", _DEFAULT_COUNSELOR_PREF),
        # Which detector produced this rating — the rule engine's phrase
        # match, the trained model, or both (see scoring._detect_categories).
        # Persisted rather than recomputed because the model can be
        # retrained or swapped: a case reviewed months from now should
        # show what the system actually concluded at the time, not what
        # today's model would conclude on the same transcript. Without
        # this stored, an officer cannot tell a Critical reached from a
        # known phrase apart from one reached purely by model inference,
        # and those warrant different amounts of trust.
        "detection": result.get("detection"),
        # Where the case came from: "chat" (the web chat) or "call" (a
        # finished 14566 call ingested by calls.py, with its call details).
        "channel": result.get("channel", "chat"),
        "call": result.get("call"),
    })


def insert_case(consent_given: bool, language: str, transcript: str, result: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO cases (created_at, consent_given, language, transcript,
            svi, risk_category, breakdown_json, keyword_hits_json,
            recommended_actions_json, meta_json, status)
        VALUES (?,?,?,?,?,?,?,?,?,?, 'pending_review')
        """,
        (
            time.time(),
            int(consent_given),
            language,
            transcript,
            result["svi"],
            result["risk_category"],
            json.dumps(result["breakdown"]),
            json.dumps(result["keyword_hits"]),
            json.dumps(result["recommended_actions"]),
            _meta_json(result),
        ),
    )
    conn.commit()
    case_id = cur.lastrowid
    conn.close()
    return case_id


def list_cases():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM cases ORDER BY svi DESC, created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_case(case_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_case(case_id: int, transcript: str, result: dict):
    conn = get_conn()
    conn.execute(
        """
        UPDATE cases SET transcript=?, svi=?, risk_category=?,
            breakdown_json=?, keyword_hits_json=?, recommended_actions_json=?,
            meta_json=?
        WHERE id=?
        """,
        (
            transcript,
            result["svi"],
            result["risk_category"],
            json.dumps(result["breakdown"]),
            json.dumps(result["keyword_hits"]),
            json.dumps(result["recommended_actions"]),
            _meta_json(result),
            case_id,
        ),
    )
    conn.commit()
    conn.close()


def update_review(case_id: int, action_taken: str, officer_email: str = None,
                   officer_name: str = None, role_type: str = None):
    """Updates the case's own latest-review columns AND appends to the
    append-only review_log audit trail — see init_db()'s comment on
    that table for why both exist."""
    conn = get_conn()
    now = time.time()
    conn.execute(
        """UPDATE cases SET status='reviewed', action_taken=?,
           reviewed_by_email=?, reviewed_by_name=?, reviewed_at=? WHERE id=?""",
        (action_taken, officer_email, officer_name, now, case_id),
    )
    conn.execute(
        """INSERT INTO review_log (case_id, officer_email, officer_name, role_type, action_taken, reviewed_at)
           VALUES (?,?,?,?,?,?)""",
        (case_id, officer_email, officer_name, role_type, action_taken, now),
    )
    conn.commit()
    conn.close()


def get_review_log(case_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM review_log WHERE case_id=? ORDER BY reviewed_at DESC", (case_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- 14566 call ingestion (see calls.py) ----------

def insert_call_event(provider: str, payload_json: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO call_events (received_at, provider, payload_json) VALUES (?,?,?)",
        (time.time(), provider, payload_json),
    )
    conn.commit()
    event_id = cur.lastrowid
    conn.close()
    return event_id


def get_call(call_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_call(call: dict, case_id, status: str):
    now = time.time()
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO calls (call_id, provider, case_id, caller_masked, duration_s, started_at,
                           status, first_received_at, last_received_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(call_id) DO UPDATE SET
            case_id=COALESCE(excluded.case_id, calls.case_id),
            status=excluded.status,
            last_received_at=excluded.last_received_at
        """,
        (call["call_id"], call["provider"], case_id, call["caller_masked"], call["duration_s"],
         call["started_at"], status, now, now),
    )
    conn.commit()
    conn.close()


def call_stats():
    conn = get_conn()
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM calls GROUP BY status").fetchall()
    conn.close()
    return {r["status"]: r["n"] for r in rows}


def log_action(case_id: int, officer: dict, action: str):
    """Append-only audit entry that does NOT mark the case reviewed
    (update_review does both) — used for uploads and transcript edits."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO review_log (case_id, officer_email, officer_name, role_type, action_taken, reviewed_at) "
        "VALUES (?,?,?,?,?,?)",
        (case_id, officer.get("email"), officer.get("name"), officer.get("role_type"), action, time.time()),
    )
    conn.commit()
    conn.close()
