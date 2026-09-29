# Sahayk — full technical description (current build)

One-line summary: a chatbot that has a conversation with a victim (by text
or voice, in English or Hindi), scores what they say for psychological/
physical risk using a transparent rule-based engine, and surfaces it to a
logged-in officer's review queue — without ever showing the score to the
person being scored.

---

## 1. Architecture at a glance

```
Browser (chat.html)  ──POST /chat/*, /analyze/*──▶  FastAPI backend
     │  (needs X-API-Key header)                        │
     │                                          ┌────────┴────────┐
     │                                          │  chat.py         │  conversation state machine
     │                                          │  scoring.py      │  SVI formula + floors
     │                                          │  lexicon.py      │  14-category keyword matcher
     │                                          │  audio_features.py│ librosa pitch/pause/rate
     │                                          └────────┬────────┘
     │                                                   │
     ▼                                          SQLite (sahayk.db) — one `cases` table
Browser (authority.html) ──GET/PATCH /cases──▶  (needs officer Bearer token from /auth/login)
```

Everything runs as a single FastAPI process (`backend/main.py`) serving
both the API and the two static HTML pages. No external services are
called at runtime except the browser's own built-in speech recognition
(client-side, not part of this backend at all).

---

## 2. File-by-file

| File | Lines | What it does |
|---|---|---|
| `backend/main.py` | ~213 | FastAPI app: routes, startup warmup, global exception handler, page serving |
| `backend/chat.py` | ~325 | The conversation state machine — see §5 |
| `backend/scoring.py` | ~157 | Combines sentiment + keywords + speech into the SVI score — see §4 |
| `backend/lexicon.py` | ~305 | 14-category keyword/phrase matcher with typo tolerance — see §3 |
| `backend/audio_features.py` | ~89 | Real pitch/pause/rate extraction from recorded audio via `librosa` |
| `backend/auth.py` | ~41 | Officer login (single demo account) + API-key check |
| `backend/config.py` | ~14 | Holds the shared `API_KEY` value |
| `backend/database.py` | ~111 | SQLite wrapper — one `cases` table, no ORM |
| `frontend/chat.html` | ~332 | Victim-facing page: consent, chat, voice recording |
| `frontend/authority.html` | ~210 | Officer login + case review queue |

---

## 3. The lexicon (`lexicon.py`) — 14 categories

Each category is a dict with a **weight** (0–1, how severe a hit in this
category is) and a list of **phrases**. Categories, in the order they
appear in the file, with their weight:

| Category | Weight | Examples of what it catches |
|---|---|---|
| `suicidal_ideation` | 1.0 | "want to die", "kill myself", "planning to suicide" + Hindi ("आत्महत्या", etc.) |
| `immediate_danger` | 1.0 | "standing right behind me", "he is here right now" — the assailant being physically present *right now* |
| `severe_trauma_fear` | 0.8 | "so scared", "im not safe"/"i am not safe", "scaring me" + Hindi |
| `intimidation_threat` | 0.85 | "threatened me", "following me", "monitors my phone" + Hindi |
| `depression_hopelessness` | 0.6 | "feel empty", "hopeless", "want to give up" + Hindi |
| `social_isolation` | 0.5 | "no one to talk to", "all alone" + Hindi |
| `physical_safety_abuse` | 0.85 | "hit me", "hitting me", "cut my finger", "choked me" + Hindi |
| `sexual_harassment_assault` | 0.9 | "touched me without", "forcing me to be with him", "harassing me" + Hindi |
| `financial_exploitation` | 0.4 | "controls my money", "demanding dowry" + Hindi |
| `legal_dispute` | 0.35 | "property dispute", "cant afford a lawyer" |
| `workplace_harassment` | 0.6 | "harassed at work", "hr ignored my complaint" |
| `cybercrime_online_abuse` | 0.55 | "leaked my photos", "hacked my account" |
| `child_safety_concern` | 0.95 | "my child is being hurt", "hurting my kids" |
| `elder_abuse` | 0.7 | "abusing my elderly parent", "neglecting my grandmother" |
| `substance_related` | 0.4 | "drunk and abusive", "violent when he drinks" |

**Matching mechanism** (`_fuzzy_phrase_in_tokens`): text is normalized
(lowercased, apostrophes stripped, then every character checked via
`unicodedata.category()` — letters/marks/digits from *any* script are
kept, everything else becomes a space; this specifically preserves
Devanagari combining vowel marks that a naive ASCII or `\w` regex would
mangle or destroy). The normalized text is split into tokens. For each
candidate phrase, a sliding window the length of the phrase is moved
across the tokens; every word in the window must match its phrase
counterpart either exactly or via `SequenceMatcher` fuzzy ratio ≥ 0.82
(only applied to words ≥5/≥4 characters, to avoid short-word false
positives like "will" fuzzy-matching "kill"). This is what lets "he
choakd me" still match "choked me" — one misspelled word doesn't kill
the whole match, but every word still has to line up in order, so it
stays precise rather than a loose bag-of-words match.

`keyword_severity_score(text)` returns `{"hits": {category: [matched
phrases]}, "score": <0-100, the highest-weight category's weight×100>}`.

---

## 4. The SVI scoring engine (`scoring.py`)

**Formula:**
```
SVI = 0.20 × text_sentiment + 0.45 × keyword_severity
    + 0.20 × speech_stress   + 0.15 × isolation_indicator
```
- `text_sentiment`: VADER's compound sentiment score, inverted and
  rescaled to 0–100 (more negative language → higher score). **Skipped
  entirely for non-Latin-script text** (VADER can't read Hindi — its
  weight is redistributed onto `keyword_severity` instead, since
  treating "unreadable" as "calm" would be actively misleading).
- `keyword_severity`: the lexicon's score (see §3).
- `speech_stress`: 0 for text-only messages; for voice, a weighted
  blend of pitch variance (40%), pause ratio (35%), and speech-rate
  deviation from a calm baseline (25%) — see §6.
- `isolation_indicator`: 100 if `social_isolation` was hit, else 0.

**Risk bands:** Low 0–25, Moderate 25–50, High 50–75, Critical 75–100.

**Severity floors** (`CATEGORY_FLOORS`) — applied *after* the weighted
formula, only ever pushing the score up, never down:
- `suicidal_ideation`, `immediate_danger`, `child_safety_concern` → floor 85 (Critical)
- `sexual_harassment_assault` → floor 60 (High)
- `physical_safety_abuse`, `elder_abuse` → floor 55 (High)

This exists because the weighted formula alone can under-score a
flatly-worded but objectively severe disclosure ("he is hitting me"
read almost neutral to VADER and only carried 45% weight from the
keyword hit) — the floors guarantee these categories can never land in
a band that undersells them, regardless of tone.

**Service routing** (`SERVICE_TAGS`) is tracked *separately* from
severity. A case can score Low overall but still need a specific
service — e.g. `legal_dispute` always adds "Legal aid referral" to the
recommended actions no matter what band the SVI lands in, because a
property dispute needs a lawyer regardless of how calm it sounds.

**Recommended actions** = the severity band's default actions
(`SEVERITY_ACTIONS`, e.g. Critical → "Immediate human review",
"Emergency support contact shown to user", "Consider police
intervention / witness protection — human decision only") **plus**
every service tag from every category that was hit, deduplicated.

---

## 5. The conversation engine (`chat.py`)

State lives in an in-memory `_sessions` dict keyed by session ID (lost
on server restart — no persistence beyond the SQLite case record).
Per-session state: message history, per-category question depth
(`asked`), turn count, crisis/danger trigger counts, current `stage`,
list of speech-stress values seen so far, and a flag for whether the
one-time generic acknowledgment has fired.

**Every turn**, the full cumulative transcript (all messages joined) is
re-scored from scratch — this means detection and severity can only
strengthen or stay the same as the conversation goes on, never weaken
just because new, calmer text was added.

**Priority order, checked before anything else, every single turn:**
1. **Exit phrases** ("bye", "stop", "gtg", etc.) → warm goodbye. If the
   session had ever shown a crisis message, the goodbye repeats the
   helpline numbers one more time before ending.
2. **Suicidal ideation, first time only** → `CRISIS_MESSAGE_FIRST`
   (KIRAN 1800-599-0019, Tele-MANAS 14416, asks if they're safe and
   with someone).
3. **Immediate danger, first time only** → `DANGER_MESSAGE_FIRST` (112
   police emergency, asks if they can get somewhere safe).
4. **Turn limit** (9 turns) closes the conversation — *unless* suicidal
   ideation or immediate danger is still active in the hits, in which
   case the limit is ignored and the conversation keeps going
   indefinitely rather than cutting off someone in crisis.

**After the first crisis/danger message**, subsequent turns fall
through into the **normal flow** (`_decide_reply`) instead of repeating
the same block forever — with a short one-line reminder
("(112 is available right now if you need it.)" or the KIRAN/Tele-MANAS
equivalent) kept in front of whatever the normal flow says, so the
resource stays visible without dominating every turn.

**The normal flow, in order:**
- If a `stage` is set (`ask_anything_else`, `ask_help_type`,
  `ask_safety_after_unclassified`, `ask_counselling`, `closing`),
  handle that stage's specific logic (e.g. `ask_help_type` checks for a
  bare "yes"/"no" and asks a clarifying rephrase instead of closing on
  an unspecific answer).
- Otherwise, walk `CATEGORY_FOLLOWUPS` in priority order (physical
  safety → sexual assault → child safety → elder abuse → intimidation
  → trauma/fear → cybercrime → legal → workplace → financial →
  substance → depression → isolation). The first category that's been
  hit and still has an unused question in its 2-question progressive
  set gets asked, with a one-time empathetic lead-in from `VALIDATIONS`
  the first time that category comes up (e.g. "I'm really sorry that's
  happening to you." before the first physical-safety question).
- If every matched category has run out of questions, or nothing was
  ever detected: a long (20+ word) message with zero hits gets treated
  as a likely detection gap rather than something minor — it gets a
  direct safety check instead of a throwaway line. Short/vague messages
  get a light generic acknowledgment. Once the conversation reaches the
  "anything else?" wrap-up with nothing having been flagged at all, it
  offers a counselling referral rather than just closing cold.

**Exit phrases, affirmative/negative word sets, and the 20-word
threshold** are all plain hardcoded sets/constants near the top of the
file — easy to tune without touching the state machine logic itself.

---

## 6. Voice pipeline

Two separate things happen for a voice message, from two separate
sources:
1. **Transcription** happens entirely in the browser via the Web
   Speech API (`SpeechRecognition`), using whichever language is
   selected in the dropdown (English, Hindi, Marathi, Tamil, Gujarati
   currently listed — any Chrome-supported BCP-47 code works). This is
   Chrome's own engine; nothing in this backend can improve its
   accuracy.
2. **The raw recording** (via `MediaRecorder`, webm/opus) is uploaded
   to `/chat/voice_message` alongside the transcript. `audio_features.py`
   uses `librosa` to compute three real signals directly from the
   waveform:
   - **pitch variance**: std-dev of the estimated fundamental frequency
     (via `librosa.pyin`), clipped/scaled to 0–1.
   - **pause ratio**: fraction of the clip that's silence, via
     `librosa.effects.split`.
   - **speech-rate delta**: onset density compared to a ~3-onsets/second
     calm baseline, via `librosa.onset`.

   These three combine (40/35/25% weights) into a single 0–100
   `speech_stress` value, which is averaged across all voice turns in a
   session and fed into the SVI formula's `speech_stress` term.

**Reliability**: every extraction step is individually wrapped in
try/except — a decoding failure (e.g. ffmpeg missing, which is required
to decode webm/opus and doesn't ship with Python) degrades to a neutral
0 signal (`audio_analysis_ok: false` in the response) rather than
crashing the request. On top of that, a global FastAPI exception
handler guarantees every response is valid JSON with a real `reply`
field no matter what fails anywhere in the stack, so the chat can never
go silent. A startup hook (`warm_up()`) runs a dummy pitch-detection
call once when the server starts, absorbing a one-time ~8-10 second
compilation cost that would otherwise land on whoever sent the first
real voice message.

---

## 7. Security

- **API key** (`config.API_KEY`, currently
  `e59d8c99f3b6b9b992b991988d458d679df9e7e475555dad`): required as an
  `X-API-Key` header on every victim-facing endpoint (`/chat/*`,
  `/analyze/*`). `chat.html` sends it automatically. This is a shared
  secret embedded in client-side JS — it stops casual/automated abuse,
  not a determined person reading the frontend's source.
- **Officer login**: real, DB-backed accounts as of round 13. `POST
  /auth/register` (name, email, role_type, password) saves a new row to
  the `authorities` table with the password hashed via
  PBKDF2-HMAC-SHA256 (100k iterations, random per-account salt) and
  signs the account in immediately. `POST /auth/login` checks the
  original hardcoded demo pair (`officer`/`changeme123`, kept working
  unchanged) first, then falls through to a DB lookup by email against
  that same table. Either path issues an in-memory bearer token;
  `/cases*` endpoints require it via `require_officer`. Tokens live only
  in memory and are wiped on server restart — same limitation as before,
  just with real accounts underneath now instead of one hardcoded pair.
  Registration is self-service with no approval step (any role,
  including "Law Enforcement" or "District Admin", can be registered by
  anyone), and there's no role-based route restriction yet — every
  logged-in authority sees the same review queue regardless of
  role_type. Both are disclosed, deliberate demo-scope choices, not
  oversights — see README round 13 for the full writeup and what a real
  deployment would add (email verification, an approval step, per-role
  RBAC, JWT instead of in-memory tokens).
- **No-store caching**: both `/` and `/authority` are served with
  `Cache-Control: no-store` so an updated file always reaches the
  browser instead of a stale cached copy.
- **Global exception handler**: catches anything unhandled anywhere and
  returns a graceful JSON reply instead of a raw error page (see §6).

---

## 8. Frontend

**`chat.html`** — victim-facing. Government-portal styling (navy/serif
letterhead, masthead), a consent checkbox + language dropdown, the chat
log itself, a text input, and a voice button (hold to speak) that runs
both the Web Speech API transcription and the `MediaRecorder` upload
described in §6. A "get help right now" hotline band sits at the bottom
of the page with real numbers (KIRAN, Tele-MANAS) always visible
regardless of conversation state. Responsive layout for mobile.

**`authority.html`** — officer-facing. Plain login form (username +
password, no forgot-password flow currently), then a review queue table
(ID, risk badge, SVI, transcript excerpt, status, action dropdown +
"mark reviewed" button) sorted by SVI descending. A small "Build check"
line at the bottom of the login screen was added specifically to help
diagnose stale-cache confusion during testing — it's harmless to leave
in.

---

## 9. Known limitations (current, honest state)

- **Lexicon is still exact/fuzzy phrase matching, not real NLP** — it
  will keep having gaps for tense, phrasing, and slang variants no
  matter how many rounds of "add this phrase" happen. A genuine fix
  would mean semantic/embedding-based matching instead of phrase lists.
- **Hindi coverage is partial** — the 8 highest-severity categories have
  a starting Hindi phrase set; legal, workplace, cybercrime, child
  safety, elder abuse, and substance categories are still English-only.
  No language beyond English/Hindi has any lexicon coverage at all
  (though the browser will still transcribe other languages — the
  *scoring* just won't understand them).
- **Voice emotion is pitch/pause/rate only**, not a trained
  emotion-classification model — that would need labeled audio data and
  a model host this environment doesn't have.
- **No real SMS/email delivery** exists anywhere in the current build
  (the OTP/password-reset feature that would have needed this was
  removed).
- **In-memory session and token state** — both chat sessions and
  officer login tokens are wiped on every server restart.
- **Single hardcoded officer account**, no real user management or
  audit logging of who reviewed what.
- **SQLite, no migrations** — schema changes require deleting
  `sahayk.db` and starting fresh (this is why testing repeatedly
  removes it before each run in the changelog).
