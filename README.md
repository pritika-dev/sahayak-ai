# Sahayk — SVI prototype

Working demo: multi-turn chat (text + voice) → adaptive follow-up questions →
SVI scoring → risk categorization → login-gated authority review queue with
real, self-registered officer accounts.

Category detection runs a **trained ML model** (`backend/models/`) alongside
the phrase lexicon — see "Round 17" below for how it was trained and what it
does and does not do.

**This is a demo-grade model, not a clinical instrument.** The scoring
weights and lexicon are illustrative and need professional review before
any real use. Nothing in this system auto-triggers police intervention
or witness protection — those stay human decisions, by design.

**The trained model does not change that, and slightly sharpens it.** Its
training labels were generated from this project's own scoring rubric, not
annotated by clinicians, so it reproduces that rubric on wording the phrase
list misses — it does not independently know what danger looks like. Every
accuracy figure below is measured on synthetic data. None of it is evidence
about real calls.

## Run it

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

- Chat (victim-facing): http://localhost:8000/
- Authority dashboard: http://localhost:8000/authority
  - Demo account: `officer` / `changeme123` (still works, unchanged)
  - Or click **"Register an account"** on the login screen: enter a name,
    email, authority type (Counsellor / Legal Aid Officer / Law Enforcement /
    District Admin), and password — you're signed in immediately, and from
    then on those exact credentials work on the normal sign-in form. See
    "Round 13 — real, DB-backed officer registration + login" below.

## Round 13 — real, DB-backed officer registration + login

Ported the registration flow from the React/Vite prototype (`sahayak-ai-
prototype.zip`) into this FastAPI build, backed by the real `cases` database
instead of a browser-only `localStorage` shim:

1. **New `authorities` table** (`database.py`): id, name, email (unique),
   role_type, password_hash, created_at.
2. **`POST /auth/register`** (`name`, `email`, `role_type`, `password`) —
   validates the email/role/password, hashes the password with
   **PBKDF2-HMAC-SHA256, 100,000 iterations, a random 16-byte salt per
   account** (Python's stdlib `hashlib`, no extra dependency), saves the
   row, and signs the new account in immediately (returns the same
   `{token, name, email, role_type}` shape as login).
3. **`POST /auth/login`** now checks two things, in order: the original
   hardcoded demo account (`officer` / `changeme123`, kept working exactly
   as before so nothing that depended on it breaks), then a lookup by email
   against the `authorities` table with the stored password hash. Whatever
   you registered with is exactly what logs you in afterwards — that's the
   whole point of the change.
4. **`authority.html`** gained a "Register an account" link on the login
   screen, a registration card (name / email / authority-type pills /
   password / confirm), and an officer badge that now shows the signed-in
   person's real name and role (e.g. "Signed in as Dr. Anjali Sharma
   (Counsellor)") instead of always saying "officer".
5. **Not carried over from the React build, on purpose, to keep this
   change minimal**: the "Manage Authorities" admin screen (add/remove
   other officers from a table) and any role-based route restriction.
   Every logged-in authority — any role — can currently see the same
   `/cases` review queue, same as before this change. Role is stored and
   shown, not yet enforced. That's flagged as next-round work below, not
   silently skipped.
6. **Still demo-grade, same caveats as the original build's auth.py**:
   in-memory bearer tokens (reset on server restart), no email
   verification, no per-role access control, self-service signup with no
   approval step (anyone can register themselves as "Law Enforcement" or
   "District Admin" right now — same open-registration tradeoff the React
   prototype had). Fine for a hackathon demo; flagged here so it's a
   deliberate, disclosed choice rather than an oversight.

**Verified**: registered a fresh account via `/auth/register`, confirmed
`/auth/login` with those exact credentials returns a valid token, confirmed
`/cases` returns 401 with no token and 200 with one, confirmed a wrong
password is rejected (401), confirmed a duplicate email is rejected (409),
and confirmed the original `officer`/`changeme123` demo login still works
unchanged.

## Round 14 — role-based case visibility (Legal Aid / Law Enforcement only see their own scope)

Previously every logged-in authority saw the exact same full case queue
regardless of role. Now:

1. **Counsellor and District Admin stay unrestricted** — they see every
   case, same as before. Counsellors are the front-line reviewer for
   everything; admins get full oversight (this is also the demo `officer`
   account's role, so nothing that depended on seeing everything breaks).
2. **Legal Aid Officer** only sees cases whose `recommended_actions`
   include a legal-aid tag (routed there by `legal_dispute`,
   `workplace_harassment`, `financial_exploitation`,
   `cybercrime_online_abuse`, `sexual_harassment_assault`, or
   `elder_abuse` in `scoring.py`'s `SERVICE_TAGS`).
3. **Law Enforcement** only sees cases tagged for police/child-protection
   action (`physical_safety_abuse`, `sexual_harassment_assault`,
   `elder_abuse`, `immediate_danger`, `child_safety_concern`).
4. A case that needs both (e.g. sexual assault — both a legal-aid and a
   police tag) correctly shows up for **both** roles, not just one — this
   is a filter on each role's view, not an exclusive assignment of the
   case to a single authority.
5. Enforced on all three endpoints, not just the dashboard list:
   `GET /cases` (filtered list), `GET /cases/{id}` and
   `PATCH /cases/{id}/review` now both return **403** if a Legal Aid or
   Law Enforcement account requests a case outside their scope — so this
   isn't just a frontend display filter a determined client could bypass.
6. `authority.html` shows a one-line notice under "Review queue" stating
   whether the current login is scoped or seeing everything, so it's
   never ambiguous why one officer's queue is shorter than another's.

**Verified live**: submitted a legal-dispute case, a physical-abuse case,
and a mild/unflagged case; registered one account per role; confirmed
Counsellor and Admin each saw all 3, Legal Aid saw only the legal-dispute
case, and Law Enforcement saw only the physical-abuse case. Also confirmed
a Legal Aid account gets 403 (not 200, not a silent empty result) when
directly requesting or trying to review a case outside its scope by ID,
while it can still fetch/review a case that *is* in scope.

## Round 15 — counsellor gender/age preference, asked right after consent

Right after consent is given (the moment `/chat/start` fires), the bot now
asks two setup questions before the normal opening line:

1. "Would you feel more comfortable speaking with a female or male
   counsellor, or do you have no preference either way?"
2. "Do you have an age preference for your counsellor — younger, older,
   or no preference either way?"

1. **Free-text, not forced buttons** — answers are matched against a
   keyword set (`female`/`woman`, `male`/`man`, `younger`, `older`,
   `no preference`/`either`/`any`, etc.) so plain typed or spoken
   replies work; anything unmatched is kept as raw text (`gender_raw`
   / `age_raw`) rather than silently discarded, so a reviewer can still
   read the actual answer.
2. **`chat.html` adds tappable chips** (Female / Male / No preference,
   then Younger / Older / No preference) for these two questions
   specifically, matched client-side on the bot's wording — mobile
   users aren't forced to type for something this structured.
3. **Excluded from scoring and from `MAX_TURNS`** — these two answers
   never enter the transcript that gets scored or shown as an excerpt,
   and don't count against the 9-turn conversation limit, since they're
   setup, not disclosure.
4. **An exit phrase ("bye", "stop", etc.) is still honoured immediately**
   even mid-way through these two questions — nobody is forced through
   them if they want to leave right after giving consent.
5. **Stored on the case, not auto-matched** — the answer is saved to
   `meta.counselor_pref` (`gender`, `gender_raw`, `age`, `age_raw`) and
   shown on the officer dashboard as a "👤 Prefers: ..." line under the
   excerpt. There's no counsellor directory/matching system in this
   prototype — this makes the stated preference visible to whoever
   picks up the case to honour when assigning, not something that
   auto-routes.

**Verified**: answered "Female please" / "younger if possible", then
disclosed physical abuse — confirmed the preference reached the case's
stored meta correctly, the disclosure-only transcript was unaffected,
turn count stayed at 1, and the SVI/floor logic scored normally. Also
verified typing "bye" immediately after consent (before answering
either question) exits cleanly, and an unmatched free-text answer
("not too fussed, maybe someone experienced") falls back to
`"unspecified"` while preserving the raw text instead of losing it.

## Round 16 — persistence, migrations, audit logging, and wider Hindi coverage

A round driven by a real list of limitations from testing this build, plus
one specific, high-priority complaint: an accidental page reload (or a
server restart) mid-conversation wiped everything and forced starting
over from the consent screen — genuinely frustrating for someone who
just disclosed something difficult. Fixed that, plus everything else on
the list that's actually fixable without external infrastructure this
environment doesn't have.

1. **Chat sessions are now persisted, not just kept in memory** — every
   turn writes the FULL session state (conversation stage, the
   cumulative transcript used for scoring, a replayable bot/user turn
   log, counsellor preference answers, etc.) to a new `chat_sessions`
   table, not just the `_sessions` Python dict that used to vanish on
   restart. `chat.py` falls back to loading from the DB on a cache
   miss instead of declaring the session expired.
2. **Reload/restart recovery, end to end**: `chat.html` now saves its
   `session_id` to `sessionStorage` (deliberately NOT `localStorage` —
   it survives a reload of the same tab, which is the actual reported
   scenario, but clears itself the moment the tab or browser closes,
   so a conversation this sensitive doesn't linger indefinitely on a
   shared or public device). On page load it silently calls the new
   `GET /chat/resume/{session_id}` endpoint, repaints the full
   conversation from the returned turn log, and continues exactly
   where it left off — with a small "Not you? Start a new
   conversation" link as an explicit way out. A session that's already
   finished, or genuinely doesn't exist, falls back to the normal
   consent screen.
3. **Officer login tokens now survive a server restart too** —
   persisted to a new `officer_tokens` table; `require_officer` falls
   back to the DB on an in-memory cache miss and re-warms the cache.
   They still don't expire (no TTL/rotation) — flagged as a real-
   deployment caveat, same as the plaintext demo API key.
4. **Real schema migrations — no more deleting `sahayk.db` to change
   the schema.** `database.py` now has an idempotent
   `_ensure_columns`/`_MIGRATIONS` mechanism: every column added after
   a table's original release goes through `ALTER TABLE ... ADD
   COLUMN`, which only runs for a column that isn't already there.
   Tested against a database built with the OLD schema (missing
   `gender`/`age`/`reviewed_by_*`) containing real cases and an
   officer account — migrated in place on the next `init_db()` call
   with zero data loss.
5. **Audit logging of who reviewed what** — new `review_log` table
   (append-only: every review action, which officer, what action,
   when) plus `reviewed_by_email` / `reviewed_by_name` / `reviewed_at`
   columns on `cases` itself for quick display. `authority.html` now
   shows "by `<name>` · `<timestamp>`" under a reviewed case's action.
6. **Wider Hindi lexicon coverage** — added a starting Hindi phrase set
   to all 6 categories that were previously English-only (legal,
   workplace, cybercrime, child safety, elder abuse, substance) plus
   `vulnerability_context`. All 14 categories now have *some* Hindi
   coverage; none of it has been reviewed by a native speaker yet, and
   this doesn't change the underlying approach — it's still
   exact/fuzzy phrase matching, which will keep having gaps for tense,
   phrasing, and slang no matter how many phrases are added. Genuinely
   closing that needs real semantic/embedding-based matching, which is
   out of scope for a phrase-list lexicon file.

**Deliberately NOT touched — needs real external infrastructure, not
more code:**
- **Real NLP/semantic matching** to replace exact/fuzzy phrase lists —
  needs an actual embedding model, not a bigger phrase list.
- **A trained voice-emotion classifier** to replace the current
  pitch/pause/rate heuristic — needs labeled audio training data and a
  hosted model this environment can't provide.
- **Real SMS/email delivery** — needs a live telecom/email provider
  account (Twilio, SendGrid, etc.) and credentials from the team; there
  is no OTP/notification feature in the current build to wire this
  into, so nothing was faked here.

**Verified**: a full chat conversation (including a Hindi disclosure)
survives an in-memory wipe simulating both a page reload and a server
restart, and resumes with the exact turn history intact and the
conversation still continuing correctly afterward; an officer session
survives the same simulated restart; a database built with the
pre-Round-16 schema migrates in place with its existing case and
officer data intact; a review call correctly populates the audit trail
end to end; and the new Hindi phrases score into their intended
categories (with sensible cross-category overlaps, e.g. a workplace
threat phrase also matching `intimidation_threat`).

## Round 18 — atrocity-specific detection, routing and languages (SIH26093 fit)

Checked the build against every line of the SIH26093 problem statement.
The design fit, but the lexicon had been written for general distress, so
the situations 14566 actually exists for scored **Low**: "my daughter was
gang raped" (17), "they murdered my father… will kill me if I testify"
(21.9), social boycott, displacement/arson, police refusing an FIR,
pressure to withdraw a case, and every Gujarati/Marathi/Tamil/Bengali
message (capped at Moderate 40).

1. **Seven new lexicon categories** (`lexicon.py`), English + Hinglish +
   Hindi, with some Gujarati/Marathi/Tamil/Bengali: `caste_atrocity`,
   `rape_sexual_violence`, `family_murder`, `witness_intimidation`,
   `social_boycott`, `displacement_arson`, `police_noncooperation`, plus a
   low-weight `atrocity_context` (Dalit / SC-ST / upper caste…) that nudges
   the score but never decides it.
2. **Safety floors** (`scoring.py`): sexual violence ≥ 90 and killing of a
   family member ≥ 85 (Critical); witness pressure ≥ 60, caste abuse /
   boycott / displacement ≥ 55, police non-cooperation ≥ 50 (High).
3. **Routing fixes** (`SERVICE_TAGS`): new *Witness protection referral*
   tag (fires on pressure, not only at Critical); *Medical assistance* now
   follows physical and sexual violence; new *Relief & rehabilitation
   referral*; *Escalate FIR / investigation delay to District Admin*;
   death threats now suggest police consideration.
4. **New role: Rehabilitation Officer** (`auth.py`, `main.py`,
   `authority.html`) — sees only cases tagged for relief & rehabilitation,
   403 on anything else, same server-side enforcement as Legal Aid / Police.
   Law Enforcement now also sees witness-protection cases.
5. **Regional crisis phrases** added to suicidal ideation, threats,
   physical violence and fear (Gujarati, Marathi, Tamil, Bengali,
   Hinglish); Bengali added to the chat language list.
6. **Matching fixes**: Hindi "ँ"/"ं" and nukta spellings now compare equal;
   lexicon phrases go through the same normalisation as the message (so
   hyphenated / apostrophe phrases like "माता-पिता" can finally match); typo
   tolerance now requires the same first letter, so "scraped" no longer
   matches "raped" and "misplaced" no longer matches "displaced".
7. **Chat follow-ups** for each new category (safety, medical help, safe
   place to stay tonight, access to water/food, "did this happen in front
   of others?", police station / legal aid).

**Verified** (`tests/atrocity_battery.py`): original 24-message battery
3/24 → 23/24; a 16-message set used once for tuning 1/16 → 15/16; a fresh
20-message set written afterwards and **not tuned on** 5/20 → 12/20. All
62 messages in `test-matrix.md` score exactly as before (no regressions).
Live API test: five atrocity chats scored and routed correctly; Counsellor
and Admin see all cases, Legal Aid 4/5, Police 3/5, Rehabilitation 2/5;
out-of-scope case → 403; no token → 401; demo login unchanged.

**Still honest limits**: 12/20 on unseen wording is the real ceiling of a
phrase list — new phrasings ("मैं खुद को खत्म कर लूंगा", "ammi ko bahut
maara", "the inspector wants money to register my case") still miss, and
unmatched non-Latin text falls back to Moderate 40 + mandatory human
review rather than Low. None of the new Indian-language phrases have been
reviewed by native speakers yet. The real fix is a trained model (the
Round 17 TF-IDF / MuRIL pipeline, retrained with these seven categories).

## Changelog — round 2 fixes (from live testing feedback)

1. **Critical scoring bug fixed**: "I am planing to suicide" and "can i sucide?"
   previously scored 0–10 (Low). The lexicon was missing the word "suicide"
   itself, and typos slipped through exact-match matching entirely.
   Fixed with: (a) added the missing root terms, (b) fuzzy single-word
   matching (`difflib`) to catch typos like "sucide"/"suicdal", (c) a hard
   safety floor — any suicidal-ideation hit forces SVI ≥ 85 / Critical
   regardless of what the blended weighted score would otherwise say, so
   calm-sounding phrasing can never dilute an explicit statement.
2. **Privacy fixed**: the review queue was previously open to anyone at
   `/cases`. It's now behind officer login (`/auth/login` issues a bearer
   token; `/cases*` requires it — verified returns 401 without one). The
   chat-facing endpoints (`/chat/*`) never return the score, category, or
   breakdown to the person chatting — only a conversational reply.
3. **Chat flow added**: `/chat/start` + `/chat/message` now run a real
   conversation instead of a single text box — open-ended opener, then
   adaptive follow-ups (`chat.py`) based on whichever categories the
   running score has picked up (fear/threat, isolation, hopelessness),
   in priority order. A suicidal-ideation hit immediately surfaces crisis
   resources (KIRAN 1800-599-0019, Tele-MANAS 14416) in the same turn,
   every time it's detected — it doesn't wait for the chat to finish.
4. **Voice input added**: `chat.html` records audio and uses the browser's
   own Web Speech API (`SpeechRecognition`) for live multilingual
   transcription (language dropdown — English, Hindi, Marathi, Tamil,
   Gujarati; more BCP-47 codes work with any Chrome-supported language).
   The raw recording is separately sent to `/analyze/voice/audio`, which
   uses `librosa` to compute real pitch variance, pause ratio, and a
   speech-rate proxy — tested against a synthetic wobbly-pitch clip and
   confirmed the features respond correctly. This *is* the emotional-AI
   layer you asked about — it's not a stub.

## What's real vs. stubbed

| Component | Status |
|---|---|
| Text sentiment (VADER) | Real, offline |
| Keyword severity + suicidal-ideation safety floor | Real, tested against typo/caps edge cases |
| Multi-turn adaptive chat | Real (`chat.py`) |
| Voice transcription | Real, via browser Web Speech API (no server-side ASR needed) |
| Voice pitch/pause/rate features | Real, via `librosa` on the uploaded recording |
| Voice *emotion classification* (beyond pitch/pause/rate) | Still a stub — a trained speech-emotion model is the next upgrade, needs labeled audio + a model host this environment doesn't have |
| Officer login | Real but demo-hardcoded — replace with real accounts + RBAC + JWT before any deployment |
| Multilingual scoring | Language is now tagged and transcribed correctly per-language via the browser, but the *scoring* (VADER, lexicon) is still English-only — non-English text isn't translated before scoring yet |
| Police/witness-protection trigger | Intentionally **not automated** — always a human action from the dashboard |

## Next build steps, in order

1. **Translate before scoring**: non-English transcripts currently get
   passed through as-is. Wire in Bhashini or Google Translate so Hindi/
   regional text gets scored properly instead of just transcribed.
2. **Real accounts**: replace `auth.py`'s hardcoded demo login with
   hashed passwords in the DB and per-officer roles.
3. **Audit log**: record every dashboard view/action against a case, not
   just the final review outcome.
4. **Expand + validate the lexicon**: get a counselor to review
   `lexicon.py` categories and add native (not translated) regional-
   language phrases.

## Files

```
backend/
  main.py            FastAPI app + routes (auth-gated /cases, public /chat)
  auth.py            demo officer login + bearer-token check
  chat.py            multi-turn conversation state machine
  scoring.py          SVI weighting + risk bands + safety floor
  lexicon.py           keyword severity categories + fuzzy typo matching
  audio_features.py   librosa-based pitch/pause/rate extraction
  database.py         SQLite persistence
frontend/
  chat.html           victim-facing chat (text + voice), no case data ever shown
  authority.html      login-gated review queue
```

## Changelog — round 4 fixes (from live testing feedback)

1. **Cold shutdown after "yes" to the help-type question** — fixed. A bare
   "yes"/"no" to "is there a specific kind of help you're hoping for" now
   gets a clarifying rephrase ("What would help most — someone to talk to,
   legal help, medical support, or something else?") instead of closing
   immediately.
2. **Crisis message repeating verbatim** — fixed. First suicidal-ideation
   hit shows the full message with both helplines; every hit after that in
   the same session shows a shorter, varied check-in instead. "Bye"/"stop"/
   similar exit phrases are now recognized and get a warm, crisis-aware
   goodbye instead of being pulled into the next scripted question.
3. **Typos silently scoring as safe across the whole lexicon** (not just
   suicide-related words) — fixed structurally. Matching is now word-level
   with typo tolerance on every phrase in every category ("he choakd me"
   still matches "choked me"), not just a special case for "suicide".
4. **No acknowledgment for ambiguous/mild disclosures** — fixed. Any
   message that doesn't hit a category now gets a brief "Thanks for
   telling me that" before the next question, instead of silence.
5. **Legal/workplace/cybercrime/child-safety/elder-abuse categories had no
   follow-up questions** — fixed. All 14 lexicon categories now have their
   own 2-question progressive follow-up set with an empathetic lead-in,
   not just the original 7.
6. **Mild cases ending on a cold "noted, reviewed" message** — fixed. If
   nothing concerning was ever detected, the close now offers a
   counselling referral instead of the formal review-queue wording.
7. **Voice score never reaching the actual case** — fixed. Voice turns now
   go through `/chat/voice_message`, which extracts real pitch/pause/rate
   features and folds them into the SAME session's case record (previous
   `/analyze/voice/audio` created a disconnected, orphaned case — the
   speech signal was being computed but never actually used).

## API key (for connecting your own frontend)

The victim-facing endpoints (`/chat/start`, `/chat/message`,
`/chat/voice_message`, `/analyze/text`, `/analyze/voice/audio`) now require
an `X-API-Key` header on every request. `chat.html` already sends it.

```
Key:    e59d8c99f3b6b9b992b991988d458d679df9e7e475555dad
Header: X-API-Key: e59d8c99f3b6b9b992b991988d458d679df9e7e475555dad
```

The value lives in `backend/config.py` on the server side. If you build a
different frontend, just send that same header on each request.

**Be clear-eyed about what this does and doesn't protect against**: it's a
shared secret, not per-user auth. Anyone who opens your frontend's dev tools
or view-source can read this key straight out of the JavaScript — so it
stops random bots/scanners from hitting the API, but not a determined
person who's looked at your frontend's code. That's fine for a hackathon
demo. Before any real deployment, this needs to move server-side (a small
backend-for-frontend proxy that holds the key and the browser never sees
it) or be replaced with real per-user login. Rotate the key anytime with:
```
python -c "import secrets; print(secrets.token_hex(24))"
```
then update both `backend/config.py` and the `API_KEY` constant near the
top of `chat.html`'s `<script>`.

## Round 5 — real frontend design + a bug fix found while doing it

Rebuilt `chat.html` and `authority.html` with a deliberate visual identity
instead of the earlier plain/functional styling: deep navy + oxblood
letterhead accent, Source Serif 4 / Source Sans 3 pairing, and a real
"Get help right now" band at the bottom of the chat page linking to
verified, current national helplines:

| Service | Number | Website |
|---|---|---|
| Mental health / crisis (Tele-MANAS) | 14416 / 1800-891-4416 | telemanas.mohfw.gov.in |
| Mental health (KIRAN) | 1800-599-0019 | (integrated with Tele-MANAS; no separate live site found) |
| Legal aid (NALSA) | 15100 | nalsa.gov.in |
| Women's safety (NCW) | 14490 | ncw.gov.in |
| Child safety (CHILDLINE) | 1098 | — |
| Cybercrime | 1930 | cybercrime.gov.in |
| Police emergency | 112 | — |

The authority dashboard's table now collapses into stacked cards below
720px instead of a squeezed horizontal scroll.

**Bug found and fixed while rebuilding**: the round-4 API key change added
`X-API-Key` to the `/chat/message` and `/chat/voice_message` fetch calls
but silently missed `/chat/start` (the string-replace target didn't match
due to indentation, and failed with no error) — meaning the chat couldn't
even open a session. Fixed and verified: a request to `/chat/start`
without the key now correctly gets 401, and with the key — exactly as
`chat.html` sends it — gets 200.

## Changelog — round 5 fixes (voice reliability + real officer accounts)

1. **Voice chat going silent (no bot reply at all)** — root-caused and
   fixed. Browsers record audio as webm/opus, which needs ffmpeg
   installed to decode — if it's missing (common on a fresh Windows
   setup), the old code crashed with an unhandled exception, which
   returned a non-JSON error page and broke the frontend's `res.json()`
   before it could ever show a reply. Fixed at three layers: (a)
   `audio_features.py` now wraps every extraction step so a decode
   failure degrades to a neutral signal instead of crashing, (b) the
   `/chat/voice_message` endpoint catches any remaining audio-read
   failure directly, (c) a global exception handler guarantees every
   response is valid JSON with a real reply no matter what fails.
   Verified by feeding the endpoint deliberately corrupt audio bytes —
   `audio_analysis_ok: false` came back but the transcript still got a
   correct, on-topic reply. Install ffmpeg for the full pitch/pause
   signal (`sudo apt install ffmpeg` on Linux, or download from
   ffmpeg.org and add it to PATH on Windows) — but the chat now works
   either way.
2. **Officer accounts are now real**, not a single hardcoded check:
   hashed passwords (PBKDF2, not plaintext) stored in the database,
   forced password change on first login, and a forgot-password flow
   that requires both a registered phone number and email on file
   before a reset is allowed (two factors, not one).
   **Demo limitation**: there's no SMS/email provider configured in
   this environment, so the one-time code can't actually be texted or
   emailed anywhere — it's returned directly in the API response
   (`demo_otp`) so the flow is fully testable end-to-end. Before any
   real deployment: wire in an SMS provider (e.g. Twilio) and an email
   provider (e.g. SendGrid/SES) in `auth.py` where marked, delete the
   `DEMO_MODE` branch so the code is never exposed to the client, and
   set the officer's real phone/email in place of the placeholders in
   `auth.py`.

## Changelog — round 6 fix (stuck/stale password-change screen)

The confirmed flow (fresh login with `officer` / `changeme123` → forced
password-change screen → review queue with cases) was already correct and
tested — but a restored browser tab (Chrome's "continue where you left
off") could bring back a stale in-progress screen with a session token
that no longer works after a server restart, with no way back to login.
Fixed: the page now force-resets to a clean login screen on any fresh
view or restored tab, and a stale/expired session during password change
now shows a clear "sign in again" link instead of a dead end.

## Changelog — round 7 (simplified back to plain login)

Removed the forced password-change and forgot-password/OTP flow — it added
more moving parts than this needed. Back to the simple, original login:
`officer` / `changeme123`, straight through to the review queue. The one
thing kept from that round: the page still resets to a clean login screen
if a browser restores a stale tab, since that's invisible robustness, not
added complexity.

## Changelog — round 8 (Hindi detection was silently broken, plus a general safety net)

Root-caused why a severe Hindi disclosure (stalking, coercion, financial
control) got the flattest possible response ("Thanks for telling me
that. Is there anything else..."):

1. **Normalization was ASCII-only** — `_normalize()` stripped every
   non-Latin character to nothing, so any Hindi (or other non-English
   script) message reduced to an empty string before matching even ran.
   A naive Unicode regex fix isn't enough either — it drops Devanagari
   combining vowel marks (matras), mangling words beyond recognition. Now
   uses Unicode category filtering (keeps letters, marks, and digits from
   any script), verified to preserve words correctly.
2. **Added a starting Hindi phrase set** to the highest-severity
   categories (fear, threats/stalking, physical/sexual abuse, financial
   control, hopelessness, isolation, suicidal ideation) — verified against
   the exact message that surfaced this: now correctly detects fear,
   intimidation/stalking, sexual coercion, AND financial exploitation in
   the same message. Legal, workplace, cybercrime, child-safety, elder-
   abuse, and substance categories are still English-only — flagged for
   next-round expansion.
3. **VADER sentiment was silently zeroing out non-English scores** —
   VADER can't read Hindi, so it returned "neutral" for every non-English
   message, which isn't an absent signal, it's actively misleading (it
   let the formula treat "unreadable" as "calm"), and was capping the
   score 20% below what the keyword signal alone supported. Fixed:
   non-Latin-script text now redistributes that weight onto keyword
   severity instead. His exact message went from 40.5/Moderate to
   58.5/High once fixed — verified English scoring is unaffected.
4. **General safety net, regardless of language**: a long, detailed
   message (20+ words) that still matches nothing now gets real
   acknowledgment plus a direct safety-check question, instead of the
   flat generic line — covers whatever the lexicon still doesn't catch,
   in any language, not just this specific Hindi gap.

## Changelog — round 9 (stale cached pages)

The password-change modal you saw again was already removed in round 7's
code — what you were actually looking at was a browser-cached copy of the
old page. Added `Cache-Control: no-store` to both `/` and `/authority` so
this class of "I fixed it but you're still seeing the old version"
confusion can't happen again on future updates. If a page still ever
looks wrong after unzipping a new version, a hard refresh (Ctrl+Shift+R)
will now always show what's actually in the code.

## Changelog — round 10 (active-danger scoring 0, and tone mismatch)

A message describing an assailant physically present RIGHT NOW, plus a
fresh injury, scored **0** and got a generic "that sounds like a lot to
carry" response — as severe a false negative as the round-4 suicide bug.
Root causes, all fixed:

1. **No concept of "the threat is here right now"** existed anywhere in
   the lexicon. Added a new `immediate_danger` category (phrases like
   "standing right behind me", "he is here right now") that — like
   suicidal ideation — now bypasses the normal conversation flow
   entirely: it always takes priority, always responds with an urgent
   safety message (112 police emergency), and forces a hard SVI floor of
   85/Critical, every single turn it's detected, and it never gets cut
   off by the turn limit.
2. **No phrase existed for weapon-based injury** ("cut my finger",
   "stabbed me") — added to `physical_safety_abuse`.
3. **"I am not safe" didn't match "I'm not safe"** — same contraction-
   mismatch bug class as the round-3 suicide fix, just a different
   specific phrase. Added the expanded form explicitly.
4. **Generic harassment/coercion phrasing** ("being harassed", "forcing
   me to be with him", "won't leave me alone") wasn't covered at all —
   added to `sexual_harassment_assault`.
5. **Tone**: the "long message, nothing detected" fallback said "that
   sounds like a lot to be carrying" — fine for a slow-burn disclosure,
   badly mismatched for an active emergency. Softened to lead with
   safety ("Your safety matters most right now — are you safe at this
   moment?") so it reads appropriately either way, as a second line of
   defense for whatever the lexicon still misses.

Verified against his exact 3-turn conversation: SVI went from 0 to
85/Critical, with police-intervention and legal-aid routing both
correctly triggered, and the bot now responds with real urgency instead
of a scripted "thanks for sharing" — and stays engaged rather than
closing after a fixed number of turns.

## Changelog — round 11 (severity scoring gaps, conversation loop, voice startup delay)

1. **"He is hitting me" scored 0** — the lexicon only had "he hits me"
   (simple present tense); "is hitting" is a different verb form and
   exact-phrase matching treated it as completely unrelated. Added
   gerund/continuous-tense forms across physical abuse, threats, fear,
   and harassment ("hitting me", "beating me", "scaring me", "touching
   me inappropriately", etc.) — this class of gap (tense/phrasing
   variants) has come up repeatedly, so this is a broader pass, not
   just this one phrase.
2. **Severe categories were structurally under-scored even when
   correctly detected** — physical abuse, sexual assault, and similar
   were only worth 45% of a full weighted blend, so a flatly-worded but
   genuinely severe disclosure could land at "Moderate." Added explicit
   severity floors (`CATEGORY_FLOORS` in `scoring.py`): physical abuse
   and sexual assault now floor at High (55+), child safety at Critical
   (85+) given the extreme vulnerability involved — on top of the
   existing suicide/immediate-danger floors, never replacing the
   nuanced weighted score for everything else.
3. **The conversation could get permanently stuck** repeating the exact
   same crisis/danger message forever once triggered, since that hit
   persists in the transcript for the rest of the session — this is
   what broke "the chatbot should actually counsel, not just repeat
   itself." Fixed: the full urgent message now shows only once: every
   turn after that falls through into real follow-up questions (the
   normal category flow), with a short one-line reminder kept in front
   of each reply so the emergency number/helpline stays visible without
   dominating every turn. Verified: the same conversation that
   previously looped now asks real follow-up questions turn after turn.
4. **Voice "loading so much"** — root cause found: the pitch-detection
   library has a one-time ~8-10 second compilation cost the first time
   it ever runs on a machine. That cost was landing on whoever sent the
   first voice message. Moved to a startup warmup instead — verified
   the first real voice message now takes ~0.2s, not 9+.
5. **Transcription accuracy** — this one is a genuine limitation, not
   something fixable in this code: live speech-to-text comes entirely
   from the browser's own Web Speech API (Chrome's built-in engine),
   not from anything in this backend. Its accuracy depends on
   microphone quality, background noise, and Chrome's own model for
   the selected language — there's no lever here to improve it further
   without switching to a paid cloud ASR provider (Google Cloud
   Speech-to-Text, Bhashini) with its own API key, which is a real
   option if transcription quality matters enough to justify the cost.

## Changelog — round 12 (external technical review — scoring architecture)

Implemented a detailed external review of the SVI scoring design. Verified
every claim against the actual code before changing anything.

1. **Cumulative multi-category scoring, not max-only.** Previously the
   keyword score was just the single highest-weight category hit — someone
   disclosing physical abuse AND threats AND isolation scored almost the
   same as someone disclosing only one. Replaced with a capped
   probabilistic combination (`K = 100 x (1 - product(1 - weight_i x
   confidence_i))` across every hit category — "noisy-OR"). Verified:
   "he hits me" alone = 85; adding "and no one believes me" = 92.5; adding
   "and he threatened to kill me" = 98.9. Each additional independent
   disclosure now genuinely raises the score.
2. **Negation was a real, dangerous false-positive bug.** "I do not want
   to die" was matching the same as "I want to die", since both contain
   that literal substring. Added negation detection: any negation word
   ("not", "never", "dont", etc.) in the few words before a match
   suppresses it entirely. Verified: now correctly scores 0/no hits.
3. **Hedged statements are now lower-confidence, not full-strength.**
   "I am afraid he MAY hurt me" previously scored identically to "he hurts
   me" (an active, ongoing disclosure). Hedge words ("may", "might",
   "could") now halve that match's contribution — still flagged, just not
   treated as equally certain as a direct statement.
4. **Text-only messages were structurally penalized vs. voice.** The
   formula always included the 20% speech-stress weight even when there
   was no voice data, silently treating "no data" as "calm". Fixed the
   same way the earlier Hindi/VADER fix worked: missing signals are now
   excluded from the blend entirely and the remaining weights renormalize,
   rather than a phantom 0 eating its share. This also fixed a related bug
   this surfaced: a FAILED voice decode was being treated as real "calm"
   data instead of "no data" — now correctly excluded too.
5. **Unrecognized-language messages no longer silently score low.** A
   long message that matches nothing, in a script the lexicon doesn't
   cover, is now flagged `language_confidence: "low"` and floored to at
   least Moderate (40) so it can't sink to the bottom of a score-sorted
   queue — instead of being indistinguishable from genuine calm.
6. **Escalation overrides raised to the reviewer's suggested values**:
   suicidal ideation and immediate danger now floor at 95 (was 85), child
   safety at 90 (was 85).
7. **Added a `vulnerability_context` signal** (pregnancy, disability,
   homelessness, minor/elderly self-disclosure, language isolation) that
   contributes to the cumulative score like any other category — a
   lightweight version of the reviewer's proposed context/vulnerability
   axis, integrated into the existing architecture rather than a full
   parallel scoring dimension.
8. **Officers can now see WHY a case scored the way it did** — not just a
   number. The dashboard has a "▸ why this score" toggle per case showing
   matched phrases per category with confidence (direct vs. uncertain),
   the full breakdown, whether a floor was applied and which category
   triggered it, and an explicit unverified-language warning when
   relevant. A "needs review" flag now shows separately from the risk
   band, since language-confidence issues force review regardless of
   the numeric score.

**Deliberately NOT implemented** (flagged as genuine next-deployment work,
not something to hack into a demo): full negation/timing/immediacy
extraction beyond the negation/hedge check above (would need real NLP,
not phrase matching); a full separate context/vulnerability scoring axis
(the lightweight category above captures the intent, not the full 20%-
weighted redesign); encrypted storage, RBAC, audit logging, rate limiting,
real SMS/email delivery, and independent security testing — all
correctly identified as pre-deployment requirements, none of which belong
in a hackathon prototype's codebase as a fake/simulated version of
themselves.

## Round 17 — a trained model replaces phrase matching for category detection

The lexicon's own docstring predicted this round: phrase matching "will keep
having gaps for tense, phrasing, and slang variants no matter how many
phrases are added; only semantic/embedding-based matching (a real model, not
a phrase list) would genuinely close that." Round 17 does that, and the
earlier "deliberately NOT implemented" note about negation needing real NLP
is now partly addressed too.

**What changed, and what deliberately did not.** Only the *matching* step is
learned. The model outputs a probability per category; everything downstream
— the per-category weights, the noisy-OR aggregation, `CATEGORY_FLOORS`,
`SERVICE_TAGS`, role routing — is untouched and consumes those probabilities
exactly as it consumed phrase-match confidences. That boundary is the point:
`scoring.py` argues the value of a rubric is that "a reviewer can see and
adjust exactly why a case scored the way it did", and a model emitting one
opaque 0-100 number would destroy that. Emitting per-category evidence keeps
it. Officers still see which categories fired, at what confidence, and the
weights are still editable in one readable dict.

**Detection is a union, not a replacement.** A category counts as present if
*either* the lexicon or the model finds it, at whichever confidence is
higher. On the held-out set the model caught ~3,900 disclosures the rules
missed entirely, and the rules caught none the model missed — but the union
is kept anyway so this round is strictly additive: nothing the rules used to
catch can regress.

### Results (held-out, 5,000 messages)

| system | band acc | escalation recall | critical recall | false escalation | negation FP | severity MAE |
|---|---|---|---|---|---|---|
| rules only (before) | 0.20 | 0.09 | 0.08 | 0.02 | 0.05 | 56.5 |
| model, no denial gate | 0.82 | 0.98 | 0.97 | 0.30 | 0.88 | 9.7 |
| **served pipeline** | 0.85 | 0.98 | 0.96 | 0.12 | 0.32 | 7.2 |

- *escalation recall* — of cases truly High/Critical, the share placed in
  High/Critical. **This is the safety metric**; the rest are secondary.
- *false escalation* — of truly Low cases, the share wrongly escalated. The
  cost the reviewer pays for that recall.
- *negation FP* — of explicitly DENIED disclosures ("it is not the case that
  he hits me"), the share still recorded.

Per-category detection on the same split: micro-F1 **0.931**
(precision 0.896, recall 0.970),
macro-F1 **0.933** across all 16 categories. Full numbers in
`training/metrics.json` and `training/pipeline_metrics.json`.

### Read those numbers carefully

The rules' 0.09 escalation recall is **not** their real-world performance.
The evaluation corpus was deliberately written in wording outside the phrase
list, so a near-total miss there is by construction. It measures the size of
the blind spot the model covers, not a claim that the rules fail 90% of real
calls.

Equally, the model's 0.85 band accuracy is measured on unseen *arrangements*
of vocabulary it has seen. It is not evidence that it handles words absent
from training — it does not, and no bag-of-words model can.

### The negation problem, and why the gate exists

A linear bag-of-words model cannot scope negation: in a weighted sum, the
clause's own words outvote the denial cue. Unaided, this model recorded the
category anyway on **88%** of explicitly
denied disclosures. Training on far more negated examples was tried and moved
it only to ~79% while degrading every other metric — the limit is the model
class, not the data.

So `ml_model.py` handles it structurally instead: the message is split into
clauses, clauses carrying an explicit *denial* cue are dropped, and the model
scores what remains. That cuts negation false positives to
**32%** and false escalation from
30% to 12%, costing
about a point of critical recall.

Two honest caveats. The cues are *metalinguistic denials* ("it is not the
case that"), never general negation words — gating on "not"/"nahi" would
suppress "he will NOT let me leave the house", a real disclosure of control.
And those cues were written from the same denial phrasings used to build the
evaluation corpus, so the measured improvement is optimistic. Real callers
deny things in ways no list anticipates. **Proper negation scoping needs a
model that reads syntax** — that is the main reason to run
`training/train_transformer.py`.

### How it was trained

- **Corpus**: 22,000 training / 5,000 test messages, synthetic,
  across English (9,188), Hindi (4,330), Gujarati
  (2,814) and romanised Hinglish/Gujlish (5,668). No
  real caller data was used, and none should be. Gujarati and romanised text
  had **zero** lexicon coverage before this round.
- **Labels**: derived from this project's rubric — the same noisy-OR over the
  same weights in `lexicon.py`. Rule-derived, not clinician-annotated.
- **Split**: held out at the *expression* level, not the row level. A
  phrasing in training never appears in test. Splitting rows at random would
  let near-duplicates straddle the split and inflate every metric above.
- **Features**: word TF-IDF (1-2 grams) + character TF-IDF (2-5 grams). The
  character half is what carries Devanagari and Gujarati with no tokeniser,
  absorbs romanised spelling variation ("chillata"/"chilata"/"chillaata"),
  and survives keypad typos and Web Speech API mistranscriptions — all of
  which break exact phrase matching outright.
- **Thresholds**: tuned per category on a validation split carved out of
  training, never on test. The six categories with safety floors are tuned
  for F2 (recall weighted 4:1) and capped at 0.35, for the reason already
  written into `CATEGORY_FLOORS`: for these, over-reacting to an uncertain
  mention beats under-reacting to a real one.

Reproduce:

```bash
cd training
python generate_dataset.py --n 22000 --n-test 5000 --report-lexicon-recall
python train.py                # ~2 min on 2 CPU cores, writes the .joblib
python evaluate_pipeline.py    # scores the pipeline that actually runs
```

The corpus itself (`training/data/*.jsonl`, ~11 MB) is **not** in this zip.
The generator is seeded and was verified byte-deterministic, so the first
command above reproduces the exact corpus these metrics were measured on —
shipping it would just be 11 MB of something regenerable in ten seconds.
`training/data/meta.json` (class balance, language mix, lexicon baseline) is
kept so the corpus can be sanity-checked without regenerating it.

### Optional upgrade: fine-tuned transformer

`training/train_transformer.py` fine-tunes MuRIL (or XLM-R) on the identical
corpus with the identical evaluation code, so results are directly
comparable. It is a separate script because HuggingFace downloads were
blocked in the environment this model was trained in — the shipping model is
the offline path.

MuRIL is the default because it is one of the few public models pretrained on
*transliterated* Indic text, which is what Sahayk's real input largely is. It
should beat the shipping model on the two things TF-IDF structurally cannot
do: negation scoping, and vocabulary absent from training.

```bash
pip install transformers torch     # not in requirements.txt — ~2 GB
cd training && python train_transformer.py --model google/muril-base-cased --epochs 4
# then copy the output dir to backend/models/transformer/
```

`ml_model.py` picks it up automatically, with no code change, and falls back
to the TF-IDF model if torch is missing or the directory fails to load. It
will not make the labels more correct — a transformer trained on this rubric
reproduces the rubric more faithfully, wrong parts included.

### Files added

```
backend/ml_model.py              inference wrapper, denial gating, fails soft to rules
backend/models/sahayk_triage.joblib   the trained model (4 MB, ~2 ms/message)
training/expressions.py          hand-written idiomatic disclosures (the hard cases)
training/templates.py            compositional slot templates (the bulk of the corpus)
training/generate_dataset.py     corpus builder + expression-level split
training/train.py                trains and evaluates the shipping model
training/evaluate_pipeline.py    scores the served path, vs. both baselines
training/train_transformer.py    optional MuRIL/XLM-R fine-tune (needs GPU + network)
training/metrics.json            classifier-level results
training/pipeline_metrics.json   served-pipeline results
```

`backend/scoring.py` gained `_detect_categories()` (the union) and a
`detection` block in its output so the dashboard can show whether a rating
came from a matched phrase or a model inference — a Critical rating reached
purely by model inference is exactly the kind of decision that has to stay
contestable by a human.

### Still deliberately not done

The model does not read audio, does not use conversation history (each
message is scored independently), and has no calibration against real
consented transcripts — which remains the single thing that would most
improve it, and the one thing no amount of synthetic data substitutes for.

## Round 19 — trained model merged and retrained for atrocity cases, 14566 call webhook

Three things in this round: the Round 17 model is merged into the Round 18
build, the model is retrained to know the Round 18 atrocity categories, and
finished 14566 calls now arrive on the dashboard automatically.

### 1. Model merged into the Round 18 build

`ml_model.py`, `models/`, `training/` and the Round 17 changes to
`scoring.py`, `database.py`, `authority.html` and `requirements.txt` are
carried over unchanged in intent. The Round 17 `chat.html` was older than the
Round 18 one (no logo, no Bengali), so it was **not** taken.

### 2. Retrained on 24 categories

- `training/atrocity_data.py` (new) adds synthetic training data for the 7
  atrocity categories plus `atrocity_context`, in English, Hindi, Gujarati and
  Hinglish, and ~130 extra harmless messages (administrative questions and
  "hard negatives" that share words with harm categories — police, pending,
  temple, tomorrow, tired, a kitchen fire).
- `generate_dataset.py`: mirrors the new floors; Gujarati lexicon phrases now
  go into the Gujarati training pool (they were being filed under English);
  harmless share raised from 22% to 30%.
- Corpus: 30,000 training / 6,000 test messages, expression-level split as in
  Round 17. Model file 9 MB, about 11 ms per message on CPU.

**New guard — `MODEL_FLOOR_MIN_P = 0.80` in `scoring.py`.** When only the
model (not the phrase list) finds a category and it is less than 80% sure,
the category still counts at hedged confidence but cannot trigger a safety
floor. Without this, short everyday messages were being floored to High or
Critical ("I scraped my knee" → rape category at p=0.57, "we went to the
temple" → caste abuse at p=0.51). The dashboard lists these as
`model_only_unconfirmed`. It is one constant: set it to 0 to switch the guard off.

### Results (held-out synthetic test, 6,000 messages)

| system | band acc | escalation recall | critical recall | false escalation | negation FP |
|---|---|---|---|---|---|
| rules only | 0.31 | 0.19 | 0.21 | 0.04 | 0.13 |
| model, no denial gate | 0.80 | 0.98 | 0.98 | 0.34 | 0.87 |
| union, no guard | 0.84 | 0.98 | 0.97 | 0.20 | 0.33 |
| **served pipeline (with guard)** | 0.83 | 0.96 | 0.95 | 0.14 | 0.33 |

Per-category micro-F1 0.90 (macro 0.90) across 24 categories. The guard
costs ~1.5 points of escalation recall and cuts false escalations by about a
third; cases it holds back still reach the queue at Moderate, flagged.

**Hand-written checks** (`tests/atrocity_battery.py`):

| set | FINAL (rules) | Round 18 (rules) | Round 19 |
|---|---|---|---|
| FRESH2 — written after training, never tuned on: 17 real-style cases | 0/17 | 4/17 | **11/17** |
| FRESH2 — 7 harmless messages kept Low | 6/7 | 6/7 | 4/7 |
| BATTERY / HELDOUT / FRESH (seen while building) | — | 23/24 · 15/16 · 12/20 | 23/24 · 15/16 · 15/20 |

All 62 messages in `test-matrix.md`: none scores lower than before; 16 score higher.

**Honest limits.** Labels are rubric-derived, not clinician-annotated; every
number above is on synthetic or hand-written data, not real calls. A bag of
words still misreads context: "the police registered my FIR and arrested
them" scores as police inaction (p=1.00). Misses on FRESH2 are mostly
wording it never saw ("set my uncle on fire", "bulldozed", "lick the floor").
The transformer path (MuRIL, `train_transformer.py`) is the fix and needs a GPU.

### 3. 14566 call webhook (`calls.py`, new)

`POST /webhook/calls?provider=<name>` — the voice/IVR platform calls this
when a call ends.

- **Adapters** for Vapi, Retell, Bland, ElevenLabs, Twilio (form fields),
  OmniDimension (best-effort) and a documented generic JSON shape; provider is
  auto-detected if `?provider=` is missing, with generic field sniffing as a
  fallback.
- **Only the caller's words are scored** — agent/bot turns are dropped, or
  every call would pick up the agent's own "are you safe?".
- **Raw payload stored first** (`call_events`, append-only) so calls can be
  re-scored later. `calls` links each `call_id` to its case; a retried webhook
  **updates** the case instead of duplicating it.
- Consent explicitly refused → logged, not analysed, no case. Empty caller
  transcript → logged, no case.
- Caller number masked to the last 4 digits. The receipt returned to the
  platform never includes the transcript or the score.
- **Auth:** `SAHAYK_WEBHOOK_KEY` env var (demo value in `config.py`), sent as
  `X-API-Key`, `Authorization: Bearer`, or `?api_key=`.
- **Dashboard:** call cases show "📞 14566 call · provider · duration · masked
  number". Counsellors and District Admins get a **Simulate a 14566 call**
  button (`POST /demo/simulate_call`) that runs a sample call through the
  same ingest path; other roles get 403. `tools/demo_calls.py` posts all five
  sample formats to any running server.
- **Security fix while here:** case text now comes from an outside platform, so
  the dashboard HTML-escapes transcript excerpts and denied-clause text before
  inserting them (a `<img onerror>` payload was tested and does not run).

**Verified live:** all five provider formats create cases; no key → 401;
Bearer and query-key both accepted; resent call_id → "updated", same case;
consent refused / empty transcript → no case; Twilio form body works;
auto-detect works; bad JSON → 400; simulate as admin works, as police → 403;
chat flow, role scoping (Legal / Police / Rehab) and 403s unchanged. Server
memory with the model loaded ≈ 360 MB (fits Render's free 512 MB).

## Round 20 — call recordings: the caller's voice is analysed, not just the words

SIH26093 asks for analysis of "voice interactions, speech patterns, pauses,
pitch variation" for people reaching 14566 — a phone line. Until now only
chat voice notes got that; a 14566 call was scored on its transcript alone.

**What happens now** (`calls.py`, `analyse_call_audio`): if the finished-call
webhook includes a recording, Sahayak downloads it, measures the caller's
pitch variation, pauses and speech rate with the same `audio_features.py`
code the chat uses, and scores the call on words **and** voice (speech stress
= 20% of the SVI, as in chat).

- **Recording sources:** a link (`recording_url`, Vapi `recordingUrl` /
  `stereoRecordingUrl`, Retell `recording_url`, Bland `recording_url`,
  Twilio `RecordingUrl` — `.wav` is appended for Twilio), or audio sent inline
  as base64 (`recording_base64`, ElevenLabs `full_audio`). If the audio
  arrives in a second webhook after the transcript (ElevenLabs
  `post_call_audio`), the existing case is re-scored with the voice added.
- **Caller's channel only, when possible:** a stereo recording usually has
  caller and agent on separate channels. `SAHAYK_CALL_AUDIO_CALLER_CHANNEL`
  = `0` or `1` picks the caller; empty = mixed. Tested: calm caller on one
  channel, distressed on the other → 9.0 vs 57.2 speech stress; mixed → 25.5.
  Check your platform's layout with one test call.
- **Safe download (SSRF protection):** http/https only; the host must resolve
  only to public internet addresses (localhost, 10.x, 192.168.x and the cloud
  metadata address 169.254.169.254 are refused); every redirect is re-checked;
  size capped (`SAHAYK_CALL_AUDIO_MAX_MB`, default 25) and time capped
  (`SAHAYK_CALL_AUDIO_TIMEOUT_S`, default 15 s). Only the first
  `SAHAYK_CALL_AUDIO_MAX_SECONDS` (default 120) are analysed, to keep it fast.
- **Protected links:** set `SAHAYK_RECORDING_AUTH_USER` / `_PASS` (e.g. Twilio
  Account SID + Auth Token) and they are sent as HTTP Basic auth.
- **Never blocks scoring:** any failure (no recording, bad link, unreadable
  audio, download refused) is recorded as the reason and the call is scored
  on words alone, exactly as before. `SAHAYK_CALL_AUDIO=off` turns it off.
- The webhook now runs ingest in a worker thread so a download doesn't
  stall other requests.
- **Dashboard:** under the 📞 badge, "🎙 voice analysed · stress 57/100
  (pitch …, pauses …)" or the reason it wasn't.
- **Demo:** `backend/demo_audio/` holds two **synthetic** clips (generated
  tones shaped like calm vs strained speech — not human voices) used by
  "Simulate a 14566 call" via `demo://…` links.

**Verified live:** calm vs distressed demo recording on the same words →
SVI 53.1 vs 62.7 (speech stress 9.0 vs 57.2); real human speech downloaded
from the internet analysed (speech stress 32.7 and 19.2 on two clips);
base64 audio; ElevenLabs two-step re-score; unreadable audio → words only;
blocked: 127.0.0.1, localhost, 10.0.0.5, 169.254.169.254, file://, ftp://,
`demo://../../etc/passwd`, an unresolvable host, and a redirect to the
metadata address. Chat voice notes, chat, roles and all earlier webhook
checks unchanged; test battery unchanged.

**Honest limits:** these are three acoustic measurements, not a trained
speech-emotion model. On a mono recording the agent's voice is mixed in
(a steady bot voice dilutes the caller's signal rather than inflating it).
When no recording is sent, the voice weight is left out of the blend
rather than counted as "calm" — so a call without audio can score slightly
higher than the same words with a calm voice; that is intended.

## Round 21 — officers can upload a call recording from the dashboard

For helpline staff whose phone system has no webhook, or for recordings
made earlier: **Counsellors and District Admins** get an **🎧 Upload a call
recording** button. Other roles get 403.

**What happens** (`calls.ingest_upload`, `POST /cases/upload_call`):

1. The officer picks the file (WAV best; MP3/M4A/OGG/WebM work when ffmpeg is
   installed on the server; max 25 MB). They can also set the language and
   which stereo side the caller is on, and add the caller's number (stored
   masked) and a note.
2. They must tick "the caller was told the call is recorded and agreed to
   it being analysed". Without it the upload is refused.
3. The **voice** is analysed exactly like chat voice notes and webhook calls.
4. The **words** come from the transcript box, or from automatic
   speech-to-text (`stt.py`) when the server has it. If neither is
   available, the case is filed **voice-only**: SVI = max(40, voice stress),
   so never below Moderate. It is flagged for review with "Listen to the
   recording and add a transcript".
5. The result shows immediately (case number, risk band, SVI, voice
   measurements, detected categories, suggested actions). The case joins
   the queue with a 🎧 badge naming the uploader.
6. **Add or correct the transcript later.** In "why this score", counsellors
   and admins get a box that re-scores the case (`POST /cases/{id}/transcript`),
   keeping the voice measurement. This works for uploaded and webhook call
   cases, not chat cases.
7. Every upload and transcript edit goes into the append-only audit log
   (`review_log`) under the officer's name, without marking the case reviewed.

**Speech-to-text (`stt.py`)** — optional, so the free deploy still fits:
- `pip install -r requirements-stt.txt` adds faster-whisper (open-source
  Whisper, CPU, multilingual). The model downloads from Hugging Face on
  first use. Pick the size with `SAHAYK_STT_MODEL`: tiny (default), base or
  small; bigger gives better Hindi and Gujarati and runs slower.
- It needs roughly 300–600 MB extra RAM, so it does **not** fit Render's free
  512 MB plan. Use a laptop or a paid instance for auto-transcription;
  without it, officers type or paste the words.
- `SAHAYK_STT=off` disables it. A failed model download is retried after
  10 minutes. The dashboard says whether auto-transcription is on.
- Bhashini (MeitY's Indian-language speech API) is the intended production
  engine and plugs into the same `transcribe()` function once credentials
  exist.

**Verified live:**
- Same typed words: strained voice → SVI 69.5, calm voice → 59.8.
- No transcript and no speech-to-text → voice-only, High 57.2, flagged.
  Then a counsellor added the words → Critical 90 with the voice kept, and
  the audit log shows both actions.
- An MP3 upload works (ffmpeg present).
- The transcription path was checked end to end with a stand-in model
  (Hindi text → Critical 90, "auto-transcribed (whisper tiny)"). **Real
  Whisper weights could not be downloaded in the build sandbox** (Hugging
  Face blocked), so run one real upload on a machine with internet before
  relying on it.
- Refusals: police uploading → 403; no consent → 400; not-audio file with
  no words → 400 with a clear message; not-audio with words → words-only
  score; >25 MB → 400; bad channel → 400; no login → 401; police adding a
  transcript → 403; transcript on a chat case → 400.
- The dashboard upload flow was clicked through in a real browser with no
  page errors.
