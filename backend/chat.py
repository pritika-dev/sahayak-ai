"""
Conversational intake. The bot asks an open question, validates what it
hears, then adapts its follow-ups to whatever situation the running SVI
score has picked up — each category has its own progressive question
set (2 questions deep) that only fires when that category is actually
detected, covering all 14 lexicon categories now, not just the original
7. When nothing more specific is detected, a short wrap-up sequence
checks for anything else to share, asks what kind of help would
actually be useful (and follows up if the answer is just "yes"/"no"
instead of specifics), and — if nothing concerning was ever flagged —
gently offers a counselling referral rather than just closing.

Suicidal-ideation handling is separate from all of this: it always
takes priority, always surfaces crisis resources, and varies its
wording after the first time so it doesn't just repeat the same block
verbatim every turn.

Privacy: /chat/message and /chat/voice_message only ever return
{reply, done}. The SVI score, category, and breakdown are written to
the case record for the authority dashboard and never sent to the
person chatting.

Session state is persisted, not just kept in the in-memory _sessions
dict below — every turn writes the full session (stage, cumulative
messages, the visual bot/user turn log, counsellor preference, etc.)
to the `chat_sessions` table (database.py), and a lookup miss in
_sessions (e.g. right after a server restart) falls back to loading it
from there instead of declaring the session expired. See
_persist_session / _load_session and main.py's /chat/resume, which
chat.html calls on page load so an accidental reload mid-conversation
resumes exactly where it left off instead of restarting from consent.
"""
import json
import re
import uuid
from scoring import score_text
import database as db

_sessions = {}


def _persist_session(session_id: str, session: dict):
    db.save_chat_session(session_id, session["case_id"], json.dumps(session))


def _load_session(session_id: str):
    """Rehydrates a session from the DB on an in-memory cache miss —
    the normal case is a server restart, where _sessions starts back
    at {} but every session that was ever active is still on disk."""
    row = db.get_chat_session(session_id)
    if not row:
        return None
    try:
        session = json.loads(row["state_json"])
    except (ValueError, TypeError):
        return None
    _sessions[session_id] = session
    return session


def get_session_log(session_id: str):
    """Used by /chat/resume: the ordered {who, text} turn log needed to
    repaint the chat window after a reload, plus whether the
    conversation had already ended. Returns None if the session truly
    doesn't exist (never started, or long enough ago it's not worth
    resuming — this prototype keeps everything indefinitely, but a
    real deployment might prune old rows)."""
    session = _sessions.get(session_id) or _load_session(session_id)
    if not session:
        return None
    return {"log": session.get("log", []), "done": session.get("done", False)}

OPENING = (
    "Hi, I'm here to listen and help connect you with the right support. "
    "What's been going on? Share as much or as little as you're comfortable with."
)

CRISIS_MESSAGE_FIRST = (
    "I want to make sure you're safe right now. If you're thinking about harming "
    "yourself, please reach out immediately: KIRAN Mental Health Helpline (India) — "
    "1800-599-0019, 24x7, free — or Tele-MANAS — 14416 / 1-800-891-4416, 24x7, "
    "multilingual. Are you safe at this moment, and is there someone with you?"
)
CRISIS_MESSAGE_REPEAT = (
    "I'm still here with you. The helplines are always available — KIRAN: "
    "1800-599-0019, Tele-MANAS: 14416, both 24x7. Do you want to tell me more "
    "about what's going on?"
)

DANGER_MESSAGE_FIRST = (
    "Your safety comes first, right now. If you're able to safely do so, please "
    "call 112 for immediate police help, or get to a safe place. I'm right here — "
    "can you tell me if you're able to get somewhere safe at this moment?"
)
DANGER_MESSAGE_REPEAT = (
    "I'm still here with you. 112 is available right now for immediate police "
    "help if you need it. Are you safe at this moment?"
)

HELP_TYPE_Q = (
    "Is there a specific kind of help that would be most useful right now — "
    "someone to talk to, legal help, medical support, or something else?"
)
HELP_TYPE_CLARIFY_Q = (
    "What would help most right now — someone to talk to, legal help, "
    "medical support, or something else?"
)
ANYTHING_ELSE_Q = "Is there anything else about the situation you'd like to share?"
COUNSELLING_OFFER_Q = (
    "It sounds like this has been weighing on you. Even without anything urgent "
    "going on, talking to a counsellor can genuinely help — would that be "
    "something you're open to?"
)
SUBSTANTIAL_UNCLASSIFIED_MSG = (
    "Thank you for telling me this. Your safety matters most right now — "
    "are you safe at this moment?"
)

# Asked once, right after consent, before the normal opening question —
# this is a stated PREFERENCE recorded on the case for whoever reviews
# it to honour when assigning a counsellor. Gender is always just a
# recorded preference (no gender-matching logic below), but an exact
# numeric age answer ("25") IS matched against currently-registered
# counsellor accounts — see _closest_counselor below and main.py's
# live re-match at dashboard-read time.
COUNSELOR_GENDER_Q = (
    "Before we begin — would you feel more comfortable speaking with a "
    "female or male counsellor, or do you have no preference either way?"
)
COUNSELOR_AGE_Q = (
    "Thank you. And do you have an age preference for your counsellor — "
    "a specific age, younger, older, or no preference either way?"
)

# Free-text answers are matched against these keyword sets rather than
# forced into a fixed set of buttons, since chat.html still accepts
# plain typed replies (voice included) — "no preference"/"either"/"any"
# etc. are treated the same way. Anything that matches nothing is kept
# as-is in the *_raw field so a reviewer can still read the actual
# answer rather than losing it to "unspecified".
GENDER_PREF_KEYWORDS = {
    "female": ["female", "woman", "women", "girl", " f "],
    "male": ["male", "man", "men", "boy", " m "],
    "no_preference": ["no preference", "either", "either is fine", "any",
                       "doesnt matter", "dont mind", "whoever", "no particular",
                       "not fussed", "not bothered"],
}
AGE_PREF_KEYWORDS = {
    "younger": ["younger", "young"],
    "older": ["older", "old", "elder", "senior"],
    "no_preference": ["no preference", "either", "any", "doesnt matter",
                       "dont mind", "whoever", "no particular", "not fussed",
                       "not bothered"],
}


def _match_pref(raw_text: str, keyword_map: dict) -> str:
    cleaned = " " + raw_text.strip().lower().replace("'", "") + " "
    for pref, keywords in keyword_map.items():
        for kw in keywords:
            if kw in cleaned:
                return pref
    return "unspecified"


# A specific number ("25", "around 25") takes priority over the
# younger/older/no-preference keywords below — someone who names an
# exact age wants that honoured as precisely as the system can, not
# collapsed into a vague bucket. Bounded to a plausible counsellor-age
# range (matches auth.py's MIN_AGE/MAX_AGE for registering an
# authority) so an unrelated number in the sentence — "I've called
# twice before" — doesn't get misread as an age.
_AGE_NUMBER_RE = re.compile(r"\d{1,3}")
_MIN_PLAUSIBLE_AGE, _MAX_PLAUSIBLE_AGE = 18, 100


def _extract_age_number(raw_text: str):
    for match in _AGE_NUMBER_RE.finditer(raw_text):
        n = int(match.group())
        if _MIN_PLAUSIBLE_AGE <= n <= _MAX_PLAUSIBLE_AGE:
            return n
    return None


# Short empathetic acknowledgment said once, the first time a category
# is detected, before its first follow-up question.
VALIDATIONS = {
    "physical_safety_abuse": "I'm really sorry that's happening to you.",
    "sexual_harassment_assault": "I'm sorry you went through that — you didn't deserve it.",
    "child_safety_concern": "Thank you for telling me — your child's safety matters.",
    "elder_abuse": "Thank you for telling me, that's a serious concern.",
    "intimidation_threat": "That sounds frightening. Thank you for telling me.",
    "severe_trauma_fear": "That sounds really hard to carry.",
    "cybercrime_online_abuse": "That sounds violating — I'm sorry that's happening to you.",
    "legal_dispute": "That sounds stressful to be dealing with.",
    "workplace_harassment": "I'm sorry you're dealing with that at work.",
    "financial_exploitation": "That's a really difficult position to be in.",
    "substance_related": "That sounds like a difficult situation at home.",
    "depression_hopelessness": "Thank you for sharing that with me.",
    "social_isolation": "That sounds really lonely to go through.",
    "rape_sexual_violence": "I'm so sorry this happened. It was not your fault, and you did the right thing by telling someone.",
    "family_murder": "I'm so deeply sorry for your loss.",
    "witness_intimidation": "No one has the right to pressure you over your case. Thank you for telling me.",
    "displacement_arson": "I'm sorry — losing your home like that is devastating.",
    "social_boycott": "Being cut off like that is a serious wrong, and you don't deserve it.",
    "caste_atrocity": "I'm sorry you were treated that way. What happened to you is not okay, and it is against the law.",
    "police_noncooperation": "You have a right to be heard by the police. Thank you for telling me.",
}

# Priority order: highest-need situations get asked about first when
# more than one is newly detected in the same turn. Each list digs one
# level deeper each time that situation keeps showing up.
CATEGORY_FOLLOWUPS = [
    ("rape_sexual_violence", [
        "Are you, or the person this happened to, somewhere safe right now — and is medical help needed?",
        "Has this been reported to the police yet?",
    ]),
    ("family_murder", [
        "Are you and the rest of your family safe right now?",
        "Has a police case been registered for this?",
    ]),
    ("witness_intimidation", [
        "Can the people pressuring you reach you or your family right now?",
        "Has this pressure been reported to the police or the court?",
    ]),
    ("physical_safety_abuse", [
        "Are you physically safe right now, at this moment?",
        "Has this happened before, or is this the first time?",
    ]),
    ("sexual_harassment_assault", [
        "Do you feel okay sharing whether this was someone you know?",
        "Has this happened more than once?",
    ]),
    ("child_safety_concern", [
        "Is your child safe at this moment?",
        "Has anyone else been told about this yet — family, school, or the police?",
    ]),
    ("elder_abuse", [
        "Is your relative safe right now?",
        "Do they currently live with the person you're concerned about?",
    ]),
    ("intimidation_threat", [
        "Is this person able to reach you right now? Are you somewhere safe today?",
        "Have they threatened anyone else close to you, like family?",
    ]),
    ("displacement_arson", [
        "Do you and your family have a safe place to stay tonight?",
        "Have you been able to report the damage to the police or local officials?",
    ]),
    ("social_boycott", [
        "Are you able to get water, food and daily essentials right now?",
        "How long has this been going on?",
    ]),
    ("caste_atrocity", [
        "Did this happen in front of other people?",
        "Has this happened to you or your family before?",
    ]),
    ("severe_trauma_fear", [
        "Has this been happening recently, or over a longer time?",
        "Is there a specific place or person that makes you feel unsafe?",
    ]),
    ("cybercrime_online_abuse", [
        "Is the content still online, or has it been taken down?",
        "Do you know who's responsible for this?",
    ]),
    ("police_noncooperation", [
        "Do you know which police station, or the name of the officer involved?",
        "Would help from a legal aid lawyer be useful right now?",
    ]),
    ("legal_dispute", [
        "Have you already spoken to a lawyer or filed anything officially about this?",
        "Is anyone threatening or pressuring you in relation to this dispute?",
    ]),
    ("workplace_harassment", [
        "Has this been formally reported to HR or someone above your manager?",
        "Would connecting with a legal aid service about this be helpful?",
    ]),
    ("financial_exploitation", [
        "Do you currently have access to your own money or resources?",
        "Is there someone controlling decisions about your finances?",
    ]),
    ("substance_related", [
        "Are you safe when they've been drinking or using?",
        "Has this been affecting anyone else in the household?",
    ]),
    ("depression_hopelessness", [
        "How long have you been feeling this way?",
        "Has it been affecting your day-to-day — sleep, eating, work?",
    ]),
    ("social_isolation", [
        "Is there anyone — family, a friend, a neighbour — you could reach out to?",
        "Do you live with anyone, or are you on your own right now?",
    ]),
]

AFFIRMATIVE = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "yess", "ya", "yah"}
NEGATIVE = {"no", "nope", "nah", "not really", "no thanks", "nothing"}
EXIT_PHRASES = {
    "bye", "goodbye", "bye bye", "stop", "gtg", "got to go", "have to go",
    "not now", "leave me alone", "i dont want to talk anymore", "exit",
    "im done talking", "i need to go", "talk later",
}

MAX_TURNS = 9


def _clean(text: str) -> str:
    return text.strip().lower().strip(".,!?\\ ")


def _is_affirmative(text: str) -> bool:
    return _clean(text) in AFFIRMATIVE


def _is_negative(text: str) -> bool:
    return _clean(text) in NEGATIVE


def _is_exit(text: str) -> bool:
    return _clean(text) in EXIT_PHRASES


def _word_count(text: str) -> int:
    return len(text.split())


def start_session(language: str = "en") -> dict:
    session_id = str(uuid.uuid4())
    dummy_result = {"svi": 0.0, "risk_category": "Low", "breakdown": {}, "keyword_hits": {}, "recommended_actions": []}
    case_id = db.insert_case(True, language, "", dummy_result)
    session = {
        "case_id": case_id, "messages": [], "asked": {}, "turns": 0,
        "crisis_shown": 0, "danger_shown": 0, "stage": "ask_counselor_gender_pref",
        "speech_stress_values": [], "acknowledged_generic": False,
        "counselor_pref": {"gender": None, "gender_raw": None, "age": None, "age_raw": None, "age_value": None},
        "log": [{"who": "bot", "text": COUNSELOR_GENDER_Q}], "done": False,
    }
    _sessions[session_id] = session
    _persist_session(session_id, session)
    # Consent has just been given (this is what triggers /chat/start) —
    # the first thing asked is who they'd feel comfortable talking to,
    # before anything about the situation itself.
    return {"session_id": session_id, "reply": COUNSELOR_GENDER_Q, "done": False}


def send_message(session_id: str, text: str, speech_stress: float = None, audio_ok: bool = True) -> dict:
    session = _sessions.get(session_id) or _load_session(session_id)
    if not session:
        return {"reply": "This session has expired — please refresh to start again.", "done": True}
    _sessions[session_id] = session

    session.setdefault("log", []).append({"who": "user", "text": text})
    reply_obj = _send_message_core(session, text, speech_stress, audio_ok)
    session["log"].append({"who": "bot", "text": reply_obj["reply"]})
    session["done"] = reply_obj.get("done", False)
    _persist_session(session_id, session)
    return reply_obj


def _send_message_core(session: dict, text: str, speech_stress: float = None, audio_ok: bool = True) -> dict:
    # The two preference questions are setup, not part of the disclosure
    # itself — they shouldn't eat into the same MAX_TURNS budget as the
    # actual conversation, or count toward the cumulative transcript
    # that gets scored (an answer like "male" or "younger" has nothing
    # to do with SVI/keyword scoring and would just be dead weight in
    # every transcript excerpt an officer reads).
    stage_at_entry = session["stage"]
    is_pref_turn = stage_at_entry in ("ask_counselor_gender_pref", "ask_counselor_age_pref")

    def _persist(transcript: str, result: dict):
        # Called right before every return, so it always reflects
        # whatever _decide_reply just set this turn (e.g. a preference
        # answered a moment ago) rather than a stale one-turn-old value.
        result = dict(result)
        result["counselor_pref"] = dict(session["counselor_pref"])
        db.update_case(session["case_id"], transcript, result)

    # An exit phrase is honoured immediately no matter what's being
    # asked — including mid-way through the two preference questions.
    # Someone who wants to leave right after giving consent shouldn't
    # be forced through two more questions first.
    if _is_exit(text):
        transcript = " ".join(session["messages"])
        result = score_text(transcript)
        _persist(transcript, result)
        if session["crisis_shown"] > 0:
            reply = (
                "Okay. Please remember KIRAN — 1800-599-0019 — and Tele-MANAS — "
                "14416 — are there for you anytime, day or night. I hope you'll "
                "reach out. Take care of yourself."
            )
        else:
            reply = "Okay, take care of yourself. I'm here anytime you'd like to talk."
        return {"reply": reply, "done": True}

    if is_pref_turn:
        reply_obj = _decide_reply(session, text, {}, " ".join(session["messages"]))
        _persist(" ".join(session["messages"]), score_text(" ".join(session["messages"])))
        return reply_obj

    session["messages"].append(text)
    session["turns"] += 1
    if speech_stress is not None and audio_ok:
        # audio_ok=False means the recording failed to decode/analyze —
        # in that case speech_stress is a meaningless 0.0, not a real
        # "calm" reading, so it must NOT be recorded as if it were real
        # data (see scoring.py's has_speech_data — this is what feeds it).
        session["speech_stress_values"].append(speech_stress)

    transcript = " ".join(session["messages"])
    avg_speech_stress = (
        sum(session["speech_stress_values"]) / len(session["speech_stress_values"])
        if session["speech_stress_values"] else 0.0
    )
    has_speech_data = bool(session["speech_stress_values"])
    result = score_text(transcript, speech_stress=avg_speech_stress, has_speech_data=has_speech_data)
    hits = result["keyword_hits"]
    _persist(transcript, result)

    # Suicidal ideation, and a person causing harm being physically
    # present right now, both get the full urgent message ONLY the
    # first time — repeating that identical block forever (which is
    # what happened before this fix, since these hits persist in the
    # cumulative transcript for the rest of the session) trapped the
    # conversation in a loop with no way to ever move into an actual
    # follow-up question again. Now: first hit = full urgent message.
    # Every hit after that falls through into the NORMAL flow below
    # (so the conversation keeps progressing/counselling) with a short
    # reminder line kept in front of whatever it says, so the resources
    # stay visible without dominating every single turn.
    if "suicidal_ideation" in hits and session["crisis_shown"] == 0:
        session["crisis_shown"] = 1
        return {"reply": CRISIS_MESSAGE_FIRST, "done": False}

    if "immediate_danger" in hits and session["danger_shown"] == 0:
        session["danger_shown"] = 1
        return {"reply": DANGER_MESSAGE_FIRST, "done": False}

    still_critical = "suicidal_ideation" in hits or "immediate_danger" in hits

    if session["turns"] >= MAX_TURNS and not still_critical:
        return {"reply": "Thank you for sharing this with me. I've noted everything, and a support coordinator will review it and reach out about next steps.", "done": True}

    reply_obj = _decide_reply(session, text, hits, transcript)

    if session["crisis_shown"] > 0 and "suicidal_ideation" in hits:
        reply_obj["reply"] = f"(KIRAN: 1800-599-0019, Tele-MANAS: 14416 — always available.) {reply_obj['reply']}"
    elif session["danger_shown"] > 0 and "immediate_danger" in hits:
        reply_obj["reply"] = f"(112 is available right now if you need it.) {reply_obj['reply']}"

    return reply_obj


def _decide_reply(session: dict, text: str, hits: dict, transcript: str) -> dict:
    stage = session["stage"]

    if stage == "ask_counselor_gender_pref":
        session["counselor_pref"]["gender"] = _match_pref(text, GENDER_PREF_KEYWORDS)
        session["counselor_pref"]["gender_raw"] = text.strip()
        session["stage"] = "ask_counselor_age_pref"
        return {"reply": COUNSELOR_AGE_Q, "done": False}

    if stage == "ask_counselor_age_pref":
        age_number = _extract_age_number(text)
        if age_number is not None:
            session["counselor_pref"]["age"] = "exact"
            session["counselor_pref"]["age_value"] = age_number
        else:
            session["counselor_pref"]["age"] = _match_pref(text, AGE_PREF_KEYWORDS)
            session["counselor_pref"]["age_value"] = None
        session["counselor_pref"]["age_raw"] = text.strip()
        session["stage"] = None
        return {"reply": OPENING, "done": False}

    if stage == "ask_anything_else":
        if _is_affirmative(text):
            session["stage"] = None
            return {"reply": "Go ahead, I'm listening.", "done": False}
        session["stage"] = "ask_help_type" if hits else "ask_counselling"
        return {"reply": HELP_TYPE_Q if hits else COUNSELLING_OFFER_Q, "done": False}

    if stage == "ask_safety_after_unclassified":
        # Whatever they answered, move forward into the normal wrap-up —
        # the point of this stage was the safety check itself, not
        # branching on the answer.
        session["stage"] = "ask_help_type"
        return {"reply": HELP_TYPE_Q, "done": False}

    if stage == "ask_help_type":
        if _is_affirmative(text):
            session["stage"] = "closing"
            return {"reply": HELP_TYPE_CLARIFY_Q, "done": False}
        session["stage"] = "closing"
        return {"reply": "Thank you — I've noted that, and a support coordinator will review it and reach out about next steps.", "done": True}

    if stage == "closing":
        return {"reply": "Thank you for sharing this with me. I've noted everything, and a support coordinator will review it and reach out about next steps.", "done": True}

    if stage == "ask_counselling":
        msg = ("Noted — I've flagged that you're open to a counselling referral, "
               "and someone will reach out.") if _is_affirmative(text) else (
               "Understood, I've recorded what you've shared. Support is "
               "always available if you change your mind.")
        return {"reply": msg, "done": True}

    # Dig deeper into whichever detected situation still has questions
    # left, in priority order — with a one-time empathetic lead-in.
    for cat, questions in CATEGORY_FOLLOWUPS:
        if cat in hits:
            depth = session["asked"].get(cat, 0)
            if depth < len(questions):
                session["asked"][cat] = depth + 1
                lead_in = VALIDATIONS.get(cat, "") if depth == 0 else ""
                reply = f"{lead_in} {questions[depth]}".strip()
                return {"reply": reply, "done": False}

    # No more category-specific questions left — move into the wrap-up.
    # If nothing was ever detected at all, don't just move on with a flat
    # acknowledgment — a LONG, detailed message that still matched
    # nothing is more likely a detection gap (wording or a language the
    # lexicon doesn't cover yet) than something genuinely minor, so it
    # gets real weight and a direct safety check instead of "thanks,
    # anything else?". Short/vague messages still get the light-touch
    # generic acknowledgment, since treating "work's been stressful" as
    # a serious unclassified disclosure would be its own kind of wrong.
    if not hits and not session["acknowledged_generic"]:
        session["acknowledged_generic"] = True
        if _word_count(transcript) >= 20:
            session["stage"] = "ask_safety_after_unclassified"
            return {"reply": SUBSTANTIAL_UNCLASSIFIED_MSG, "done": False}
        session["stage"] = "ask_anything_else"
        return {"reply": f"Thanks for telling me that. {ANYTHING_ELSE_Q}", "done": False}

    session["stage"] = "ask_anything_else"
    return {"reply": ANYTHING_ELSE_Q, "done": False}
