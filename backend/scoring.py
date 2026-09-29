from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicon import CATEGORIES, keyword_severity_score
import ml_model

_analyzer = SentimentIntensityAnalyzer()

# Weights are intentionally explicit and editable — the whole point of a
# rubric-based score over a black-box model is that a reviewer can see
# and adjust exactly why a case scored the way it did.
WEIGHTS = {
    "text_sentiment": 0.20,
    "keyword_severity": 0.45,
    "speech_stress": 0.20,
    "isolation_indicator": 0.15,
}
RISK_BANDS = [
    (0, 25, "Low"),
    (25, 50, "Moderate"),
    (50, 75, "High"),
    (75, 101, "Critical"),
]

SEVERITY_ACTIONS = {
    "Low": ["Self-help resources", "Optional counselling follow-up"],
    "Moderate": ["Counselling referral", "Periodic check-in"],
    "High": ["Priority counselling", "Flag for human review within 24h"],
    "Critical": [
        "Immediate human review", "Emergency support contact shown to user",
        "Consider police intervention / witness protection — human decision only",
    ],
}

# Service routing is deliberately SEPARATE from the severity band. A
# case can read as psychologically "calm" (low SVI) but still clearly
# need a specific service — a property/legal dispute is the example
# that surfaced this: it scored Low and got only "self-help resources"
# even though what the person actually needs is a legal aid referral.
# These tags always get added when their category is hit, regardless
# of what band the overall SVI lands in.
SERVICE_TAGS = {
    "legal_dispute": ["Legal aid referral"],
    "workplace_harassment": ["Legal aid referral"],
    "financial_exploitation": ["Legal aid referral"],
    "cybercrime_online_abuse": ["Legal aid referral / cybercrime cell"],
    "physical_safety_abuse": ["Police intervention consideration — human decision only", "Medical assistance referral"],
    "sexual_harassment_assault": ["Legal aid referral", "Police intervention consideration — human decision only"],
    "child_safety_concern": ["Immediate human review", "Child protection referral"],
    "elder_abuse": ["Legal aid referral", "Police intervention consideration — human decision only"],
    "substance_related": ["Medical assistance referral"],
    "intimidation_threat": ["Police intervention consideration — human decision only"],
    "immediate_danger": ["Immediate human review", "Police intervention consideration — human decision only", "Emergency support contact shown to user"],
    # Atrocity-specific routing (SIH26093). "Relief & rehabilitation
    # referral" = relief/compensation and rehabilitation under the SC/ST
    # (PoA) Act rules; "Witness protection referral" is its own tag so it
    # fires whenever someone is pressured over their case, not only when
    # the overall band happens to reach Critical.
    "caste_atrocity": ["Legal aid referral", "Police intervention consideration — human decision only", "Relief & rehabilitation referral"],
    "rape_sexual_violence": ["Immediate human review", "Medical assistance referral", "Police intervention consideration — human decision only", "Legal aid referral", "Relief & rehabilitation referral"],
    "family_murder": ["Immediate human review", "Police intervention consideration — human decision only", "Legal aid referral", "Relief & rehabilitation referral"],
    "witness_intimidation": ["Witness protection referral — human decision only", "Police intervention consideration — human decision only", "Legal aid referral"],
    "social_boycott": ["Legal aid referral", "Police intervention consideration — human decision only", "Relief & rehabilitation referral"],
    "displacement_arson": ["Relief & rehabilitation referral", "Police intervention consideration — human decision only", "Legal aid referral"],
    "police_noncooperation": ["Legal aid referral", "Escalate FIR / investigation delay to District Admin"],
}

# Some categories are severe enough on their own that they should never
# land in a band that undersells them, regardless of how flat or calm
# the surrounding sentence reads — "he is hitting me" shouldn't score
# lower than a florid but vague complaint just because the weighted
# blend below is tuned for nuance across ambiguous cases. Each forces a
# minimum SVI the moment that category is hit (at ANY confidence level,
# including a hedged "he might hurt me" — for these specific categories
# it is safer to over-react to an uncertain mention than risk under-
# reacting to a real one), on top of (never instead of) the normal
# weighted score.
CATEGORY_FLOORS = {
    "suicidal_ideation": 95,          # Critical — hard escalation override
    "immediate_danger": 95,           # Critical — hard escalation override
    "child_safety_concern": 90,       # Critical — extreme vulnerability
    "sexual_harassment_assault": 60,  # High
    "physical_safety_abuse": 55,      # High
    "elder_abuse": 55,                # High
    # Atrocity-specific floors (SIH26093)
    "rape_sexual_violence": 90,       # Critical
    "family_murder": 85,              # Critical
    "witness_intimidation": 60,       # High
    "caste_atrocity": 55,             # High
    "social_boycott": 55,             # High
    "displacement_arson": 55,         # High
    "police_noncooperation": 50,      # High
}


def _sentiment_stress_score(text: str) -> float:
    """VADER compound score in [-1, 1] -> stress score in [0, 100].
    More negative sentiment = higher stress score."""
    compound = _analyzer.polarity_scores(text)["compound"]
    return round(max(0.0, -compound) * 100, 1)


def _is_mostly_non_latin(text: str) -> bool:
    """VADER's lexicon is English-only — it silently returns ~0 (neutral)
    for a script it can't read at all, like Devanagari. Treating that as
    a genuine "calm" reading is wrong: it's not evidence of low severity,
    it's an absent signal, and blending it in anyway wastes 20% of the
    score's weight on nothing for every non-English message (this is
    exactly what capped a Critical-reading Hindi disclosure at
    Moderate before this fix). If the message is mostly non-Latin
    script, text_sentiment is excluded from the blend entirely below,
    rather than silently contributing a false "not negative" reading."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if c.isascii())
    return (latin / len(letters)) < 0.3


MODEL_FLOOR_MIN_P = 0.80


def _detect_categories(text: str) -> dict:
    """Runs BOTH detectors and takes the union of what they find.

    The two fail in opposite directions, which is the reason to keep
    both rather than replace one with the other:

      - The phrase list is precise and auditable but brittle. On the
        held-out evaluation set it found nothing at all on ~90% of
        messages that genuinely disclosed something, because the
        wording was not on the list. That is the gap lexicon.py's own
        docstring predicts.
      - The model generalises across wording, script and typos, but it
        is a linear bag-of-words classifier: it cannot scope negation
        on its own (ml_model.py gates denials structurally instead),
        and it cannot recognise vocabulary it never saw in training.

      Union, not average: a category counts as present if EITHER finds
      it, at whichever confidence is higher. Averaging would let a
      silent detector dilute a positive finding from the other one,
      which for a triage tool means a real disclosure losing to a
      shrug. Measured on the held-out set, the model caught ~3,900
      disclosures the rules missed entirely, while the rules caught
      none the model missed — so the union is meaningfully better than
      either alone, and strictly no worse than the rules were.

    Returns merged hits, merged confidences, the combined noisy-OR
    severity, and both sources kept separate so the dashboard can show
    a reviewer which detector fired.
    """
    kw = keyword_severity_score(text)
    ml = ml_model.predict(text)

    confidences, hits = dict(kw["confidences"]), {}
    for cat, matched in kw["hits"].items():
        hits[cat] = [f"rule: {m}" for m in matched]

    for cat, conf in ml.get("confidences", {}).items():
        if conf > confidences.get(cat, 0.0):
            confidences[cat] = conf
        hits.setdefault(cat, []).extend(ml["hits"][cat])

    # Guard for model-only detections (Round 19). When the phrase list
    # found nothing for a category and the model is only moderately sure
    # (p < MODEL_FLOOR_MIN_P), the category still counts — at hedged
    # confidence — but it cannot trigger a safety floor on its own. On
    # short everyday messages the character n-grams can latch onto one
    # fragment ("scraped" -> "raped", "temple" -> caste abuse, "pending"
    # -> police inaction) with p around 0.4-0.7, and letting those force
    # a Critical/High floor buries real cases under false alarms. Real
    # disclosures the model has learned score p >= 0.9, so they keep
    # their floor; anything the rules match keeps it unconditionally.
    probs = ml.get("probabilities", {})
    no_floor = []
    for cat in list(confidences):
        if cat not in kw["hits"] and probs.get(cat, 0.0) < MODEL_FLOOR_MIN_P:
            confidences[cat] = min(confidences[cat], 0.5)
            no_floor.append(cat)

    survival = 1.0
    for cat, conf in confidences.items():
        survival *= (1 - CATEGORIES[cat]["weight"] * conf)
    score = round(100 * (1 - survival), 1)

    return {
        "no_floor": sorted(no_floor),
        "hits": hits,
        "confidences": confidences,
        "score": score,
        "rule_score": kw["score"],
        "ml_score": ml.get("score", 0.0),
        "ml_available": ml.get("available", False),
        "ml_probabilities": ml.get("probabilities", {}),
        "denied_clauses": ml.get("denied_clauses", []),
    }


def _band_for(score: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= score < hi:
            return label
    return "Critical"


def _recommended_actions(band: str, hits: dict) -> list:
    actions = list(SEVERITY_ACTIONS[band])
    for cat in hits:
        for tag in SERVICE_TAGS.get(cat, []):
            if tag not in actions:
                actions.append(tag)
    return actions


def score_text(text: str, speech_stress: float = 0.0, has_speech_data: bool = False) -> dict:
    """
    speech_stress: 0-100 score computed elsewhere from pitch variance /
    pause ratio / speech rate (see /chat/voice_message).
    has_speech_data: whether speech_stress reflects a real, successfully
    analyzed recording. Defaults to False for text-only submissions —
    and is ALSO False if voice analysis was attempted but failed to
    decode (see main.py), which matters: without this, a failed
    decode's speech_stress=0.0 would be blended in as if it meant
    "calm", silently deflating the score for exactly the wrong reason.
    """
    kw = _detect_categories(text)
    isolation_score = 100.0 if "social_isolation" in kw["hits"] else 0.0
    non_latin = _is_mostly_non_latin(text)
    sentiment_score = 0.0 if non_latin else _sentiment_stress_score(text)

    # Only signals that are ACTUALLY AVAILABLE go into the blend, each
    # with its normal weight — and the result is normalized by the sum
    # of those available weights, not a fixed 1.0. This is the same
    # principle applied twice: VADER can't read Hindi, so a non-Latin
    # message excludes text_sentiment; a text-only (or failed-audio)
    # message excludes speech_stress. Either way, a missing signal is
    # treated as absent information, never as evidence of calm — an
    # unusable "0" would otherwise silently eat its share of the total
    # weight and deflate the score for no real reason.
    components = [(WEIGHTS["keyword_severity"], kw["score"]),
                  (WEIGHTS["isolation_indicator"], isolation_score)]
    if not non_latin:
        components.append((WEIGHTS["text_sentiment"], sentiment_score))
    if has_speech_data:
        components.append((WEIGHTS["speech_stress"], speech_stress))

    total_weight = sum(w for w, _ in components)
    svi = sum(w * v for w, v in components) / total_weight
    svi = round(min(100.0, svi), 1)

    # Safety floors: certain categories must never be diluted down by an
    # otherwise "calm-reading" sentiment/speech score — see
    # CATEGORY_FLOORS above for which ones and why. Tracked explicitly
    # (floor_applied) so an officer can see a Critical rating was a
    # hard override, not just the weighted formula happening to agree.
    floor_applied = None
    for cat, floor in CATEGORY_FLOORS.items():
        if cat in kw["hits"] and cat not in kw.get("no_floor", ()) and svi < floor:
            svi = floor
            floor_applied = cat

    # A long-ish message that matches nothing at all is far more likely
    # to be a genuine coverage gap — wrong language, unfamiliar phrasing
    # — than real evidence of calm, especially for non-English text
    # where lexicon coverage is known to be partial. Flagging this
    # explicitly (rather than silently reporting a low score) is what
    # keeps an unrecognized disclosure from sinking to the bottom of a
    # queue sorted by score.
    #
    # This still fires when BOTH detectors come up empty, but it is a
    # weaker signal than it was. The model was trained on Devanagari and
    # Gujarati as well as English, so non-Latin text with no hits is no
    # longer almost-certainly a coverage gap — for those two scripts it
    # now more often means the message genuinely carries no risk. The
    # escalation is kept because the guarantee it provides (an
    # unreadable disclosure never sinks to the bottom of the queue)
    # matters more than the review time a false one costs, and because
    # neither detector covers any script beyond these three at all.
    language_confidence = "low" if (non_latin and not kw["hits"]) else "ok"
    if language_confidence == "low":
        svi = max(svi, 40.0)  # keep it out of the bottom of a score-sorted queue

    band = _band_for(svi)
    human_review_required = band in ("High", "Critical") or language_confidence == "low"

    return {
        "svi": svi,
        "risk_category": band,
        "breakdown": {
            "text_sentiment": sentiment_score,
            "keyword_severity": kw["score"],
            "speech_stress": speech_stress if has_speech_data else None,
            "isolation_indicator": isolation_score,
        },
        "keyword_hits": kw["hits"],
        "category_confidences": kw["confidences"],
        "recommended_actions": _recommended_actions(band, kw["hits"]),
        "language_confidence": language_confidence,
        "human_review_required": human_review_required,
        "floor_applied": floor_applied,
        # Detection provenance. An officer reviewing a Critical rating
        # should be able to see whether it came from a phrase the
        # lexicon recognised or from a model inference, because those
        # warrant different amounts of trust — and because a case
        # escalated purely on a model call is exactly the kind of
        # decision that has to remain contestable by a human.
        "detection": {
            "rule_score": kw["rule_score"],
            "ml_score": kw["ml_score"],
            "ml_available": kw["ml_available"],
            "denied_clauses": kw["denied_clauses"],
            "model_only_unconfirmed": kw.get("no_floor", []),
        },
    }
