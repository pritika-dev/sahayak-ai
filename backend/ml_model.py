"""
Inference wrapper for the trained triage model.

WHAT THIS REPLACES
------------------
Only the MATCHING step. `lexicon.py` decides whether a category is
present by looking for phrases from a hand-written list; this module
decides the same thing with a trained classifier. Everything after that
— the severity weights, the noisy-OR aggregation, the safety floors,
the service-routing tags — stays in `scoring.py`, untouched, and works
on the model's output exactly as it works on the rule engine's.

That boundary is deliberate. It keeps the property `scoring.py` calls
the whole point of a rubric: "a reviewer can see and adjust exactly why
a case scored the way it did". An officer still sees a list of
categories with confidences and an editable weight for each. The model
changes how the categories are *found*, not how the score is *built*.

FAILS SOFT, ALWAYS
------------------
If the model file is missing, unreadable, or built by a different
scikit-learn version, `available()` returns False and `scoring.py`
falls back to pure rule matching. A triage line does not go down
because a pickle would not load.
"""
import re
import threading
import unicodedata
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "sahayk_triage.joblib"
TRANSFORMER_DIR = Path(__file__).parent / "models" / "transformer"

_bundle = None
_load_attempted = False
_load_error = None
_lock = threading.Lock()


def _load_transformer():
    """Loads a fine-tuned transformer if one has been dropped into
    models/transformer/ (see training/train_transformer.py).

    Returns None — quietly, not fatally — when the directory is absent
    or torch/transformers are not installed, which is the normal case:
    the shipping requirements.txt does not pull in a ~2 GB deep-learning
    stack for a prototype that runs fine without it. Installing
    `transformers torch` and copying the directory in is the whole
    upgrade path; no code changes, no config."""
    meta_path = TRANSFORMER_DIR / "sahayk_meta.json"
    if not meta_path.exists():
        return None
    import json
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR))
    model.eval()
    return {
        "kind": "transformer", "meta": meta, "tokenizer": tok, "model": model,
        "torch": torch, "categories": meta["categories"],
        "thresholds": meta["thresholds"], "hedge_thresholds": meta["hedge_thresholds"],
        "version": meta.get("version"), "trained_at": meta.get("trained_at"),
        "provenance": meta.get("provenance"), "base_model": meta.get("base_model"),
    }


def _load_tfidf():
    import joblib
    bundle = joblib.load(MODEL_PATH)
    bundle["kind"] = "tfidf"
    return bundle


def _load():
    """Transformer first when present, TF-IDF otherwise. Any failure in
    the transformer path falls through to the TF-IDF model rather than
    propagating: a half-installed upgrade should degrade the quality of
    triage, never take the line down."""
    global _bundle, _load_attempted, _load_error
    with _lock:
        if _load_attempted:
            return _bundle
        _load_attempted = True
        errors = []
        for loader in (_load_transformer, _load_tfidf):
            try:
                loaded = loader()
                if loaded is not None:
                    _bundle = loaded
                    _load_error = "; ".join(errors) or None
                    return _bundle
            except Exception as exc:                  # noqa: BLE001 — never fatal
                errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
        _load_error = "; ".join(errors) or "no model found"
        _bundle = None
        return _bundle


def _probabilities(bundle, text: str) -> dict:
    """Per-category probability, from whichever backend is loaded."""
    if bundle["kind"] == "transformer":
        torch = bundle["torch"]
        enc = bundle["tokenizer"]([text], padding=True, truncation=True,
                                  max_length=bundle["meta"].get("max_len", 128),
                                  return_tensors="pt")
        with torch.no_grad():
            probs = torch.sigmoid(bundle["model"](**enc).logits)[0].tolist()
        return {cat: float(p) for cat, p in zip(bundle["categories"], probs)}

    from scipy.sparse import hstack
    X = hstack([bundle["word_vectorizer"].transform([text]),
                bundle["char_vectorizer"].transform([text])]).tocsr()
    return {cat: float(bundle["classifiers"][cat].predict_proba(X)[0, 1])
            for cat in bundle["categories"]}


def available() -> bool:
    return _load() is not None


def status() -> dict:
    b = _load()
    if b is None:
        return {"available": False, "error": _load_error, "path": str(MODEL_PATH)}
    return {
        "available": True,
        "backend": b.get("kind"),
        "base_model": b.get("base_model"),
        "version": b.get("version"),
        "trained_at": b.get("trained_at"),
        "categories": len(b.get("categories", [])),
        "provenance": b.get("provenance"),
    }


# --------------------------------------------------------------------------
# Denial gating
# --------------------------------------------------------------------------
# A linear bag-of-words model cannot scope negation. Measured on the
# held-out set, it recorded the category anyway on ~87% of explicitly
# DENIED disclosures ("i want to be clear that it is not the case that
# he hits me"), because the clause's own words outweigh the denial cue
# in a linear sum. Training on far more negated examples was tried and
# moved it only to ~79% while degrading everything else — the limit is
# the model class, not the data.
#
# So denial is handled structurally instead: the message is split into
# clauses, clauses carrying an explicit DENIAL cue are dropped, and the
# model scores what remains. If a disclosure survives only inside a
# denied clause, it does not survive at all.
#
# These cues are metalinguistic denials — a speaker saying a thing is
# not so — NOT general negation words. That distinction is the whole
# design. Gating on "not"/"nahi" would suppress "he will NOT let me
# leave the house", which is a genuine disclosure of control, and
# suppressing those is far worse than the false positives this is
# fixing. `lexicon.py` handles ordinary negation separately with its
# own 4-token lookbehind window, which is the right tool for that case.
#
# HONEST CAVEAT: these cues were written from the same denial frames
# used to generate the evaluation corpus, so the improvement this
# produces on that corpus is optimistic. Real callers deny things in
# ways no list anticipates. Proper negation scoping needs a model that
# reads syntax — see training/train_transformer.py.
_DENIAL_CUES = [
    # English
    "it is not the case", "its not the case", "is not true that", "isnt true that",
    "not true that", "i am not saying", "im not saying", "that is not my situation",
    "thats not my situation", "has never happened", "never happened",
    "do not note that", "dont note that", "please do not write", "please dont write",
    "it would be wrong to", "none of that applies", "not about me",
    "nothing like this", "the answer is no", "it is not like that",
    "its not like that", "not as if", "i want to correct", "does not apply to me",
    # Hindi
    "ऐसा नहीं है कि", "ऐसा कुछ नहीं", "सच नहीं है कि", "यह नहीं कह रही",
    "यह नहीं कह रहा", "कभी नहीं हुआ", "मत लिखिए", "लिखना गलत होगा",
    "वैसी नहीं है", "जवाब है नहीं", "बात बिल्कुल नहीं",
    # Gujarati
    "એવું નથી કે", "એવું કંઈ નથી", "સાચું નથી કે", "એવું નથી કહેતી",
    "ક્યારેય નથી થયું", "ન લખશો", "ખોટું ગણાશે", "એવી નથી", "જવાબ ના છે",
    # Romanised
    "aisa nahi hai ki", "aisa kuch nahi", "sach nahi hai ki", "yeh nahi keh rahi",
    "kabhi nahi hua", "mat likhiye", "likhna galat hoga", "waisi nahi hai",
    "jawab hai nahi", "evu nathi ke", "kyarey nathi thayu", "nathi kehti ke",
]

# Clause boundaries: sentence punctuation plus the conjunctions the
# corpus joins clauses with, so a denial in one clause does not take a
# genuine disclosure in the next one down with it.
_CLAUSE_SPLIT = re.compile(
    r"(?:[.!?;।]+)|(?:\s+(?:and also|and on top of that|aur uske alawa|"
    r"साथ ही|और इसके अलावा|અને એ ઉપરાંત|and|aur|और|અને|plus)\s+)",
    re.IGNORECASE,
)


def _normalize_for_cue(text: str) -> str:
    text = text.lower().replace("'", "").replace("’", "")
    return "".join(
        ch if (unicodedata.category(ch)[0] in ("L", "M", "N") or ch.isspace()) else " "
        for ch in text
    )


def split_clauses(text: str):
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]


def is_denial(clause: str) -> bool:
    norm = " ".join(_normalize_for_cue(clause).split())
    return any(" ".join(_normalize_for_cue(cue).split()) in norm for cue in _DENIAL_CUES)


def _gate(text: str):
    """Returns (text_to_score, denied_clauses). If every clause is a
    denial there is nothing left to score, and the caller gets no hits
    rather than hits drawn from denied material."""
    clauses = split_clauses(text)
    if len(clauses) <= 1:
        c = clauses[0] if clauses else text
        return ("", [c]) if is_denial(c) else (text, [])
    denied = [c for c in clauses if is_denial(c)]
    kept = [c for c in clauses if not is_denial(c)]
    return (" . ".join(kept), denied)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict(text: str) -> dict:
    """Returns the same shape `lexicon.keyword_severity_score` produces
    — hits, confidences, and per-category probability — so `scoring.py`
    can consume both through one code path.

    Confidence is snapped to the two values the rubric defines (1.0
    direct, 0.5 hedged) rather than passed through as a raw
    probability: the severity weights were calibrated against those two
    readings of a sentence, not against a degree of belief. See
    training/train.py:to_confidence.
    """
    bundle = _load()
    if bundle is None or not text or not text.strip():
        return {"available": False, "hits": {}, "confidences": {},
                "probabilities": {}, "score": 0.0, "denied_clauses": []}

    scored_text, denied = _gate(text)
    if not scored_text.strip():
        return {"available": True, "hits": {}, "confidences": {}, "probabilities": {},
                "score": 0.0, "denied_clauses": denied}

    raw = _probabilities(bundle, scored_text)

    hits, confidences, probabilities = {}, {}, {}
    for cat in bundle["categories"]:
        p = raw[cat]
        probabilities[cat] = round(p, 4)
        if p >= bundle["thresholds"][cat]:
            conf = 1.0 if p >= bundle["hedge_thresholds"][cat] else 0.5
            confidences[cat] = conf
            hits[cat] = [f"model: p={p:.2f}" + ("" if conf == 1.0 else " (uncertain)")]

    # Same noisy-OR the rule engine uses, so the two severity numbers
    # are directly comparable and can be blended in scoring.py.
    from lexicon import CATEGORIES as LEX
    survival = 1.0
    for cat, conf in confidences.items():
        survival *= (1 - LEX[cat]["weight"] * conf)
    score = round(100 * (1 - survival), 1)

    return {"available": True, "hits": hits, "confidences": confidences,
            "probabilities": probabilities, "score": score, "denied_clauses": denied}
