"""
Builds the labelled training corpus for Sahayk's ML triage model.

METHOD (read this before trusting any number the training run prints)
---------------------------------------------------------------------
Every sample here is SYNTHETIC. No real caller data was used, and none
should be — helpline transcripts are among the most sensitive data that
exists. What the model learns from this corpus is therefore not "what
real distress looks like"; it is "how to generalise Sahayk's existing
severity rubric to phrasings the rule engine's phrase list does not
contain".

That is a real and useful thing to learn, and it is exactly the gap
`backend/lexicon.py` names in its own docstring: phrase matching "will
keep having gaps for tense, phrasing, and slang variants no matter how
many phrases are added". But it is a bounded claim, and the honest
version of it is:

  - Labels are RULE-DERIVED, not clinician-annotated. The severity
    target is computed from the same noisy-OR formula and the same
    per-category weights the rule engine uses. The model inherits
    whatever is wrong with those weights. A high test score means the
    model reproduces the rubric on unseen wording — NOT that the rubric
    is clinically correct.
  - Generalisation is measured honestly. Expressions are held out at
    the EXPRESSION level, not the sample level: a phrasing that appears
    in training never appears in test. Splitting rows at random would
    let near-duplicate sentences straddle the split and inflate every
    metric — the usual way synthetic-data benchmarks end up meaningless.
  - `--report-lexicon-recall` measures how many held-out test samples
    the existing rule engine misses entirely. That is the number worth
    quoting: it is the gap the model is there to close.

Usage:
    python generate_dataset.py --n 9000 --out data/
"""
import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from lexicon import CATEGORIES as LEXICON_CATEGORIES  # noqa: E402

from expressions import (  # noqa: E402
    EXPRESSIONS, NEUTRAL, OPENERS, CLOSERS, NEGATION_FRAMES, HEDGE_FRAMES,
    THIRD_PERSON_FRAMES, CO_OCCURRENCE, JOINERS, LANGUAGES, LANGUAGE_MIX,
)
from templates import TEMPLATES  # noqa: E402

CATEGORY_NAMES = sorted(EXPRESSIONS.keys())


def expand_templates(per_cell, seed):
    """Fills every slot pattern in templates.py to produce the bulk of
    the expression pool: many distinct sentences per category that share
    vocabulary but differ in arrangement.

    `per_cell` caps how many are kept per (category, language) so one
    category with a large slot product cannot dominate the corpus."""
    rng = random.Random(seed)
    pools = {}
    for cat, langs in TEMPLATES.items():
        pools[cat] = {}
        for lang, spec in langs.items():
            seen = set()
            slots, patterns = spec["slots"], spec["patterns"]
            # Deterministic sweep first so every slot VALUE appears at
            # least once — random sampling alone can leave rare values
            # out of training entirely, which is how a category ends up
            # blind to one particular word for the abuser or the act.
            for pattern in patterns:
                names = [n for n in slots if "{" + n + "}" in pattern]
                if not names:
                    seen.add(pattern)
                    continue
                longest = max(names, key=lambda n: len(slots[n]))
                for value in slots[longest]:
                    filled = pattern
                    for n in names:
                        v = value if n == longest else rng.choice(slots[n])
                        filled = filled.replace("{" + n + "}", v)
                    seen.add(filled)
            # Then random combinations up to the cap.
            guard = 0
            while len(seen) < per_cell and guard < per_cell * 30:
                guard += 1
                pattern = rng.choice(patterns)
                filled = pattern
                for n in slots:
                    filled = filled.replace("{" + n + "}", rng.choice(slots[n]))
                seen.add(filled)
            pools[cat][lang] = sorted(seen)
    return pools


# Verb-agreement augmentation. Hindi and Gujarati inflect the verb for
# the subject's gender and number, so the same disclosure has several
# equally ordinary forms: "mujhe marta hai" (one man), "mujhe marti hai"
# (one woman), "mujhe marte hain" (plural, or the respectful form a
# caller often uses even for one person). Writing only one of them into
# the slot vocabulary is how a model ends up confidently detecting
# abuse from a husband and missing the identical sentence about
# in-laws — which is exactly what the first spot-check of this model
# did. These rules generate the other forms mechanically.
#
# Deliberately shallow: suffix substitution, not morphological
# analysis. It produces a few forms that a native speaker would call
# slightly off, which is acceptable here — extra near-miss spellings in
# the TRAINING pool only widen what the character n-grams tolerate. It
# would not be acceptable in anything the caller reads.
_AGREEMENT_RULES = {
    "rom": [("ta hai", ["ti hai", "te hain", "ta tha", "ti thi"]),
            ("ti hai", ["ta hai", "te hain", "ti thi"]),
            ("te hain", ["ta hai", "ti hai", "te the"]),
            ("ta hu", ["ti hu", "ta hun", "ti hun"]),
            ("ti hu", ["ta hu", "ti hun", "ta hun"]),
            ("chahti", ["chahta"]), ("chahta", ["chahti"]),
            ("rahi hu", ["raha hu", "rahi hun"]),
            ("gayi", ["gaya"]), ("diya", ["di", "diye"]),
            ("chhe", ["chhe", "hase"])],
    "hi": [("ता है", ["ती है", "ते हैं", "ता था"]),
           ("ती है", ["ता है", "ते हैं", "ती थी"]),
           ("ते हैं", ["ता है", "ती है", "ते थे"]),
           ("चाहती", ["चाहता"]), ("चाहता", ["चाहती"]),
           ("रही हूं", ["रहा हूं"]), ("रहा हूं", ["रही हूं"]),
           ("गई", ["गया"]), ("हूं", ["हूँ"])],
    "gu": [("છે", ["છે"]), ("માંગું", ["માગું"]),
           ("કરે છે", ["કરે છે", "કરતો હતો"]),
           ("મારે છે", ["મારતો હતો", "મારે છે"]),
           ("છું", ["છુ"])],
    "en": [],
}


def agreement_variants(sentence, lang, rng, max_extra=2):
    out = []
    for src, targets in _AGREEMENT_RULES.get(lang, []):
        if src in sentence:
            for tgt in targets:
                v = sentence.replace(src, tgt)
                if v != sentence:
                    out.append(v)
    rng.shuffle(out)
    return out[:max_extra]


def merged_pools(per_cell, seed):
    """Template-generated sentences plus the hand-written idiomatic ones
    from expressions.py. The hand-written set is the harder half — the
    indirect disclosures no pattern produces — so it is kept rather than
    replaced."""
    generated = expand_templates(per_cell, seed)
    merged = {}
    for cat in CATEGORY_NAMES:
        merged[cat] = {}
        for lang in LANGUAGES:
            merged[cat][lang] = list(EXPRESSIONS[cat][lang]) + \
                list(generated.get(cat, {}).get(lang, []))
    return merged

# Mirrors scoring.py. Duplicated deliberately rather than imported:
# training must not silently change meaning when the serving code is
# edited, and a drift between the two should show up as a failing
# assertion in train.py, not as a quietly mislabelled corpus.
RISK_BANDS = [(0, 25, "Low"), (25, 50, "Moderate"), (50, 75, "High"), (75, 101, "Critical")]
CATEGORY_FLOORS = {
    "suicidal_ideation": 95, "immediate_danger": 95, "child_safety_concern": 90,
    "sexual_harassment_assault": 60, "physical_safety_abuse": 55, "elder_abuse": 55,
    # SIH26093 atrocity categories (Round 19) — mirrors scoring.py
    "rape_sexual_violence": 90, "family_murder": 85, "witness_intimidation": 60,
    "caste_atrocity": 55, "social_boycott": 55, "displacement_arson": 55,
    "police_noncooperation": 50,
}


def band_for(score):
    for lo, hi, label in RISK_BANDS:
        if lo <= score < hi:
            return label
    return "Critical"


def severity_from(active):
    """Noisy-OR over active categories — identical to
    lexicon.keyword_severity_score's aggregation. `active` maps a
    category to its confidence (1.0 direct, 0.5 hedged)."""
    survival = 1.0
    for cat, conf in active.items():
        survival *= (1 - LEXICON_CATEGORIES[cat]["weight"] * conf)
    return round(100 * (1 - survival), 1)


def band_with_floors(severity, active):
    """The band a reviewer actually sees, after safety floors. Used as
    the classification target so the 4-class metrics reported by
    train.py describe the decision the system really makes."""
    svi = severity
    for cat, floor in CATEGORY_FLOORS.items():
        if cat in active and svi < floor:
            svi = floor
    return band_for(svi), svi


# --------------------------------------------------------------------------
# Surface noise. Real input arrives from a phone keypad and from the Web
# Speech API, neither of which produces clean text.
# --------------------------------------------------------------------------

_KEYBOARD_NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "g": "fh", "h": "gj",
    "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip",
    "p": "o", "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe",
    "y": "tu",
}


def _typo(word, rng):
    if len(word) < 4:
        return word
    kind = rng.random()
    i = rng.randrange(1, len(word) - 1)
    if kind < 0.35:                                  # dropped character
        return word[:i] + word[i + 1:]
    if kind < 0.60:                                  # doubled character
        return word[:i] + word[i] + word[i:]
    if kind < 0.80:                                  # transposition
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    nb = _KEYBOARD_NEIGHBOURS.get(word[i].lower())   # adjacent key
    return word[:i] + rng.choice(nb) + word[i + 1:] if nb else word


def _add_noise(text, rng):
    """Typos, dropped punctuation, ASR-style run-on. Applied to a subset
    of samples so the model sees both clean and degraded input — the
    char_wb n-gram features in train.py are chosen largely to survive
    this, since a phrase matcher does not."""
    if rng.random() < 0.30:
        words = text.split()
        for _ in range(max(1, len(words) // 12)):
            j = rng.randrange(len(words))
            words[j] = _typo(words[j], rng)
        text = " ".join(words)
    if rng.random() < 0.35:
        text = text.replace(",", "").replace(".", " ").replace("।", " ")
        text = " ".join(text.split())
    if rng.random() < 0.12:
        text = text.upper() if rng.random() < 0.3 else text.lower()
    return text


# --------------------------------------------------------------------------
# Expression-level train/test split
# --------------------------------------------------------------------------

def _lexicon_phrases_by_lang():
    """The rule engine's own phrases, bucketed by script. These are
    added to the TRAINING pool only — never to test.

    Why add them at all: the expression pools are written to avoid the
    lexicon, so a model trained purely on them would learn the novel
    phrasings and have no particular reason to get the *known* ones
    right. Since the deployed scorer runs both signals together, a
    model that regressed on the phrasings the rules already catch would
    be a net loss. Training on both keeps the model a strict addition.

    Why never in test: including them would let the model score points
    for reproducing phrases it was trained on, and would deflate the
    lexicon-baseline comparison into meaninglessness (the rules match
    their own phrases by definition). Test stays 100% novel wording."""
    buckets = {"en": [], "hi": [], "gu": [], "rom": []}
    for cat, spec in LEXICON_CATEGORIES.items():
        for phrase in spec["phrases"]:
            names = [unicodedata.name(c, "") for c in phrase]
            if any("DEVANAGARI" in n for n in names):
                script = "hi"
            elif any("GUJARATI" in n for n in names):
                script = "gu"
            elif any(("TAMIL" in n) or ("BENGALI" in n) for n in names):
                continue   # no tam/ben training language yet; the lexicon covers these
            else:
                script = "en"
            buckets[script].append((cat, phrase))
    return buckets


def split_expressions(holdout_frac, seed, include_lexicon=True, per_cell=260):
    """Partitions the expression pools themselves, so no phrasing is
    shared between the train and test corpora. This is the difference
    between measuring generalisation and measuring memorisation.

    Note what this does and does not guarantee. A held-out sentence is
    never seen in training, but it is built from the same slot
    vocabularies, so individual words recur. That is the intended
    setting: the question being asked is whether the model recognises an
    unseen ARRANGEMENT of familiar language, which is precisely what a
    phrase list cannot do. It is not a claim that the model handles
    vocabulary it has never encountered — it does not, and no
    bag-of-words model can. Closing that second gap needs pretrained
    multilingual embeddings; see train_transformer.py."""
    rng = random.Random(seed)
    all_pools = merged_pools(per_cell, seed)
    train_pool, test_pool = {}, {}
    for cat, langs in all_pools.items():
        train_pool[cat], test_pool[cat] = {}, {}
        for lang, exprs in langs.items():
            shuffled = list(exprs)
            rng.shuffle(shuffled)
            n_test = max(1, int(round(len(shuffled) * holdout_frac)))
            test_pool[cat][lang] = shuffled[:n_test]
            train_pool[cat][lang] = shuffled[n_test:]
    neutral_train, neutral_test = {}, {}
    for lang, exprs in NEUTRAL.items():
        shuffled = list(exprs)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * holdout_frac)))
        neutral_test[lang] = shuffled[:n_test]
        neutral_train[lang] = shuffled[n_test:]

    if include_lexicon:
        for lang, pairs in _lexicon_phrases_by_lang().items():
            for cat, phrase in pairs:
                if cat in train_pool and lang in train_pool[cat]:
                    train_pool[cat][lang].append(phrase)

    # Agreement variants are added AFTER the split and to the TRAIN
    # side only. Generating them before splitting would put "mujhe
    # marta hai" in train and its variant "mujhe marte hain" in test —
    # a near-duplicate straddling the split, which is the exact leak
    # the expression-level holdout exists to prevent, and it would
    # inflate every metric in the report.
    aug_rng = random.Random(seed + 99)
    for cat in train_pool:
        for lang in train_pool[cat]:
            extra = []
            for sentence in train_pool[cat][lang]:
                extra.extend(agreement_variants(sentence, lang, aug_rng))
            train_pool[cat][lang] = sorted(set(train_pool[cat][lang]) | set(extra))

    return (train_pool, neutral_train), (test_pool, neutral_test)


# --------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------

def _pick_categories(rng):
    """1-3 categories, drawn through the co-occurrence graph rather than
    uniformly, so the label correlations resemble the domain."""
    primary = rng.choice(CATEGORY_NAMES)
    chosen = [primary]
    n_extra = rng.choices([0, 1, 2], weights=[0.62, 0.30, 0.08])[0]
    partners = list(CO_OCCURRENCE.get(primary, []))
    rng.shuffle(partners)
    for p in partners[:n_extra]:
        if p not in chosen:
            chosen.append(p)
    return chosen


def make_sample(pool, neutral_pool, rng):
    lang = rng.choices(LANGUAGES, weights=LANGUAGE_MIX)[0]

    # ~30% neutral: the negative class that keeps precision honest.
    # (Raised from 22% in Round 19 together with the hard negatives in
    # atrocity_data.py — the first 24-category run over-escalated
    # harmless messages.)
    if rng.random() < 0.30:
        text = rng.choice(neutral_pool[lang])
        if rng.random() < 0.25:
            text = rng.choice(OPENERS[lang]) + " " + text
        return {
            "text": _add_noise(text, rng), "lang": lang, "categories": [],
            "confidences": {}, "negated": [], "severity": 0.0, "band": "Low",
            "svi_after_floors": 0.0,
        }

    cats = _pick_categories(rng)
    active, parts, negated = {}, [], []

    for cat in cats:
        options = pool[cat][lang]
        if not options:                     # holdout emptied this cell
            options = pool[cat]["en"] or EXPRESSIONS[cat]["en"]
        clause = rng.choice(options)

        roll = rng.random()
        if roll < 0.10:
            # Negated: the category is mentioned but explicitly denied,
            # so it must NOT appear in the label. These are the samples
            # that stop the model becoming a keyword spotter.
            parts.append(rng.choice(NEGATION_FRAMES[lang]).format(x=clause))
            negated.append(cat)
            continue
        if roll < 0.23:
            parts.append(rng.choice(HEDGE_FRAMES[lang]).format(x=clause))
            active[cat] = 0.5
        elif roll < 0.33:
            parts.append(rng.choice(THIRD_PERSON_FRAMES[lang]).format(x=clause))
            active[cat] = 1.0
        else:
            parts.append(clause)
            active[cat] = 1.0

    rng.shuffle(parts)
    text = rng.choice(JOINERS[lang]).join(parts)
    if rng.random() < 0.45:
        text = rng.choice(OPENERS[lang]) + " " + text
    if rng.random() < 0.35:
        text = text + " " + rng.choice(CLOSERS[lang])

    severity = severity_from(active)
    band, svi = band_with_floors(severity, active)
    return {
        "text": _add_noise(text, rng), "lang": lang,
        "categories": sorted(active.keys()),
        "confidences": {k: v for k, v in sorted(active.items())},
        "negated": sorted(set(negated)),
        "severity": severity, "band": band, "svi_after_floors": svi,
    }


def build(n, pools, seed):
    pool, neutral_pool = pools
    rng = random.Random(seed)
    seen, rows = set(), []
    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        s = make_sample(pool, neutral_pool, rng)
        key = s["text"].strip().lower()
        if key in seen:
            continue           # exact duplicates would leak across the split
        seen.add(key)
        rows.append(s)
    return rows


# --------------------------------------------------------------------------
# Baseline measurement: what does the CURRENT rule engine miss?
# --------------------------------------------------------------------------

def lexicon_recall(rows):
    """Runs the existing phrase matcher over a corpus and reports how
    often it finds nothing at all on a sample that genuinely carries
    risk. This is the honest 'why bother with a model' number."""
    from lexicon import keyword_severity_score
    total = missed = 0
    per_lang = {}
    for r in rows:
        if not r["categories"]:
            continue
        total += 1
        hit = bool(keyword_severity_score(r["text"])["hits"])
        d = per_lang.setdefault(r["lang"], {"n": 0, "missed": 0})
        d["n"] += 1
        if not hit:
            missed += 1
            d["missed"] += 1
    return {
        "risk_samples": total,
        "completely_missed": missed,
        "miss_rate": round(missed / total, 4) if total else 0.0,
        "per_language": {
            k: {"n": v["n"], "miss_rate": round(v["missed"] / v["n"], 4)}
            for k, v in sorted(per_lang.items())
        },
    }


def summarise(rows, name):
    bands, langs, cats = {}, {}, {}
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
        for c in r["categories"]:
            cats[c] = cats.get(c, 0) + 1
    return {"split": name, "n": len(rows), "bands": dict(sorted(bands.items())),
            "languages": dict(sorted(langs.items())), "categories": dict(sorted(cats.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000, help="training rows")
    ap.add_argument("--n-test", type=int, default=2400, help="test rows (unseen phrasings)")
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="fraction of EXPRESSIONS reserved for the test split")
    ap.add_argument("--per-cell", type=int, default=260,
                    help="max template-generated sentences per (category, language)")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", default="data")
    ap.add_argument("--report-lexicon-recall", action="store_true")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / args.out
    out.mkdir(parents=True, exist_ok=True)

    train_pools, test_pools = split_expressions(args.holdout_frac, args.seed, per_cell=args.per_cell)
    train_rows = build(args.n, train_pools, args.seed)
    test_rows = build(args.n_test, test_pools, args.seed + 1)

    for rows, fname in ((train_rows, "train.jsonl"), (test_rows, "test.jsonl")):
        with open(out / fname, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "categories": CATEGORY_NAMES,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "note": "Synthetic corpus. Labels derived from the Sahayk rubric, not clinician-annotated.",
        "baseline_caveat": (
            "lexicon_baseline_on_test measures the rule engine on wording chosen "
            "specifically to fall outside its phrase list. A near-total miss rate "
            "is therefore BY CONSTRUCTION and is not an estimate of real-world "
            "recall. Read it as: this is the size of the blind spot the model is "
            "meant to cover, not as a claim that the rules fail 99% of real calls."
        ),
        "splits": [summarise(train_rows, "train"), summarise(test_rows, "test")],
    }
    if args.report_lexicon_recall:
        meta["lexicon_baseline_on_test"] = lexicon_recall(test_rows)

    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
