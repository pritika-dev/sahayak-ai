"""
Trains Sahayk's ML triage model and reports honest held-out metrics.

WHAT IS ACTUALLY LEARNED
------------------------
Only one thing: the MATCHING step. The model predicts, for each of the
16 categories, the probability that the message discloses it. Everything
downstream of that — the noisy-OR aggregation, the per-category severity
weights, the safety floors, the service-routing tags — stays in
`scoring.py` exactly as it was, and consumes the model's probabilities
the same way it consumes the rule engine's match confidences.

This is a deliberate architecture choice, not an accident of
convenience. `scoring.py` says the point of a rubric over a black-box
model is that "a reviewer can see and adjust exactly why a case scored
the way it did". A model that emitted a single opaque 0-100 number
would destroy that. A model that emits *per-category evidence* keeps it
completely: an officer still sees which categories fired and at what
confidence, and the weights remain editable in one readable dict. The
model closes the recall gap in phrase matching without taking the
rubric away from the humans who have to defend it.

FEATURES
--------
Word TF-IDF (1-2 grams) unioned with character TF-IDF (2-5 grams,
word-boundary aware). The character half is what makes this work across
Sahayk's actual input distribution:
  - Devanagari and Gujarati need no tokeniser or stemmer to be useful.
  - Romanised Hindi/Gujarati has no fixed spelling — "chillata",
    "chilata", "chillaata" all share character n-grams and collapse to
    nearly the same feature vector. Word-level features treat them as
    three unrelated tokens.
  - Keypad typos and Web Speech API mistranscriptions degrade character
    n-grams gracefully and destroy exact phrase matches outright.

WHY NOT A TRANSFORMER
---------------------
It should be one, and `train_transformer.py` in this folder does
exactly that against the same corpus. It is a separate script because
HuggingFace model downloads were blocked in the environment this was
trained in, so no pretrained multilingual weights (MuRIL, XLM-R) could
be fetched. Training a transformer from scratch on ~9k synthetic rows
would be strictly worse than this — a transformer's value is the
pretraining, and without it there is nothing to fine-tune.

Usage:
    python train.py --data data/ --out ../backend/models/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from lexicon import CATEGORIES as LEXICON_CATEGORIES, keyword_severity_score  # noqa: E402

from generate_dataset import CATEGORY_FLOORS, band_for, severity_from, band_with_floors  # noqa: E402

BAND_ORDER = ["Low", "Moderate", "High", "Critical"]

# Categories where a miss is far more costly than a false alarm. Their
# decision thresholds are tuned for F2 (recall weighted 2x) rather than
# F1, and floored low. This mirrors the reasoning already written into
# scoring.py's CATEGORY_FLOORS: "for these specific categories it is
# safer to over-react to an uncertain mention than risk under-reacting
# to a real one". A triage tool that quietly drops a suicide disclosure
# to protect its precision score is not a triage tool.
RECALL_FIRST = set(CATEGORY_FLOORS.keys())


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def build_features(train_texts):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, max_features=120_000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=250_000)
    word.fit(train_texts)
    char.fit(train_texts)
    return word, char


def transform(word, char, texts):
    return hstack([word.transform(texts), char.transform(texts)]).tocsr()


def tune_threshold(y_true, probs, recall_first):
    """Picks the probability cut-off per category on a validation split
    held out of training — never on the test set, which would make
    every number that follows meaningless."""
    beta2 = 4.0 if recall_first else 1.0   # F2 vs F1
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.91, 0.01):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        score = (1 + beta2) * prec * rec / (beta2 * prec + rec)
        if score > best_score:
            best_score, best_t = score, float(t)
    if recall_first:
        best_t = min(best_t, 0.35)   # hard ceiling: never gate a floor category behind high confidence
    return round(best_t, 3)


def to_confidence(p, threshold, hedge_threshold):
    """Maps a probability onto the confidence values the rubric actually
    defines: 1.0 for a direct disclosure, 0.5 for a hedged one.

    Passing the raw probability through instead looks tempting but is
    wrong in a specific way. The rule engine's confidence is not a
    degree of belief — it is a reading of the sentence ("he hurts me"
    vs "he MIGHT hurt me"), and the severity weights were tuned against
    those two values. A 0.83 probability fed in as 0.83 silently
    discounts a direct disclosure the model is simply slightly unsure
    about, and that discount compounds through the noisy-OR. Snapping
    to the two defined levels keeps the model's uncertainty in the
    threshold decision, where it belongs, instead of leaking it into a
    severity number the rubric never meant to receive."""
    return 1.0 if p >= hedge_threshold else 0.5


def predicted_severity(cat_probs, categories, thresholds, hedge_thresholds):
    """Feeds model output into the EXISTING rubric: a category counts as
    present once it clears its threshold, and its confidence enters the
    same noisy-OR the rule engine uses.
    Returns (severity, band, svi_after_floors, active dict)."""
    active = {}
    for i, cat in enumerate(categories):
        p = float(cat_probs[i])
        if p >= thresholds[cat]:
            active[cat] = to_confidence(p, thresholds[cat], hedge_thresholds[cat])
    sev = severity_from(active)
    band, svi = band_with_floors(sev, active)
    return sev, band, svi, active


def tune_hedge_threshold(rows, probs, cat):
    """Picks the probability above which a detected category is treated
    as a DIRECT (confidence 1.0) rather than hedged (0.5) disclosure,
    by matching the confidence recorded in the corpus. Tuned on the
    validation split only.

    Scored by a WEIGHTED balanced accuracy, for two separate reasons.

    Balanced rather than plain, because hedged disclosures are only
    ~13% of instances and plain accuracy is therefore maximised by
    calling everything direct — which is exactly what the first version
    of this did, driving every threshold to the sweep floor and
    inflating severity on every hedged message.

    Weighted rather than even, because the two errors do not cost the
    same. Reading a direct disclosure as hedged HALVES its confidence,
    which propagates through the noisy-OR and can drop a case a whole
    band — a real disclosure sinking down a queue sorted by score. The
    reverse error scores a tentative message as if it were certain,
    which puts a case in front of a human who did not strictly need to
    see it. An evenly balanced objective treats those as equivalent and
    pushed the cut-offs to ~0.95, costing 7 points of Critical recall
    to buy 3 points of false-alarm rate. That is the wrong trade for a
    triage tool, so misses are weighted 3:1 against false alarms."""
    truth = [(p, r["confidences"].get(cat)) for p, r in zip(probs, rows)
             if r["confidences"].get(cat) is not None]
    direct = [p for p, c in truth if c == 1.0]
    hedged = [p for p, c in truth if c == 0.5]
    if len(direct) < 20 or len(hedged) < 20:
        return 0.5
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.20, 0.96, 0.01):
        tpr = np.mean([p >= t for p in direct])      # direct called direct
        tnr = np.mean([p < t for p in hedged])       # hedged called hedged
        score = 0.75 * tpr + 0.25 * tnr
        if score > best_score:
            best_score, best_t = score, float(t)
    return round(best_t, 3)


def lexicon_prediction(text):
    kw = keyword_severity_score(text)
    active = dict(kw["confidences"])
    sev = kw["score"]
    band, svi = band_with_floors(sev, active)
    return sev, band, svi, set(active.keys())


def evaluate(rows, categories, cat_probs, thresholds, hedge_thresholds):
    y_true = np.array([[1 if c in r["categories"] else 0 for c in categories] for r in rows])
    y_pred = np.array([[1 if cat_probs[i][j] >= thresholds[c] else 0
                        for j, c in enumerate(categories)] for i in range(len(rows))])

    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_cat = {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                   "f1": round(float(f[i]), 3), "support": int(sup[i]),
                   "threshold": thresholds[c], "hedge_threshold": hedge_thresholds[c],
                   "recall_first": c in RECALL_FIRST}
               for i, c in enumerate(categories)}
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    model_bands, true_bands, lex_bands = [], [], []
    neg_total = neg_model_wrong = neg_lex_wrong = 0
    sev_err, lex_sev_err = [], []
    per_lang = {}
    lex_missed_but_model_caught = 0
    model_missed_but_lex_caught = 0

    for i, row in enumerate(rows):
        m_sev, m_band, _, m_active = predicted_severity(
            cat_probs[i], categories, thresholds, hedge_thresholds)
        l_sev, l_band, _, l_active = lexicon_prediction(row["text"])
        model_bands.append(m_band)
        true_bands.append(row["band"])
        lex_bands.append(l_band)
        sev_err.append(abs(m_sev - row["severity"]))
        lex_sev_err.append(abs(l_sev - row["severity"]))

        true_set = set(row["categories"])
        if true_set:
            if not (l_active & true_set) and (set(m_active) & true_set):
                lex_missed_but_model_caught += 1
            if (l_active & true_set) and not (set(m_active) & true_set):
                model_missed_but_lex_caught += 1

        for cat in row.get("negated", []):
            neg_total += 1
            neg_model_wrong += int(cat in m_active)
            neg_lex_wrong += int(cat in l_active)

        d = per_lang.setdefault(row["lang"], {"n": 0, "band_ok": 0, "lex_band_ok": 0, "err": []})
        d["n"] += 1
        d["band_ok"] += int(m_band == row["band"])
        d["lex_band_ok"] += int(l_band == row["band"])
        d["err"].append(abs(m_sev - row["severity"]))

    def band_acc(pred):
        return round(float(np.mean([a == b for a, b in zip(pred, true_bands)])), 4)

    # Escalation safety: of the cases that genuinely belong in the
    # High/Critical part of the queue, how many does each system put
    # there? This is the metric that decides whether the tool is safe
    # to put in front of a reviewer, and it is far more important than
    # overall accuracy.
    def escalation_recall(pred):
        idx = [i for i, t in enumerate(true_bands) if t in ("High", "Critical")]
        if not idx:
            return 0.0
        return round(sum(1 for i in idx if pred[i] in ("High", "Critical")) / len(idx), 4)

    def critical_recall(pred):
        idx = [i for i, t in enumerate(true_bands) if t == "Critical"]
        if not idx:
            return 0.0
        return round(sum(1 for i in idx if pred[i] == "Critical") / len(idx), 4)

    # False-alarm rate on genuinely low-risk traffic — the number that
    # decides whether the queue stays workable for a human.
    def false_escalation(pred):
        idx = [i for i, t in enumerate(true_bands) if t == "Low"]
        if not idx:
            return 0.0
        return round(sum(1 for i in idx if pred[i] in ("High", "Critical")) / len(idx), 4)

    cm = confusion_matrix(true_bands, model_bands, labels=BAND_ORDER).tolist()
    cm_lex = confusion_matrix(true_bands, lex_bands, labels=BAND_ORDER).tolist()

    return {
        "n_test": len(rows),
        "categories": per_cat,
        "multilabel": {
            "micro_f1": round(float(micro[2]), 4), "micro_precision": round(float(micro[0]), 4),
            "micro_recall": round(float(micro[1]), 4), "macro_f1": round(float(macro[2]), 4),
        },
        "band_accuracy": {"model": band_acc(model_bands), "lexicon_baseline": band_acc(lex_bands)},
        "escalation_recall_high_or_critical": {
            "model": escalation_recall(model_bands), "lexicon_baseline": escalation_recall(lex_bands)},
        "critical_recall": {"model": critical_recall(model_bands),
                            "lexicon_baseline": critical_recall(lex_bands)},
        "false_escalation_rate_on_low_risk": {
            "model": false_escalation(model_bands), "lexicon_baseline": false_escalation(lex_bands)},
        "severity_mae": {"model": round(float(np.mean(sev_err)), 2),
                         "lexicon_baseline": round(float(np.mean(lex_sev_err)), 2)},
        "disagreement": {
            "lexicon_missed_model_caught": lex_missed_but_model_caught,
            "model_missed_lexicon_caught": model_missed_but_lex_caught,
        },
        "negation_handling": {
            "note": ("Explicitly DENIED disclosures ('it is not the case that he hits me'). "
                     "Counted as an error when the system records the category anyway. "
                     "The rule engine only inspects a 4-token window before a match, so "
                     "long-range denials are invisible to it; the model sees the whole "
                     "sentence."),
            "negated_mentions": neg_total,
            "false_positive_rate_model": round(neg_model_wrong / neg_total, 4) if neg_total else 0.0,
            "false_positive_rate_lexicon": round(neg_lex_wrong / neg_total, 4) if neg_total else 0.0,
        },
        "confusion_matrix_labels": BAND_ORDER,
        "confusion_matrix_model": cm,
        "confusion_matrix_lexicon": cm_lex,
        "per_language": {
            k: {"n": v["n"],
                "band_accuracy_model": round(v["band_ok"] / v["n"], 4),
                "band_accuracy_lexicon": round(v["lex_band_ok"] / v["n"], 4),
                "severity_mae_model": round(float(np.mean(v["err"])), 2)}
            for k, v in sorted(per_lang.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="../backend/models")
    ap.add_argument("--C", type=float, default=4.0)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    data = here / args.data
    out = (here / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    train_rows = load(data / "train.jsonl")
    test_rows = load(data / "test.jsonl")
    categories = sorted(LEXICON_CATEGORIES.keys())

    # Sanity check: training labels and serving rubric must agree on
    # the category set, or the model's outputs will be silently
    # misaligned with scoring.py's weights.
    assert categories == json.loads((data / "meta.json").read_text())["categories"], \
        "category set drifted between lexicon.py and the generated corpus"

    # Validation split, carved out of TRAIN only, used purely to tune
    # decision thresholds.
    tr, va = train_test_split(train_rows, test_size=0.15, random_state=7,
                              stratify=[r["band"] for r in train_rows])

    print(f"train={len(tr)}  val={len(va)}  test={len(test_rows)}  categories={len(categories)}")
    t0 = time.time()
    word, char = build_features([r["text"] for r in tr])
    Xtr = transform(word, char, [r["text"] for r in tr])
    Xva = transform(word, char, [r["text"] for r in va])
    Xte = transform(word, char, [r["text"] for r in test_rows])
    print(f"features: {Xtr.shape[1]:,} dims  ({time.time() - t0:.1f}s)")

    models, thresholds, hedge_thresholds = {}, {}, {}
    for cat in categories:
        ytr = np.array([1 if cat in r["categories"] else 0 for r in tr])
        yva = np.array([1 if cat in r["categories"] else 0 for r in va])
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, ytr)
        models[cat] = clf
        pva = clf.predict_proba(Xva)[:, 1]
        thresholds[cat] = tune_threshold(yva, pva, cat in RECALL_FIRST)
        hedge_thresholds[cat] = tune_hedge_threshold(va, pva, cat)
        print(f"  {cat:<28} thr={thresholds[cat]:.2f}  hedge={hedge_thresholds[cat]:.2f}  pos={int(ytr.sum())}")

    cat_probs = np.column_stack([models[c].predict_proba(Xte)[:, 1] for c in categories])
    report = evaluate(test_rows, categories, cat_probs, thresholds, hedge_thresholds)
    report["train_size"] = len(tr)
    report["val_size"] = len(va)
    report["feature_dims"] = int(Xtr.shape[1])
    report["train_seconds"] = round(time.time() - t0, 1)

    bundle = {
        "version": 1,
        "categories": categories,
        "thresholds": thresholds,
        "hedge_thresholds": hedge_thresholds,
        "word_vectorizer": word,
        "char_vectorizer": char,
        "classifiers": models,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "Synthetic corpus, rubric-derived labels. Not clinically validated.",
    }
    model_path = out / "sahayk_triage.joblib"
    joblib.dump(bundle, model_path, compress=3)
    report["model_file"] = str(model_path.name)
    report["model_size_mb"] = round(model_path.stat().st_size / 1e6, 2)

    (here / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in report.items()
                             if k not in ("categories",)}, indent=2))
    print(f"\nwrote {model_path}  ({report['model_size_mb']} MB)")


if __name__ == "__main__":
    main()
