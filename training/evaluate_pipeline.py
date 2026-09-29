"""
Evaluates the PIPELINE THAT ACTUALLY RUNS, not just the classifier.

train.py measures the model in isolation. This measures what a caller's
message really passes through: backend/ml_model.py (clause splitting,
denial gating, confidence snapping) unioned with the rule engine, all
the way to the SVI and risk band scoring.py returns. The two differ
enough to matter — the denial gate alone moves the false-escalation
rate by more than 15 points — so shipping train.py's numbers as though
they described the product would be misleading.

Reports the served pipeline against two baselines: the rule engine
alone (what Sahayk did before), and the model without the denial gate
(what the model does unaided). Comparing all three is what shows which
component earns its place.

Usage:
    python evaluate_pipeline.py            # writes pipeline_metrics.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_model                                            # noqa: E402
from lexicon import CATEGORIES as LEX, keyword_severity_score  # noqa: E402
from generate_dataset import CATEGORY_FLOORS, band_for, band_with_floors, severity_from  # noqa: E402

BANDS = ["Low", "Moderate", "High", "Critical"]


def _band(active):
    return band_with_floors(severity_from(active), active)


def _ungated(text, bundle):
    """The model's raw opinion: no clause splitting, no denial gate."""
    from scipy.sparse import hstack
    X = hstack([bundle["word_vectorizer"].transform([text]),
                bundle["char_vectorizer"].transform([text])]).tocsr()
    active = {}
    for cat in bundle["categories"]:
        p = float(bundle["classifiers"][cat].predict_proba(X)[0, 1])
        if p >= bundle["thresholds"][cat]:
            active[cat] = 1.0 if p >= bundle["hedge_thresholds"][cat] else 0.5
    return active


def _union(text):
    """Exactly what scoring._detect_categories does."""
    kw = keyword_severity_score(text)
    ml = ml_model.predict(text)
    active = dict(kw["confidences"])
    for cat, conf in ml.get("confidences", {}).items():
        if conf > active.get(cat, 0.0):
            active[cat] = conf
    return active


def _guarded(text):
    """The pipeline as served since Round 19: the union above, plus the
    guard that stops a moderately-sure, model-only detection from
    triggering a safety floor (see scoring._detect_categories)."""
    import scoring
    d = scoring._detect_categories(text)
    return d["confidences"], set(d["no_floor"])


def _band_guarded(active, no_floor):
    svi = severity_from(active)
    for cat, floor in CATEGORY_FLOORS.items():
        if cat in active and cat not in no_floor and svi < floor:
            svi = floor
    return band_for(svi), svi


def run():
    rows = [json.loads(line) for line in
            open(Path(__file__).parent / "data" / "test.jsonl", encoding="utf-8")]
    bundle = ml_model._load()
    if bundle is None:
        raise SystemExit("model not loaded — run train.py first")

    systems = {
        "rules_only": lambda t: keyword_severity_score(t)["confidences"],
        "model_ungated": lambda t: _ungated(t, bundle),
        "union_no_guard": _union,
        "served_pipeline": _guarded,
    }
    acc = {name: {"band_ok": 0, "esc_hit": 0, "crit_hit": 0, "low_fp": 0,
                  "neg_fp": 0, "abs_err": 0.0, "cm": [[0] * 4 for _ in range(4)]}
           for name in systems}
    esc_n = crit_n = low_n = neg_n = 0

    for r in rows:
        true_band = r["band"]
        esc_n += int(true_band in ("High", "Critical"))
        crit_n += int(true_band == "Critical")
        low_n += int(true_band == "Low")
        neg_n += len(r.get("negated", []))
        for name, fn in systems.items():
            out = fn(r["text"])
            if isinstance(out, tuple):
                active, no_floor = out
                band, svi = _band_guarded(active, no_floor)
            else:
                active = out
                band, svi = _band(active)
            a = acc[name]
            a["band_ok"] += int(band == true_band)
            a["abs_err"] += abs(severity_from(active) - r["severity"])
            a["cm"][BANDS.index(true_band)][BANDS.index(band)] += 1
            if true_band in ("High", "Critical"):
                a["esc_hit"] += int(band in ("High", "Critical"))
            if true_band == "Critical":
                a["crit_hit"] += int(band == "Critical")
            if true_band == "Low":
                a["low_fp"] += int(band in ("High", "Critical"))
            for cat in r.get("negated", []):
                a["neg_fp"] += int(cat in active)

    n = len(rows)
    out = {
        "n_test": n,
        "note": ("Held-out synthetic corpus, rubric-derived labels. Unseen phrasings "
                 "built from seen vocabulary. Not a clinical validation and not a "
                 "measurement on real calls."),
        "metric_notes": {
            "escalation_recall": "of cases truly High/Critical, the share placed in High/Critical — the safety metric",
            "critical_recall": "of cases truly Critical, the share placed in Critical",
            "false_escalation_rate": "of cases truly Low, the share wrongly escalated — the reviewer-workload cost",
            "negation_false_positive_rate": "of explicitly DENIED disclosures, the share still recorded",
        },
        "systems": {},
    }
    for name, a in acc.items():
        out["systems"][name] = {
            "band_accuracy": round(a["band_ok"] / n, 4),
            "severity_mae": round(a["abs_err"] / n, 2),
            "escalation_recall": round(a["esc_hit"] / esc_n, 4) if esc_n else None,
            "critical_recall": round(a["crit_hit"] / crit_n, 4) if crit_n else None,
            "false_escalation_rate": round(a["low_fp"] / low_n, 4) if low_n else None,
            "negation_false_positive_rate": round(a["neg_fp"] / neg_n, 4) if neg_n else None,
            "confusion_matrix": a["cm"],
        }
    out["confusion_matrix_labels"] = BANDS

    path = Path(__file__).parent / "pipeline_metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
