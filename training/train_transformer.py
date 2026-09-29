"""
Fine-tunes a pretrained multilingual transformer on the same corpus.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
`train.py` produces the model that currently ships: TF-IDF + logistic
regression, 4 MB, 2 ms per message, CPU-only, no downloads. It was
trained in an environment where HuggingFace was blocked by network
policy, so no pretrained multilingual weights could be fetched. This
script is the path that environment could not take. Run it anywhere
with a GPU and network access — Colab's free tier is enough — and it
fine-tunes MuRIL or XLM-R against the identical dataset, with the
identical evaluation code, so the two are directly comparable.

WHAT IT SHOULD FIX
------------------
The shipping model has one weakness that is structural, not a matter of
more data (documented in ml_model.py and measured in
pipeline_metrics.json): a linear bag-of-words classifier cannot scope
negation. Even behind the denial gate it still records roughly a third
of explicitly DENIED disclosures — "it is not the case that he hits
me" — because it has no representation of what the denial attaches to.
Training on far more negated examples was tried and moved it from ~87%
to ~79% unaided while degrading every other metric. A transformer with
pretrained attention reads the dependency and should handle this
properly, which is the main reason to bother.

The second gain is vocabulary. TF-IDF only generalises across
arrangements of words it has seen; pretrained subword embeddings carry
meaning for words that never appear in this corpus at all. For
romanised Hindi and Gujarati especially, that is the difference between
covering the phrasings someone thought to write down and covering the
language.

WHICH BASE MODEL
----------------
  google/muril-base-cased   (default) — Google's Indic BERT, pretrained
      on 17 Indian languages INCLUDING transliterated text. That last
      part is the reason it is the default here: Sahayk's real input is
      full of romanised Hindi and Gujarati, and MuRIL is one of very
      few public models that saw transliteration during pretraining.
  xlm-roberta-base          — stronger general multilingual model, much
      weaker on romanised Indic text. Better if the deployment is
      mostly Devanagari/Gujarati script.
  bert-base-multilingual-cased — smallest and weakest; a fallback.

HONEST EXPECTATION
------------------
This will beat the shipping model on negation handling and on unseen
vocabulary. It will NOT make the labels more correct: they are still
derived from Sahayk's own rubric, so a transformer trained on them
reproduces that rubric more faithfully, including wherever the rubric
is wrong. Clinical validity still comes from professional review of the
weights in scoring.py and calibration against real consented
transcripts — not from a bigger model.

COLAB
-----
    !pip install -q transformers torch
    # upload the training/ folder (needs data/, generate_dataset.py,
    # train.py, expressions.py, templates.py and backend/lexicon.py)
    !python train_transformer.py --model google/muril-base-cased --epochs 4

Expect roughly 10-20 minutes on a T4 for 4 epochs over ~22k rows.
The saved directory is ~1 GB; download it and drop it into
backend/models/transformer/, and ml_model.py will pick it up
automatically (see USE_TRANSFORMER there).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lexicon import CATEGORIES as LEXICON_CATEGORIES  # noqa: E402
# Reusing train.py's evaluation code verbatim is the point: if the two
# models were scored by separately written functions, any difference
# between them could just be a difference between the two scorers.
from train import (  # noqa: E402
    RECALL_FIRST, evaluate, load, tune_threshold, tune_hedge_threshold,
)


def build_labels(rows, categories):
    return np.array([[1.0 if c in r["categories"] else 0.0 for c in categories]
                     for r in rows], dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="../backend/models/transformer")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128,
                    help="tokens; helpline messages are short, 128 covers almost all")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap training rows")
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency ({exc}). Install with:\n"
            "    pip install transformers torch\n"
            "This script is meant to run on a machine with a GPU and network "
            "access; train.py is the CPU/offline path."
        )

    here = Path(__file__).resolve().parent
    data = here / args.data
    out = (here / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    categories = sorted(LEXICON_CATEGORIES.keys())
    train_rows = load(data / "train.jsonl")
    test_rows = load(data / "test.jsonl")
    if args.limit:
        train_rows = train_rows[:args.limit]

    # Validation split carved out of TRAIN only. Thresholds are tuned on
    # it and never on test — the same discipline train.py follows, and
    # the reason its reported numbers mean anything.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_rows))
    n_val = int(len(train_rows) * args.val_frac)
    va_rows = [train_rows[i] for i in idx[:n_val]]
    tr_rows = [train_rows[i] for i in idx[n_val:]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. This will take hours on CPU — "
              "train.py is the CPU path, not this script.")

    print(f"base model : {args.model}")
    print(f"device     : {device}")
    print(f"train/val  : {len(tr_rows)} / {len(va_rows)}   test: {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(categories),
        problem_type="multi_label_classification",   # independent sigmoids, not softmax
        # A bare pretrained checkpoint has no classification head, so one
        # is created at the right size and this flag never matters. It
        # matters when --model points at a checkpoint that DOES carry a
        # head: another fine-tune, or this script's own output being
        # trained for more epochs. Without it, transformers refuses to
        # load rather than resizing, and the run dies after the download
        # has already been paid for. The head is meant to be replaced
        # here in every case, so discarding a mismatched one is correct.
        ignore_mismatched_sizes=True,
    ).to(device)

    class Rows(Dataset):
        def __init__(self, rows):
            self.texts = [r["text"] for r in rows]
            self.y = build_labels(rows, categories)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            return self.texts[i], self.y[i]

    def collate(batch):
        texts, ys = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=args.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(np.stack(ys))
        return enc

    train_loader = DataLoader(Rows(tr_rows), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)

    # Positive-class weighting. Categories are imbalanced (some appear in
    # ~4% of rows), and unweighted BCE on a rare label is minimised by
    # predicting "absent" always — which for suicidal_ideation means a
    # model that scores well and never fires. This is the transformer's
    # equivalent of class_weight="balanced" in train.py.
    y_tr = build_labels(tr_rows, categories)
    pos = y_tr.sum(axis=0)
    pos_weight = torch.tensor(
        np.clip((len(y_tr) - pos) / np.maximum(pos, 1.0), 1.0, 20.0),
        dtype=torch.float32, device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += float(loss)
            if step % 50 == 0:
                print(f"  epoch {epoch + 1}/{args.epochs}  step {step}/{len(train_loader)}"
                      f"  loss {running / (step + 1):.4f}")
        print(f"epoch {epoch + 1} done  mean loss {running / len(train_loader):.4f}"
              f"  ({time.time() - t0:.0f}s elapsed)")

    @torch.no_grad()
    def probs_for(rows):
        model.eval()
        out = []
        for i in range(0, len(rows), 64):
            chunk = [r["text"] for r in rows[i:i + 64]]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=args.max_len, return_tensors="pt").to(device)
            out.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
        return np.vstack(out)

    print("tuning thresholds on validation split...")
    p_va = probs_for(va_rows)
    thresholds, hedge_thresholds = {}, {}
    for j, cat in enumerate(categories):
        y = np.array([1 if cat in r["categories"] else 0 for r in va_rows])
        thresholds[cat] = tune_threshold(y, p_va[:, j], cat in RECALL_FIRST)
        hedge_thresholds[cat] = tune_hedge_threshold(va_rows, p_va[:, j], cat)
        print(f"  {cat:<28} thr={thresholds[cat]:.2f}  hedge={hedge_thresholds[cat]:.2f}")

    print("evaluating on held-out test split...")
    report = evaluate(test_rows, categories, probs_for(test_rows),
                      thresholds, hedge_thresholds)
    report["base_model"] = args.model
    report["epochs"] = args.epochs
    report["train_size"] = len(tr_rows)
    report["train_seconds"] = round(time.time() - t0, 1)
    report["device"] = device

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "sahayk_meta.json").write_text(json.dumps({
        "version": 1,
        "kind": "transformer",
        "base_model": args.model,
        "categories": categories,
        "thresholds": thresholds,
        "hedge_thresholds": hedge_thresholds,
        "max_len": args.max_len,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "Synthetic corpus, rubric-derived labels. Not clinically validated.",
    }, indent=2), encoding="utf-8")

    (here / "transformer_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + json.dumps({k: v for k, v in report.items()
                             if k != "categories"}, indent=2))
    print(f"\nwrote {out}")
    print("Compare against metrics.json (the shipping TF-IDF model). The number "
          "worth checking first is negation_handling.false_positive_rate_model — "
          "that is the weakness this model exists to fix.")


if __name__ == "__main__":
    main()
