"""Data & Training Lead: preprocessing -> training -> ONNX export -> validation.

Usage: python train.py [--data data/SMSSpamCollection.txt] [--out models]
Exits non-zero if the accuracy SLA (>= 90%) is not met, so CI fails the build.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from features import FEATURES, msg_features, normalize

ACCURACY_SLA = 0.90


def load_and_validate(path: str) -> pd.DataFrame:
    """Cleaning, missing-value imputation and schema validation."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     dtype=str, keep_default_na=False)
    # Schema validation
    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns, got {df.shape[1]}")
    bad = set(df["label"].unique()) - {"ham", "spam"}
    if bad:
        raise ValueError(f"Unexpected labels: {bad}")
    # Missing-value handling: text imputed to "", then empty rows dropped
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""]
    # Cleaning: collapse whitespace, remove exact duplicates
    df["raw"] = df["text"]  # keep original casing for drift stats
    df["text"] = df["text"].map(normalize)
    df = df.drop_duplicates().reset_index(drop=True)
    df["target"] = (df["label"] == "spam").astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/SMSSpamCollection.txt")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    df = load_and_validate(args.data)
    print(f"Clean rows: {len(df)}  (spam: {df.target.sum()})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"])

    # TF-IDF applies L2 normalisation = feature scaling for text
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(classification_report(y_te, pipe.predict(X_te), target_names=["ham", "spam"]))
    print(f"Test accuracy: {acc:.4f} (SLA >= {ACCURACY_SLA})")

    # ---- ONNX export + validation ----
    onx = convert_sklearn(
        pipe, initial_types=[("message", StringTensorType([None, 1]))],
        options={id(pipe.steps[-1][1]): {"zipmap": False}}, target_opset=15)
    model_path = out / "spam_model.onnx"
    onnx.save(onx, model_path)
    onnx.checker.check_model(onnx.load(model_path))
    print("onnx.checker: OK")

    # Parity check: ONNX Runtime vs scikit-learn on the test set
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    sample = X_te.tolist()
    onnx_pred = sess.run(None, {"message": np.array(sample, dtype=object).reshape(-1, 1)})[0]
    match = float(np.mean(onnx_pred == pipe.predict(X_te)))
    print(f"ONNX/sklearn prediction agreement: {match:.4f}")
    if match < 0.99:
        sys.exit("ONNX parity check failed")

    # ---- Reference stats for drift monitoring ----
    feats = pd.DataFrame([msg_features(t) for t in df.loc[X_tr.index, "raw"]])
    ref = {f: {"mean": float(feats[f].mean()), "std": float(feats[f].std())} for f in FEATURES}
    (out / "reference_stats.json").write_text(json.dumps(ref, indent=2))
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))

    if acc < ACCURACY_SLA:
        sys.exit(f"Accuracy SLA violated: {acc:.4f} < {ACCURACY_SLA}")


if __name__ == "__main__":
    main()
