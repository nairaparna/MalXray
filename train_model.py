"""
train_model.py — Trains and evaluates the XGBoost models for MalXray.

MSc REVISION. What changed and why:

  1. Data-sufficiency guards. The old script assumed hundreds of rows per
     class and crashed on tiny datasets. It now checks class counts and
     skips (with an explanation) any model it cannot train honestly.
  2. Real vs synthetic awareness (uses the `source` column written by
     generate_training_data.py):
       - >= MIN_REAL_SAMPLES_FOR_TRAINING real rows  -> train/evaluate on
         REAL data only. This is the path for your dissertation numbers.
       - fewer real rows -> train on SYNTHETIC data (pipeline test only)
         and keep the real rows OUT of training, using them purely as a
         transfer check: "what does the synthetic-trained model say about
         actual malware?" Real rows are never in the training set, so
         there is no leakage.
  3. The transfer check uses the STATIC-ONLY model, because real samples
     that were only statically analysed have no dynamic features
     (dynamic_data_available = 0). Feeding them to the fused model would
     compare against a distribution it never saw.
  4. The static-only model is saved to models/ so ml_classifier.py can use
     it whenever a sample has no dynamic data.
  5. Family classifier ignores samples whose family is "unknown".

Outputs (results/): classification reports, confusion matrices, feature
importance plots, static-vs-dynamic-vs-fused table, real_sample_check.json
Outputs (models/): binary_detector, family_classifier, family_label_encoder,
                    static_only_detector, static_feature_columns.json
"""

import argparse
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_fusion import DYNAMIC_FEATURE_KEYS, get_feature_order

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = "training_data.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

MIN_REAL_SAMPLES_FOR_TRAINING = 200   # below this, real rows are a check set only
MIN_PER_CLASS = 20                    # per class, needed for a hold-out split


class InsufficientData(Exception):
    """Raised when a model can't be trained/evaluated honestly."""


# ---------------------------------------------------------------- helpers

def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _feature_groups():
    all_cols = get_feature_order()
    dynamic = [c for c in all_cols if c in DYNAMIC_FEATURE_KEYS]  # canonical list
    static = [c for c in all_cols if c not in dynamic]
    return static, dynamic, all_cols


def _require_classes(y, name, min_count=MIN_PER_CLASS):
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < min_count:
        raise InsufficientData(
            f"{name}: needs >= 2 classes with >= {min_count} samples each "
            f"to hold out a test set; got {counts.to_dict()}."
        )


def _plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(model, feature_names, title, out_path, top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([feature_names[i] for i in idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------- data selection

def select_training_data(df: pd.DataFrame):
    """Decide what to train on and what (if anything) to hold out as a
    real-sample check. Returns (train_df, holdout_real_df, mode_label)."""
    if "source" not in df.columns:
        df = df.assign(source="synthetic")

    real = df[df["source"] == "real"]
    synth = df[df["source"] == "synthetic"]

    if len(real) >= MIN_REAL_SAMPLES_FOR_TRAINING:
        print(f"\n[data] {len(real)} real samples -> training and evaluating on "
              f"REAL data only (synthetic rows ignored).")
        return real, None, "real"

    if len(synth) > 0:
        print(f"\n[data] Only {len(real)} real sample(s) (< "
              f"{MIN_REAL_SAMPLES_FOR_TRAINING} needed to train).")
        print("[data] Training on SYNTHETIC data. Metrics below prove the "
              "pipeline runs; they are NOT real detection performance.")
        if len(real):
            print("[data] Real samples are held OUT of training and used only "
                  "as a transfer check at the end.")
        return synth, (real if len(real) else None), "synthetic"

    raise InsufficientData(
        f"Only {len(real)} real sample(s) and no synthetic rows. Add more "
        f"samples (>= {MIN_REAL_SAMPLES_FOR_TRAINING}) or run "
        f"generate_training_data.py --mode both."
    )


# ---------------------------------------------------------------- models

def train_binary_detector(df: pd.DataFrame, feature_cols: list):
    y = df["label"].values
    _require_classes(y, "Binary detector")
    X = df[feature_cols].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    names = ["benign", "malicious"]
    report = classification_report(y_test, preds, target_names=names, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    _plot_confusion_matrix(cm, names, "Binary Detection — Confusion Matrix",
                            f"{RESULTS_DIR}/binary_confusion_matrix.png")
    _plot_feature_importance(model, feature_cols,
                              "Binary Detector — Top Feature Importances",
                              f"{RESULTS_DIR}/binary_feature_importance.png")
    joblib.dump(model, f"{MODEL_DIR}/binary_detector.joblib")
    with open(f"{RESULTS_DIR}/binary_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Binary Detector (benign vs malicious) ===")
    print(classification_report(y_test, preds, target_names=names))
    return model


def train_family_classifier(df: pd.DataFrame, feature_cols: list):
    mal = df[(df["label"] == 1) & (df["family"] != "unknown")].copy()
    y_raw = mal["family"].values
    _require_classes(y_raw, "Family classifier (known families only)")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    X = mal[feature_cols].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    labels = list(encoder.classes_)
    report = classification_report(y_test, preds, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    _plot_confusion_matrix(cm, labels, "Family Classification — Confusion Matrix",
                            f"{RESULTS_DIR}/family_confusion_matrix.png")
    _plot_feature_importance(model, feature_cols,
                              "Family Classifier — Top Feature Importances",
                              f"{RESULTS_DIR}/family_feature_importance.png")
    joblib.dump(model, f"{MODEL_DIR}/family_classifier.joblib")
    joblib.dump(encoder, f"{MODEL_DIR}/family_label_encoder.joblib")

    # Static-only twin, used by ml_classifier.py when a sample has no
    # dynamic data (same split, static columns only).
    static_cols, _, _ = _feature_groups()
    Xs_train, _, ys_train, _ = train_test_split(
        mal[static_cols].values, y, test_size=0.25, random_state=42, stratify=y)
    static_family = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       eval_metric="mlogloss", random_state=42)
    static_family.fit(Xs_train, ys_train)
    joblib.dump(static_family, f"{MODEL_DIR}/family_classifier_static.joblib")
    with open(f"{RESULTS_DIR}/family_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Family Classifier (malicious samples, known families) ===")
    print(classification_report(y_test, preds, target_names=labels))
    return model


def compare_static_dynamic_fused(df: pd.DataFrame):
    """Static-only vs dynamic-only vs fused on the binary task.
    Also exports the static-only model. Returns that model (or None)."""
    static_cols, dynamic_cols, all_cols = _feature_groups()
    y = df["label"].values
    _require_classes(y, "Static/dynamic/fused comparison")

    has_dynamic = df["dynamic_data_available"].sum() > 0
    if has_dynamic:
        sets = [("static_only", static_cols), ("dynamic_only", dynamic_cols),
                ("fused", all_cols)]
    else:
        print("\n[compare] No sample has dynamic data (nothing was detonated), so "
              "dynamic-only and fused comparisons are skipped — they would be "
              "meaningless. Static-only model will still be trained.")
        sets = [("static_only", static_cols)]

    results, static_model = {}, None
    for name, cols in sets:
        X = df[cols].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
        model = xgb.XGBClassifier(n_estimators=150, max_depth=5,
                                   eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted")
        results[name] = {"accuracy": round(accuracy_score(y_test, preds), 4),
                         "precision": round(p, 4), "recall": round(r, 4),
                         "f1": round(f1, 4), "num_features": len(cols)}
        if name == "static_only":
            static_model = model

    joblib.dump(static_model, f"{MODEL_DIR}/static_only_detector.joblib")
    with open(f"{MODEL_DIR}/static_feature_columns.json", "w") as f:
        json.dump(static_cols, f)
    with open(f"{RESULTS_DIR}/static_vs_dynamic_vs_fused.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Static vs Dynamic vs Fused Comparison ===")
    print(pd.DataFrame(results).T)
    return static_model


# ------------------------------------------------- real-sample transfer

def real_sample_check(holdout: pd.DataFrame, static_model):
    """Score real samples (never seen in training) with the static-only model."""
    static_cols, _, _ = _feature_groups()
    proba = static_model.predict_proba(holdout[static_cols].values)[:, 1]

    rows = []
    for (_, r), p in zip(holdout.iterrows(), proba):
        rows.append({
            "sha256": r["sha256"][:16],
            "true_label": "malicious" if r["label"] == 1 else "benign",
            "family": r["family"],
            "is_pe": int(r["is_pe"]),
            "file_entropy": round(float(r["file_entropy"]), 2),
            "p_malicious": round(float(p), 3),
            "predicted": "malicious" if p >= 0.5 else "benign",
        })
    out = pd.DataFrame(rows)
    correct = int((out["true_label"] == out["predicted"]).sum())

    with open(f"{RESULTS_DIR}/real_sample_check.json", "w") as f:
        json.dump({"note": "Synthetic-trained static-only model scored on real "
                           "samples held out of training. A handful of samples "
                           "is an anecdote, not an evaluation.",
                   "correct": correct, "total": len(out),
                   "rows": rows}, f, indent=2)

    print("\n=== Real-sample transfer check (static-only model, trained on synthetic) ===")
    print(out.to_string(index=False))
    print(f"\n{correct}/{len(out)} correct. With this few samples this is an "
          f"anecdote, not an evaluation — but a miss here is a real signal that "
          f"synthetic data does not represent real malware.")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train MalXray XGBoost models.")
    ap.add_argument("--data", default=CSV_PATH, help="CSV from generate_training_data.py")
    args = ap.parse_args()

    _ensure_dirs()
    df = pd.read_csv(args.data)
    _, _, feature_cols = _feature_groups()

    try:
        train_df, holdout_real, mode = select_training_data(df)
    except InsufficientData as e:
        raise SystemExit(f"\n[stop] {e}")

    trained = []
    for label, fn in [("binary detector", lambda: train_binary_detector(train_df, feature_cols)),
                      ("family classifier", lambda: train_family_classifier(train_df, feature_cols))]:
        try:
            fn()
            trained.append(label)
        except InsufficientData as e:
            print(f"\n[skip] {label}: {e}")

    static_model = None
    try:
        static_model = compare_static_dynamic_fused(train_df)
        trained.append("static/dynamic/fused comparison")
    except InsufficientData as e:
        print(f"\n[skip] comparison: {e}")

    if holdout_real is not None and static_model is not None:
        real_sample_check(holdout_real, static_model)

    print(f"\nTrained on: {mode.upper()} data. Completed: {', '.join(trained) or 'nothing'}.")
    print(f"Models -> '{MODEL_DIR}/', results -> '{RESULTS_DIR}/'.")