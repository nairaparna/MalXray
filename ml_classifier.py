"""
ml_classifier.py — Inference: classifies one fused feature dict using the
models trained by train_model.py.

MSc REVISION. What changed and why:

  1. Static-only fallback. If a sample has no dynamic data
     (dynamic_data_available == 0, i.e. it was never detonated in a
     sandbox), the fused models would be scoring it against a feature
     distribution they never saw. The classifier now switches to the
     static-only binary + family models and reports which mode it used
     ("fused" or "static_only") so the report can say so.
  2. Per-prediction feature contributions. The old version listed the
     model's global feature importances next to this sample's values,
     which does not explain THIS prediction. It now uses XGBoost's
     pred_contribs (per-sample, tree-SHAP-style contributions), so
     "top_features" are the features that actually pushed this verdict.
  3. Fails soft. Missing models -> a clear "Unavailable" result rather
     than a crash, so the CLI still works before training.

Returned dict keys:
  ml_available          bool
  mode                  "fused" | "static_only"
  verdict               "Malicious" | "Benign" | "Unavailable"
  malicious_confidence  float 0-1
  predicted_family      str            (Malicious only)
  family_confidence     float 0-1      (Malicious only)
  top_features          [(name, value)]        features that pushed the verdict most
  top_feature_contributions  [(name, contribution)]  signed; + = toward malicious
  reliability           "normal" | "low"  (low = not a PE; verdict unreliable)
  warnings              [str]   caveats the report should show
"""

import json
import logging
import os
import joblib
import numpy as np
import xgboost as xgb

from feature_fusion import DYNAMIC_FEATURE_KEYS, get_feature_order, to_vector

MODEL_DIR = os.environ.get("MALXRAY_MODEL_DIR", "models")
PATHS = {
    "binary_fused": os.path.join(MODEL_DIR, "binary_detector.joblib"),
    "binary_static": os.path.join(MODEL_DIR, "static_only_detector.joblib"),
    "family_fused": os.path.join(MODEL_DIR, "family_classifier.joblib"),
    "family_static": os.path.join(MODEL_DIR, "family_classifier_static.joblib"),
    "encoder": os.path.join(MODEL_DIR, "family_label_encoder.joblib"),
    "static_cols": os.path.join(MODEL_DIR, "static_feature_columns.json"),
}

logging.basicConfig(filename="pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

_cache = {}
_load_attempted = False


def _load_models():
    """Load whatever models exist. Missing individual models are tolerated;
    classify() decides what it can still do."""
    global _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    for key, path in PATHS.items():
        if not os.path.exists(path):
            logging.warning(f"Model file missing: {path}")
            continue
        try:
            _cache[key] = (json.load(open(path)) if path.endswith(".json")
                           else joblib.load(path))
        except Exception as e:
            logging.error(f"Could not load {path}: {e}")
    logging.info(f"ML models loaded: {sorted(_cache)}")


def _contributions(model, vector: np.ndarray, feature_names: list, top_n: int = 5):
    """Per-sample signed feature contributions (log-odds space) via XGBoost's
    built-in pred_contribs. Returns [(name, contribution)] sorted by |value|."""
    try:
        contribs = model.get_booster().predict(xgb.DMatrix(vector), pred_contribs=True)[0]
        contribs = contribs[:-1]  # last entry is the bias term
        idx = np.argsort(np.abs(contribs))[::-1][:top_n]
        return [(feature_names[i], round(float(contribs[i]), 4)) for i in idx
                if contribs[i] != 0]
    except Exception as e:
        logging.warning(f"pred_contribs failed: {e}")
        return []


def classify(fused_features: dict) -> dict:
    _load_models()

    dynamic_ok = int(fused_features.get("dynamic_data_available", 0)) == 1
    mode = "fused" if dynamic_ok else "static_only"
    warnings = []

    binary = _cache.get("binary_fused" if dynamic_ok else "binary_static")
    family = _cache.get("family_fused" if dynamic_ok else "family_static")
    encoder = _cache.get("encoder")

    # Fused requested but its model is missing -> fall back to static-only.
    if binary is None and dynamic_ok and "binary_static" in _cache:
        mode, dynamic_ok = "static_only", False
        binary, family = _cache["binary_static"], _cache.get("family_static")
        warnings.append("Fused model not found; used the static-only model.")

    if binary is None:
        return {
            "ml_available": False, "verdict": "Unavailable", "mode": mode,
            "note": "ML models not found. Run generate_training_data.py then "
                    "train_model.py (ideally on a real dataset such as EMBER "
                    "or BODMAS).",
        }

    # Column set must match what the chosen model was trained on.
    if mode == "static_only":
        cols = _cache.get("static_cols") or [c for c in get_feature_order()
                                              if c not in DYNAMIC_FEATURE_KEYS]
        vector = np.array([[fused_features.get(c, 0) for c in cols]])
        warnings.append("No dynamic (runtime) data for this sample: verdict is "
                        "based on static features only.")
    else:
        cols = get_feature_order()
        vector = np.array([to_vector(fused_features)])

    if not fused_features.get("is_pe", 0):
        warnings.append("Not a Windows PE file: PE-based features are empty, so "
                        "this verdict is unreliable (e.g. macro documents, "
                        "scripts and archives look like near-empty files).")

    proba = float(binary.predict_proba(vector)[0][1])
    verdict = "Malicious" if proba >= 0.5 else "Benign"
    result = {
        "ml_available": True, "mode": mode, "verdict": verdict,
        "malicious_confidence": round(proba, 4), "warnings": warnings,
        # "low" when PE features can't describe the file (docs, scripts, archives)
        "reliability": "normal" if fused_features.get("is_pe", 0) else "low",
    }

    contribs = _contributions(binary, vector, cols)
    if contribs:
        result["top_feature_contributions"] = contribs
        result["top_features"] = [(n, fused_features.get(n, 0)) for n, _ in contribs]

    if verdict == "Malicious" and family is not None and encoder is not None:
        fam_proba = family.predict_proba(vector)[0]
        i = int(np.argmax(fam_proba))
        result["predicted_family"] = str(encoder.inverse_transform([i])[0])
        result["family_confidence"] = round(float(fam_proba[i]), 4)
    elif verdict == "Malicious":
        warnings.append("Family model unavailable; no family prediction.")

    return result


if __name__ == "__main__":
    import sys
    from feature_fusion import fuse_features

    target = sys.argv[1] if len(sys.argv) > 1 else __file__
    print(json.dumps(classify(fuse_features(target)), indent=2, default=str))