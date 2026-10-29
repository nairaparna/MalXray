"""
feature_fusion.py — Combines static + dynamic features into one fixed-order
numeric vector for the ML classifier.

NEW MODULE. This is the "Feature Fusion and Preprocessing Module" your
architecture diagram (Figure 1) already names but the BSc code never
implemented — MalXray.py previously just concatenated dicts into a PDF,
it never built anything a model could consume.

Design choice: simple concatenation with a fixed feature order (not a
learned embedding). This is a deliberate, defensible simplicity for a
masters-level dissertation — it keeps every feature individually
interpretable for the "feature importance" discussion the paper's
Discussion section (Section V) promises, rather than hiding the fusion
inside an opaque neural encoder.
"""

from static_features import extract_static_features, _empty_feature_set as _empty_static
import logging

logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed dynamic feature schema — derived from behavior_monitor.py's
# "Summary" block plus a couple of binary flags. Keeping this list here
# (not inferred at runtime) is what makes the feature vector length
# stable across samples, which XGBoost/scikit-learn requires.
DYNAMIC_FEATURE_KEYS = [
    "num_new_processes",
    "num_new_connections",
    "num_dropped_files",
    "num_suspicious_process_flags",
    "num_suspicious_connection_flags",
    "num_suspicious_file_flags",
    "dynamic_data_available",  # 0 if dynamic analysis was skipped/unavailable
]


def _dynamic_features_from_behavior(behavior_data: dict) -> dict:
    """Reduce behavior_monitor.py's delta output to the fixed dynamic schema.
    Returns zeros with dynamic_data_available=0 if no usable data was given
    (e.g. dynamic analysis was skipped because no isolated sandbox was
    available) — this lets the fused model still run in "static-only mode"
    gracefully, which matters for the static-vs-dynamic-vs-fused comparison
    the research paper's evaluation needs."""
    features = {k: 0 for k in DYNAMIC_FEATURE_KEYS}
    if not behavior_data or "Summary" not in behavior_data:
        return features

    summary = behavior_data["Summary"]
    for key in DYNAMIC_FEATURE_KEYS:
        if key == "dynamic_data_available":
            continue
        features[key] = summary.get(key, 0)
    features["dynamic_data_available"] = 1
    return features


def get_feature_order() -> list:
    """Canonical, stable ordering of every feature name. Must be identical
    between training (train_model.py) and inference (ml_classifier.py),
    or the model will silently misread the vector."""
    static_keys = sorted(_empty_static().keys())
    return static_keys + DYNAMIC_FEATURE_KEYS


def fuse_features(file_path: str, file_data: bytes = None,
                   behavior_data: dict = None) -> dict:
    """Build the full fused feature dict for one sample."""
    static = extract_static_features(file_path, file_data=file_data)
    dynamic = _dynamic_features_from_behavior(behavior_data)
    fused = {**static, **dynamic}
    logging.info(f"Fused feature vector built for: {file_path} ({len(fused)} features)")
    return fused


def to_vector(fused_features: dict) -> list:
    """Convert the fused dict into an ordered numeric list matching
    get_feature_order(), for direct use with sklearn/XGBoost."""
    order = get_feature_order()
    return [fused_features.get(k, 0) for k in order]


if __name__ == "__main__":
    import sys
    import json
    target = sys.argv[1] if len(sys.argv) > 1 else __file__
    fused = fuse_features(target)
    print(json.dumps(fused, indent=2))
    print(f"\nVector length: {len(to_vector(fused))}")