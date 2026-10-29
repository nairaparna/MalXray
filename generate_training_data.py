"""
generate_training_data.py — Builds the training CSV for train_model.py.

MSc REVISION: now supports THREE modes.

  synthetic  Made-up statistics per family. Only proves the ML pipeline
             runs. NOT real malware data; never quote metrics from it.
  real       Static features extracted from actual samples you place in
             samples/benign/ and samples/malware/ (see below).
  both       Real + synthetic rows in one CSV, tagged in a `source`
             column. train_model.py uses this to (a) train on synthetic
             and (b) test how that model behaves on your real samples.

REAL-SAMPLE FOLDER LAYOUT
    samples/
      benign/    any files you know are clean (e.g. copies of Windows
                 system .exe files)                        -> label 0
      malware/   sample files or password-protected .zip archives
                 (MalwareBazaar-style zips, password "infected")
                                                             -> label 1
      labels.csv optional; columns: filename,family
                 (filename = the file/zip name inside malware/).
                 Unlisted malware gets family "unknown" and is used for
                 the binary detector only, never the family classifier.

SAFETY: samples are only READ. Zips are opened in memory and the bytes go
to static_features.py (pefile parsing + entropy). Nothing is executed and
nothing is written to disk. Dynamic features are left at 0 with
dynamic_data_available=0, because no sample was detonated.

For your Results chapter you still need a far larger real corpus
(EMBER, BODMAS, or a MalwareBazaar + VirusTotal-labelled set). Two or
three samples cannot train or evaluate a classifier.
"""

import argparse
import hashlib
import logging
import os

import numpy as np
import pandas as pd

from feature_fusion import get_feature_order
from static_features import extract_static_features

try:
    import pyzipper  # reads both AES and legacy-encrypted zips
except ImportError:
    pyzipper = None

FAMILIES = ["benign", "trojan", "ransomware", "spyware", "worm"]
N_PER_CLASS = 400
RANDOM_SEED = 42

# Conventional password for malware-sample archives (MalwareBazaar etc.)
DEFAULT_ZIP_PASSWORD = b"infected"
MAX_MEMBER_BYTES = 200 * 1024 * 1024  # guard against zip bombs

def _sample_family(family: str, feature_order: list, rng: np.random.Generator) -> dict:
    """Draw one synthetic feature row with class-dependent distributions,
    loosely motivated by real malware characteristics described in the
    literature review (higher entropy = packing, more suspicious API
    flags for trojans/spyware, more dropped files for worms, etc.)."""
    row = {}

    is_malicious = family != "benign"
    row["is_pe"] = 1
    row["file_entropy"] = rng.normal(5.0 if not is_malicious else 6.8,
                                      0.6 if not is_malicious else 0.9)
    row["max_section_entropy"] = rng.normal(4.5 if not is_malicious else 7.2, 0.8)
    row["avg_section_entropy"] = row["max_section_entropy"] * rng.uniform(0.5, 0.8)
    row["min_section_entropy"] = row["avg_section_entropy"] * rng.uniform(0.1, 0.4)
    row["num_sections"] = rng.integers(3, 7) if not is_malicious else rng.integers(4, 12)
    row["size_of_code"] = int(abs(rng.normal(50_000, 20_000)))
    row["size_of_headers"] = int(abs(rng.normal(1024, 200)))
    row["size_of_image"] = int(abs(rng.normal(200_000, 80_000)))
    row["entry_point"] = int(abs(rng.normal(4096, 1000)))
    row["timestamp"] = int(abs(rng.normal(1_650_000_000, 50_000_000)))
    row["num_imported_dlls"] = rng.integers(2, 10)
    row["num_imported_functions"] = rng.integers(10, 120)
    row["has_debug_dir"] = rng.integers(0, 2)
    row["is_signed"] = 1 if not is_malicious and rng.random() > 0.3 else 0
    row["num_resources"] = rng.integers(0, 15)

    # Family-specific API flag tendencies
    api_bias = {
        "benign": [],
        "trojan": ["api_InternetOpen", "api_HttpSendRequest", "api_ShellExecute", "api_CreateMutex"],
        "ransomware": ["api_CryptEncrypt", "api_RegSetValueEx", "api_OpenProcess"],
        "spyware": ["api_GetAsyncKeyState", "api_SetWindowsHookEx", "api_InternetConnect"],
        "worm": ["api_WinExec", "api_CreateRemoteThread", "api_VirtualAllocEx",
                 "api_WriteProcessMemory"],
    }
    for key in feature_order:
        if key.startswith("api_"):
            base_p = 0.03  # small false-flag rate even for benign
            if key in api_bias.get(family, []):
                base_p = 0.75
            row[key] = int(rng.random() < base_p)

    # Dynamic features — malicious families show more spawned processes/
    # connections/dropped files; benign mostly quiet.
    row["dynamic_data_available"] = 1
    if not is_malicious:
        row["num_new_processes"] = rng.integers(0, 2)
        row["num_new_connections"] = rng.integers(0, 1)
        row["num_dropped_files"] = rng.integers(0, 1)
    else:
        row["num_new_processes"] = rng.integers(1, 6)
        row["num_new_connections"] = rng.integers(0, 5) if family in ("trojan", "spyware", "worm") else rng.integers(0, 2)
        row["num_dropped_files"] = rng.integers(0, 4) if family in ("worm", "ransomware") else rng.integers(0, 2)

    row["num_suspicious_process_flags"] = min(row["num_new_processes"], rng.integers(0, row["num_new_processes"] + 1))
    row["num_suspicious_connection_flags"] = min(row["num_new_connections"], rng.integers(0, row["num_new_connections"] + 1))
    row["num_suspicious_file_flags"] = min(row["num_dropped_files"], rng.integers(0, row["num_dropped_files"] + 1))

    # Fill any schema key not explicitly set above
    for key in feature_order:
        if key not in row:
            row[key] = 0

    return row



def generate_synthetic(n_per_class: int = N_PER_CLASS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feature_order = get_feature_order()
    rows = []
    for family in FAMILIES:
        for _ in range(n_per_class):
            row = _sample_family(family, feature_order, rng)
            row["family"] = family
            row["label"] = 0 if family == "benign" else 1
            row["source"] = "synthetic"
            row["sha256"] = ""
            rows.append(row)
    return pd.DataFrame(rows)


def _iter_sample_payloads(path: str, zip_password: bytes):
    """Yield (name, raw_bytes) for a sample path. Zips are read in memory
    only — nothing is extracted to disk or executed."""
    if path.lower().endswith(".zip"):
        if pyzipper is None:
            print(f"  ! pyzipper not installed, skipping {os.path.basename(path)} "
                  f"(pip install pyzipper)")
            return
        with pyzipper.AESZipFile(path) as zf:
            zf.setpassword(zip_password)
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.file_size > MAX_MEMBER_BYTES:
                    print(f"  ! skipping oversized zip member: {info.filename}")
                    continue
                yield info.filename, zf.read(info)
    else:
        with open(path, "rb") as f:
            yield os.path.basename(path), f.read()


def _load_family_labels(samples_dir: str) -> dict:
    labels_path = os.path.join(samples_dir, "labels.csv")
    if not os.path.exists(labels_path):
        return {}
    df = pd.read_csv(labels_path)
    return {str(r["filename"]).strip(): str(r["family"]).strip().lower()
            for _, r in df.iterrows()}


def build_real_dataset(samples_dir: str = "samples",
                        zip_password: bytes = DEFAULT_ZIP_PASSWORD) -> pd.DataFrame:
    """Static-feature rows for real samples. See module docstring for layout."""
    feature_order = get_feature_order()
    family_map = _load_family_labels(samples_dir)
    rows = []

    for folder_name, label in (("benign", 0), ("malware", 1)):
        folder = os.path.join(samples_dir, folder_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath) or fname == "labels.csv":
                continue
            try:
                for inner_name, data in _iter_sample_payloads(fpath, zip_password):
                    feats = extract_static_features(inner_name, file_data=data)
                    row = {k: feats.get(k, 0) for k in feature_order}
                    # No detonation happened, so dynamic features stay zero.
                    row["dynamic_data_available"] = 0
                    row["label"] = label
                    row["family"] = ("benign" if label == 0
                                      else family_map.get(fname, "unknown"))
                    row["source"] = "real"
                    row["sha256"] = hashlib.sha256(data).hexdigest()
                    rows.append(row)
                    print(f"  + {folder_name}/{fname} -> {inner_name[:24]}  "
                          f"(pe={feats['is_pe']}, entropy={feats['file_entropy']})")
            except Exception as e:
                logging.error(f"[Real sample load error] {fpath}: {e}")
                print(f"  ! could not read {fpath}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset="sha256")  # avoid train/test leakage
        if len(df) < before:
            print(f"  (dropped {before - len(df)} duplicate sample(s) by SHA-256)")
    return df


def generate_dataset(out_path: str = "training_data.csv", mode: str = "both",
                      samples_dir: str = "samples",
                      zip_password: bytes = DEFAULT_ZIP_PASSWORD) -> pd.DataFrame:
    frames = []
    if mode in ("synthetic", "both"):
        frames.append(generate_synthetic())
    if mode in ("real", "both"):
        print(f"Reading real samples from '{samples_dir}/' ...")
        real = build_real_dataset(samples_dir, zip_password)
        if real.empty:
            print("  No real samples found.")
        else:
            frames.append(real)
    if not frames:
        raise SystemExit("Nothing to write: no synthetic or real rows produced.")

    df = pd.concat(frames, ignore_index=True)
    ordered = get_feature_order() + ["label", "family", "source", "sha256"]
    df = df[ordered]
    df.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}: {len(df)} rows, {len(get_feature_order())} features")
    print(df.groupby(["source", "family"]).size().to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the MalXray training CSV.")
    ap.add_argument("--mode", choices=["synthetic", "real", "both"], default=None,
                    help="default: 'both' if the samples folder exists, else 'synthetic'")
    ap.add_argument("--samples-dir", default="samples")
    ap.add_argument("--out", default="training_data.csv")
    ap.add_argument("--zip-password", default="infected")
    args = ap.parse_args()

    mode = args.mode or ("both" if os.path.isdir(args.samples_dir) else "synthetic")
    generate_dataset(args.out, mode, args.samples_dir, args.zip_password.encode())