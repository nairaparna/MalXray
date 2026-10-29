"""
static_features.py — Static (PE structural) feature extraction.

NEW MODULE for the MSc extension of MalXray.

The BSc version (file_info.py) only extracted hashes/MIME type/size.
This module fills the "Static Analysis Module" the architecture diagram
already names but never implemented: PE header fields, section entropy,
and imported API/DLL usage — the same feature families described in the
research paper (Section IV.4.2 / Discussion, Section V).

Output is a flat, fixed-schema dict of numeric/categorical features so it
can be handed straight to feature_fusion.py and then to the ML classifier.
Non-PE files (scripts, archives, docs) degrade gracefully to a zeroed
feature vector with an "is_pe" flag of 0, rather than raising.
"""

import math
import os
import logging

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False

logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed list of "interesting" imported APIs commonly associated with
# malicious behaviour (process injection, keylogging, persistence,
# anti-debugging, network C2). Presence/absence of these becomes a
# binary feature per API — a lightweight stand-in for a full API
# embedding, and easy to defend/explain in a viva.
SUSPICIOUS_APIS = [
    "VirtualAlloc", "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
    "SetWindowsHookEx", "GetAsyncKeyState", "RegSetValueEx", "RegCreateKeyEx",
    "InternetOpen", "InternetConnect", "HttpSendRequest", "URLDownloadToFile",
    "CreateMutex", "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "ShellExecute", "WinExec", "CryptEncrypt", "GetSystemDirectory",
    "AdjustTokenPrivileges", "OpenProcess",
]


def shannon_entropy(data: bytes) -> float:
    """Byte-level Shannon entropy of a buffer, 0.0 (uniform) to 8.0 (random/packed)."""
    if not data:
        return 0.0
    occurrences = [0] * 256
    for byte in data:
        occurrences[byte] += 1
    entropy = 0.0
    length = len(data)
    for count in occurrences:
        if count == 0:
            continue
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def _empty_feature_set() -> dict:
    """Zeroed schema returned for non-PE or unreadable files, so the
    fused feature vector always has a consistent shape."""
    features = {
        "is_pe": 0,
        "file_entropy": 0.0,
        "num_sections": 0,
        "size_of_code": 0,
        "size_of_headers": 0,
        "size_of_image": 0,
        "entry_point": 0,
        "num_imported_dlls": 0,
        "num_imported_functions": 0,
        "has_debug_dir": 0,
        "is_signed": 0,
        "num_resources": 0,
        "max_section_entropy": 0.0,
        "avg_section_entropy": 0.0,
        "min_section_entropy": 0.0,
        "timestamp": 0,
    }
    for api in SUSPICIOUS_APIS:
        features[f"api_{api}"] = 0
    return features


def extract_static_features(file_path: str, file_data: bytes = None) -> dict:
    """
    Extract a fixed-schema static feature dict for one file.

    Parameters
    ----------
    file_path : path to the sample on disk
    file_data : raw bytes, reused from file_info.py if already read
                (avoids reading the file twice)

    Returns
    -------
    dict of numeric/binary features, safe to pass to feature_fusion.py
    even when the file is not a valid PE.
    """
    features = _empty_feature_set()

    try:
        if file_data is None:
            with open(file_path, "rb") as f:
                file_data = f.read()

        features["file_entropy"] = shannon_entropy(file_data)

        if not PEFILE_AVAILABLE:
            logging.warning("pefile not installed; static PE features skipped.")
            return features

        try:
            pe = pefile.PE(data=file_data, fast_load=True)
        except pefile.PEFormatError:
            # Not a PE file (e.g. script, archive, document) — expected,
            # not an error condition.
            return features

        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE'],
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DEBUG'],
        ])

        features["is_pe"] = 1
        features["num_sections"] = len(pe.sections)
        features["size_of_code"] = getattr(pe.OPTIONAL_HEADER, "SizeOfCode", 0)
        features["size_of_headers"] = getattr(pe.OPTIONAL_HEADER, "SizeOfHeaders", 0)
        features["size_of_image"] = getattr(pe.OPTIONAL_HEADER, "SizeOfImage", 0)
        features["entry_point"] = getattr(pe.OPTIONAL_HEADER, "AddressOfEntryPoint", 0)
        features["timestamp"] = getattr(pe.FILE_HEADER, "TimeDateStamp", 0)

        # Section entropy — the classic packing/obfuscation signal the
        # paper's Discussion section refers to.
        section_entropies = []
        for section in pe.sections:
            try:
                section_entropies.append(section.get_entropy())
            except Exception:
                continue
        if section_entropies:
            features["max_section_entropy"] = round(max(section_entropies), 4)
            features["min_section_entropy"] = round(min(section_entropies), 4)
            features["avg_section_entropy"] = round(
                sum(section_entropies) / len(section_entropies), 4
            )

        # Imports
        imported_dlls = set()
        imported_funcs = set()
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode(errors="ignore") if entry.dll else ""
                imported_dlls.add(dll_name.lower())
                for imp in entry.imports:
                    if imp.name:
                        fname = imp.name.decode(errors="ignore")
                        imported_funcs.add(fname)
                        if fname in SUSPICIOUS_APIS:
                            features[f"api_{fname}"] = 1

        features["num_imported_dlls"] = len(imported_dlls)
        features["num_imported_functions"] = len(imported_funcs)

        features["has_debug_dir"] = int(hasattr(pe, "DIRECTORY_ENTRY_DEBUG") and
                                         len(pe.DIRECTORY_ENTRY_DEBUG) > 0)
        features["num_resources"] = (len(pe.DIRECTORY_ENTRY_RESOURCE.entries)
                                      if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE") else 0)

        # Authenticode signature presence (very rough: security directory populated)
        try:
            sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
                pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']
            ]
            features["is_signed"] = int(sec_dir.VirtualAddress != 0 and sec_dir.Size != 0)
        except Exception:
            features["is_signed"] = 0

        pe.close()
        logging.info(f"Static features extracted for: {file_path}")

    except Exception as e:
        logging.error(f"[Static Feature Extraction Error] {file_path}: {e}")

    return features


if __name__ == "__main__":
    import sys
    import json
    target = sys.argv[1] if len(sys.argv) > 1 else __file__
    print(json.dumps(extract_static_features(target), indent=2))