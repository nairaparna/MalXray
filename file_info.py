"""
file_info.py — improved archive handling + password support
"""

import hashlib
import json
import mimetypes
import os
import zipfile
import logging
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Common passwords used by malware sharing sites
COMMON_ZIP_PASSWORDS = [b"infected", b"malware", b"virus", b"password", b""]

try:
    from magic import Magic
    _magic = Magic(mime=True)
except ImportError:
    try:
        import magic as _magic_mod
        _magic = _magic_mod
    except ImportError:
        _magic = None

try:
    import pyzipper
    PYZIPPER_AVAILABLE = True
except ImportError:
    PYZIPPER_AVAILABLE = False


def detect_file_type(file_path: str) -> str:
    try:
        if _magic:
            if hasattr(_magic, "from_file"):
                return _magic.from_file(file_path) or "unknown"
            return _magic.from_file(file_path, mime=True) or "unknown"
        mime, _ = mimetypes.guess_type(file_path)
        return mime or "unknown"
    except Exception as e:
        logging.error(f"[File Type Detection] {file_path}: {e}")
        return "unknown"


def _try_extract_zip(file_path: str, output_dir: str) -> bool:
    """Try normal zip first, then password-protected with common passwords."""
    # 1. Try normal (unencrypted) zip
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(output_dir)
        logging.info(f"Extracted normal ZIP: {file_path}")
        return True
    except RuntimeError:
        pass  # encrypted
    except Exception as e:
        logging.debug(f"Normal zip failed: {e}")

    # 2. Try pyzipper with common passwords
    if not PYZIPPER_AVAILABLE:
        logging.warning("pyzipper not installed — cannot open password-protected zips")
        return False

    for pwd in COMMON_ZIP_PASSWORDS:
        try:
            with pyzipper.AESZipFile(file_path) as zf:
                zf.setpassword(pwd)
                zf.extractall(output_dir)
            logging.info(f"Extracted password-protected ZIP (pwd tried) → {file_path}")
            return True
        except Exception:
            continue

    logging.error(f"Could not extract archive (wrong password or unsupported): {file_path}")
    return False


def extract_archive(file_path: str, output_dir: str = "extracted_files") -> bool:
    os.makedirs(output_dir, exist_ok=True)
    lower = file_path.lower()

    if lower.endswith(".zip"):
        return _try_extract_zip(file_path, output_dir)

    # Fallback for rar / 7z (requires patool + external tools)
    try:
        import patoolib
        patoolib.extract_archive(file_path, outdir=output_dir, verbosity=-1)
        logging.info(f"Extracted with patoolib: {file_path}")
        return True
    except Exception as e:
        logging.error(f"[Archive Extraction] {e}")
        return False


def extract_file_info(file_path: str, output_dir: str = None) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Returns (file_bytes, info_dict).
    For archives, also populates "Extracted Contents".
    """
    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), "malxray_extract")

    try:
        with open(file_path, "rb") as f:
            file_data = f.read()

        mime_type = detect_file_type(file_path)
        file_info = {
            "File Path": file_path,
            "MD5": hashlib.md5(file_data).hexdigest(),
            "SHA-256": hashlib.sha256(file_data).hexdigest(),
            "File Type": mime_type,
            "Size (bytes)": len(file_data),
        }

        # Handle archives
        if mime_type in ("application/zip", "application/x-rar", "application/x-rar-compressed",
                         "application/x-7z-compressed") or file_path.lower().endswith((".zip", ".rar", ".7z")):
            if extract_archive(file_path, output_dir):
                extracted = {}
                for root, _, files in os.walk(output_dir):
                    for filename in files:
                        full = os.path.join(root, filename)
                        try:
                            with open(full, "rb") as ef:
                                data = ef.read()
                            extracted[filename] = {
                                "File Path": full,
                                "MD5": hashlib.md5(data).hexdigest(),
                                "SHA-256": hashlib.sha256(data).hexdigest(),
                                "File Type": detect_file_type(full),
                                "Size (bytes)": len(data),
                            }
                        except Exception as e:
                            extracted[filename] = {"Error": str(e)}
                file_info["Extracted Contents"] = extracted

        return file_data, file_info

    except Exception as e:
        logging.error(f"[File Info] {file_path}: {e}")
        return None, {"File Path": file_path, "Error": str(e)}


# Need this import at top if you use the default temp dir
import tempfile