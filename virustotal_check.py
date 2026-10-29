"""
virustotal_check.py — MSc revision (fixed)
"""

import os
import requests
import time
import logging

logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Correct way: read from environment variable
API_KEY = os.environ.get("cae9eb4012ea974676832655bc8461fbb0da9439bd1e56d3a977393e24a21882")  # GUI Settings can also set this at runtime
BASE_URL = "https://www.virustotal.com/api/v3/files/"


def check_virustotal(hash_value: str) -> dict:
    """Secondary / validation signal only. Does not decide the verdict."""
    if not API_KEY:
        return {
            "Error": "VIRUSTOTAL_API_KEY environment variable not set. "
                     "VT check skipped; ML classifier result still applies."
        }

    url = f"{BASE_URL}{hash_value}"
    headers = {"x-apikey": API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return parse_vt_response(response.json())
        elif response.status_code == 404:
            logging.info(f"Hash {hash_value} not found in VirusTotal.")
            return {"Error": "File not found in VirusTotal."}
        elif response.status_code == 403:
            return {"Error": "Access forbidden. Check API key or permissions."}
        elif response.status_code == 429:
            logging.warning("VT rate limit reached. Retrying in 15 s…")
            time.sleep(15)
            return check_virustotal(hash_value)
        else:
            return {"Error": f"Unhandled response: {response.status_code}"}
    except requests.RequestException as e:
        return {"Error": f"Network error: {e}"}


def parse_vt_response(vt_data: dict) -> dict:
    try:
        attributes = vt_data["data"]["attributes"]
        stats = attributes.get("last_analysis_stats", {})
        total_engines = sum(stats.values())
        malicious_count = stats.get("malicious", 0)
        suspicious_count = stats.get("suspicious", 0)

        threat = attributes.get("popular_threat_classification", {}) or {}
        malware_family = threat.get("suggested_threat_label", "Unknown")

        return {
            "Total AV Engines": total_engines,
            "Malicious Detections": malicious_count,
            "Suspicious Detections": suspicious_count,
            "Malware Family": malware_family,
            "Reputation Score": attributes.get("reputation", "Unknown"),
            "Scan Date": attributes.get("last_analysis_date", "N/A"),
        }
    except Exception as e:
        return {"Error": f"Failed to parse VT response: {e}"}


def upload_file_for_scan(file_path: str) -> dict:
    if not API_KEY:
        return {"Error": "VIRUSTOTAL_API_KEY environment variable not set."}

    url = "https://www.virustotal.com/api/v3/files"
    headers = {"x-apikey": API_KEY}
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files, headers=headers, timeout=60)
            if response.status_code == 200:
                logging.info("File uploaded to VirusTotal for scanning.")
                return response.json()
            return {"Error": f"Upload failed: {response.status_code}"}
    except Exception as e:
        return {"Error": f"Failed to upload file: {e}"}