"""
behavior_monitor.py — Delta-based runtime monitor (isolated VM only)
"""
import os
import time
import shutil
import tempfile
import logging
import subprocess
import psutil

logging.basicConfig(filename="pipeline.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

SUSPICIOUS_KEYWORDS = [
    "keylog", "ransom", "stealer", "inject", "crypt", "miner",
    "backdoor", "trojan", "worm", "spy", "logger", "payload"
]
SUSPICIOUS_PORTS = {21, 22, 23, 25, 445, 1433, 3306, 3389, 4444, 5555, 6667, 8080, 8443}
DROPPED_EXTS = (".exe", ".dll", ".sys", ".js", ".vbs", ".ps1", ".bat", ".cmd", ".scr")


class SandboxEnvironment:
    def __init__(self, watch_dir: str = None):
        self.temp_dir = tempfile.mkdtemp(prefix="malxray_sandbox_")
        self.watch_dir = watch_dir or self.temp_dir

    def clean_up(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _snap_procs(self):
        snap = {}
        for p in psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
            try:
                snap[p.info["pid"]] = p.info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return snap

    def _snap_conns(self):
        conns = set()
        try:
            for c in psutil.net_connections(kind="inet"):
                if c.raddr:
                    conns.add((c.laddr.port if c.laddr else None,
                               c.raddr.ip, c.raddr.port, c.status))
        except (psutil.AccessDenied, PermissionError):
            pass
        return conns

    def _snap_files(self):
        files = set()
        for root, _, names in os.walk(self.watch_dir):
            for n in names:
                full = os.path.join(root, n)
                try:
                    files.add((full, os.path.getmtime(full)))
                except OSError:
                    continue
        return files

    def execute_and_monitor(self, sample_path: str, timeout: int = 20,
                            auto_execute: bool = False) -> dict:
        if not auto_execute:
            return {"Error": "auto_execute=False – refusing to run sample."}

        before_p = self._snap_procs()
        before_c = self._snap_conns()
        before_f = self._snap_files()

        proc = None
        try:
            proc = subprocess.Popen(
                [sample_path], cwd=self.watch_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(timeout)
        except Exception as e:
            return {"Error": f"Execution failed: {e}"}
        finally:
            if proc and proc.poll() is None:
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except Exception:
                    proc.terminate()

        return self._diff(before_p, before_c, before_f)

    def _diff(self, before_p, before_c, before_f) -> dict:
        after_p = self._snap_procs()
        after_c = self._snap_conns()
        after_f = self._snap_files()

        new_procs = []
        for pid in set(after_p) - set(before_p):
            info = after_p[pid]
            name = (info.get("name") or "").lower()
            cmd = " ".join(info.get("cmdline") or []).lower()
            entry = {
                "PID": pid,
                "Name": info.get("name"),
                "Command Line": " ".join(info.get("cmdline") or []),
                "Parent PID": info.get("ppid"),
            }
            if any(k in name or k in cmd for k in SUSPICIOUS_KEYWORDS):
                entry["Flag"] = "Suspicious name/cmdline"
            new_procs.append(entry)

        new_conns = []
        for lport, rip, rport, status in after_c - before_c:
            entry = {"Remote IP": rip, "Remote Port": rport, "Status": status}
            if rport in SUSPICIOUS_PORTS:
                entry["Flag"] = f"Suspicious port {rport}"
            new_conns.append(entry)

        dropped = []
        for path, _ in after_f - before_f:
            entry = {"Path": path}
            if path.lower().endswith(DROPPED_EXTS):
                entry["Flag"] = "Dropped executable/script"
            dropped.append(entry)

        return {
            "New Processes": new_procs,
            "New Network Connections": new_conns,
            "Dropped Files": dropped,
            "Summary": {
                "num_new_processes": len(new_procs),
                "num_new_connections": len(new_conns),
                "num_dropped_files": len(dropped),
                "num_suspicious_process_flags": sum(1 for p in new_procs if "Flag" in p),
                "num_suspicious_connection_flags": sum(1 for c in new_conns if "Flag" in c),
                "num_suspicious_file_flags": sum(1 for f in dropped if "Flag" in f),
            }
        }