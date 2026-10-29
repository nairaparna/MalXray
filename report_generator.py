"""
report_generator.py — PDF report for a MalXray analysis.

MSc REVISION. Same entry point as before:

    generate_report(file_data, behavior_data, virustotal_data, ml_result,
                    verdict, report_path, highlights=None)

What changed and why:
  * Reports what the new pipeline actually produces: the ML verdict, mode
    (fused / static-only), confidence, family estimate, and the features
    that pushed THIS verdict (signed contributions from ml_classifier).
  * VirusTotal is shown as a secondary cross-check, with an explicit
    agrees / disagrees / not available line. VT errors (no API key, hash
    not found) are shown as "not available", not as findings.
  * Behaviour section reads the delta-monitor schema (new processes,
    connections, dropped files) and says plainly when the sample was NOT
    executed, instead of printing an empty section.
  * Honest caveats: a "Benign" verdict on a non-PE file (macro documents,
    scripts, archives) is flagged as unreliable, and Benign is never
    described as "safe".
  * Layout fixes: emoji removed (Helvetica draws them as black boxes),
    verdict colours cover Unknown/Safe, only flagged lines are red, and
    reportlab Platypus handles wrapping/pagination so long hashes and
    command lines can't run off the page. Every page has a footer.
"""

import os
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

MAX_LIST_ROWS = 25          # cap long process/connection/file tables
MAX_CMD_CHARS = 180

VERDICT_COLORS = {
    "Malicious": colors.HexColor("#c0392b"),
    "Potentially Harmful": colors.HexColor("#e67e22"),
    "Benign": colors.HexColor("#27ae60"),
    "Unknown": colors.HexColor("#7f8c8d"),
}
RED = colors.HexColor("#c0392b")
GREY = colors.HexColor("#555555")

_base = getSampleStyleSheet()
S_TITLE = ParagraphStyle("t", parent=_base["Title"], fontSize=24, spaceAfter=6)
S_H1 = ParagraphStyle("h1", parent=_base["Heading2"], fontSize=14, spaceBefore=14,
                      spaceAfter=4, textColor=colors.HexColor("#1f2d3d"), keepWithNext=1)
S_BODY = ParagraphStyle("b", parent=_base["BodyText"], fontSize=10, leading=13)
S_NOTE = ParagraphStyle("n", parent=S_BODY, fontSize=8.5, leading=11,
                        textColor=GREY, fontName="Helvetica-Oblique")
S_SUB = ParagraphStyle("sub", parent=S_BODY, fontName="Helvetica-Bold", keepWithNext=1)
S_NOTE_KEEP = ParagraphStyle("nk", parent=S_NOTE, keepWithNext=1)
S_CELL = ParagraphStyle("c", parent=S_BODY, fontSize=9, leading=11)
S_MONO = ParagraphStyle("m", parent=S_CELL, fontName="Courier", fontSize=8)
S_CENTER = ParagraphStyle("ctr", parent=S_BODY, alignment=1)
S_BANNER = ParagraphStyle("ban", parent=S_BODY, alignment=1, fontSize=15,
                          fontName="Helvetica-Bold", textColor=colors.white, leading=20)


# ---------------------------------------------------------------- helpers

def _clean(text) -> str:
    """Drop characters Helvetica can't draw (emoji etc.) rather than show boxes."""
    return str(text).encode("cp1252", "ignore").decode("cp1252")


def _esc(text) -> str:
    return escape(_clean(text))


def _p(text, style=S_CELL, color=None, bold=False):
    t = _esc(text)
    if bold:
        t = f"<b>{t}</b>"
    if color is not None:
        t = f'<font color="{color.hexval().replace("0x", "#")}">{t}</font>'
    return Paragraph(t, style)


def _table(rows, col_widths, header=False, flag_col=None):
    """rows: list of lists of str/Paragraph. Strings become Paragraphs."""
    data = [[c if not isinstance(c, str) else _p(c, bold=(header and i == 0))
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd2")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f4")))
    else:
        style.append(("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef1f4")))
    t.setStyle(TableStyle(style))
    return t


def _kv(rows, widths=(1.7 * inch, 4.8 * inch)):
    return _table(rows, list(widths))


def _normalise_verdict(verdict: str) -> str:
    v = (verdict or "Unknown").strip()
    return "Benign" if v.lower() in ("safe", "benign") else v


def _fmt_bytes(n):
    return f"{n:,} bytes ({round(n / 1024, 2)} KB)"


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(letter[0] / 2, 0.45 * inch,
                             f"MalXray (MSc build)  |  TLP: WHITE  |  Page {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------- sections

def _vt_agreement(ml_result, vt):
    """One-line comparison of the model's verdict with VirusTotal's."""
    if not vt or "Error" in vt:
        return "Not available"
    positives = vt.get("Malicious Detections", 0)
    ml_mal = (ml_result or {}).get("verdict") == "Malicious"
    if not (ml_result or {}).get("ml_available"):
        return "n/a (no model verdict to compare)"
    if ml_mal and positives > 0:
        return f"Agrees ({positives} engines flagged the file)"
    if not ml_mal and positives == 0:
        return "Agrees (no engines flagged the file)"
    if ml_mal and positives == 0:
        return "Disagrees: model says malicious, no VirusTotal engine flagged it"
    return f"Disagrees: model says benign, {positives} VirusTotal engine(s) flagged it"


def _summary_section(verdict, ml_result, vt, highlights, story):
    ml = ml_result or {}
    if ml.get("ml_available"):
        basis = ("ML classifier, fused static + runtime features" if ml.get("mode") == "fused"
                 else "ML classifier, static features only (sample was not executed)")
    else:
        basis = "ML model unavailable"

    rows = [["Verdict", verdict],
            ["Assessment basis", basis]]
    if ml.get("ml_available"):
        prob = f"{ml.get('malicious_confidence', 0):.1%}"
        if ml.get("reliability") == "low":
            prob += "  (unreliable: not a Windows PE file)"
        rows.append(["Malicious probability", prob])
        if ml.get("predicted_family"):
            rows.append(["Family (model estimate)",
                         f"{ml['predicted_family']} ({ml.get('family_confidence', 0):.1%})"])
    rows.append(["VirusTotal cross-check", _vt_agreement(ml, vt)])
    story.append(_kv(rows))

    msg = (highlights or {}).get("Verdict")
    if msg:
        story += [Spacer(1, 6), _p(msg, S_BODY)]


def _file_section(file_data, story):
    story.append(_p("File information", S_H1))
    rows = []
    if "Size (bytes)" in file_data:
        rows.append(["File size", _fmt_bytes(file_data["Size (bytes)"])])
    if "File Type" in file_data:
        rows.append(["File type", str(file_data["File Type"])])
    if "SHA-256" in file_data:
        rows.append(["SHA-256", Paragraph(_esc(file_data["SHA-256"]), S_MONO)])
    if "MD5" in file_data:
        rows.append(["MD5 (checksum only)", Paragraph(_esc(file_data["MD5"]), S_MONO)])
    story.append(_kv(rows) if rows else _p("No file information available.", S_NOTE))

    extracted = file_data.get("Extracted Contents")
    if extracted:
        story += [Spacer(1, 6), _p("Files extracted from the archive", S_SUB)]
        er = [["Name", "Type", "Size", "SHA-256"]]
        for name, info in list(extracted.items())[:MAX_LIST_ROWS]:
            if "Error" in info:
                er.append([name, "error", "", info["Error"]])
            else:
                er.append([name, info.get("File Type", ""),
                           f"{info.get('Size (bytes)', 0):,} B",
                           Paragraph(_esc(info.get("SHA-256", "")), S_MONO)])
        story.append(_table(er, [1.3 * inch, 1.3 * inch, 0.8 * inch, 3.1 * inch], header=True))
        if len(extracted) > MAX_LIST_ROWS:
            story.append(_p(f"... and {len(extracted) - MAX_LIST_ROWS} more.", S_NOTE))


def _ml_section(ml, story):
    story.append(_p("ML classification", S_H1))
    story.append(_p("Verdict and family estimate from the trained XGBoost models. "
                    "The feature table shows what pushed this specific verdict.", S_NOTE_KEEP))
    if not ml or not ml.get("ml_available"):
        story.append(_p((ml or {}).get("note", "ML model unavailable."), S_BODY))
        return

    rows = [["Analysis mode", "Fused (static + runtime)" if ml["mode"] == "fused"
             else "Static only"],
            ["Verdict", ml["verdict"]],
            ["Malicious probability", f"{ml.get('malicious_confidence', 0):.1%}"]]
    if ml.get("predicted_family"):
        rows += [["Predicted family", ml["predicted_family"]],
                 ["Family confidence", f"{ml.get('family_confidence', 0):.1%}"]]
    story.append(_kv(rows))

    contribs = ml.get("top_feature_contributions") or []
    values = dict(ml.get("top_features") or [])
    if contribs:
        story += [Spacer(1, 8), _p("Features that most influenced this verdict", S_SUB)]
        fr = [["Feature", "Value", "Effect", "Contribution"]]
        for name, c in contribs:
            toward = "toward malicious" if c > 0 else "toward benign"
            fr.append([name, str(values.get(name, "")),
                       _p(toward, color=RED if c > 0 else GREY), f"{c:+.2f}"])
        story.append(_table(fr, [2.3 * inch, 1.2 * inch, 1.7 * inch, 1.3 * inch], header=True))
        story.append(_p("Contribution is in log-odds: positive values raise the "
                        "malicious score, negative values lower it.", S_NOTE))

    warns = ml.get("warnings") or []
    if warns:
        story += [Spacer(1, 8), _p("Caveats", S_SUB)]
        for w in warns:
            story.append(_p("- " + w, S_BODY, color=colors.HexColor("#b9770e")))


def _behaviour_section(behavior_data, ml, story):
    story.append(_p("Behaviour analysis", S_H1))
    mode = (ml or {}).get("mode")
    dynamic_ran = (mode == "fused") if mode else bool(behavior_data and "Summary" in behavior_data)

    if not dynamic_ran or not behavior_data:
        story.append(_p("No runtime analysis was performed: the sample was not "
                        "executed, so this verdict rests on static features only. "
                        "Detonate the sample inside an isolated VM/sandbox to add "
                        "behavioural evidence.", S_BODY))
        return

    s = behavior_data.get("Summary", {})
    story.append(_kv([
        ["New processes", str(s.get("num_new_processes", 0))],
        ["New network connections", str(s.get("num_new_connections", 0))],
        ["Files dropped", str(s.get("num_dropped_files", 0))],
        ["Flagged (proc / conn / file)",
         f"{s.get('num_suspicious_process_flags', 0)} / "
         f"{s.get('num_suspicious_connection_flags', 0)} / "
         f"{s.get('num_suspicious_file_flags', 0)}"],
    ]))

    def add_list(title, items, header, widths, row_fn):
        if not items:
            return
        story.append(Spacer(1, 6))
        story.append(_p(title, S_SUB))
        rows = [header] + [row_fn(i) for i in items[:MAX_LIST_ROWS]]
        story.append(_table(rows, widths, header=True))
        if len(items) > MAX_LIST_ROWS:
            story.append(_p(f"... and {len(items) - MAX_LIST_ROWS} more not shown.", S_NOTE))

    def flag(i):
        return _p(i["Flag"], color=RED) if "Flag" in i else ""

    add_list("New processes", behavior_data.get("New Processes", []),
             ["PID", "Name", "Command line", "Flag"],
             [0.6 * inch, 1.2 * inch, 3.3 * inch, 1.4 * inch],
             lambda i: [str(i.get("PID", "")), str(i.get("Name", "")),
                        str(i.get("Command Line", ""))[:MAX_CMD_CHARS], flag(i)])
    add_list("New network connections", behavior_data.get("New Network Connections", []),
             ["Remote IP", "Port", "State", "Flag"],
             [1.7 * inch, 0.8 * inch, 1.3 * inch, 2.7 * inch],
             lambda i: [str(i.get("Remote IP", "")), str(i.get("Remote Port", "")),
                        str(i.get("Status", "")), flag(i)])
    add_list("Dropped files", behavior_data.get("Dropped Files", []),
             ["Path", "Flag"], [4.6 * inch, 1.9 * inch],
             lambda i: [str(i.get("Path", "")), flag(i)])


def _vt_section(vt, story):
    story.append(_p("VirusTotal cross-check (secondary signal)", S_H1))
    story.append(_p("Independent second opinion from 70+ antivirus engines. It does "
                    "not decide the verdict.", S_NOTE_KEEP))
    if not vt or "Error" in vt:
        reason = (vt or {}).get("Error", "No VirusTotal data.")
        story.append(_p(f"Not available: {reason}", S_BODY, color=GREY))
        return
    scan = vt.get("Scan Date", "N/A")
    if isinstance(scan, (int, float)):
        scan = datetime.fromtimestamp(scan, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(_kv([
        ["Engines", str(vt.get("Total AV Engines", "N/A"))],
        ["Flagged malicious", str(vt.get("Malicious Detections", 0))],
        ["Flagged suspicious", str(vt.get("Suspicious Detections", 0))],
        ["Threat label", str(vt.get("Malware Family", "Unknown"))],
        ["Reputation", str(vt.get("Reputation Score", "Unknown"))],
        ["Last analysed", str(scan)],
    ]))


def _recommendations(verdict, ml, story):
    story.append(_p("Recommendations", S_H1))
    low = (ml or {}).get("reliability") == "low"
    static_only = (ml or {}).get("mode") == "static_only"

    if verdict == "Malicious":
        recs = ["Quarantine the file and do not run it outside an isolated analysis environment.",
                "Block or search for the SHA-256 hash across your endpoints and mail gateway.",
                "If it was already executed on a real machine, run a full scan with an "
                "updated antivirus and change credentials that may have been exposed.",
                "Confirm the family label through manual or sandbox analysis before "
                "acting on it; it is a model estimate."]
    elif verdict == "Potentially Harmful":
        recs = ["Review where the file came from and whether it was expected.",
                "Analyse it further inside an isolated sandbox before trusting it.",
                "Monitor for unusual activity if it has already been run."]
    elif verdict == "Benign":
        if low:
            recs = ["Do not treat this file as clean. It is not a Windows executable, so the "
                    "model's features cannot describe it and a Benign verdict is unreliable.",
                    "Inspect it with format-specific tools (for example, an Office macro "
                    "analyser for documents) or detonate it in a sandbox.",
                    "Check the hash on VirusTotal if you have not already."]
        else:
            recs = ["No malicious indicators were found by the model, which is not proof "
                    "of safety.",
                    "Keep antivirus definitions up to date."]
            if static_only:
                recs.append("For higher confidence, add runtime analysis in an isolated sandbox.")
    else:
        recs = ["A verdict could not be produced. Train the ML models "
                "(train_model.py) and/or set VIRUSTOTAL_API_KEY, then re-run the analysis."]

    for r in recs:
        story.append(_p("- " + r, S_BODY))


# ------------------------------------------------------------ entry point

def generate_report(file_data, behavior_data, virustotal_data, ml_result,
                    verdict, report_path, highlights=None):
    verdict = _normalise_verdict(verdict)
    ml = ml_result or {}
    base_name = os.path.splitext(os.path.basename(report_path))[0]
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    story = [_p("Malware Analysis Report", S_TITLE),
             Paragraph(f"File: <b>{_esc(base_name)}</b>", S_CENTER),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", S_CENTER),
             Spacer(1, 12)]

    # Verdict banner; qualified when the model itself says it can't be trusted.
    label = f"Verdict: {verdict}"
    if verdict == "Benign" and ml.get("reliability") == "low":
        label += "  (low reliability)"
    # An unreliable "Benign" must never look reassuringly green.
    unreliable_benign = verdict == "Benign" and ml.get("reliability") == "low"
    banner_color = (VERDICT_COLORS["Potentially Harmful"] if unreliable_benign
                    else VERDICT_COLORS.get(verdict, VERDICT_COLORS["Unknown"]))
    banner = Table([[Paragraph(_esc(label), S_BANNER)]], colWidths=[6.5 * inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [banner, Spacer(1, 10)]

    _summary_section(verdict, ml, virustotal_data, highlights, story)
    _file_section(file_data or {}, story)
    _ml_section(ml, story)
    _behaviour_section(behavior_data, ml, story)
    _vt_section(virustotal_data, story)
    _recommendations(verdict, ml, story)
    story += [Spacer(1, 14),
              _p("Automated result generated by MalXray. Classifier output is a "
                 "probability estimate, not proof, and should be confirmed by manual "
                 "analysis before any action is taken.", S_NOTE)]

    doc = SimpleDocTemplate(report_path, pagesize=letter,
                            leftMargin=1 * inch, rightMargin=1 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.9 * inch,
                            title=f"MalXray report - {base_name}", author="MalXray")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Analysis complete! Report available at: {report_path}")
    return report_path