"""Segment the official compilation 'Fjalime të z. Fatos Nano në Kuvendin e Shqipërisë'
(Kuvendi i Shqipërisë, Tiranë 2010; 617 pp.) into sessions, speeches and paragraphs,
using the typography of the edition:

  Bold            session headers (SEANCË PLENARE, date), legislature/session lines,
                  and sub-headings inside long programme speeches  -> kept, flagged 'heading'
  BoldItalic      editorial summaries and intertitles               -> removed; mark speech boundaries
  Italic, long    deputies' questions read aloud, stage directions  -> removed
  Italic, short   emphasis and foreign expressions                  -> kept
  Regular         Nano's own words                                  -> kept

Usage: python parse.py nano.pdf corpus.json
"""
import json, re, sys, unicodedata
import pymupdf

SRC = sys.argv[1]; OUT = sys.argv[2]
MONTHS = {"janar": 1, "shkurt": 2, "mars": 3, "prill": 4, "maj": 5, "qershor": 6,
          "korrik": 7, "gusht": 8, "shtator": 9, "tetor": 10, "nëntor": 11, "dhjetor": 12}
STAGE = re.compile(r"ndërhyr|ndërhyn|të qeshura|duartrokit|ndërpret seancën|ndërpritet seanca|zhurmë|salla|"
                   r"kundërshtime|dëgjohet zë|u drejtohet", re.I)

doc = pymupdf.open(SRC)
toc_pno = next(i for i in range(len(doc)) if "PËRMBAJTJA" in unicodedata.normalize("NFC", doc[i].get_text())[:200])

def cls(font):
    f = font.split("+")[-1]
    if f.endswith("BoldItalic"): return "E"      # editorial
    if f.endswith(",Italic"): return "I"
    if "Bold" in f: return "B"
    return "R"

# 1. flatten to lines: (pno, x0, [(cls, text), ...])
lines = []
for pno in range(toc_pno):
    for b in doc[pno].get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [(cls(s["font"]), unicodedata.normalize("NFC", s["text"])) for s in l["spans"]]
            txt = "".join(t for _, t in spans).strip()
            if not txt:
                continue
            lines.append((pno + 1, round(l["bbox"][1], 1), l["bbox"][0], spans, txt))
lines.sort(key=lambda r: (r[0], r[1], r[2]))   # reading order: page, y, x (stream order is unreliable)
lines = [(p, x, s, t) for p, y, x, s, t in lines]

# 2. walk lines, build sessions -> speeches -> paragraphs (as lists of (cls,text) runs)
sessions, cur, state, legis, sess = [], None, None, None, None
EDITORIAL_START = re.compile(
    r"^(E merr|Replikë|Kryeministri|Kryetari i Këshillit|Diskutim|Mocion|Interpelancë|Përgjigje|Bën |"
    r"Sugjeron|Kërkon|Falënderon|Vazhdon|Deputet|Njeh|Pas |Me kërkesë|Në vijim|Edhe në|Kjo seancë|"
    r"U përgjigjet|Zoti |Në këtë seancë|I ngarkuar|Raporton|Paraqet|Shpreh|Informon|I përgjigjet)")
pending_E = []
def new_speech(intertitle=""):
    cur["speeches"].append({"intertitle": intertitle, "paragraphs": []})
def flush_E():
    """Bold-italic material inside a session: an editorial intertitle (new speech unit)
    or a sub-heading of a long programme speech (kept as a heading paragraph)."""
    global pending_E
    t = " ".join(pending_E).strip(); pending_E = []
    if not t: return
    if EDITORIAL_START.match(t) or len(t.split()) > 25:
        new_speech(t)
    else:
        cur["speeches"][-1]["paragraphs"].append({"heading": True, "runs": [["R", t + " "]]})
for pno, x0, spans, txt in lines:
    if re.fullmatch(r"\d{1,3}", txt):
        continue
    kinds = {k for k, t in spans if t.strip()}
    e_chars = sum(len(t.strip()) for k, t in spans if k == "E")
    if "E" in kinds and e_chars >= 0.8 * len(txt):   # line set (almost) entirely in bold italic
        kinds = {"E"}
    if kinds <= {"B"}:
        if txt.startswith("LEGJISLATURA"): legis = txt.split()[1]; continue
        if re.fullmatch(r"SESIONI [IVX]+", txt): sess = txt.split()[1]; continue
        if re.fullmatch(r"[A-ZËÇ][a-zëç]+-[a-zëç]+ \d{4}", txt): continue
        if txt == "SEANCË PLENARE":
            cur = {"legislature": legis, "session": sess, "date": None, "summary": "",
                   "speeches": [], "pdf_page": pno}
            sessions.append(cur); state = "date"; continue
        if state == "date":
            d, mo, y = re.fullmatch(r"(\d{1,2})\s+([a-zë]+)\s+(\d{4})", txt).groups()
            cur["date"] = f"{int(y):04d}-{MONTHS[mo]:02d}-{int(d):02d}"; state = "summary"; continue
    if cur is None:
        continue
    if kinds <= {"E"}:                                   # editorial summary / intertitle / sub-heading
        if state == "summary":
            cur["summary"] = (cur["summary"] + " " + txt).strip()
        else:
            pending_E.append(txt); state = "inter"
        continue
    # regular body line
    if state in ("summary", "date"):
        new_speech(); state = "body"
    if state == "inter":
        flush_E(); state = "body"
    paras = cur["speeches"][-1]["paragraphs"]
    is_heading = kinds <= {"B"}
    indent = x0 > 75 and x0 < 110
    if indent or not paras or is_heading or paras[-1]["heading"]:
        paras.append({"heading": is_heading, "runs": []})
    runs = paras[-1]["runs"]
    for k, t in spans:
        if runs and runs[-1][0] == k:
            runs[-1][1] += t
        else:
            runs.append([k, t])
    # line break = space
    runs[-1][1] += " "

# 3. resolve runs into text; drop editorial and quoted questions
stats = {"italic_kept_words": 0, "italic_removed_words": 0, "stage_removed": 0, "frames_removed": 0,
         "editorial_inline_removed_words": 0}
removed_log = []
for c in sessions:
    for sp in c["speeches"]:
        out = []
        for p in sp["paragraphs"]:
            parts = []
            for k, t in p["runs"]:
                w = len(t.split())
                if k == "E":
                    stats["editorial_inline_removed_words"] += w; removed_log.append(("E", t.strip())); continue
                if k == "I":
                    if STAGE.search(t) or t.strip().startswith("("):
                        stats["stage_removed"] += 1; removed_log.append(("STAGE", t.strip())); continue
                    if w >= 6:
                        stats["italic_removed_words"] += w; removed_log.append(("Q", t.strip())); continue
                    stats["italic_kept_words"] += w
                parts.append(t)
            text = re.sub(r"\s+", " ", "".join(parts)).strip()
            text = re.sub(r"\(\s*\)", "", text)
            text = re.sub(r"\s+([,.;:?!])", r"\1", text).strip()
            # dangling question frame e.g. 'Pyetje tjetër:' whose italic question was removed
            if text.endswith(":") and re.search(r"[Pp]yet", text) and len(text.split()) <= 15 and \
               any(k == "I" for k, _ in p["runs"]):
                stats["frames_removed"] += 1; removed_log.append(("FRAME", text)); continue
            m = re.match(r"^(.{0,120}?[Pp]yet\w*[^:]{0,60}:)\s*$", text)
            if text:
                out.append({"heading": p["heading"], "text": text})
        sp["paragraphs"] = out
    c["speeches"] = [sp for sp in c["speeches"] if sp["paragraphs"]]

def term(date):
    y = int(date[:4]); return "T1" if y == 1991 else ("T2" if y in (1997, 1998) else "T3")
k = 0
for i, c in enumerate(sessions):
    c["id"] = f"S{i+1:02d}"; c["term"] = term(c["date"])
    for j, sp in enumerate(c["speeches"]):
        sp["id"] = f"{c['id']}.{j+1}"

json.dump(sessions, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(OUT.replace(".json", "_removed.log"), "w", encoding="utf-8") as f:
    for kk, t in removed_log: f.write(f"{kk}\t{t}\n")
tot = sum(len(p["text"].split()) for c in sessions for sp in c["speeches"] for p in sp["paragraphs"])
print("sessions", len(sessions), "speeches", sum(len(c["speeches"]) for c in sessions), "words", tot, stats)
for c in sessions:
    print(c["id"], c["date"], c["term"], len(c["speeches"]),
          [sum(len(p["text"].split()) for p in sp["paragraphs"]) for sp in c["speeches"]])
