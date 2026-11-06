"""Section 11 — the Marx comparison corpus.

Source: the HTML mirror of the Marx-Engels-Werke (MEW, Dietz Verlag, Berlin 1956-1990) and of English translations in
the public repository github.com/MARX-EBOOKS/mlraw, pinned at commit a3a7d3ea087e9ca069f10f5101103c8abe8afb22
(folders mewerke/ and en/). The texts are not redistributed with this package; get_marx.sh fetches them.

German originals (MEW), Marx's own prose:
  written  Manifest der Kommunistischen Partei (1848, with Engels), Lohnarbeit und Kapital (1849),
           Die Klassenkämpfe in Frankreich I-IV (1850), Der achtzehnte Brumaire I-VII (1852),
           Zur Kritik der Politischen Ökonomie, Vorwort (1859), Das Kapital I: Vorwort (1867), ch. 1, 4, 24,
           Kritik des Gothaer Programms (1875)
  spoken   the two defence speeches before the Cologne jury, 7 and 8 February 1849 (Marx's parts only)
English translations of four of the same works (Manifesto, Class Struggles, Eighteenth Brumaire, Gotha critique),
used only to measure how far translation moves the measures of Section 11.

Cleaning: editorial matter is removed (small-print blocks, footnotes, page numbers <123> and |123|, editorial
insertions in angle brackets, bracketed editorial lines, headings); hyphenation at page breaks is joined. Sentences
that quote another author at length (a quotation of eight or more words) and sentences containing formulas (=) are
dropped, so that the remaining sentences are Marx's own.
Output: results/marx_corpus_stats.json (counts only) and marx_corpus.json (local, not distributed).
"""
import os, re, html, json, sys, hashlib
from common import save, HERE

SRC = os.environ.get("MARX_MLRAW", os.path.join(HERE, "..", "marx-ebooks", "mlraw"))
MEW = os.path.join(SRC, "mewerke")
EN = os.path.join(SRC, "en")

WORKS_DE = {
    "written": [("Manifest der Kommunistischen Partei", 1848, ["me04/me04_459.htm"]),
                ("Lohnarbeit und Kapital", 1849, ["me06/me06_397.htm"]),
                ("Die Klassenkämpfe in Frankreich", 1850, ["me07/me07_012.htm", "me07/me07_035.htm",
                                                            "me07/me07_064.htm", "me07/me07_095.htm"]),
                ("Der achtzehnte Brumaire des Louis Bonaparte", 1852,
                 [f"me08/me08_{p}.htm" for p in ("115", "124", "135", "149", "159", "176", "194")]),
                ("Zur Kritik der Politischen Ökonomie, Vorwort", 1859, ["me13/me13_007.htm"]),
                ("Das Kapital I (Vorwort, Kap. 1, 4, 24)", 1867,
                 ["me23/me23_011.htm", "me23/me23_049.htm", "me23/me23_161.htm", "me23/me23_741.htm"]),
                ("Kritik des Gothaer Programms", 1875, ["me19/me19_013.htm"])],
    "spoken": [("Verteidigungsrede im ersten Preßprozeß", 1849, ["me06/me06_223.htm"]),
               ("Verteidigungsrede im Prozeß gegen den Kreisausschuß", 1849, ["me06/me06_240.htm"])],
}
WORKS_EN = [("Manifesto of the Communist Party", "Manifest der Kommunistischen Partei",
             [f"Manifesto_of_the_Communist_Party/{d}" for d in ("1", "2", "3", "4")]),
            ("The Class Struggles in France", "Die Klassenkämpfe in Frankreich",
             ["The_Class_Struggles_in_France,_1848_to_1850.html"]),
            ("The Eighteenth Brumaire of Louis Bonaparte", "Der achtzehnte Brumaire des Louis Bonaparte",
             ["The_Eighteenth_Brumaire_of_Louis_Bonaparte.html"]),
            ("Critique of the Gotha Programme", "Kritik des Gothaer Programms", ["Critique_of_the_Gotha_Programme.html"])]

# first and last sentence of the text proper (inclusive), where the source page carries prefaces, notes or letters
BOUNDS = {"Manifest der Kommunistischen Partei": (r"^Ein Gespenst geht um", r"^Proletarier aller Länder"),
          "Kritik des Gothaer Programms": (r"^Die Arbeit ist nicht die Quelle alles Reichtums", r"^Dixi et salvavi"),
          "Manifesto of the Communist Party": (r"^A spectre is haunting", r"^Working Men of All Countries, Unite"),
          "The Class Struggles in France": (r"^With the exception of only a few chapters", None),
          "The Eighteenth Brumaire of Louis Bonaparte": (r"^Hegel remarks somewhere", r"bronze statue of Napoleon"),
          "Critique of the Gotha Programme": (r"^Labou?r is not the source of all wealth", None)}
ADDRESS = re.compile(r"^(?:Meine\s+)?(?:Herr(?:e)?n|Bürger)(?:\s+Geschwornen)?!$")

WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
ABBR_DE = {"z", "b", "d", "h", "u", "s", "w", "usw", "etc", "bd", "vgl", "nr", "ca", "dr", "hr", "hrn", "st", "resp",
           "bzw", "p", "pp", "sog", "ztr", "lond", "mr", "mrs", "ch", "kap", "aufl", "ebd", "ibid", "a", "o", "dgl",
           "pfd", "sh", "fr", "mill", "doll", "thlr", "sgr", "pf", "art", "sir", "m", "cit", "op", "l", "c", "vol",
           "esq", "jun", "sen", "gen", "lt", "col", "capt", "rev", "no", "ff", "f", "i", "e", "tom", "liv", "chap",
           "ed", "edit", "pag", "t", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
ABBR_EN = {"mr", "mrs", "messrs", "dr", "st", "e", "g", "i", "etc", "vol", "p", "pp", "cf", "no", "m", "viz", "ibid",
           "op", "cit", "esq", "gen", "col", "capt", "rev", "jun", "sen", "ch", "chap", "ed", "ii", "iii", "iv", "v",
           "vi", "vii", "viii", "ix", "x", "l", "c", "s", "d", "sh", "fr", "mme", "mlle", "art"}


def clean_mew(raw, part=None):
    t = raw
    t = re.split(r"<P[^>]*>\s*Fu(?:&szlig;|ß)noten\s*</P>", t, flags=re.I)[0]          # footnotes
    t = re.sub(r"<FONT SIZE=2>.*?</FONT>", " ", t, flags=re.S | re.I)                    # small print
    t = re.sub(r"<H\d[^>]*>.*?</H\d>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<div class=\"seitenzahlen\">.*?</div>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<A NAME=\"Z[^\"]*\"><A HREF=\"#M[^\"]*\">\([^)]*\)</A></A>", " ", t, flags=re.I)  # note markers
    # page numbers inside a hyphenated word: "Rela- <51> tives" -> "Relatives"
    t = re.sub(r"(\w)-\s*(?:<B>)?\s*<A NAME=\"S\d+\">(?:<B>)?(?:&lt;|<|\|)\d+(?:&gt;|>|\|)</A>(?:</B>)?\s*", r"\1", t,
               flags=re.I)
    t = re.sub(r"<A NAME=\"S\d+\">.*?</A>", " ", t, flags=re.S | re.I)
    paras = []
    for m in re.finditer(r"<P[^>]*>(.*?)</P>", t, re.S | re.I):
        p = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        p = re.sub(r"<[^<>]{0,300}>", " ", p)                  # editorial insertions <Siehe ...>, <123>
        p = re.sub(r"\|[^|]{1,200}\|", " ", p)              # page numbers |123| and editorial glosses |...|
        p = re.sub(r"\{\d+\}", " ", p)                         # note markers {1}
        p = re.sub(r"(\w)- (?!(?:und|oder|bis|sowie|wie|noch|als|resp|zu)\b)([a-zäöüß])", r"\1\2", p)   # line-break hyphens
        p = re.sub(r"\s+", " ", p).strip()
        paras.append(p)
    if part == "marx_only":                                    # 1849 press trial: Marx's speech, not Engels'
        i0 = next(i for i, p in enumerate(paras) if p.startswith("[Verteidigungsrede von Karl Marx"))
        i1 = next(i for i, p in enumerate(paras) if p.startswith("[Verteidigungsrede von Friedrich Engels"))
        paras = paras[i0 + 1:i1]
    out = []
    for p in paras:
        if not p or (p.startswith("[") and p.endswith("]")) or p in ("Inhalt:",) or len(WORD.findall(p)) < 4:
            continue
        out.append(p)
    return out


def clean_en(raw):
    i = raw.find('id="mw-content-text"'); t = raw[i:] if i >= 0 else raw
    t = re.split(r"<h2[^>]*>\s*(?:<span[^>]*>)?\s*(?:Notes|References|Footnotes)", t, flags=re.I)[0]
    t = re.sub(r"<sup[^>]*>.*?</sup>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<(table|style|script)[^>]*>.*?</\1>", " ", t, flags=re.S | re.I)
    paras = []
    for m in re.finditer(r"<p[^>]*>(.*?)</p>", t, re.S | re.I):
        p = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        p = re.sub(r"\[\d+\]|\[[a-z]\]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p and len(WORD.findall(p)) >= 4 and not (p.startswith("[") and p.endswith("]")):
            paras.append(p)
    return paras


def split_sentences(par, abbr):
    out, start = [], 0
    for m in re.finditer(r"([.!?…]+)([”“\"»«’'„)\]]*)\s+(?=[\"„“»«(\[]?[A-ZÄÖÜÉÈ\d])", par):
        pre = par[start:m.start()]
        last = re.findall(r"([\w.]+)$", pre); lw = last[0].lower().rstrip(".") if last else ""
        if m.group(1) == "." and (lw in abbr or re.fullmatch(r"\d{1,4}", lw) or re.fullmatch(r"(\w\.)+\w", lw)
                                  or re.fullmatch(r"[^\W\d_]", lw)):
            continue
        out.append(par[start:m.end()].strip()); start = m.end()
    if par[start:].strip(): out.append(par[start:].strip())
    return out


def split_clauses(sent):
    return [p for p in re.split(r"\s*[,;:()]\s*|\s+[–—-]\s+", sent) if WORD.search(p)]


QUOTE = re.compile(r"[„\"“»]([^„\"“”»«]{1,600})[“\"”«]")


def keep(sent):
    if "=" in sent or " | " in sent: return False             # formulas; navigation lines
    if not re.search(r"[.!?…][\"„“”»«’')\]]*$", sent): return False   # headings, datelines, captions
    if re.fullmatch(r"\(.*\)\.?", sent): return False           # stage notes such as (Beifallsruf im Auditorium.)
    if ADDRESS.match(sent): return False                      # vocative address, excluded as in the Nano corpus
    for q in QUOTE.findall(sent):
        if len(WORD.findall(q)) >= 8: return False
    return True


def sentences_of(paras, abbr, title=None):
    S = []
    for pi, p in enumerate(paras):
        for s in split_sentences(p, abbr):
            toks = WORD.findall(s)
            if not toks or not keep(s): continue
            S.append({"par": pi, "text": s, "tokens": toks, "clauses": [WORD.findall(c) for c in split_clauses(s)]})
    if title in BOUNDS:
        a, b = BOUNDS[title]
        i0 = next(i for i, x in enumerate(S) if re.search(a, x["text"]))
        i1 = next((i for i, x in enumerate(S) if b and i >= i0 and re.search(b, x["text"])), len(S) - 1)
        S = S[i0:i1 + 1]
    return S


def build():
    corpus = {"de": [], "en": []}
    for mode, works in WORKS_DE.items():
        for title, year, files in works:
            paras = []
            for f in files:
                raw = open(os.path.join(MEW, f), encoding="utf-8-sig", errors="replace").read()
                paras += clean_mew(raw, "marx_only" if f.endswith("me06_223.htm") else None)
            corpus["de"].append({"title": title, "year": year, "mode": mode, "files": files,
                                 "sentences": sentences_of(paras, ABBR_DE, title)})
    for title, de_title, files in WORKS_EN:
        paras = []
        for f in files:
            p = os.path.join(EN, f)
            fl = sorted(os.path.join(p, x) for x in os.listdir(p)) if os.path.isdir(p) else [p]
            for x in fl: paras += clean_en(open(x, encoding="utf-8", errors="replace").read())
        corpus["en"].append({"title": title, "de_title": de_title, "files": files,
                             "sentences": sentences_of(paras, ABBR_EN, title)})
    return corpus


if __name__ == "__main__":
    C = build()
    json.dump(C, open(os.path.join(HERE, "marx_corpus.json"), "w", encoding="utf-8"), ensure_ascii=False)
    stats = {lang: [{"title": w["title"], "mode": w.get("mode"), "year": w.get("year"), "sentences": len(w["sentences"]),
                     "words": sum(len([t for t in s["tokens"] if not t[0].isdigit()]) for s in w["sentences"])}
                    for w in C[lang]] for lang in C}
    stats["source"] = {"repository": "https://github.com/MARX-EBOOKS/mlraw",
                       "commit": "a3a7d3ea087e9ca069f10f5101103c8abe8afb22"}
    save("marx_corpus_stats.json", stats)
    for lang in ("de", "en"):
        for w in stats[lang]: print(lang, w)
