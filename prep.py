"""Tokenisation, sentence and clause segmentation, syllabification and genre coding.

Input : corpus.json (from parse.py)
Output: units.json  -- one record per speech unit with sentences -> clauses -> tokens
"""
import json, re, unicodedata, os, collections

ABBR = {"nr", "dr", "z", "zj", "prof", "fq", "shek", "p.sh", "sh.a", "sh", "pra"}  # 'pra' never abbreviated; filtered below
ABBR.discard("pra")
WORD_RE = re.compile(r"[A-Za-zËëÇç]+(?:['\-][A-Za-zËëÇç]+)*'?|\d+(?:[.,]\d+)*")
VOWELS = "aeëiouy"
DIPHTHONGS = ("ie", "ye", "ua", "ue")
DIGRAPHS = ("dh", "gj", "ll", "nj", "rr", "sh", "th", "xh", "zh")


def norm(t):
    t = unicodedata.normalize("NFC", t)
    return t.replace("’", "'").replace("‘", "'").replace("´", "'").replace("«", "“").replace("»", "”")


def split_sentences(par):
    """Split a paragraph into sentences at . ! ? … followed by space + capital/quote/digit."""
    out, start = [], 0
    # dates printed with spaces ("3. 07. 2003") are one token, not three sentences
    par = re.sub(r"\b(\d{1,2})\.\s+(\d{1,2})\.\s+(\d{4})\b", r"\1.\2.\3", par)
    for m in re.finditer(r"([.!?…]+)([”\")]*)\s+(?=[A-ZËÇ“\"(\d–-])", par):
        pre = par[start:m.start()]
        last = re.findall(r"([\w.]+)$", pre)
        lw = last[0].lower() if last else ""
        if m.group(1) == "." and (lw in ABBR or re.fullmatch(r"[A-ZËÇ]", last[0] if last else "") or
                                  re.fullmatch(r"(\w\.)+\w", lw)):
            continue
        # a list numeral ("1.", "06.") standing alone is the label of the item that follows, not a sentence
        if m.group(1) == "." and re.fullmatch(r"\d{1,3}", pre.strip()):
            continue
        out.append(par[start:m.end()].strip()); start = m.end()
    tail = par[start:].strip()
    if tail:
        out.append(tail)
    return out


def split_clauses(sent):
    """Punctuation-delimited segments (the 'kola' of the period)."""
    parts = re.split(r"\s*[,;:()]\s*|\s+[–—-]\s+", sent)
    return [p for p in parts if WORD_RE.search(p)]


def tokens(s):
    return [m.group(0) for m in WORD_RE.finditer(s)]


def is_word(t):
    return not t[0].isdigit()


def phonemes(w):
    """Number of phonemes, counting Albanian digraphs as one segment."""
    w = w.lower().replace("'", "").replace("-", "")
    n, i = 0, 0
    while i < len(w):
        if w[i:i + 2] in DIGRAPHS:
            i += 2
        else:
            i += 1
        n += 1
    return n


def syllables(w, phonetic=False):
    """Count vowel nuclei; ie/ye/ua/ue are treated as diphthongs.
    phonetic=True drops a word-final unstressed -ë after a consonant in polysyllables
    (the final -ë is commonly reduced or elided in connected standard Albanian)."""
    w = w.lower().replace("'", "").replace("-", "")
    nuclei = []
    i = 0
    while i < len(w):
        if w[i] in VOWELS:
            if w[i:i + 2] in DIPHTHONGS:
                nuclei.append(i); i += 2; continue
            nuclei.append(i)
        i += 1
    n = len(nuclei)
    if phonetic and n >= 2 and w.endswith("ë") and len(w) >= 2 and w[-2] not in VOWELS:
        n -= 1
    return max(n, 1) if any(ch.isalpha() for ch in w) else 0


def genre(summary, intertitle, first):
    t = (intertitle or summary).lower()
    if "replikë" in t: return "rejoinder"
    if "dorëheqj" in t or "votëbes" in t: return "debate"          # resignation and confidence-vote sittings
    if "merr përsëri fjalën" in t or "merr fjalën përsëri" in t or "merr fjalën pas" in t or \
       "bën një sqarim" in t or "sqarime" in t and intertitle: return "rejoinder"
    if "interpelanc" in t: return "interpellation"
    if "mocion" in t: return "motion"
    if "pyetje" in t: return "questions"
    if "buxhet" in t or "paketën fiskale" in t: return "budget"
    if "program" in t: return "programme"
    if any(k in t for k in ("informon", "raporton", "situatën", "qëndrimin e qeverisë", "flet për")): return "report"
    return "debate"


def main():
  corpus = json.load(open("corpus.json", encoding="utf-8"))
  residual = []
  units = []
  for c in corpus:
      for j, sp in enumerate(c["speeches"]):
          sents = []
          label = ""   # a list numeral printed as a paragraph of its own ("1.") labels the next paragraph
          for pi, p in enumerate(sp["paragraphs"]):
              if p["heading"]:
                  continue
              if re.fullmatch(r"\s*\d{1,3}\.\s*", p["text"]):
                  label = p["text"].strip() + " "; continue
              text, label = label + norm(p["text"]), ""
              for si, s in enumerate(split_sentences(text)):
                  toks = tokens(s)
                  if not toks:
                      continue
                  # residual deputies' questions set in roman type ('Pyetje: ...?')
                  if re.match(r"^(\w+\s+){0,4}?[Pp]yetj\w*\b[^:]{0,70}:", s) and s.rstrip().endswith("?"):
                      residual.append(len([t for t in toks if is_word(t)]))
                      continue
                  kind = "address" if (s.rstrip().endswith(",") and len(toks) <= 8) else (
                         "question" if s.rstrip().rstrip("”\")").endswith("?") else (
                         "exclamation" if s.rstrip().rstrip("”\")").endswith("!") else "declarative"))
                  cls = [tokens(cl) for cl in split_clauses(s)]
                  sents.append({"par": pi, "text": s, "kind": kind, "tokens": toks, "clauses": cls})
          units.append({"id": sp["id"], "session": c["id"], "date": c["date"], "term": c["term"],
                        "legislature": c["legislature"], "genre": genre(c["summary"], sp["intertitle"], j == 0),
                        "summary": sp["intertitle"] or c["summary"], "sentences": sents})

  json.dump(units, open("units.json", "w", encoding="utf-8"), ensure_ascii=False)
  # words removed at extraction (parse.py log): read-aloud questions, question frames, stage directions, editorial lines
  from collections import Counter as _Counter
  rc, rw = _Counter(), _Counter()
  for line in open("corpus_removed.log", encoding="utf-8"):
      kind, _, txt = line.rstrip("\n").partition("\t"); rc[kind] += 1
      ws = [t for t in tokens(norm(txt))]; rw[kind + "_tokens"] += len(ws); rw[kind + "_words"] += sum(1 for t in ws if is_word(t))
  os.makedirs("results", exist_ok=True)
  json.dump({"items": dict(rc), "counts": dict(rw), "residual_roman_questions": len(residual),
             "residual_roman_question_words": sum(residual)}, open("results/removed_counts.json", "w"), indent=1)
  import collections
  nt = sum(len(s["tokens"]) for u in units for s in u["sentences"])
  nw = sum(sum(is_word(t) for t in s["tokens"]) for u in units for s in u["sentences"])
  print("units", len(units), "sentences", sum(len(u["sentences"]) for u in units), "tokens", nt, "words", nw)
  print(collections.Counter(u["genre"] for u in units))
  print(collections.Counter(s["kind"] for u in units for s in u["sentences"]))
  by = collections.defaultdict(lambda: [0, 0, 0])
  for u in units:
      by[u["term"]][0] += 1; by[u["term"]][1] += len(u["sentences"])
      by[u["term"]][2] += sum(sum(is_word(t) for t in s["tokens"]) for s in u["sentences"])
  print(dict(by))


if __name__ == "__main__":
    main()
