"""Section 11 — Nano and Marx.

Nano's parliamentary corpus (Albanian) is compared with Marx's German originals (marx_corpus.py): written works and
the two Cologne defence speeches of February 1849. Because the languages differ, only measures that do not depend on
the scale of a language's words are compared, each against a null model of its own text:
  shape of the sentence-length distribution (coefficient of variation, Weibull shape, share of sentences at least
  twice the median length); long-range order of sentence lengths (DFA against shuffles); anaphora and epistrophe at
  sentence and clause level against permutation nulls; the arrangement of repeated word pairs within a sentence
  (chiastic ABBA against parallel ABAB, which a random order makes equally likely); questions and exclamations;
  the figure "not only ... but also".
Translation check: four works are measured in the German original and in an English translation; a measure that
moves little under translation is a candidate for comparison across languages.
Vocabulary: the concepts of political economy in Marx (per 10,000 words) and in Nano by premiership.
Output: results/marx.json
"""
import json, os, re, collections
import zlib
import numpy as np
from scipy import stats
from common import load_units, save, HERE, TERMS
from rhythm_dfa import dfa

rng = np.random.default_rng(1848)
U = load_units()
MC = json.load(open(os.path.join(HERE, "marx_corpus.json"), encoding="utf-8"))
R = {}


def words(s): return [t for t in s["tokens"] if not t[0].isdigit()]


# ------------------------------------------------------------------ documents (lists of sentences, in order)
# guard: a sentence without a word would have no length in words (prep.py no longer produces any)
nano_docs = [[s for s in u["sentences"] if s["kind"] != "address" and words(s)] for u in U]
nano_docs_term = {t: [d for d, u in zip(nano_docs, U) if u["term"] == t] for t in TERMS}
de = sorted(MC["de"], key=lambda w: w["year"])
marx_w = [[x for x in w["sentences"] if words(x)] for w in de if w["mode"] == "written"]
marx_s = [[x for x in w["sentences"] if words(x)] for w in de if w["mode"] == "spoken"]
pairs_tr = [(next(w["sentences"] for w in MC["de"] if w["title"] == e["de_title"]), e["sentences"], e["title"])
            for e in MC["en"]]
CORPORA = {"nano": nano_docs, "marx_written": marx_w, "marx_spoken": marx_s,
           "transl_de": [p[0] for p in pairs_tr], "transl_en": [p[1] for p in pairs_tr]}
for t in TERMS: CORPORA[f"nano_{t}"] = nano_docs_term[t]


# ------------------------------------------------------------------ measures
def boot(fn, sents, n=500):
    vals = [fn([sents[i] for i in rng.integers(0, len(sents), len(sents))]) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def length_shape(sents):
    L = np.array([len(words(s)) for s in sents], float); L = L[L > 0]
    return L


def cv(sents): L = length_shape(sents); return float(L.std() / L.mean())
def long_share(sents): L = length_shape(sents); return float((L >= 2 * np.median(L)).mean())
def wshape(sents): L = length_shape(sents); return float(stats.weibull_min.fit(L, floc=0)[0])


def dfa_test(docs, n=100):
    x = np.log(np.concatenate([length_shape(d) for d in docs]))
    scales = np.unique(np.round(np.logspace(np.log10(4), np.log10(len(x) / 8), 22)).astype(int))
    small = scales <= 64
    a = lambda F, sc: np.polyfit(np.log(sc), np.log(F), 1)[0]
    Fo = dfa(x, scales); alpha = a(Fo, scales); alpha_s = a(Fo[small], scales[small])
    bounds = np.cumsum([0] + [len(length_shape(d)) for d in docs])
    sh, ws = [], []
    for _ in range(n):
        sh.append(a(dfa(rng.permutation(x), scales), scales))
        y = x.copy()
        for b0, b1 in zip(bounds, bounds[1:]): y[b0:b1] = rng.permutation(y[b0:b1])
        ws.append(a(dfa(y, scales[small]), scales[small]))
    return {"n": int(len(x)), "alpha": float(alpha), "null_mean": float(np.mean(sh)), "null_sd": float(np.std(sh)),
            "z": float((alpha - np.mean(sh)) / np.std(sh)), "alpha_small": float(alpha_s),
            "small_within_null_mean": float(np.mean(ws)), "small_within_null_sd": float(np.std(ws)),
            "z_small_within": float((alpha_s - np.mean(ws)) / np.std(ws))}


def first(toks):
    w = [t.lower() for t in toks if not t[0].isdigit()]; return w[0] if w else None
def last(toks):
    w = [t.lower() for t in toks if not t[0].isdigit()]; return w[-1] if w else None
def adj(seq): return sum(1 for a, b in zip(seq, seq[1:]) if a is not None and a == b)


def sentence_figure(docs, fn, n=300):
    seqs = [[fn(s["tokens"]) for s in d] for d in docs]
    obs = sum(adj(x) for x in seqs)
    null = [sum(adj(list(rng.permutation(np.array(x, dtype=object)))) for x in seqs) for _ in range(n)]
    return {"observed": obs, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "ratio": float(obs / np.mean(null)), "z": float((obs - np.mean(null)) / np.std(null))}


def expected_adj(x):
    """Expected number of equal adjacent items when the sequence x is randomly permuted."""
    c = collections.Counter(v for v in x if v is not None); n = len(x)
    return sum(k * (k - 1) for k in c.values()) / n if n > 1 else 0.0


def clause_figure(docs, fn, n=200):
    seqs = [[fn(c) for c in s["clauses"] if c] for d in docs for s in d]
    seqs = [x for x in seqs if len(x) > 1]
    O = np.array([adj(x) for x in seqs], float); E = np.array([expected_adj(x) for x in seqs])
    null = [sum(adj(list(rng.permutation(np.array(x, dtype=object)))) for x in seqs) for _ in range(n)]
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(seqs), len(seqs)); bs.append(O[i].sum() / E[i].sum())
    return {"observed": int(O.sum()), "expected": float(E.sum()), "null_mean": float(np.mean(null)),
            "null_sd": float(np.std(null)), "ratio": float(O.sum() / E.sum()),
            "ratio_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "z": float((O.sum() - np.mean(null)) / np.std(null))}


def stem(w): return w[:5]


def frequent(docs, k=100):
    c = collections.Counter(t.lower() for d in docs for s in d for t in words(s)); return {w for w, _ in c.most_common(k)}


SPAN = 10


def local_pairs(sent, stop):
    """Pairs of distinct content stems whose four occurrences fall within SPAN words, classified as chiastic
    (a b ... b a) or parallel (a b ... a b). A random order of the four positions makes the two equally likely."""
    ws = [t.lower() for t in words(sent)]
    pos = collections.defaultdict(list)
    for i, t in enumerate(ws):
        if len(t) >= 4 and t not in stop: pos[stem(t)].append(i)
    rep = [(k, v) for k, v in pos.items() if len(v) >= 2]
    chi, par = set(), set()
    for i in range(len(rep)):
        for j in range(i + 1, len(rep)):
            for a1 in rep[i][1]:
                for a2 in (x for x in rep[i][1] if x > a1):
                    for b1 in rep[j][1]:
                        for b2 in (x for x in rep[j][1] if x > b1):
                            if max(a2, b2) - min(a1, b1) > SPAN: continue
                            key = (rep[i][0], rep[j][0])
                            if a1 < b1 < b2 < a2 or b1 < a1 < a2 < b2: chi.add(key)
                            elif a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2: par.add(key)
    return len(chi), len(par)


def chiasmus(docs):
    stop = frequent(docs); sents = [s for d in docs for s in d]
    po = np.array([local_pairs(s, stop) for s in sents]); nw = np.array([len(words(s)) for s in sents])
    idx = lambda P: P[:, 0].sum() / max(1, P.sum())
    ci_i, ci_r = [], []
    for _ in range(500):
        i = rng.integers(0, len(po), len(po)); ci_i.append(idx(po[i])); ci_r.append(1e4 * po[i, 0].sum() / nw[i].sum())
    ex = [s["text"] for s, (c, p) in zip(sents, po) if c and len(words(s)) <= 40]
    return {"span": SPAN, "chiastic": int(po[:, 0].sum()), "parallel": int(po[:, 1].sum()), "index": float(idx(po)),
            "index_ci": [float(np.percentile(ci_i, 2.5)), float(np.percentile(ci_i, 97.5))],
            "chiastic_per_10k_words": float(1e4 * po[:, 0].sum() / nw.sum()),
            "chiastic_per_10k_ci": [float(np.percentile(ci_r, 2.5)), float(np.percentile(ci_r, 97.5))],
            "examples": ex[:40]}


NOT_ONLY = {"sq": re.compile(r"\bjo\s+(?:vetëm|thjesht|vetem)\b[^.!?;]*?\bpor\b", re.I),
            "de": re.compile(r"\bnicht\s+(?:nur|allein|bloß|bloss|einfach)\b[^.!?;]*?\bsondern\b", re.I),
            "en": re.compile(r"\bnot\s+(?:only|merely|simply|just)\b[^.!?;]*?\bbut\b", re.I)}


def lang_of(name): return "sq" if name.startswith("nano") else ("en" if name == "transl_en" else "de")


def surface(docs, lang):
    S = [s for d in docs for s in d]
    q = np.mean([s["text"].rstrip("”“\"»«’') ").endswith("?") for s in S])
    e = np.mean([s["text"].rstrip("”“\"»«’') ").endswith("!") for s in S])
    hits = sum(bool(NOT_ONLY[lang].search(s["text"])) for s in S); nw = sum(len(words(s)) for s in S)
    return {"questions": float(q), "exclamations": float(e), "not_only": int(hits),
            "not_only_per_1000": float(1000 * hits / len(S)), "not_only_per_10k_words": float(1e4 * hits / nw)}


for name, docs in CORPORA.items():
    # each corpus draws its own random stream, so its nulls and intervals do not depend on the others
    rng = np.random.default_rng([1848, zlib.crc32(name.encode())])
    S = [s for d in docs for s in d]; L = length_shape(S)
    r = {"documents": len(docs), "sentences": len(S), "words": int(L.sum()), "mean_len": float(L.mean()),
         "median_len": float(np.median(L)), "cv": cv(S), "cv_ci": boot(cv, S), "weibull_k": wshape(S),
         "weibull_k_ci": boot(wshape, S, 200), "long_share": long_share(S), "long_share_ci": boot(long_share, S)}
    r["dfa"] = dfa_test(docs)
    r["anaphora_sentence"] = sentence_figure(docs, first)
    r["epistrophe_sentence"] = sentence_figure(docs, last)
    r["anaphora_clause"] = clause_figure(docs, first)
    r["epistrophe_clause"] = clause_figure(docs, last)
    r["pairs"] = chiasmus(docs)
    r.update(surface(docs, lang_of(name)))
    R[name] = r
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (dict, list))},
          "| dfa", round(r["dfa"]["alpha"], 3), round(r["dfa"]["z"], 1), "| ana", round(r["anaphora_sentence"]["ratio"], 2),
          round(r["anaphora_clause"]["ratio"], 2), "| epi", round(r["epistrophe_sentence"]["ratio"], 2),
          round(r["epistrophe_clause"]["ratio"], 2), "| chi", round(r["pairs"]["index"], 3), r["pairs"]["chiastic"],
          r["pairs"]["parallel"], round(r["pairs"]["chiastic_per_10k_words"], 2), flush=True)

# ------------------------------------------------------------------ the vocabulary of political economy
CONCEPTS = [  # concept, German (Marx; case-sensitive), Albanian (Nano; case-insensitive)
    ("class", r"\bKlasse\w*|\bKlassen\w*|\bklassen\w*", r"\bklas(?:a|ë|ës|ën|at|ave|ash|e)\b"),
    ("capital", r"\b[Kk]apital\w*", r"\bkapital\w*"),
    ("bourgeoisie", r"\b[Bb]ourgeois\w*", r"\bborgjez\w*"),
    ("proletariat", r"\b[Pp]roletari\w*", r"\bproletar\w*"),
    ("revolution", r"\b[Rr]evolution\w*", r"\brevolucion\w*"),
    ("communism", r"\b[Kk]ommunis\w*", r"\bkomunis\w*|\bkomuniz\w*"),
    ("worker", r"\bArbeiter\w*|\barbeiter\w*", r"\bpunëtor\w*"),
    ("commodity", r"\bWare\w*|\bWaren\w*", r"\bmall(?:ra|rat|rave|rash)\b"),
    ("property", r"\bEigentum\w*|\beigentum\w*", r"\bpron(?:a|ë|ës|ën|at|ave|ash|ësi\w*)\b"),
    ("market", r"\bM[aä]rkt\w*", r"\btreg(?:u|ut|un|je|jet|jeve|jesh)?\b"),
    ("privatisation", r"\b[Pp]rivatisier\w*", r"\bprivatiz\w*"),
]
SPECIAL_SQ = {"political_class": r"\bklas\w*\s+(?:e\s+|të\s+)?politik\w*",
              "market_economy": r"\bekonomi\w*\s+(?:e\s+|të\s+)?tregut\b",
              "marxist_doctrine": r"\bmarks\w*|\bleninist\w*|\bsocializ\w*|\bmbivler\w*|\bluft\w*\s+(?:e\s+|të\s+)?klasave|"
                                  r"\bforca\w*\s+prodhuese|\bmjet\w*\s+(?:e\s+|të\s+)?prodhimit|\bmarrëdhëni\w*\s+(?:e\s+|të\s+)?prodhimit|"
                                  r"\bdiktatur\w*\s+(?:e\s+|të\s+)?proletariatit|\bmaterializ\w*|\bdialektik\w*"}


def per10k(docs, pat, flags=re.I):
    S = [s for d in docs for s in d]; n = sum(len(words(s)) for s in S)
    c = sum(len(re.findall(pat, s["text"], flags)) for s in S)
    return {"count": c, "per_10k": 1e4 * c / n}


R["vocabulary"] = {c: {"marx_written": per10k(marx_w, d, 0), "marx_spoken": per10k(marx_s, d, 0),
                       **{f"nano_{t}": per10k(nano_docs_term[t], a) for t in TERMS}, "nano": per10k(nano_docs, a)}
                   for c, d, a in CONCEPTS}
R["vocabulary_nano_special"] = {k: {**{t: per10k(nano_docs_term[t], p) for t in TERMS}, "all": per10k(nano_docs, p)}
                               for k, p in SPECIAL_SQ.items()}
# Nano's few uses of the doctrinal words, in context
ctx = []
for u in U:
    for s in u["sentences"]:
        if re.search(r"\bkapitaliz\w*|\bborgjez\w*|\bkomuniz\w*|“epërsi", s["text"], re.I):
            ctx.append({"date": u["date"], "session": u["session"], "text": s["text"]})
R["vocabulary_contexts"] = ctx
save("marx.json", R)
for c in R["vocabulary"]: print(c, {k: round(v["per_10k"], 2) for k, v in R["vocabulary"][c].items()})
print({k: {t: round(v[t]["per_10k"], 2) for t in v} for k, v in R["vocabulary_nano_special"].items()})
