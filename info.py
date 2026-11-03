"""Section 8 (model-free part) — entropy, redundancy and the Birkhoff–Bense index at several levels;
Shannon block entropies of letters; compression-based entropy rate and word-order information
(Montemurro–Zanette style); normalised compression distances between terms and genres."""
import collections, math, lzma, bz2, random
import numpy as np
from common import *

U = load_units()
R = {}
rng = random.Random(3)

def H_plugin(counts):
    n = sum(counts); p = np.array([c / n for c in counts if c > 0]); return float(-(p * np.log2(p)).sum())
def H_mm(counts):            # Miller–Madow bias correction
    n = sum(counts); k = sum(1 for c in counts if c > 0); return H_plugin(counts) + (k - 1) / (2 * n * math.log(2))
def H_cs(counts):            # Chao–Shen coverage-adjusted estimator
    counts = np.array([c for c in counts if c > 0], float); n = counts.sum()
    f1 = (counts == 1).sum(); C = 1 - f1 / n if f1 < n else 1 - (f1 - 1) / n
    pa = C * counts / n; la = 1 - (1 - pa) ** n
    return float(-np.sum(pa * np.log2(pa) / la))
def level(counts, V=None):
    V = V or sum(1 for c in counts if c > 0)
    h = H_cs(counts); hmax = math.log2(V); r = 1 - h / hmax
    return {"H": h, "H_plugin": H_plugin(counts), "H_MM": H_mm(counts), "Hmax": hmax, "R": r, "M_B": r / h, "V": V,
            "n": int(sum(counts))}

# ---------------------------------------------------------------- level 1: letters (+ word boundary)
ALPH = set("abcçdeëfghijklmnopqrstuvxyz")
def letter_stream(units):
    out = []
    for u in units:
        for s in u["sentences"]:
            for t in s["tokens"]:
                if t[0].isdigit(): continue
                out.extend(ch for ch in t.lower() if ch in ALPH); out.append(" ")
    return "".join(out)
LS = letter_stream(U)
R["letters"] = level(list(collections.Counter(LS).values()), V=len(ALPH) + 1)
# Shannon's F_N: conditional block entropies F_N = H_N - H_{N-1}
blocks = {}
for n in range(1, 7):
    c = collections.Counter(LS[i:i + n] for i in range(len(LS) - n + 1))
    blocks[n] = H_cs(list(c.values()))
R["letters_FN"] = {n: blocks[n] - (blocks[n - 1] if n > 1 else 0) for n in blocks}
R["letters_block_H"] = blocks

# ---------------------------------------------------------------- level 2: word forms (by term, matched N)
N0 = 30000
def words_of(units): return [w for u in units for w in unit_words(u)]
W = words_of(U)
R["words_all"] = level(list(collections.Counter(W).values()))
R["words_by_term_matched"] = {}
for t in TERMS:
    tw = words_of([u for u in U if u["term"] == t])
    offs = np.linspace(0, len(tw) - N0, 10).astype(int)
    vals = [level(list(collections.Counter(tw[o:o + N0]).values())) for o in offs]
    R["words_by_term_matched"][t] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals])))
                                     for k in ("H", "R", "M_B", "V")}
# ---------------------------------------------------------------- level 3: sentence openings; level 4: length classes
def sent_list(units): return [s for u in units for s in u["sentences"] if s["kind"] != "address"]
def first_words(units):
    out = []
    for s in sent_list(units):
        ws = [t.lower() for t in s["tokens"] if not t[0].isdigit()]
        if ws: out.append(ws[0])
    return out
def clause_first(units):
    return [c[0].lower() for s in sent_list(units) for c in s["clauses"] if c and not c[0][0].isdigit()]
def len_classes(units):
    return [min(len(s["tokens"]) // 5, 20) for s in sent_list(units)]
R["sentence_openings"] = level(list(collections.Counter(first_words(U)).values()))
R["clause_openings"] = level(list(collections.Counter(clause_first(U)).values()))
R["length_classes"] = level(list(collections.Counter(len_classes(U)).values()), V=21)
R["by_term_levels"] = {}
for t in TERMS:
    ut = [u for u in U if u["term"] == t]
    R["by_term_levels"][t] = {"sentence_openings": level(list(collections.Counter(first_words(ut)).values())),
                              "clause_openings": level(list(collections.Counter(clause_first(ut)).values())),
                              "length_classes": level(list(collections.Counter(len_classes(ut)).values()), V=21),
                              "letters": level(list(collections.Counter(letter_stream(ut)).values()), V=len(ALPH) + 1)}
# matched-size sentence openings (entropy is sample-size dependent): 1,000 sentences per term, 50 draws
R["openings_matched"] = {}
for t in TERMS:
    fw = first_words([u for u in U if u["term"] == t]); vals = []
    for _ in range(50):
        smp = rng.sample(fw, 1000); vals.append(level(list(collections.Counter(smp).values())))
    R["openings_matched"][t] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals])))
                                for k in ("H", "R", "M_B")}

# ---------------------------------------------------------------- compression: entropy rate and word-order information
def ids_bytes(seq, vocab):
    # fixed-width 2-byte code per word id (V < 65536): the compressor sees word symbols, not letters
    return b"".join(vocab[w].to_bytes(2, "little") for w in seq)
vocab = {w: i for i, w in enumerate(sorted(set(W)))}
def comp_bits(b):
    return 8 * len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))
def word_order_info(seq, reps=5):
    orig = comp_bits(ids_bytes(seq, vocab))
    sh = []
    for _ in range(reps):
        s2 = seq[:]; rng.shuffle(s2); sh.append(comp_bits(ids_bytes(s2, vocab)))
    return {"bits_per_word_orig": orig / len(seq), "bits_per_word_shuffled": np.mean(sh) / len(seq),
            "D_bits_per_word": (np.mean(sh) - orig) / len(seq)}
R["compression_words_all"] = word_order_info(W)
R["compression_words_by_term"] = {}
for t in TERMS:
    tw = words_of([u for u in U if u["term"] == t])[:N0]
    R["compression_words_by_term"][t] = word_order_info(tw)
# letter-level compression of the running text (bits per character), per term, matched length
R["compression_letters_by_term"] = {}
for t in TERMS:
    ls = letter_stream([u for u in U if u["term"] == t])[:180000].encode("utf-8")
    R["compression_letters_by_term"][t] = {"lzma_bits_per_char": comp_bits(ls) / 180000,
                                           "bz2_bits_per_char": 8 * len(bz2.compress(ls, 9)) / 180000}

# ---------------------------------------------------------------- normalised compression distance (Cilibrasi & Vitanyi)
def C(b): return len(lzma.compress(b, preset=9))
def ncd(x, y):
    cx, cy, cxy = C(x), C(y), C(x + y); return (cxy - min(cx, cy)) / max(cx, cy)
samples = {}
for t in TERMS:
    txt = " ".join(" ".join(s["tokens"]).lower() for u in U if u["term"] == t for s in u["sentences"])
    samples[t] = txt[:150000].encode("utf-8")
R["ncd_terms"] = {f"{a}-{b}": ncd(samples[a], samples[b]) for a, b in (("T1", "T2"), ("T2", "T3"), ("T1", "T3"))}
gen = collections.defaultdict(str)
for u in U: gen[u["genre"]] += " " + " ".join(" ".join(s["tokens"]).lower() for s in u["sentences"])
G = sorted(g for g in gen if len(gen[g]) > 60000)
R["ncd_genres"] = {"genres": G, "matrix": [[ncd(gen[a][:60000].encode(), gen[b][:60000].encode()) if a != b else 0.0
                                             for b in G] for a in G]}
save("info.json", R)
print(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:6000])
