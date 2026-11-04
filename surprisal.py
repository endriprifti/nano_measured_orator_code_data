"""Section 8 (model-based part) — analysis of cross-validated word surprisal: information rate,
word-order information, Birkhoff–Bense index at the model level, information contour of the sentence and
arc of the speech, uniformity of information density, word length vs information, surprisal at
anaphoric clause openings and at sentence ends, extreme sentences."""
import collections, math
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.formula.api as smf
from common import *
from prep import syllables, phonemes

U = load_units(); S = J = None
SU = json.load(open(os.path.join(RES, "surprisal.json")))
meta, SUN = SU["meta"], SU["units"]
INFO = json.load(open(os.path.join(RES, "info.json")))
MODELS = ("kn", "gpt")
R = {"meta": meta}

# ---------------------------------------------------------------- sentences with word surprisal
sents_all = []      # list of dicts: unit, term, genre, words, surprisal per model, pos in speech
for u in U:
    rec = SUN[u["id"]]; toks, isw = rec["tokens"], rec["isword"]
    cur = {"w": [], "kn": [], "gpt": [], "punct_after": []}; k = 0; sl = []
    for j, t in enumerate(toks):
        if t == "¶":
            sl.append(cur); cur = {"w": [], "kn": [], "gpt": [], "punct_after": []}; continue
        if isw[j]:
            cur["w"].append(t); cur["kn"].append(rec["kn"][j]); cur["gpt"].append(rec["gpt"][j])
            cur["punct_after"].append(toks[j + 1] if j + 1 < len(toks) and toks[j + 1] in ",;:.!?" else "")
    # the stream has one segment per sentence of units.json (checked: 7,125 = 7,125), so kinds and clauses carry over
    assert len(sl) == len(u["sentences"]), u["id"]
    na = [j for j, x in enumerate(u["sentences"]) if x["kind"] != "address"]; rank = {j: r for r, j in enumerate(na)}
    for i, s in enumerate(sl):
        if s["w"]:
            src = u["sentences"][i]
            s.update({"unit": u["id"], "term": u["term"], "genre": u["genre"], "kind": src["kind"],
                      "idx": rank.get(i, -1), "n_in_unit": len(na), "src_idx": i,
                      "clause_sizes": [sum(1 for t in c if not t[0].isdigit()) for c in src["clauses"] if c]})
            sents_all.append(s)
# model-level rates use every scored word; all sentence-level analyses use the same sentence set as Sections 6-7
# (vocative address lines excluded)
sents = [s for s in sents_all if s["kind"] != "address"]
R["n_sentences_all"] = len(sents_all); R["n_sentences"] = len(sents)
allw = {m: np.array([x for s in sents_all for x in s[m] if x == x]) for m in MODELS}
R["rate"] = {m: {"bits_per_word": float(allw[m].mean()), "sd": float(allw[m].std()), "median": float(np.median(allw[m])),
                 "perplexity_per_word": float(2 ** allw[m].mean()), "n_words": int(len(allw[m])),
                 "q": np.percentile(allw[m], [5, 25, 75, 95, 99])} for m in MODELS}
# all tokens incl. punctuation & boundaries, bits per character of the running text
for m in MODELS:
    tot_bits = 0.0; chars = 0
    for u in U:
        rec = SUN[u["id"]]
        v = np.array(rec[m], float); tot_bits += np.nansum(v)
        chars += sum(len(t) + 1 for t in rec["tokens"])
    R["rate"][m]["bits_per_char"] = tot_bits / chars
both = np.array([(a, b) for s in sents_all for a, b in zip(s["kn"], s["gpt"]) if a == a and b == b])
R["kn_gpt_corr"] = {"pearson": float(np.corrcoef(both[:, 0], both[:, 1])[0, 1]),
                    "spearman": float(stats.spearmanr(both[:, 0], both[:, 1])[0])}
best = min(MODELS, key=lambda m: R["rate"][m]["bits_per_word"]); R["best_model"] = best

# ---------------------------------------------------------------- word-order information and M_B at model level
Huni = INFO["words_all"]["H"]; V = INFO["words_all"]["V"]
for m in MODELS:
    H = R["rate"][m]["bits_per_word"]
    R.setdefault("order", {})[m] = {"H_unigram": Huni, "H_model": H, "D_lower_bound": Huni - H,
                                    "R_model": 1 - H / math.log2(V), "M_B_model": (1 - H / math.log2(V)) / H}

# ---------------------------------------------------------------- by term and genre (speech-unit means)
rows = []
for u in U:
    ss = [s for s in sents if s["unit"] == u["id"]]
    w = [x for s in ss for x in s[best] if x == x]
    if len(w) >= 300:
        rows.append({"unit": u["id"], "term": u["term"], "genre": u["genre"], "n": len(w), "mean": np.mean(w),
                     "sd_within_sent": np.mean([np.std(s[best]) for s in ss if len(s[best]) >= 8])})
df = pd.DataFrame(rows)
df["genre2"] = df["genre"].replace({"rejoinder": "adversarial", "interpellation": "adversarial", "motion": "adversarial",
                                    "questions": "adversarial", "programme": "expository", "report": "expository",
                                    "budget": "expository", "debate": "debate"})
R["by_term_units"] = df.groupby("term")["mean"].agg(["median", "mean", "count"]).to_dict()
R["by_genre_units"] = df.groupby("genre2")["mean"].agg(["median", "mean", "count"]).to_dict()
mod = smf.ols("mean ~ C(term, Treatment('T1')) + C(genre2, Treatment('expository')) + np.log(n)", data=df).fit()
R["ols_unit_surprisal"] = {"coef": mod.params.to_dict(), "p": mod.pvalues.to_dict(), "r2": mod.rsquared, "n": len(df)}
R["by_term_tokens"] = {t: float(np.mean([x for s in sents if s["term"] == t for x in s[best] if x == x])) for t in TERMS}

# ---------------------------------------------------------------- information contour of the sentence
cont = collections.defaultdict(list); first_vs = []
for s in sents:
    v = s[best]; n = len(v)
    if n < 10: continue
    for i, x in enumerate(v):
        if x == x: cont[min(int(10 * i / n), 9)].append(x)
R["sentence_contour"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v) / math.sqrt(len(v)))} for k, v in sorted(cont.items())}
# position effects: first word, last word, word before a comma
fw = [s[best][0] for s in sents if len(s[best]) >= 6 and s[best][0] == s[best][0]]
lw = [s[best][-1] for s in sents if len(s[best]) >= 6 and s[best][-1] == s[best][-1]]
mid = [x for s in sents if len(s[best]) >= 6 for x in s[best][1:-1] if x == x]
R["positions"] = {"first": float(np.mean(fw)), "last": float(np.mean(lw)), "internal": float(np.mean(mid)),
                  "p_last_vs_internal": stats.mannwhitneyu(lw, mid).pvalue}
# 'informational Menzerath': per-word surprisal vs sentence length
sl = np.array([len(s[best]) for s in sents if len(s[best]) >= 3])
ms = np.array([np.nanmean(s[best]) for s in sents if len(s[best]) >= 3])
R["length_vs_rate"] = {"spearman": stats.spearmanr(sl, ms)[0], "p": stats.spearmanr(sl, ms)[1],
                       "by_bin": {b: float(ms[(sl >= lo) & (sl < hi)].mean()) for b, (lo, hi) in
                                  {"3-10": (3, 10), "10-20": (10, 20), "20-30": (20, 30), "30-45": (30, 45),
                                   "45-70": (45, 70), "70+": (70, 999)}.items()}}

# ---------------------------------------------------------------- arc of the speech
arc = collections.defaultdict(list)
for s in sents:
    if s["n_in_unit"] >= 30 and len(s[best]) >= 3:
        arc[min(int(10 * s["idx"] / s["n_in_unit"]), 9)].append(np.nanmean(s[best]))
R["speech_arc"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v) / math.sqrt(len(v))), "n": len(v)} for k, v in sorted(arc.items())}

# ---------------------------------------------------------------- uniform information density
uid = []
for s in sents:
    v = np.array([x for x in s[best] if x == x])
    if len(v) >= 8:
        uid.append({"term": s["term"], "genre": s["genre"], "n": len(v), "var": v.var(), "mad": np.abs(np.diff(v)).mean(),
                    "mean": v.mean()})
ud = pd.DataFrame(uid)
# null: same words, random order inside the sentence, surprisal values permuted (tests ordering of information)
rng = np.random.default_rng(1)
obs_mad = ud["mad"].mean(); null_mad = []
vs = [np.array([x for x in s[best] if x == x]) for s in sents]
vs = [v for v in vs if len(v) >= 8]
for _ in range(100):
    null_mad.append(np.mean([np.abs(np.diff(rng.permutation(v))).mean() for v in vs]))
R["uid"] = {"mean_var": ud["var"].mean(), "mean_mad": obs_mad, "null_mad_mean": float(np.mean(null_mad)),
            "null_mad_sd": float(np.std(null_mad)), "z_mad": float((obs_mad - np.mean(null_mad)) / np.std(null_mad)),
            "by_term_var": ud.groupby("term")["var"].mean().to_dict(), "by_term_mad": ud.groupby("term")["mad"].mean().to_dict()}
# lag-1 autocorrelation of surprisal within sentences (smoothing vs alternation)
ac = [np.corrcoef(v[:-1], v[1:])[0, 1] for v in vs if v.std() > 0]
R["uid"]["lag1_autocorr_mean"] = float(np.nanmean(ac))

# ---------------------------------------------------------------- word length vs information (Piantadosi et al. 2011)
F = collections.Counter(w for s in sents for w in s["w"]); Ntok = sum(F.values())
ctx = collections.defaultdict(list)
for s in sents:
    for w, x in zip(s["w"], s[best]):
        if x == x: ctx[w].append(x)
types = [w for w, c in F.items() if c >= 20]
length = np.array([phonemes(w) for w in types]); info = np.array([np.mean(ctx[w]) for w in types])
freqinfo = np.array([-math.log2(F[w] / Ntok) for w in types])
def partial(x, y, z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    return stats.pearsonr(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))
R["length_info"] = {"n_types": len(types),
                    "spearman_len_context_info": stats.spearmanr(length, info)[0],
                    "spearman_len_unigram_info": stats.spearmanr(length, freqinfo)[0],
                    "partial_len_context_given_unigram": partial(length, info, freqinfo)[0],
                    "partial_len_unigram_given_context": partial(length, freqinfo, info)[0]}

# ---------------------------------------------------------------- surprisal at anaphoric clause openings
# re-align clause boundaries from punctuation recorded after each word
ana_first, other_first, ana_rest, other_rest = [], [], [], []
n_fallback = 0
for s in sents:
    v = s[best]
    if sum(s["clause_sizes"]) == len(s["w"]):          # clause segmentation of prep.py (punctuation, brackets, dashes)
        clauses, k = [], 0                                # a clause without words (a numeral) breaks adjacency, as in figures.py
        for n in s["clause_sizes"]:
            clauses.append(list(zip(s["w"][k:k + n], v[k:k + n]))); k += n
    else:                                              # tokenisation mismatch: fall back to commas, semicolons, colons
        n_fallback += 1; clauses = [[]]
        for i, (w, p) in enumerate(zip(s["w"], s["punct_after"])):
            clauses[-1].append((w, v[i]))
            if p in (",", ";", ":"): clauses.append([])
        clauses = [c for c in clauses if c]
    for j in range(1, len(clauses)):
        c, prev = clauses[j], clauses[j - 1]
        if not c or not prev: continue
        if c[0][1] != c[0][1]: continue
        rest = [x for _, x in c[1:] if x == x]
        if c[0][0] == prev[0][0]:
            ana_first.append(c[0][1]); ana_rest.extend(rest)
        else:
            other_first.append(c[0][1]); other_rest.extend(rest)
R["anaphora_surprisal"] = {"n_anaphoric": len(ana_first), "n_clause_fallback": n_fallback, "first_anaphoric": float(np.mean(ana_first)),
                           "first_other": float(np.mean(other_first)),
                           "p_first": stats.mannwhitneyu(ana_first, other_first).pvalue,
                           "rest_anaphoric": float(np.mean(ana_rest)), "rest_other": float(np.mean(other_rest)),
                           "p_rest": stats.mannwhitneyu(ana_rest, other_rest).pvalue}

# ---------------------------------------------------------------- peaks and extreme sentences
v = allw[best]; thr = v.mean() + 2 * v.std()
R["peaks"] = {"threshold_bits": float(thr), "share": float((v > thr).mean())}
pk = collections.defaultdict(list)
for s in sents:
    n = len(s[best])
    if n < 10: continue
    for i, x in enumerate(s[best]):
        if x == x: pk[min(int(10 * i / n), 9)].append(x > thr)
R["peaks"]["by_decile"] = {k: float(np.mean(v_)) for k, v_ in sorted(pk.items())}
texts = {(u["id"], i): s["text"] for u in U for i, s in enumerate(u["sentences"])}
scored = [(np.nanmean(s[best]), s["unit"], s["src_idx"], len(s["w"])) for s in sents if 12 <= len(s["w"]) <= 40]
scored.sort()
def sentence_text(unit, idx):
    u = next(x for x in U if x["id"] == unit); return u["sentences"][idx]["text"]
R["most_predictable"] = [(float(a), b, sentence_text(b, c)) for a, b, c, _ in scored[:6]]
R["most_surprising"] = [(float(a), b, sentence_text(b, c)) for a, b, c, _ in scored[-6:]]
save("surprisal_analysis.json", R)
print(json.dumps({k: v for k, v in R.items() if k not in ("most_predictable", "most_surprising")}, indent=1,
                 ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
for x in R["most_predictable"] + R["most_surprising"]: print(round(x[0], 2), x[1], x[2][:200])

# ---------------------------------------------------------------- self-repetition across speeches (set pieces)
occ = collections.defaultdict(set)
for u in U:
    for s in u["sentences"]:
        w = words(s)
        if s["kind"] != "address" and len(w) >= 10: occ[" ".join(w)].add(u["id"])
G = collections.defaultdict(set)
for u in U:
    w = unit_words(u)
    for i in range(len(w) - 7): G[tuple(w[i:i + 8])].add(u["id"])
cov_t = {t: [0, 0] for t in TERMS}
for u in U:
    w = unit_words(u); cov = [False] * len(w)
    for i in range(len(w) - 7):
        if len(G[tuple(w[i:i + 8])]) >= 2:
            for j in range(i, i + 8): cov[j] = True
    cov_t[u["term"]][0] += sum(cov); cov_t[u["term"]][1] += len(w)
R["self_repetition"] = {
    "sentences_ge10": sum(1 for u in U for s in u["sentences"] if s["kind"] != "address" and len(words(s)) >= 10),
    "verbatim_repeated_distinct": sum(1 for v in occ.values() if len(v) >= 2),
    "share_tokens_in_shared_8grams": sum(c for c, _ in cov_t.values()) / sum(n for _, n in cov_t.values()),
    "share_by_term": {t: c / n for t, (c, n) in cov_t.items()}}
save("surprisal_analysis.json", R)
print(R["self_repetition"])
