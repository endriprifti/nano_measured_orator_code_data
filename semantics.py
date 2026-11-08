"""Section 9 — semantic geometry: skip-gram embeddings, SIF sentence vectors, trajectory measures
(speed, spread, circuitousness) against shuffled-order nulls, local coherence, ring composition,
and an NMF topic model of paragraphs with topic prevalence by term."""
import collections, math, random
import numpy as np
from scipy import stats
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from common import *

U = load_units()
R = {}
rng = np.random.default_rng(9)
sents_all = [words(s) for u in U for s in u["sentences"] if s["kind"] != "address"]
F = collections.Counter(w for s in sents_all for w in s); Ntok = sum(F.values())

w2v = Word2Vec(sentences=sents_all, vector_size=100, window=5, min_count=3, sg=1, negative=10,
               epochs=30, workers=1, seed=17)
R["w2v"] = {"vocab": len(w2v.wv), "dim": 100, "window": 5, "epochs": 30}
probe = {}
for w in ("reforma", "integrimi", "opozita", "qeveria", "demokracia", "shqiptarët", "kriza", "ligji"):
    if w in w2v.wv: probe[w] = [x for x, _ in w2v.wv.most_similar(w, topn=6)]
R["neighbours"] = probe

# ---------------------------------------------------------------- SIF sentence embeddings (Arora et al. 2017)
a = 1e-3
def sif(ws):
    vs = [a / (a + F[w] / Ntok) * w2v.wv[w] for w in ws if w in w2v.wv]
    return np.mean(vs, 0) if vs else None
E, owner = [], []
for ui, u in enumerate(U):
    for s in u["sentences"]:
        if s["kind"] == "address": continue
        v = sif(words(s))
        if v is not None: E.append(v); owner.append(ui)
E = np.array(E); owner = np.array(owner)
pc = np.linalg.svd(E - E.mean(0), full_matrices=False)[2][0]
E = E - np.outer(E @ pc, pc)                                     # remove common component
E = E / np.linalg.norm(E, axis=1, keepdims=True)

# ---------------------------------------------------------------- trajectories per speech unit
def tsp_len(P):
    """Approximate shortest Hamiltonian path (nearest neighbour from every start + 2-opt) — lower bound proxy."""
    n = len(P); D = np.linalg.norm(P[:, None] - P[None], axis=-1); best = 1e18
    for s in range(min(n, 8)):
        path = [s]; left = set(range(n)) - {s}
        while left:
            j = min(left, key=lambda k: D[path[-1], k]); path.append(j); left.remove(j)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    if D[path[i - 1], path[i]] + D[path[j], path[j + 1]] > D[path[i - 1], path[j]] + D[path[i], path[j + 1]] + 1e-12:
                        path[i:j + 1] = path[i:j + 1][::-1]; improved = True
        best = min(best, sum(D[path[k], path[k + 1]] for k in range(n - 1)))
    return best
def chunks(X, size=3):
    m = len(X) // size
    return np.array([X[i * size:(i + 1) * size].mean(0) for i in range(m)])
traj = []
rng_ring = np.random.default_rng(77)
for ui, u in enumerate(U):
    X = E[owner == ui]
    if len(X) < 45: continue
    C = chunks(X)
    sp = np.linalg.norm(np.diff(C, axis=0), axis=1).mean()
    spread = np.linalg.norm(C - C.mean(0), axis=1).mean()
    path = np.linalg.norm(np.diff(C, axis=0), axis=1).sum(); short = tsp_len(C)
    # null: shuffle sentence order then chunk
    nsp, ncirc = [], []
    for _ in range(50):
        Cs = chunks(X[rng.permutation(len(X))])
        nsp.append(np.linalg.norm(np.diff(Cs, axis=0), axis=1).mean())
        ncirc.append(np.linalg.norm(np.diff(Cs, axis=0), axis=1).sum() / short)
    # ring composition: first vs last decile similarity against random decile pairs
    k = max(1, len(X) // 10)
    first, last = X[:k].mean(0), X[-k:].mean(0)
    ring = float(first @ last / np.linalg.norm(first) / np.linalg.norm(last))
    nring = []
    for _ in range(200):          # unmatched null of the first version (random pairs, overlap allowed); kept for comparison
        i, j = rng.choice(len(X) - k, 2, replace=False)
        f2, l2 = X[i:i + k].mean(0), X[j:j + k].mean(0)
        nring.append(float(f2 @ l2 / np.linalg.norm(f2) / np.linalg.norm(l2)))
    # matched null: non-overlapping segments of the same size at least half a speech apart (separate random stream)
    gap = max(k, (len(X) - k) // 2); mring = []
    while len(mring) < 200:
        i = int(rng_ring.integers(0, len(X) - k - gap + 1)); j = int(rng_ring.integers(i + gap, len(X) - k + 1))
        if i == 0 and j == len(X) - k: continue           # exclude the first-last pair itself
        f2, l2 = X[i:i + k].mean(0), X[j:j + k].mean(0)
        mring.append(float(f2 @ l2 / np.linalg.norm(f2) / np.linalg.norm(l2)))
    # local coherence: adjacent sentences vs random pairs of distinct sentences in the same speech
    adj = np.mean(np.sum(X[:-1] * X[1:], 1))
    rp = rng.integers(0, len(X), (2000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]
    rnd = np.mean(np.sum(X[rp[:, 0]] * X[rp[:, 1]], 1))
    traj.append({"id": u["id"], "term": u["term"], "genre": u["genre"], "n_sent": len(X),
                 "speed": sp, "speed_null": np.mean(nsp), "speed_ratio": sp / np.mean(nsp),
                 "spread": spread, "circuitousness": path / short, "circ_null": np.mean(ncirc),
                 "ring": ring, "ring_null_mean": np.mean(nring), "ring_pct_unmatched": float(np.mean(np.array(nring) < ring)),
                 "ring_null_matched_mean": np.mean(mring), "ring_pct": float(np.mean(np.array(mring) < ring)),
                 "coh_adjacent": adj, "coh_random": rnd})
R["trajectories"] = traj
arr = lambda k: np.array([t[k] for t in traj])
R["trajectory_summary"] = {
    "n_units": len(traj),
    "speed_ratio_median": float(np.median(arr("speed_ratio"))),
    "speed_ratio_wilcoxon_p": stats.wilcoxon(arr("speed") - arr("speed_null")).pvalue,
    "circ_obs_median": float(np.median(arr("circuitousness"))), "circ_null_median": float(np.median(arr("circ_null"))),
    "circ_wilcoxon_p": stats.wilcoxon(arr("circuitousness") - arr("circ_null")).pvalue,
    "ring_pct_median": float(np.median(arr("ring_pct"))),
    "ring_pct_unmatched_median": float(np.median(arr("ring_pct_unmatched"))),
    "ring_unmatched_share_above_null_median": float(np.mean(arr("ring_pct_unmatched") > 0.5)),
    "ring_share_above_null_median": float(np.mean(arr("ring_pct") > 0.5)),
    "ring_binom_p": stats.binomtest(int(np.sum(arr("ring_pct") > 0.5)), len(traj), 0.5).pvalue,
    "coh_adjacent_mean": float(arr("coh_adjacent").mean()), "coh_random_mean": float(arr("coh_random").mean()),
    "coh_wilcoxon_p": stats.wilcoxon(arr("coh_adjacent") - arr("coh_random")).pvalue}
R["trajectory_by_term"] = {t: {k: float(np.median([x[k] for x in traj if x["term"] == t]))
                               for k in ("speed", "speed_ratio", "spread", "circuitousness", "ring_pct", "coh_adjacent")}
                           | {"n": sum(1 for x in traj if x["term"] == t)} for t in TERMS}
for k in ("speed", "spread", "circuitousness"):
    R["trajectory_by_term"][f"kruskal_{k}"] = stats.kruskal(*[[x[k] for x in traj if x["term"] == t] for t in TERMS]).pvalue

# ---------------------------------------------------------------- topic model on paragraphs
pars, pterm, punit, pgenre = [], [], [], []
for u in U:
    d = collections.defaultdict(list)
    for s in u["sentences"]:
        if s["kind"] != "address": d[s["par"]].extend(words(s))
    for p, ws in d.items():
        if len(ws) >= 30: pars.append(" ".join(ws)); pterm.append(u["term"]); punit.append(u["id"]); pgenre.append(u["genre"])
stop = {w for w, _ in F.most_common(150)}
tf = TfidfVectorizer(stop_words=list(stop), min_df=5, max_df=0.4, token_pattern=r"(?u)\b[\w'ëç-]{3,}\b", sublinear_tf=True)
X = tf.fit_transform(pars); vocab = np.array(tf.get_feature_names_out())
K = 12
nmf = NMF(n_components=K, init="nndsvda", random_state=1, max_iter=600, l1_ratio=0.0)
Wt = nmf.fit_transform(X); Ht = nmf.components_
topics = [list(vocab[np.argsort(-Ht[k])[:12]]) for k in range(K)]
share = Wt / np.maximum(Wt.sum(1, keepdims=True), 1e-12)
prev = {t: share[np.array(pterm) == t].mean(0) for t in TERMS}
R["topics"] = {"n_paragraphs": len(pars), "K": K, "top_words": topics,
               "prevalence_by_term": {t: prev[t] for t in TERMS}}
# thematic concentration: entropy of topic mixture per paragraph
ent = -(np.where(share > 0, share * np.log2(np.where(share > 0, share, 1)), 0)).sum(1)
R["topic_entropy_by_term"] = {t: float(ent[np.array(pterm) == t].mean()) for t in TERMS}
# the same contrast at constant genre: OLS of paragraph entropy on premiership and the three-level genre factor,
# with standard errors clustered by speech unit (paragraphs of one speech are not independent)
import pandas as pd, statsmodels.formula.api as smf
G2 = {"rejoinder": "adversarial", "interpellation": "adversarial", "motion": "adversarial", "questions": "adversarial",
      "programme": "expository", "report": "expository", "budget": "expository", "debate": "debate"}
_d = pd.DataFrame({"ent": ent, "term": pterm, "genre2": [G2[g] for g in pgenre], "unit": punit})
from patsy import EvalEnvironment   # a clean namespace: this script uses the name C for an array
_m = smf.ols("ent ~ C(term, Treatment('T1')) + C(genre2, Treatment('expository'))", data=_d,
             eval_env=EvalEnvironment([{}])).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(_d["unit"])[0]})
R["topic_entropy_ols"] = {"coef": _m.params.to_dict(), "p": _m.pvalues.to_dict(),
                          "ci": {k: list(v) for k, v in _m.conf_int().T.to_dict("list").items()},
                          "n_paragraphs": int(len(_d)), "n_units": int(_d["unit"].nunique()), "r2": float(_m.rsquared)}
R["topic_entropy_by_genre"] = {g: float(_d.loc[_d.genre2 == g, "ent"].mean()) for g in sorted(set(_d.genre2))}
save("semantics.json", R)
print(json.dumps({k: R[k] for k in ("w2v", "neighbours", "trajectory_summary", "trajectory_by_term", "topic_entropy_by_term")},
                 indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
for k, t in enumerate(topics):
    print(k, " ".join(t[:10]), [round(float(prev[x][k]), 3) for x in TERMS])
