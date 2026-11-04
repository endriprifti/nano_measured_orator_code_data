"""Section 10 — diachronic stylometry: Burrows' Delta on most-frequent-word profiles of 2,000-word
chunks, PCA, nearest-centroid attribution of premiership (leave-one-speech-out), a Mantel-type test of
stylistic distance against temporal distance, and genre-adjusted negative-binomial models for markers."""
import collections, math, re, datetime
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from common import *

U = load_units()
R = {}
rng = np.random.default_rng(21)
F = collections.Counter(all_words(U))
MFW = [w for w, _ in F.most_common(150)]
idx = {w: i for i, w in enumerate(MFW)}

# chunks of 2,000 words within speech units, units shorter than 2,000 words are pooled with their
# chronological neighbours of the same term and genre
CH = 2000
chunks = []
buf, meta = [], None
for u in U:
    w = unit_words(u)
    key = (u["term"], u["genre"])
    if meta and meta[0] != key:
        buf = []           # do not let a chunk straddle terms or genres
    for x in w:
        buf.append(x)
        if len(buf) == CH:
            chunks.append({"words": buf, "term": u["term"], "genre": u["genre"], "date": u["date"], "unit": u["id"]})
            buf = []
    meta = (key,)
R["n_chunks"] = len(chunks)
R["chunks_by_term"] = collections.Counter(c["term"] for c in chunks)
M = np.zeros((len(chunks), len(MFW)))
for i, c in enumerate(chunks):
    cnt = collections.Counter(c["words"])
    for w, j in idx.items(): M[i, j] = cnt[w] / CH
Z = (M - M.mean(0)) / M.std(0)
D = np.abs(Z[:, None, :] - Z[None, :, :]).mean(-1)          # Burrows' Delta
terms = np.array([c["term"] for c in chunks]); units = np.array([c["unit"] for c in chunks])
dates = np.array([datetime.date.fromisoformat(c["date"]).toordinal() for c in chunks])

# mean Delta within / between terms
R["delta_within_between"] = {}
for a in TERMS:
    for b in TERMS:
        if a > b: continue
        m = D[np.ix_(terms == a, terms == b)]
        if a == b: m = m[np.triu_indices_from(m, 1)]
        R["delta_within_between"][f"{a}-{b}"] = float(m.mean())

# nearest-centroid attribution of term, leave-one-speech-out (chunks of the held-out speech removed)
def attribute(Zm, labels, groups, K=TERMS):
    correct = 0; pred = []
    for i in range(len(Zm)):
        tr = groups != groups[i]
        cents = {t: Zm[tr & (labels == t)].mean(0) for t in K if np.any(tr & (labels == t))}
        dist = {t: np.abs(Zm[i] - c).mean() for t, c in cents.items()}
        p = min(dist, key=dist.get); pred.append(p); correct += p == labels[i]
    return correct / len(Zm), pred
acc, pred = attribute(Z, terms, units)
cm = pd.crosstab(pd.Series(terms, name="true"), pd.Series(pred, name="pred"))
null = []
for _ in range(200):
    perm = rng.permutation(terms); null.append(attribute(Z, perm, units)[0])
R["attribution"] = {"accuracy": acc, "null_mean": float(np.mean(null)), "null_95": float(np.percentile(null, 95)),
                    "p_perm": (1 + sum(n >= acc for n in null)) / 201, "confusion": cm.to_dict()}
# balanced accuracy
rec = [np.mean(np.array(pred)[terms == t] == t) for t in TERMS]
R["attribution"]["balanced_accuracy"] = float(np.mean(rec)); R["attribution"]["recall_by_term"] = dict(zip(TERMS, rec))

# attribution within genre (holds the rhetorical situation constant)
G2 = {"rejoinder": "adversarial", "interpellation": "adversarial", "motion": "adversarial", "questions": "adversarial",
      "programme": "expository", "report": "expository", "budget": "expository", "debate": "debate"}
genre2 = np.array([G2[c["genre"]] for c in chunks])
R["attribution_within_genre"] = {}
rng_g = np.random.default_rng(4)
for g in ("expository", "adversarial"):
    m = genre2 == g
    a_, _ = attribute(Z[m], terms[m], units[m])
    nl = [attribute(Z[m], rng_g.permutation(terms[m]), units[m])[0] for _ in range(300)]
    R["attribution_within_genre"][g] = {"n_by_term": collections.Counter(terms[m]), "accuracy": a_,
                                        "null_mean": float(np.mean(nl)), "p_perm": (1 + sum(n >= a_ for n in nl)) / 301}

# PCA
Zc = Z - Z.mean(0); Uu, S, Vt = np.linalg.svd(Zc, full_matrices=False)
pcs = Zc @ Vt[:2].T; expl = S ** 2 / np.sum(S ** 2)
R["pca_explained"] = expl[:5]
load = sorted(zip(MFW, Vt[0]), key=lambda x: x[1])
R["pc1_loadings_neg"] = load[:12]; R["pc1_loadings_pos"] = load[-12:]
R["pc1_vs_date_spearman"] = stats.spearmanr(dates, pcs[:, 0])
_df = pd.DataFrame({"pc1": pcs[:, 0], "term": terms, "genre2": genre2})
_m = smf.ols("pc1 ~ C(term, Treatment('T1')) + C(genre2, Treatment('expository'))", data=_df).fit()
R["pc1_ols"] = {"coef": _m.params.to_dict(), "p": _m.pvalues.to_dict(), "r2": _m.rsquared}
R["chunk_scores"] = [{"pc1": float(pcs[i, 0]), "pc2": float(pcs[i, 1]), "term": c["term"], "date": c["date"],
                      "genre": c["genre"], "unit": c["unit"]} for i, c in enumerate(chunks)]

# Mantel test: Delta vs. absolute time distance
T = np.abs(dates[:, None] - dates[None, :]).astype(float)
iu = np.triu_indices(len(D), 1)
r_obs = stats.spearmanr(D[iu], T[iu])[0]
rs = []
for _ in range(500):
    p = rng.permutation(len(D)); rs.append(stats.spearmanr(D[np.ix_(p, p)][iu], T[iu])[0])
R["mantel"] = {"spearman_r": r_obs, "p_perm": (1 + sum(r >= r_obs for r in rs)) / 501}
# within T3 only (removes the 1991/1997 gap): drift inside the third premiership
m3 = terms == "T3"; D3 = D[np.ix_(m3, m3)]; T3m = T[np.ix_(m3, m3)]; iu3 = np.triu_indices(m3.sum(), 1)
r3 = stats.spearmanr(D3[iu3], T3m[iu3])[0]; rs3 = []
for _ in range(500):
    p = rng.permutation(m3.sum()); rs3.append(stats.spearmanr(D3[np.ix_(p, p)][iu3], T3m[iu3])[0])
R["mantel_T3"] = {"spearman_r": r3, "p_perm": (1 + sum(r >= r3 for r in rs3)) / 501}

# ---------------------------------------------------------------- genre-adjusted models per speech unit
MARK = {
    "pron_1sg": r"\b(unë|mua|im|ime|tim|time|imi|imja)\b",
    "pron_2pl": r"\b(ju|juve|juaj|tuaj|tuaja|tuajat)\b",
    "pron_1pl": r"\b(ne|neve|ynë|jonë|tonë|tanë|sonë|tona|tanët|tonat|na)\b",
    "future_do_te": r"\bdo t[ëa]\b|\bdo t'",
    # a name after zoti/zonja (title word matched in any case, so sentence-initial "Zoti" counts; titles of office excluded)
    "address_zoti_name": r"\b(?i:zoti|zonja|zotit|zonjës)\s+(?!(?:Kryetar|President)\w*\b)[A-ZËÇ]\w+",
    "antithesis_jo_vetem_por": r"\bjo vetëm\b[^.?!]{0,250}?\bpor\b",
    "adversative_por": r"\bpor\b",
}
rows = []
for u in U:
    n = len(unit_words(u))
    if n < 300: continue
    text = " ".join(s["text"] for s in u["sentences"])
    r = {"unit": u["id"], "term": u["term"], "genre": u["genre"], "n": n,
         "msl": np.mean([len(s["tokens"]) for s in u["sentences"] if s["kind"] != "address"])}
    for k, p in MARK.items():
        r[k] = len(re.findall(p, text, 0 if k == "address_zoti_name" else re.I))
    rows.append(r)
df = pd.DataFrame(rows)
df["genre2"] = df["genre"].replace({"rejoinder": "adversarial", "interpellation": "adversarial", "motion": "adversarial",
                                    "questions": "adversarial", "programme": "expository", "report": "expository",
                                    "budget": "expository", "debate": "debate"})
R["glm_n_units"] = len(df)
R["glm"] = {}
for k in MARK:
    try:
        # NB2 model with the dispersion alpha estimated by maximum likelihood, exposure = words in the unit
        mod = smf.negativebinomial(f"{k} ~ C(term, Treatment('T1')) + C(genre2, Treatment('expository'))", data=df,
                                   exposure=df["n"]).fit(disp=0, maxiter=500)
        ci = mod.conf_int(); keep = [n for n in mod.params.index if n != "alpha"]
        R["glm"][k] = {"irr": {n: float(np.exp(mod.params[n])) for n in keep},
                       "p": {n: float(mod.pvalues[n]) for n in keep},
                       "ci": {n: [float(np.exp(ci.loc[n, 0])), float(np.exp(ci.loc[n, 1]))] for n in keep},
                       "alpha": float(mod.params["alpha"]), "alpha_ci": [float(ci.loc["alpha", 0]), float(ci.loc["alpha", 1])],
                       "converged": bool(mod.mle_retvals.get("converged", True))}
    except Exception as e:
        R["glm"][k] = {"error": str(e)}
mod = smf.ols("np.log(msl) ~ C(term, Treatment('T1')) + C(genre2, Treatment('expository'))", data=df).fit()
R["ols_log_msl"] = {"coef": {n: float(v) for n, v in mod.params.items()}, "p": {n: float(v) for n, v in mod.pvalues.items()},
                    "r2": mod.rsquared}
R["genre2_counts"] = df.groupby(["term", "genre2"]).size().to_dict()
R["genre2_counts"] = {f"{a}|{b}": int(v) for (a, b), v in R["genre2_counts"].items()}
save("stylometry.json", R)

# figure: PCA of chunks
fig, ax = plt.subplots(figsize=(3.6, 3.0))
for t in TERMS:
    m = terms == t
    ax.scatter(pcs[m, 0], pcs[m, 1], s=16, marker=TERM_MK[t], color=TERM_COL[t], edgecolor="white", lw=0.4,
               label=f"{TERM_LABEL[t]} (n = {m.sum()})")
ax.set_xlabel(f"PC1 ({100*expl[0]:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({100*expl[1]:.1f}%)")
ax.set_title("Function-word profiles of 2,000-word chunks"); ax.legend(fontsize=7, loc="best")
savefig(fig, "fig_pca")
print(json.dumps({k: v for k, v in R.items()}, indent=1, ensure_ascii=False,
                 default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:7000])
