"""Section 5 — lexical economy: Zipf, Zipf–Mandelbrot, two-regime Zipf, Heaps,
frequency-spectrum estimators, lexical-richness indices (whole corpus, by term at matched N,
and per speech unit)."""
import collections, math, random
import numpy as np
from scipy import optimize, stats
import powerlaw
from common import *

U = load_units()
W = all_words(U)
N = len(W)
F = collections.Counter(W)
V = len(F)
freqs = np.array(sorted(F.values(), reverse=True), dtype=float)
ranks = np.arange(1, V + 1, dtype=float)
R = {"N": N, "V": V}

# ---------------------------------------------------------------- frequency spectrum
spec = collections.Counter(F.values())
V1, V2, V3 = spec[1], spec[2], spec[3]
R["spectrum"] = {"V1": V1, "V2": V2, "V3": V3, "hapax_share_types": V1 / V, "hapax_share_tokens": V1 / N}
R["good_turing_unseen_mass"] = V1 / N
R["chao1"] = V + V1 ** 2 / (2 * V2)
# Efron–Thisted expected new types in t*N further tokens, Euler-smoothed (truncated alternating series)
def efron_thisted(t, kmax=25):
    s = 0.0
    for k in range(1, kmax + 1):
        # binomial smoothing (Efron & Thisted 1976, eq. for t>1 requires smoothing; t<=1 is stable)
        s += (-1) ** (k + 1) * t ** k * spec.get(k, 0)
    return s
R["efron_thisted_new_types_t0.5"] = efron_thisted(0.5)
R["efron_thisted_new_types_t1"] = efron_thisted(1.0)

# ---------------------------------------------------------------- Zipf fits on rank-frequency
def zm_negll(params, f, r):
    a, b = params
    if a <= 0 or b <= -0.99: return 1e18
    logp = -a * np.log(r + b)
    logZ = np.log(np.sum(np.exp(logp - logp.max()))) + logp.max()
    return -np.sum(f * (logp - logZ))

def fit_zipf(f):
    r = np.arange(1, len(f) + 1, dtype=float)
    z = optimize.minimize_scalar(lambda a: zm_negll((a, 0.0), f, r), bounds=(0.3, 3), method="bounded")
    zm = optimize.minimize(zm_negll, x0=(1.0, 1.0), args=(f, r), method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    # OLS on log-log over ranks 1..V (descriptive, for comparability with older literature)
    sl, ic, rr, *_ = stats.linregress(np.log10(r), np.log10(f))
    return {"zipf_a_mle": z.x, "zipf_negll": z.fun, "zm_a": zm.x[0], "zm_b": zm.x[1], "zm_negll": zm.fun,
            "ols_slope": -sl, "ols_r2": rr ** 2,
            "lr_zm_vs_zipf": 2 * (z.fun - zm.fun)}
R["zipf_corpus"] = fit_zipf(freqs)

# block bootstrap over speech units for CI of exponents
rng = np.random.default_rng(42)
uw = [unit_words(u) for u in U]
boot = []
for b in range(200):
    idx = rng.integers(0, len(uw), len(uw))
    c = collections.Counter(w for i in idx for w in uw[i])
    f = np.array(sorted(c.values(), reverse=True), float)
    fz = fit_zipf(f); boot.append((fz["zipf_a_mle"], fz["zm_a"], fz["zm_b"]))
boot = np.array(boot)
R["zipf_corpus"]["boot_ci_zipf_a"] = np.percentile(boot[:, 0], [2.5, 97.5])
R["zipf_corpus"]["boot_ci_zm_a"] = np.percentile(boot[:, 1], [2.5, 97.5])
R["zipf_corpus"]["boot_ci_zm_b"] = np.percentile(boot[:, 2], [2.5, 97.5])

# two-regime (kernel vs. specialised lexicon) segmented fit in log-log, ranks >= 2
lr_, lf_ = np.log10(ranks), np.log10(freqs)
def seg_sse(bp):
    m1 = lr_ <= bp; m2 = ~m1
    if m1.sum() < 10 or m2.sum() < 10: return 1e18, None
    # continuous piecewise: f = c + s1*x for x<=bp, c + s1*bp + s2*(x-bp) after
    X = np.column_stack([np.ones_like(lr_), np.minimum(lr_, bp), np.maximum(lr_ - bp, 0)])
    # weight by inverse density of points in log space: use binned representative points
    beta, *_ = np.linalg.lstsq(X[sel], lf_[sel], rcond=None)
    res = lf_[sel] - X[sel] @ beta
    return float(res @ res), beta
# log-binned representative points avoid the dominance of the hapax plateau
edges = np.unique(np.round(np.logspace(0, np.log10(V), 60)).astype(int))
sel = np.zeros(V, bool); sel[edges - 1] = True
best = min(((seg_sse(bp)[0], bp) for bp in np.linspace(1.5, 3.8, 231)))
sse, bp = best; _, beta = seg_sse(bp)
R["two_regime"] = {"break_rank": 10 ** bp, "exp_kernel": -beta[1], "exp_tail": -(beta[1] + beta[2])}

# frequency distribution P(f) via Clauset–Shalizi–Newman (powerlaw package)
fit = powerlaw.Fit(freqs, discrete=True, verbose=False)
Rln, pln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
Rtr, ptr = fit.distribution_compare("power_law", "truncated_power_law", normalized_ratio=True)
R["csn"] = {"alpha": fit.power_law.alpha, "sigma": fit.power_law.sigma, "xmin": fit.power_law.xmin,
            "n_tail": int((freqs >= fit.power_law.xmin).sum()),
            "implied_zipf_a": 1 / (fit.power_law.alpha - 1),
            "LR_vs_lognormal": Rln, "p_vs_lognormal": pln, "LR_vs_truncated": Rtr, "p_vs_truncated": ptr}

# ---------------------------------------------------------------- Heaps' law
def heaps_curve(seq, step=500):
    seen, out = set(), []
    for i, w in enumerate(seq, 1):
        seen.add(w)
        if i % step == 0: out.append((i, len(seen)))
    return np.array(out, float)
hc = heaps_curve(W)
def heaps_fit(hc, nmin=5000):
    m = hc[:, 0] >= nmin
    sl, ic, rr, *_ = stats.linregress(np.log(hc[m, 0]), np.log(hc[m, 1]))
    return {"beta": sl, "K": math.exp(ic), "r2": rr ** 2}
R["heaps_chrono"] = heaps_fit(hc)
shuf = W[:]; random.Random(7).shuffle(shuf)
hcs = heaps_curve(shuf)
R["heaps_shuffled"] = heaps_fit(hcs)
# Herdan's C and its inverse relation to Zipf a under the Zipf->Heaps link (beta ≈ 1/a for a>1)
R["herdan_C"] = math.log(V) / math.log(N)

# ---------------------------------------------------------------- lexical richness indices
def yule_k(c):
    n = sum(c.values()); s2 = sum(v * v for v in c.values())
    return 1e4 * (s2 - n) / (n * n)
def simpson_d(c):
    n = sum(c.values()); return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))
def honore_h(c):
    n = sum(c.values()); v = len(c); v1 = sum(1 for x in c.values() if x == 1)
    return 100 * math.log(n) / (1 - v1 / v) if v1 < v else float("nan")
def sichel_s(c):
    v = len(c); return sum(1 for x in c.values() if x == 2) / v
def brunet_w(c):
    n = sum(c.values()); v = len(c); return n ** (v ** -0.165)
def guiraud(c):
    return len(c) / math.sqrt(sum(c.values()))
def mattr(seq, w=500):
    if len(seq) < w: return float("nan")
    c = collections.Counter(seq[:w]); vals = [len(c)]
    for i in range(w, len(seq)):
        c[seq[i]] += 1; old = seq[i - w]; c[old] -= 1
        if c[old] == 0: del c[old]
        vals.append(len(c))
    return float(np.mean(vals)) / w
def mtld_one(seq, thr=0.72):
    factors, types, toks = 0.0, set(), 0
    for w in seq:
        toks += 1; types.add(w)
        if len(types) / toks <= thr:
            factors += 1; types, toks = set(), 0
    if toks > 0:
        ttr = len(types) / toks
        factors += (1 - ttr) / (1 - thr)
    return len(seq) / factors if factors > 0 else float("nan")
def mtld(seq):
    return (mtld_one(seq) + mtld_one(seq[::-1])) / 2
def hdd(c, sample=42):
    n = sum(c.values()); s = 0.0
    for f in c.values():
        p0 = stats.hypergeom.pmf(0, n, f, sample)
        s += (1 - p0) / sample
    return s
def richness(seq):
    c = collections.Counter(seq)
    return {"N": len(seq), "V": len(c), "TTR": len(c) / len(seq), "Guiraud_R": guiraud(c),
            "Herdan_C": math.log(len(c)) / math.log(len(seq)), "Yule_K": yule_k(c), "Simpson_D": simpson_d(c),
            "Honore_H": honore_h(c), "Sichel_S": sichel_s(c), "Brunet_W": brunet_w(c),
            "MATTR500": mattr(seq), "MTLD": mtld(seq), "HDD42": hdd(c)}
R["richness_corpus"] = richness(W)

# by term: (a) full texts; (b) matched-N contiguous windows (N0 tokens) with many offsets
N0 = 30000
R["by_term"] = {}
for t in TERMS:
    tw = term_words(U, t)
    d = {"full": richness(tw)}
    # contiguous windows of N0 tokens at 20 evenly spaced offsets (T1 has ~32.5k tokens, so offsets overlap)
    offs = np.linspace(0, len(tw) - N0, 20).astype(int)
    rows = [richness(tw[o:o + N0]) for o in offs]
    d["matched"] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
                    for k in rows[0] if k not in ("N",)}
    # Zipf on matched windows
    zz = [fit_zipf(np.array(sorted(collections.Counter(tw[o:o + N0]).values(), reverse=True), float)) for o in offs]
    d["zipf_matched"] = {k: (float(np.mean([z[k] for z in zz])), float(np.std([z[k] for z in zz])))
                         for k in ("zipf_a_mle", "zm_a", "zm_b", "ols_slope")}
    hct = heaps_curve(tw[:N0], step=250)
    d["heaps_first_N0"] = heaps_fit(hct, nmin=2000)
    d["heaps_curve"] = hct
    R["by_term"][t] = d

# per speech unit (>= 1000 words): MATTR, MTLD, Yule K vs date
per = []
for u in U:
    w = unit_words(u)
    if len(w) >= 1000:
        c = collections.Counter(w)
        per.append({"id": u["id"], "date": u["date"], "term": u["term"], "genre": u["genre"], "N": len(w),
                    "MATTR500": mattr(w), "MTLD": mtld(w), "Yule_K": yule_k(c)})
R["per_unit"] = per
import datetime
dnum = [datetime.date.fromisoformat(p["date"]).toordinal() for p in per]
for k in ("MATTR500", "MTLD", "Yule_K"):
    rho, p = stats.spearmanr(dnum, [x[k] for x in per])
    R.setdefault("per_unit_trend", {})[k] = {"spearman_rho": rho, "p": p, "n": len(per)}
    groups = [[x[k] for x in per if x["term"] == t] for t in TERMS]
    H, pk = stats.kruskal(*groups)
    R["per_unit_trend"][k].update({"kruskal_H": H, "kruskal_p": pk,
                                   "medians": [float(np.median(g)) for g in groups],
                                   "n_by_term": [len(g) for g in groups]})

# genre-adjusted models of per-unit richness (premiership + genre + log length)
import pandas as pd, statsmodels.formula.api as smf
_df = pd.DataFrame(per)
_df["genre2"] = _df["genre"].replace({"rejoinder": "adversarial", "interpellation": "adversarial", "motion": "adversarial",
                                      "questions": "adversarial", "programme": "expository", "report": "expository",
                                      "budget": "expository", "debate": "debate"})
R["per_unit_ols"] = {}
for k in ("MTLD", "Yule_K", "MATTR500"):
    _m = smf.ols(f"{k} ~ C(term, Treatment('T1')) + C(genre2, Treatment('expository')) + np.log(N)", data=_df).fit()
    R["per_unit_ols"][k] = {"coef": _m.params.to_dict(), "p": _m.pvalues.to_dict(), "r2": _m.rsquared, "n": len(_df)}

# top words (for the paper's table)
R["top30"] = F.most_common(30)
R["curve_rank_freq"] = np.column_stack([ranks, freqs])[sel]
R["heaps_curve_chrono"] = hc[::4]
save("lexical.json", R)

# ---------------------------------------------------------------- figure: Zipf + Heaps
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
a = ax[0]
a.loglog(ranks, freqs, ".", ms=1.6, color=GREY, alpha=0.6, rasterized=True, label="observed")
zm = R["zipf_corpus"]
r = ranks
p = (r + zm["zm_b"]) ** (-zm["zm_a"]); p = p / p.sum() * N
a.loglog(r, p, "-", color=ACCENT, lw=1.4, label=f"Zipf–Mandelbrot  a = {zm['zm_a']:.2f}, b = {zm['zm_b']:.2f}")
a.axvline(R["two_regime"]["break_rank"], color=DARK, lw=0.6, ls=":")
a.text(R["two_regime"]["break_rank"] * 1.15, freqs[0] * 0.5,
       f"regime break\n≈ rank {R['two_regime']['break_rank']:.0f}", fontsize=7.5, color=DARK, va="top")
a.set_xlabel("rank r"); a.set_ylabel("frequency f(r)")
a.set_title(f"(a) Rank–frequency, N = {N:,} tokens, V = {V:,} types")
a.legend(loc="lower left", fontsize=7.5)
b_ = ax[1]
for t in TERMS:
    h = R["by_term"][t]["heaps_curve"]
    b_.plot(h[:, 0] / 1000, h[:, 1] / 1000, TERM_LS[t], color=TERM_COL[t], lw=1.5,
            label=f"{TERM_LABEL[t]}  β = {R['by_term'][t]['heaps_first_N0']['beta']:.3f}")
b_.set_xlabel("tokens (thousands)"); b_.set_ylabel("types (thousands)")
b_.set_title(f"(b) Vocabulary growth, first {N0//1000}k tokens of each term")
b_.legend(loc="lower right", fontsize=7.5)
savefig(fig, "fig_zipf_heaps")
print(json.dumps({k: R[k] for k in ("N", "V", "spectrum", "chao1", "zipf_corpus", "two_regime", "csn",
                                     "heaps_chrono", "heaps_shuffled", "herdan_C")}, default=str, indent=1))
for t in TERMS:
    print(t, {k: (round(v[0], 4), round(v[1], 4)) for k, v in R["by_term"][t]["matched"].items()})
    print("  ", R["by_term"][t]["zipf_matched"], R["by_term"][t]["heaps_first_N0"])
print(R["per_unit_trend"])
print(R["efron_thisted_new_types_t0.5"], R["efron_thisted_new_types_t1"])
