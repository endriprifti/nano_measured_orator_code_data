"""Section 6 — rhythm and periodic architecture: sentence and clause lengths, distribution fits,
Menzerath–Altmann at two levels, detrended fluctuation analysis of sentence length,
normalised pairwise variability (nPVI) of syllables and of clause lengths, cadence of sentence ends."""
import collections, math, datetime
import numpy as np
from scipy import stats, optimize
from common import *
from prep import syllables, phonemes

U = load_units()
rng = np.random.default_rng(11)
R = {}

SENTS = [(u, s) for u in U for s in u["sentences"] if s["kind"] != "address"]
L = np.array([len(s["tokens"]) for _, s in SENTS])
R["n_sentences"] = len(L)
R["kinds"] = collections.Counter(s["kind"] for u in U for s in u["sentences"])
R["sent_len"] = {"mean": L.mean(), "sd": L.std(ddof=1), "median": float(np.median(L)),
                 "q": np.percentile(L, [10, 25, 75, 90, 99]), "max": int(L.max()),
                 "share_gt40": float((L > 40).mean()), "share_gt60": float((L > 60).mean()),
                 "cv": L.std(ddof=1) / L.mean()}

# ---------------------------------------------------------------- distribution fits
def aic(ll, k): return 2 * k - 2 * ll
fits = {}
x = L.astype(float)
ln_sigma, loc, ln_scale = stats.lognorm.fit(x, floc=0); fits["lognormal"] = aic(stats.lognorm.logpdf(x, ln_sigma, 0, ln_scale).sum(), 2)
a, loc, sc = stats.gamma.fit(x, floc=0); fits["gamma"] = aic(stats.gamma.logpdf(x, a, 0, sc).sum(), 2)
c, loc, sc = stats.weibull_min.fit(x, floc=0); fits["weibull"] = aic(stats.weibull_min.logpdf(x, c, 0, sc).sum(), 2)
# negative binomial on (L-1) by MLE
def nb_nll(p):
    r, q = p
    if r <= 0 or not (0 < q < 1): return 1e18
    return -stats.nbinom.logpmf(L - 1, r, q).sum()
m, v = (L - 1).mean(), (L - 1).var()
nb = optimize.minimize(nb_nll, x0=(m * m / (v - m), m / v), method="Nelder-Mead")
fits["negbin(L-1)"] = aic(-nb.fun, 2)
# discrete comparability: continuous densities evaluated at integers are approximately pmfs (bin width 1)
R["sent_len_fits_AIC"] = fits
R["sent_len_params"] = {"lognormal_sigma": ln_sigma, "lognormal_median": ln_scale,
                        "gamma_shape": a, "negbin_r": nb.x[0], "negbin_p": nb.x[1]}
R["sent_len_best"] = min(fits, key=fits.get)

# by term + genre
by = {t: np.array([len(s["tokens"]) for u, s in SENTS if u["term"] == t]) for t in TERMS}
R["sent_len_by_term"] = {t: {"n": len(v), "mean": v.mean(), "median": float(np.median(v)), "sd": v.std(ddof=1)}
                         for t, v in by.items()}
H, p = stats.kruskal(*by.values()); R["sent_len_kruskal"] = {"H": H, "p": p}
def cliffs(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(1)); lt = sum((a[:, None] < b[None, :]).sum(1))
    return (gt - lt) / (len(a) * len(b))
R["sent_len_pairwise"] = {}
for i, j in (("T1", "T2"), ("T2", "T3"), ("T1", "T3")):
    U_, pp = stats.mannwhitneyu(by[i], by[j])
    R["sent_len_pairwise"][f"{i}-{j}"] = {"p": pp, "cliffs_delta": cliffs(by[i], by[j])}
gen = collections.defaultdict(list)
for u, s in SENTS: gen[u["genre"]].append(len(s["tokens"]))
R["sent_len_by_genre"] = {g: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))} for g, v in gen.items()}

# ---------------------------------------------------------------- clauses (kola)
CL = [len(c) for _, s in SENTS for c in s["clauses"] if len(c) > 0]
CL = np.array(CL)
nCl = np.array([len([c for c in s["clauses"] if c]) for _, s in SENTS])
R["clause"] = {"n": len(CL), "mean": CL.mean(), "median": float(np.median(CL)), "sd": CL.std(ddof=1),
               "clauses_per_sentence_mean": nCl.mean(), "clauses_per_sentence_median": float(np.median(nCl))}

# ---------------------------------------------------------------- Menzerath–Altmann
def ma(x, a, b, c): return a * x ** b * np.exp(-c * x)
def fit_ma(xs, ys, w):
    p, cov = optimize.curve_fit(ma, xs, ys, p0=(ys[0], -0.1, 0.0), sigma=1 / np.sqrt(w), maxfev=20000)
    yhat = ma(xs, *p); r2 = 1 - np.sum(w * (ys - yhat) ** 2) / np.sum(w * (ys - np.average(ys, weights=w)) ** 2)
    return {"a": p[0], "b": p[1], "c": p[2], "se": np.sqrt(np.diag(cov)), "R2_weighted": r2}
# level 1: sentence length in clauses -> mean clause length in words
g = collections.defaultdict(list)
for _, s in SENTS:
    cl = [len(c) for c in s["clauses"] if c]
    g[len(cl)].append(np.mean(cl))
xs = np.array(sorted(k for k in g if len(g[k]) >= 10), float)
ys = np.array([np.mean(g[k]) for k in xs]); ws = np.array([len(g[k]) for k in xs], float)
R["menzerath_sentence_clause"] = fit_ma(xs, ys, ws)
R["menzerath_sentence_clause"]["points"] = np.column_stack([xs, ys, ws])
rho, pr = stats.spearmanr(xs, ys); R["menzerath_sentence_clause"]["spearman"] = (rho, pr)
# level 2: word length in syllables -> mean syllable length in phonemes (word types weighted by tokens)
g2 = collections.defaultdict(list)
for _, s in SENTS:
    for t in s["tokens"]:
        if t[0].isdigit(): continue
        sy = syllables(t); ph = phonemes(t)
        g2[sy].append(ph / sy)
xs2 = np.array(sorted(k for k in g2 if len(g2[k]) >= 30), float)
ys2 = np.array([np.mean(g2[k]) for k in xs2]); ws2 = np.array([len(g2[k]) for k in xs2], float)
R["menzerath_word_syllable"] = fit_ma(xs2, ys2, ws2)
R["menzerath_word_syllable"]["points"] = np.column_stack([xs2, ys2, ws2])

# ---------------------------------------------------------------- DFA of sentence-length series
def dfa(x, scales):
    y = np.cumsum(x - np.mean(x)); F = []
    for n in scales:
        k = len(y) // n
        if k < 4: F.append(np.nan); continue
        segs = y[:k * n].reshape(k, n); t = np.arange(n)
        res = []
        for seg in segs:
            co = np.polyfit(t, seg, 1); res.append(np.mean((seg - np.polyval(co, t)) ** 2))
        # also from the end, to use the whole series
        segs2 = y[len(y) - k * n:].reshape(k, n)
        for seg in segs2:
            co = np.polyfit(t, seg, 1); res.append(np.mean((seg - np.polyval(co, t)) ** 2))
        F.append(np.sqrt(np.mean(res)))
    return np.array(F)
series = np.array([len(s["tokens"]) for _, s in SENTS], float)   # chronological, speech after speech
logL = np.log(series)
scales = np.unique(np.round(np.logspace(np.log10(4), np.log10(len(series) / 8), 22)).astype(int))
Fo = dfa(logL, scales)
alpha_o = np.polyfit(np.log(scales), np.log(Fo), 1)[0]
alphas_sh = []
for _ in range(100):
    sh = rng.permutation(logL); Fs = dfa(sh, scales); alphas_sh.append(np.polyfit(np.log(scales), np.log(Fs), 1)[0])
# within-speech shuffles keep each speech's mean (tests short-range order vs. between-speech level shifts)
bounds = []; i0 = 0
for u in U:
    n = sum(1 for s in u["sentences"] if s["kind"] != "address"); bounds.append((i0, i0 + n)); i0 += n
alphas_ws = []
for _ in range(100):
    sh = logL.copy()
    for a_, b_ in bounds: sh[a_:b_] = rng.permutation(sh[a_:b_])
    Fs = dfa(sh, scales); alphas_ws.append(np.polyfit(np.log(scales), np.log(Fs), 1)[0])
# small scales only (<= 64 sentences), within speeches
small = scales <= 64
alpha_small = np.polyfit(np.log(scales[small]), np.log(Fo[small]), 1)[0]
alpha_small_ws = []
for _ in range(100):
    sh = logL.copy()
    for a_, b_ in bounds: sh[a_:b_] = rng.permutation(sh[a_:b_])
    Fs = dfa(sh, scales[small]); alpha_small_ws.append(np.polyfit(np.log(scales[small]), np.log(Fs), 1)[0])
R["dfa"] = {"alpha": alpha_o, "alpha_shuffled_mean": np.mean(alphas_sh), "alpha_shuffled_sd": np.std(alphas_sh),
            "alpha_within_speech_shuffle_mean": np.mean(alphas_ws), "alpha_within_speech_shuffle_sd": np.std(alphas_ws),
            "alpha_small_scales": alpha_small, "alpha_small_ws_mean": np.mean(alpha_small_ws),
            "alpha_small_ws_sd": np.std(alpha_small_ws),
            "z_vs_global_shuffle": (alpha_o - np.mean(alphas_sh)) / np.std(alphas_sh),
            "z_small_vs_within": (alpha_small - np.mean(alpha_small_ws)) / np.std(alpha_small_ws),
            "scales": scales, "F": Fo}
# lag-1 autocorrelation of log sentence length within speeches
ac = []
for a_, b_ in bounds:
    seg = logL[a_:b_]
    if len(seg) > 20: ac.append(np.corrcoef(seg[:-1], seg[1:])[0, 1])
R["lag1_autocorr_within_speech"] = {"mean": np.mean(ac), "n_speeches": len(ac),
                                    "t_test_p": stats.ttest_1samp(ac, 0).pvalue}

# ---------------------------------------------------------------- nPVI (syllabic rhythm)
def npvi(d):
    d = np.asarray(d, float)
    if len(d) < 2: return np.nan
    return 100 * np.mean(np.abs(np.diff(d)) / ((d[:-1] + d[1:]) / 2))
syl_s = [[syllables(t, phonetic=True) for t in s["tokens"] if not t[0].isdigit()] for _, s in SENTS]
syl_s = [x for x in syl_s if len(x) >= 4]
obs = np.mean([npvi(x) for x in syl_s])
null = []
for _ in range(200):
    null.append(np.mean([npvi(rng.permutation(x)) for x in syl_s]))
R["npvi_syllables_words"] = {"observed": obs, "null_mean": np.mean(null), "null_sd": np.std(null),
                             "z": (obs - np.mean(null)) / np.std(null),
                             "mean_syll_per_word_phonetic": float(np.mean([v for x in syl_s for v in x])),
                             "mean_syll_per_word_orth": float(np.mean([syllables(t) for _, s in SENTS for t in s["tokens"] if not t[0].isdigit()]))}
# isocolon: nPVI of consecutive clause lengths (in syllables) inside sentences vs. clauses shuffled across corpus
cl_syl = []
for _, s in SENTS:
    cs = [sum(syllables(t, phonetic=True) for t in c if not t[0].isdigit()) for c in s["clauses"] if c]
    cs = [c for c in cs if c > 0]
    if len(cs) >= 3: cl_syl.append(cs)
obs_c = np.mean([npvi(x) for x in cl_syl])
pool = np.array([v for x in cl_syl for v in x])
null_c = []
for _ in range(200):
    perm = rng.permutation(pool); k = 0; vals = []
    for x in cl_syl:
        vals.append(npvi(perm[k:k + len(x)])); k += len(x)
    null_c.append(np.mean(vals))
# within-sentence permutation (order only)
null_c2 = [np.mean([npvi(rng.permutation(x)) for x in cl_syl]) for _ in range(200)]
R["npvi_clauses"] = {"observed": obs_c, "null_across_mean": np.mean(null_c), "null_across_sd": np.std(null_c),
                     "z_across": (obs_c - np.mean(null_c)) / np.std(null_c),
                     "null_within_mean": np.mean(null_c2), "null_within_sd": np.std(null_c2),
                     "z_within": (obs_c - np.mean(null_c2)) / np.std(null_c2), "n_sentences": len(cl_syl)}
# rising/falling: is the last clause the longest? (period with weight at the end — 'cursus' of members)
last_longest = np.mean([x[-1] == max(x) for x in cl_syl]); exp_ll = np.mean([1 / len(x) for x in cl_syl])
first_longest = np.mean([x[0] == max(x) for x in cl_syl])
inc = np.mean([x[-1] > x[0] for x in cl_syl])
R["period_shape"] = {"share_last_clause_longest": last_longest, "share_first_clause_longest": first_longest,
                     "expected_if_random": exp_ll, "share_last_gt_first": inc,
                     "binom_p_last": stats.binomtest(int(last_longest * len(cl_syl)), len(cl_syl), exp_ll).pvalue}

# ---------------------------------------------------------------- cadence (clausula) of sentence ends
fin, intern = [], []
for _, s in SENTS:
    sy = [syllables(t, phonetic=True) for t in s["tokens"] if not t[0].isdigit()]
    if len(sy) >= 6:
        fin.append(sy[-1]); intern.extend(sy[1:-1])
fin = np.array(fin); intern = np.array(intern)
R["cadence"] = {"final_mean_syll": fin.mean(), "internal_mean_syll": intern.mean(),
                "final_dist": {k: float((fin == k).mean()) for k in range(1, 6)},
                "internal_dist": {k: float((intern == k).mean()) for k in range(1, 6)},
                "mannwhitney_p": stats.mannwhitneyu(fin, intern).pvalue}

# ---------------------------------------------------------------- arc of a speech: sentence length by decile
dec = collections.defaultdict(list)
for u in U:
    ss = [s for s in u["sentences"] if s["kind"] != "address"]
    if len(ss) >= 30:
        for i, s in enumerate(ss):
            dec[int(10 * i / len(ss))].append(len(s["tokens"]))
R["arc_deciles"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / math.sqrt(len(v))), "n": len(v)}
                    for k, v in sorted(dec.items())}
R["arc_first_vs_last_decile_p"] = stats.mannwhitneyu(dec[0], dec[9]).pvalue

# per speech unit mean sentence length vs date
per = []
for u in U:
    ss = [len(s["tokens"]) for s in u["sentences"] if s["kind"] != "address"]
    if len(ss) >= 20: per.append((datetime.date.fromisoformat(u["date"]).toordinal(), np.mean(ss), u["term"]))
rho, p = stats.spearmanr([a for a, _, _ in per], [b for _, b, _ in per])
R["per_unit_sentlen_trend"] = {"rho": rho, "p": p, "n": len(per)}
# ---------------------------------------------------------------- additions: Weibull parameters, empirical closure hazard, arc
_c, _, _sc = stats.weibull_min.fit(L.astype(float), floc=0)
R["weibull"] = {"shape": float(_c), "scale": float(_sc), "mean": float(_sc * math.gamma(1 + 1 / _c))}
_cnt = collections.Counter(L.tolist()); _n = len(L); _surv = _n; haz = {}
for l in range(1, int(L.max()) + 1):
    if _surv > 0: haz[l] = _cnt.get(l, 0) / _surv
    _surv -= _cnt.get(l, 0)
R["closure_hazard_bands"] = {f"{a}-{a + 9}": float(np.mean([haz[l] for l in range(a, a + 10) if l in haz]))
                             for a in range(1, 81, 10)}
R["closure_hazard_bands_n_at_risk"] = {f"{a}-{a + 9}": int((L >= a).sum()) for a in range(1, 81, 10)}
dec2 = collections.defaultdict(list); n_units2 = 0
for u in U:
    ss = [s for s in u["sentences"] if s["kind"] not in ("address", "exclamation")]
    if len(ss) >= 30:
        n_units2 += 1
        for i, s in enumerate(ss): dec2[int(10 * i / len(ss))].append(len(s["tokens"]))
R["arc_no_exclamations"] = {"n_units": n_units2, "first": float(np.mean(dec2[0])), "last": float(np.mean(dec2[9])),
                            "p": float(stats.mannwhitneyu(dec2[0], dec2[9]).pvalue)}
R["arc_n_units"] = sum(1 for u in U if len([s for s in u["sentences"] if s["kind"] != "address"]) >= 30)

save("rhythm.json", R)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
a = ax[0]
bins = np.arange(0, 121, 3)
a.hist(L[L <= 120], bins=bins, density=True, color=GREY, alpha=0.55, edgecolor="white", lw=0.4)
xx = np.linspace(1, 120, 300)
a.plot(xx, stats.lognorm.pdf(xx, ln_sigma, 0, ln_scale), color=ACCENT, lw=1.3, label="lognormal fit")
a.axvline(np.median(L), color=DARK, lw=0.6, ls=":")
a.text(np.median(L) + 2, a.get_ylim()[1] * 0.92, f"median {np.median(L):.0f}", fontsize=7.5, color=DARK)
a.set_xlabel("sentence length (tokens)"); a.set_ylabel("density"); a.set_title("(a) The period: length distribution")
a.legend(fontsize=7.5, loc="upper right")
b_ = ax[1]
P = R["menzerath_sentence_clause"]["points"]
b_.scatter(P[:, 0], P[:, 1], s=np.clip(P[:, 2] / 15, 6, 60), color=GREY, alpha=0.8, edgecolor="white", lw=0.4)
xx = np.linspace(1, P[:, 0].max(), 100); m_ = R["menzerath_sentence_clause"]
b_.plot(xx, ma(xx, m_["a"], m_["b"], m_["c"]), color=ACCENT, lw=1.3,
        label=f"y = {m_['a']:.1f}·x^{m_['b']:.2f}·e^({-m_['c']:.3f}x)")
b_.set_xlabel("sentence length (clauses)"); b_.set_ylabel("mean clause length (tokens)")
b_.set_title("(b) Menzerath–Altmann: sentence → clause"); b_.legend(fontsize=7, loc="upper right")
c_ = ax[2]
c_.loglog(scales, Fo, "o", ms=3.5, color=ACCENT, label=f"observed  α = {alpha_o:.2f}")
Fsh = dfa(rng.permutation(logL), scales)
c_.loglog(scales, Fsh, "s", ms=3, color=GREY, mfc="none", label=f"shuffled  α ≈ {np.mean(alphas_sh):.2f}")
c_.set_xlabel("window n (sentences)"); c_.set_ylabel("F(n)"); c_.set_title("(c) DFA of log sentence length")
c_.legend(fontsize=7.5, loc="upper left")
fig.tight_layout(w_pad=1.6)
savefig(fig, "fig_rhythm")
out = {k: v for k, v in R.items() if k not in ("dfa",)}
out["dfa"] = {k: v for k, v in R["dfa"].items() if k not in ("scales", "F")}
for k in ("menzerath_sentence_clause", "menzerath_word_syllable"):
    out[k] = {kk: vv for kk, vv in R[k].items() if kk != "points"}
print(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1))
