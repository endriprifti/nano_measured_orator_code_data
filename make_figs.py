"""Publication figures, drawn from the saved results (and cheap recomputation where needed).
Usage: python make_figs.py [fig1 fig2 ...]"""
import sys, collections, json, math
import numpy as np
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
def J(name): return json.load(open(os.path.join(RES, name), encoding="utf-8"))

def fig1():
    U = load_units(); W = all_words(U); N = len(W)
    f = np.array(sorted(collections.Counter(W).values(), reverse=True), float); V = len(f); r = np.arange(1, V + 1)
    L = J("lexical.json"); zm = L["zipf_corpus"]; br = L["two_regime"]["break_rank"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a = ax[0]
    a.loglog(r, f, ".", ms=1.8, color=GREY, alpha=0.55, rasterized=True, label="observed frequencies")
    p = (r + zm["zm_b"]) ** (-zm["zm_a"]); p = p / p.sum() * N
    a.loglog(r, p, "-", color=ACCENT, lw=1.4, label=f"Zipf–Mandelbrot fit, a = {zm['zm_a']:.3f}")
    a.axvline(br, color=DARK, lw=0.6, ls=":")
    a.text(br * 1.15, 400, f"break near\nrank {br:,.0f}", fontsize=7.2, color=DARK, ha="left", va="center")
    a.text(1.3, 1.5, f"N = {N:,} tokens\nV = {V:,} forms", fontsize=7.2, color=DARK)
    a.set_xlabel("rank r"); a.set_ylabel("frequency f(r)")
    a.set_title("(a) Rank–frequency distribution", loc="left")
    a.legend(loc="upper right", fontsize=7)
    b = ax[1]
    for t in TERMS:
        h = np.array(L["by_term"][t]["heaps_curve"])
        b.plot(h[:, 0] / 1000, h[:, 1] / 1000, TERM_LS[t], color=TERM_COL[t], lw=1.6,
               label=f"{TERM_LABEL[t]}   β = {L['by_term'][t]['heaps_first_N0']['beta']:.3f}")
    b.set_xlabel("word tokens (thousands)"); b.set_ylabel("word forms (thousands)")
    b.set_title("(b) Vocabulary growth, first 30,000 tokens per term", loc="left")
    b.legend(loc="lower right", fontsize=7.2, title="premiership", title_fontsize=7.2)
    fig.tight_layout(w_pad=2.0)
    savefig(fig, "fig1_zipf_heaps")


def fig2():
    from scipy import stats
    U = load_units(); Rr = J("rhythm.json")
    L = np.array([len(s["tokens"]) for u in U for s in u["sentences"] if s["kind"] != "address"])
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7))
    a = ax[0]
    a.hist(L[L <= 120], bins=np.arange(0, 121, 3), density=True, color=GREY, alpha=0.6, edgecolor="white", lw=0.4)
    xx = np.linspace(1, 120, 300); P = Rr["sent_len_params"]
    kw, _, lw_ = stats.weibull_min.fit(L.astype(float), floc=0)
    a.plot(xx, stats.weibull_min.pdf(xx, kw, 0, lw_), color=ACCENT, lw=1.5, label=f"Weibull, k = {kw:.2f}\n(best by AIC)")
    a.plot(xx, stats.lognorm.pdf(xx, P["lognormal_sigma"], 0, P["lognormal_median"]), color=DARK, lw=0.9, ls="--",
           label="lognormal\n(worst by AIC)")
    a.set_xlabel("sentence length (tokens)"); a.set_ylabel("density")
    a.set_title("(a) Length of the period", loc="left"); a.set_ylim(0, 0.034); a.legend(fontsize=6.6, loc="upper right")
    b = ax[1]
    M = Rr["menzerath_sentence_clause"]; pts = np.array(M["points"])
    b.scatter(pts[:, 0], pts[:, 1], s=np.clip(pts[:, 2] / 12, 8, 70), color=GREY, alpha=0.85, edgecolor="white", lw=0.4,
              label="observed (area ∝ n)")
    xx = np.linspace(1, pts[:, 0].max(), 100)
    b.plot(xx, M["a"] * xx ** M["b"] * np.exp(-M["c"] * xx), color=ACCENT, lw=1.4,
           label=f"fitted law, b = {M['b']:.2f}")
    b.set_xlabel("sentence length (clauses)"); b.set_ylabel("mean clause length (tokens)")
    b.set_title("(b) Menzerath–Altmann law", loc="left"); b.legend(fontsize=7, loc="upper right")
    c = ax[2]
    D = Rr["dfa"]; sc = np.array(D["scales"]); F = np.array(D["F"])
    c.loglog(sc, F, "o", ms=3.6, color=ACCENT, label=f"spoken order, α = {D['alpha']:.2f}")
    logL = np.log(L.astype(float)); rng = np.random.default_rng(0)
    import importlib; rh = importlib.import_module("rhythm_dfa")
    Fs = rh.dfa(rng.permutation(logL), sc)
    c.loglog(sc, Fs, "s", ms=3.2, color=GREY, mfc="none", label=f"shuffled, α ≈ {D['alpha_shuffled_mean']:.2f}")
    c.set_xlabel("window n (sentences)"); c.set_ylabel("fluctuation F(n)")
    c.set_title("(c) Long-range correlation", loc="left"); c.legend(fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=1.4)
    savefig(fig, "fig2_rhythm")

def fig3():
    Fg = J("figures.json")
    items = [("Sentence anaphora\n(sentences permuted in speech)", Fg["anaphora_sentence"]["ratio"], Fg["anaphora_sentence"]["z"]),
             ("Sentence epistrophe", Fg["epistrophe_sentence"]["ratio"], Fg["epistrophe_sentence"]["z"]),
             ("Clause anaphora\n(clauses permuted in sentence)", Fg["anaphora_clause"]["observed"] / Fg["anaphora_clause"]["nullA_within_mean"], Fg["anaphora_clause"]["zA"]),
             ("Clause anaphora\n(openings redistributed)", Fg["anaphora_clause"]["ratioB"], Fg["anaphora_clause"]["zB"]),
             ("Clause epistrophe\n(clauses permuted in sentence)", Fg["epistrophe_clause"]["observed"] / Fg["epistrophe_clause"]["nullA_within_mean"], Fg["epistrophe_clause"]["zA"]),
             ("Clause epistrophe\n(endings redistributed)", Fg["epistrophe_clause"]["ratioB"], Fg["epistrophe_clause"]["zB"]),
             ("Lexical recurrence, k = 20\n(words permuted in speech)", Fg["recurrence_speech_shuffle"]["ratio"], Fg["recurrence_speech_shuffle"]["z"]),
             ("Lexical recurrence, k = 20\n(words permuted in paragraph)", Fg["recurrence_paragraph_shuffle"]["ratio"], Fg["recurrence_paragraph_shuffle"]["z"]),
             ("Alliteration of adjacent\npolysyllables", Fg["alliteration"]["ratio"], Fg["alliteration"]["z"])]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.55, 1]})
    a = ax[0]; y = np.arange(len(items))[::-1]
    for yi, (lab, r, z) in zip(y, items):
        sig = abs(z) >= 3
        a.plot([1, r], [yi, yi], color=GREY, lw=0.8)
        a.plot(r, yi, "o", ms=5.5, color=ACCENT if sig else GREY, mec="white", mew=0.6)
        a.text(max(r, 1) * 1.07, yi, f"×{r:.2f}  (z = {z:.1f})", va="center", ha="left", fontsize=6.8, color=DARK)
    a.axvline(1, color=DARK, lw=0.7)
    a.set_xscale("log"); a.set_xlim(0.7, 20); a.set_yticks(y); a.set_yticklabels([i[0] for i in items], fontsize=6.6)
    a.set_xticks([1, 2, 5, 10]); a.set_xticklabels(["1", "2", "5", "10"])
    a.set_xlabel("observed / expected under the null (log scale)")
    a.set_title("(a) Repetition against permutation nulls", loc="left")
    b = ax[1]
    obs = Fg["runs_clause"]; nul = Fg["runs_clause_null"]
    ks = [2, 3, 4]; keys = ["P_ge3_given_ge2", "P_ge4_given_ge3", "P_ge5_given_ge4"]
    b.plot(ks, [obs[k] for k in keys], "o-", color=ACCENT, lw=1.5, ms=5, label="observed anaphoric runs")
    b.plot(ks, [nul[k] for k in keys], "s--", color=GREY, lw=1.2, ms=4.5, mfc="white", label="null (openings redistributed)")
    for k_, kk in zip(ks, keys):
        b.text(k_ + 0.08, obs[kk] - 0.035, f"{obs[kk]:.2f}", ha="left", fontsize=7, color=DARK)
    b.set_xticks(ks); b.set_xticklabels(["2 → 3", "3 → 4", "4 → 5"])
    b.set_ylim(0, 0.6); b.set_xlim(1.8, 4.5); b.set_xlabel("members in the run"); b.set_ylabel("probability the run continues")
    b.set_title("(b) No closure at three", loc="left"); b.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    savefig(fig, "fig3_figures")

def fig4():
    St = J("stylometry.json")["glm"]
    lab = {"pron_1sg": "1st person singular pronouns", "pron_1pl": "1st person plural pronouns",
           "pron_2pl": "2nd person plural pronouns", "address_zoti_name": "zoti / zonja + name",
           "future_do_te": "analytic future do të", "antithesis_jo_vetem_por": "jo vetëm … por",
           "adversative_por": "adversative por"}
    order = ["address_zoti_name", "pron_1sg", "pron_2pl", "antithesis_jo_vetem_por", "adversative_por", "pron_1pl", "future_do_te"]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        g = St[k]
        for off, term, col, mk in ((0.14, "T2", TERM_COL["T2"], "s"), (-0.14, "T3", TERM_COL["T3"], "^")):
            name = f"C(term, Treatment('T1'))[T.{term}]"
            irr = g["irr"][name]; lo, hi = g["ci"][name]
            ax.plot([lo, hi], [yi + off, yi + off], color=col, lw=1.2)
            ax.plot(irr, yi + off, mk, color=col, ms=5, mec="white", mew=0.5,
                    label=f"{TERM_LABEL[term]} vs 1991" if k == order[0] else None)
    ax.axvline(1, color=DARK, lw=0.7)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order], fontsize=7.2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 10, 30, 100]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "10", "30", "100"])
    ax.set_xlabel("incidence-rate ratio at constant genre (95% CI, log scale)")
    ax.set_title("Change in rhetorical markers relative to 1991", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig4_markers")


TOPIC_LABELS = ["Parliament and opposition", "Budget and macro-economy", "Kosovo and foreign relations",
                "Energy, roads, rural infrastructure", "Organised crime and corruption", "Money and accusations",
                "Donors and recovery programme", "Public order and administration", "Pyramid schemes and banking",
                "European integration (SAA)", "Labour, unions and wages", "Private enterprise and investment"]

def fig7():
    Sm = J("semantics.json"); P = np.array([Sm["topics"]["prevalence_by_term"][t] for t in TERMS]).T   # K x 3
    order = np.argsort(-(P * np.array([1, 2, 3])).sum(1) / P.sum(1))[::-1]   # by chronological centre of mass
    order = np.argsort((P * np.array([1, 2, 3])).sum(1) / P.sum(1))
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    M = P[order]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f7fb", "#2a78d6", "#0b2f5e"])
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=0.25)
    for i in range(M.shape[0]):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f"{100*v:.0f}%", ha="center", va="center", fontsize=7.4, color="white" if v > 0.13 else DARK)
    ax.set_xticks(range(3)); ax.set_xticklabels([TERM_LABEL[t] for t in TERMS])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([TOPIC_LABELS[k] for k in order], fontsize=7.4)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("Mean topic share of paragraphs, by premiership", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    cb.set_ticks([0, 0.1, 0.2]); cb.set_ticklabels(["0%", "10%", "≥20%"])
    fig.tight_layout()
    savefig(fig, "fig7_topics")

def fig8():
    import datetime
    St = J("stylometry.json"); C = St["chunk_scores"]; ex = St["pca_explained"]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    for t in TERMS:
        xs = [c["pc1"] for c in C if c["term"] == t]; ys = [c["pc2"] for c in C if c["term"] == t]
        a.scatter(xs, ys, s=18, marker=TERM_MK[t], color=TERM_COL[t], edgecolor="white", lw=0.4, label=f"{TERM_LABEL[t]} (n = {len(xs)})")
    a.set_xlabel(f"PC1 ({100*ex[0]:.1f}% of variance)"); a.set_ylabel(f"PC2 ({100*ex[1]:.1f}%)")
    a.set_title("(a) Function-word space of 2,000-word chunks", loc="left", fontsize=8.6)
    b = ax[1]
    for t in TERMS:
        d = [datetime.date.fromisoformat(c["date"]) for c in C if c["term"] == t]
        b.scatter(d, [c["pc1"] for c in C if c["term"] == t], s=18, marker=TERM_MK[t], color=TERM_COL[t], edgecolor="white", lw=0.4)
    rho = St["pc1_vs_date_spearman"][0]
    b.set_ylabel("PC1 score"); b.set_title(f"(b) PC1 over time, Spearman ρ = {rho:.2f}", loc="left", fontsize=8.6)
    b.axhline(0, color=GREY, lw=0.6)
    b.text(datetime.date(1998, 6, 1), max(c["pc1"] for c in C) * 0.98, "argumentative pole\n(se, nuk, unë,\nkëtë, sepse, besoj)", fontsize=6.6, color=DARK, va="top")
    b.text(datetime.date(1998, 6, 1), min(c["pc1"] for c in C) * 0.98, "programmatic pole\n(të, e, qeveria,\ndo, ekonomisë)", fontsize=6.6, color=DARK, va="bottom")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
    savefig(fig, "fig8_stylometry")

def fig6():
    SA = J("surprisal_analysis.json"); SU = J("surprisal.json")["units"]; best = SA["best_model"]
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.4))
    a = ax[0, 0]
    bins = np.linspace(0, 30, 61)
    for mname, col, ls, lab in (("kn", GREY, "--", "Kneser–Ney 5-gram"), ("gpt", ACCENT, "-", "Transformer")):
        v = np.array([x for u in SU.values() for x, w in zip(u[mname], u["isword"]) if w and x == x])
        h, e = np.histogram(v, bins=bins, density=True)
        a.plot((e[:-1] + e[1:]) / 2, h, ls, color=col, lw=1.4, label=f"{lab}: {v.mean():.2f} bits/word")
    a.set_xlabel("word surprisal (bits)"); a.set_ylabel("density"); a.legend(fontsize=6.8)
    a.set_title("(a) Distribution of word surprisal", loc="left")
    b = ax[0, 1]
    C = SA["sentence_contour"]; k = sorted(C, key=int)
    b.errorbar([int(x) + 0.5 for x in k], [C[x]["mean"] for x in k], yerr=[1.96 * C[x]["se"] for x in k], fmt="o-",
               color=ACCENT, ms=4, lw=1.3, capsize=2)
    b.set_xticks(range(0, 11, 2)); b.set_xticklabels([f"{10*i}%" for i in range(0, 11, 2)])
    b.set_xlabel("relative position in the sentence"); b.set_ylabel("mean surprisal (bits/word)")
    b.set_title("(b) Information contour of the period", loc="left")
    c = ax[1, 0]
    A = SA["speech_arc"]; k = sorted(A, key=int)
    c.errorbar([int(x) + 0.5 for x in k], [A[x]["mean"] for x in k], yerr=[1.96 * A[x]["se"] for x in k], fmt="s-",
               color=ACCENT, ms=4, lw=1.3, capsize=2)
    c.set_xticks(range(0, 11, 2)); c.set_xticklabels([f"{10*i}%" for i in range(0, 11, 2)])
    c.set_xlabel("relative position in the speech"); c.set_ylabel("mean sentence surprisal (bits/word)")
    c.set_title("(c) Information arc of the speech", loc="left")
    d = ax[1, 1]
    from prep import phonemes
    import collections
    words_, sur = [], collections.defaultdict(list); F = collections.Counter()
    for u in SU.values():
        for t, w, x in zip(u["tokens"], u["isword"], u[best]):
            if w and x == x: sur[t].append(x); F[t] += 1
    T = [w for w, c_ in F.items() if c_ >= 20]
    L = np.array([phonemes(w) for w in T]); I = np.array([np.mean(sur[w]) for w in T])
    d.scatter(L + np.random.default_rng(0).uniform(-0.25, 0.25, len(L)), I, s=4, color=GREY, alpha=0.45, lw=0)
    xs = sorted(set(L)); ms = [I[L == x].mean() for x in xs if (L == x).sum() >= 5]; xs = [x for x in xs if (L == x).sum() >= 5]
    d.plot(xs, ms, "o-", color=ACCENT, ms=4, lw=1.4, label="mean by length")
    LI = SA["length_info"]
    d.set_xlabel("word length (phonemes)"); d.set_ylabel("mean in-context surprisal (bits)")
    d.set_title(f"(d) Length and information, ρ = {LI['spearman_len_context_info']:.2f}", loc="left")
    d.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    savefig(fig, "fig6_surprisal")

def fig5():
    I = J("info.json")
    try:
        SA = J("surprisal_analysis.json"); best = SA["best_model"]; Om = SA["order"][best]
    except Exception:
        Om = None
    # (name, H, R, label offset (pts), ha)
    pts = [("letters", I["letters"]["H"], I["letters"]["R"], (6, 6), "left"),
           ("sentence-length classes", I["length_classes"]["H"], I["length_classes"]["R"], (4, -13), "left"),
           ("sentence openings", I["sentence_openings"]["H"], I["sentence_openings"]["R"], (-7, -12), "right"),
           ("clause openings", I["clause_openings"]["H"], I["clause_openings"]["R"], (-7, 5), "right"),
           ("word forms\n(whole corpus)", I["words_all"]["H"], I["words_all"]["R"], (-8, 4), "right")]
    if Om: pts.append(("word sequence\n(language model)", Om["H_model"], Om["R_model"], (8, -4), "left"))
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    Hs = np.linspace(3, 11, 200)
    for M in (0.02, 0.03, 0.05, 0.1):
        ax.plot(Hs, M * Hs, color="#dddcd6", lw=0.7, zorder=0)
        hx = min(10.95, 0.56 / M); ax.text(hx, M * hx + 0.006, f"M = {M:g}", fontsize=6.4, color=GREY, ha="right", va="bottom")
    for name, h, r, off, ha in pts:
        col = ACCENT if "language model" in name else DARK
        ax.plot(h, r, "D" if "language model" in name else "o", ms=6, color=col, mec="white", mew=0.6, zorder=3)
        ax.annotate(name, (h, r), xytext=off, textcoords="offset points", fontsize=7, color=col, ha=ha)
    for t in TERMS:
        w = I["words_by_term_matched"][t]
        ax.plot(w["H"][0], w["R"][0], TERM_MK[t], ms=4.5, color=TERM_COL[t], zorder=3, label=f"word forms, {TERM_LABEL[t]}")
    ax.legend(fontsize=6.6, loc="lower right", title="30,000-token windows", title_fontsize=6.6)
    ax.set_xlim(3, 11); ax.set_ylim(0, 0.6)
    ax.set_xlabel("complexity H (bits per symbol)"); ax.set_ylabel("order R = 1 − H / H_max")
    ax.set_title("The order–complexity plane (grey lines: constant M_B = R/H)", loc="left", fontsize=8.8)
    fig.tight_layout()
    savefig(fig, "fig5_order_complexity")

def fig9():
    A = J("acoustic.json"); rows = J("acoustic_ipus.json")
    from prep import syllables
    U = load_units()
    cl = np.array([c for c in (sum(syllables(t, phonetic=True) for t in c if not t[0].isdigit())
                   for u in U for s in u["sentences"] if s["kind"] != "address" for c in s["clauses"] if c) if c > 0])
    S = np.array([r["syl"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.2))
    a = ax[0, 0]
    for data, col, ls, lab in ((S, ACCENT, "-", f"spoken phrases, median {np.median(S):.0f}"),
                               (cl, GREY, "--", f"written clauses, median {np.median(cl):.0f}")):
        xs = np.sort(data); a.plot(xs, np.arange(1, len(xs) + 1) / len(xs), ls, color=col, lw=1.6, label=lab)
    a.set_xscale("log"); a.set_xlabel("length (syllables)"); a.set_ylabel("cumulative share")
    a.set_title("(a) Spoken phrase vs written clause", loc="left"); a.legend(fontsize=6.6, loc="lower right")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    a.set_xticks([1, 2, 5, 10, 20, 50, 100]); a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(NullFormatter())
    b = ax[0, 1]
    P = np.array([rows[k + 1]["a"] - rows[k]["b"] for k in range(len(rows) - 1)]); P = P[P >= 0.25]
    b.hist(P, bins=np.logspace(np.log10(0.25), np.log10(max(P.max(), 5)), 30), color=GREY, alpha=0.65, edgecolor="white", lw=0.4)
    b.axvline(np.median(P), color=ACCENT, lw=1.3, label=f"median {np.median(P):.2f} s"); b.set_xscale("log"); b.legend(fontsize=6.8, loc="upper right")
    b.set_xticks([0.25, 0.5, 1, 2, 4, 8]); b.xaxis.set_major_formatter(ScalarFormatter()); b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel("pause duration (s, log scale)"); b.set_ylabel("number of pauses")
    b.set_title("(b) Silent pauses (≥ 0.25 s)", loc="left")
    c = ax[1, 0]
    D = A["decile_profile"]; k = [d["decile"] + 0.5 for d in D]
    c.fill_between(k, [d["f0_p10"] for d in D], [d["f0_p90"] for d in D], color=ACCENT, alpha=0.15, lw=0, label="10th–90th percentile")
    c.plot(k, [d["f0_med"] for d in D], "o-", color=ACCENT, ms=4, lw=1.4, label="median F0")
    c.set_xticks(range(0, 11, 2)); c.set_xticklabels([f"{10*i}%" for i in range(0, 11, 2)])
    c.set_xlabel("relative position in the speech"); c.set_ylabel("F0 (Hz)")
    c.set_title("(c) Pitch across the speech", loc="left"); c.legend(fontsize=6.6, loc="upper left")
    d = ax[1, 1]
    sc = np.array(A["dfa_curve"]["scales"]); Fo = np.array(A["dfa_curve"]["F"])
    import importlib; rh = importlib.import_module("rhythm_dfa")
    x = np.log(S.astype(float)); Fs = rh.dfa(np.random.default_rng(1).permutation(x), sc)
    d.loglog(sc, Fo, "o", ms=3.6, color=ACCENT, label=f"spoken order, α = {A['dfa_ipu_syllables']['alpha']:.2f}")
    d.loglog(sc, Fs, "s", ms=3.2, color=GREY, mfc="none", label=f"shuffled, α ≈ {A['dfa_ipu_syllables']['null_mean']:.2f}")
    d.set_xlabel("window n (phrases)"); d.set_ylabel("fluctuation F(n)")
    d.set_yticks([0.3, 0.5, 1, 2, 3]); d.yaxis.set_major_formatter(ScalarFormatter()); d.yaxis.set_minor_formatter(NullFormatter())
    d.xaxis.set_major_formatter(ScalarFormatter())
    d.set_title("(d) Long-range correlation of phrase length", loc="left"); d.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    savefig(fig, "fig9_acoustic")

def fig10():
    """Appendix C.2: the S18 excerpt as performed (a) and five recordings compared (b-e)."""
    V = J("video.json"); al = V["fn2_alignment"]["ipus"]; recs = V["recordings"]
    import parselmouth
    from matplotlib.lines import Line2D
    snd = parselmouth.Sound(os.path.join(os.environ.get("NANO_VIDEO", os.path.join(HERE, "..", "video")), "FN2.wav"))
    pt = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    F = pt.selected_array["frequency"]; tF = pt.xs()
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.62, wspace=0.65)
    a = fig.add_subplot(gs[0, :])
    for i, r in enumerate(al, 1):
        a.axvspan(r["a"], r["b"], color=ACCENT, alpha=0.08, lw=0)
        sel = (tF >= r["a"]) & (tF <= r["b"]) & (F > 0)
        st = 12 * np.log2(F[sel] / 100); ok = st > np.median(st) - 9      # hide octave (halving) errors
        a.plot(tF[sel][ok], st[ok], ".", ms=1.6, color=ACCENT, rasterized=True)
        a.text((r["a"] + r["b"]) / 2, 13.0, str(i), ha="center", va="bottom", fontsize=7, color=DARK)
        if r["pause_after"] is not None:
            x = r["b"] + r["pause_after"] / 2
            mk = {"sentence end": ("v", DARK), "comma": ("v", DARK)}.get(r["site_after"], ("o", GREY))
            a.plot([x], [-4.6], marker=mk[0], color=mk[1], ms=4.2, mfc=mk[1] if mk[0] == "v" else "white", clip_on=False)
    a.axvline(47.8, color=DARK, lw=0.6, ls=":"); a.text(47.6, 10.6, "broadcast\nsplice", fontsize=6.4, color=DARK, va="bottom", ha="right")
    a.set_ylim(-5.5, 14.5); a.set_xlim(al[0]["a"] - 0.6, al[-1]["b"] + 0.6)
    a.set_xlabel("time in clip FN2 (s)"); a.set_ylabel("F0 (semitones re 100 Hz)")
    a.set_title("(a) Session S18, 29 July 2002: the printed text as performed (IPUs numbered as in Table C2)", loc="left")
    h = [Line2D([], [], marker="v", ls="", color=DARK, ms=4.2, label="pause at a printed comma or full stop"),
         Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=4.2, label="pause with no printed punctuation")]
    a.legend(handles=h, fontsize=6.6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    order = ["FN5", "FN1", "KPD", "FN3", "FN2"]
    lab = [recs[k]["short"] for k in order]
    panels = [("artic_est", "artic_est_ci", "syllables per second", "(b) Articulation rate"),
              ("ipu_syl_est_median", "ipu_syl_est_ci", "syllables (median)", "(c) Length of IPU"),
              ("pause_median", "pause_median_ci", "seconds (median)", "(d) Silent pause"),
              ("f0_ipu_median_hz", "f0_ipu_median_ci_hz", "Hz (median over IPUs)", "(e) Pitch level")]
    for j, (key, ci, xl, tt) in enumerate(panels):
        b = fig.add_subplot(gs[1, j])
        for y, k in enumerate(order):
            v = recs[k][key]; lo, hi = recs[k][ci]
            col = ACCENT if k == "FN2" else DARK
            b.plot([lo, hi], [y, y], "-", color=col, lw=1.0)
            b.plot([v], [y], "o", color=col, ms=4.2)
        b.set_yticks(range(len(order))); b.set_yticklabels(lab if j == 0 else [""] * len(order), fontsize=7)
        b.set_ylim(-0.6, len(order) - 0.4); b.set_xlabel(xl, fontsize=7.4); b.set_title(tt, loc="left", fontsize=8.2)
        b.tick_params(axis="x", labelsize=7); b.margins(x=0.12)
        b.grid(axis="x", color=GREY, alpha=0.25, lw=0.5)
    savefig(fig, "fig10_broadcast")

if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
