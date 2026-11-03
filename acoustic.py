"""Appendix C — acoustic pilot on a recording of Nano's intervention at the meeting of the KPD of the
Socialist Party, 3 December 2001 (segment from 2:40 to the end, as indicated by the author).

Pauses and inter-pause units (IPUs) from the intensity contour; syllable nuclei after de Jong & Wempe (2009);
speech and articulation rate; acoustic Menzerath (IPU length vs mean syllable duration); phrase-final
lengthening; nPVI of inter-nucleus intervals; F0 level, range, declination and reset; DFA of IPU durations;
comparison of IPU length in syllables with the clause length of the parliamentary corpus.

Usage: python acoustic.py nano_seg.wav
"""
import sys, json, math, collections
import numpy as np
import parselmouth
from scipy import stats, signal
from common import *
from prep import syllables

WAV = sys.argv[1]
snd = parselmouth.Sound(WAV)
R = {"duration_s": snd.duration}

# ---------------------------------------------------------------- intensity, pitch
inten = snd.to_intensity(minimum_pitch=60, time_step=0.01)
I = inten.values[0]; tI = inten.xs()
pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
F = pitch.selected_array["frequency"]; tF = pitch.xs()
def f0_at(t):
    i = np.searchsorted(tF, t); i = min(max(i, 0), len(F) - 1); return F[i]

maxI = np.percentile(I, 99)
thr = maxI - 25.0                     # de Jong & Wempe: silence threshold 25 dB below the 0.99 quantile
R["intensity_99pct_dB"] = maxI; R["silence_threshold_dB"] = thr

# ---------------------------------------------------------------- pauses and IPUs
MINPAUSE = 0.25; MINSPEECH = 0.10
speech = I > thr
# run-length encode
runs = []; cur = speech[0]; start = 0
for i in range(1, len(speech)):
    if speech[i] != cur:
        runs.append((cur, tI[start], tI[i])); cur = speech[i]; start = i
runs.append((cur, tI[start], tI[-1]))
# merge short silences (< MINPAUSE) into speech, drop very short speech blips
merged = []
for s, a, b in runs:
    if not s and (b - a) < MINPAUSE and merged:
        merged[-1] = (True, merged[-1][1], b) if merged[-1][0] else merged[-1]
        if merged[-1][0]: continue
    if merged and merged[-1][0] == s:
        merged[-1] = (s, merged[-1][1], b)
    else:
        merged.append((s, a, b))
ipus = [(a, b) for s, a, b in merged if s and (b - a) >= MINSPEECH]
# pauses = silent gaps between consecutive IPUs of at least MINPAUSE seconds
pauses = [(ipus[k][1], ipus[k + 1][0]) for k in range(len(ipus) - 1) if ipus[k + 1][0] - ipus[k][1] >= MINPAUSE]
P = np.array([b - a for a, b in pauses]); D = np.array([b - a for a, b in ipus])
R["n_ipus"] = len(ipus); R["n_pauses"] = len(pauses)
R["pause"] = {"median": float(np.median(P)), "mean": float(P.mean()), "q": np.percentile(P, [10, 25, 75, 90]).tolist(),
              "per_minute": len(P) / (snd.duration / 60), "share_ge_1s": float((P >= 1).mean())}
R["ipu_duration"] = {"median": float(np.median(D)), "mean": float(D.mean()), "q": np.percentile(D, [10, 25, 75, 90]).tolist()}
phon = D.sum(); R["phonation_time_ratio"] = phon / (ipus[-1][1] - ipus[0][0])

# ---------------------------------------------------------------- syllable nuclei (de Jong & Wempe 2009)
peaks, _ = signal.find_peaks(I, height=thr)
nuclei = []
prev_t, prev_v = None, None
for p in peaks:
    # require a dip of >= 2 dB between consecutive accepted peaks
    if nuclei:
        j = np.searchsorted(tI, nuclei[-1])
        if I[j:p + 1].min() > min(I[p], I[j]) - 2.0:
            # no sufficient dip: keep the higher peak
            if I[p] > I[j]: nuclei[-1] = tI[p]
            continue
    if f0_at(tI[p]) > 0:                    # voiced
        nuclei.append(tI[p])
nuclei = np.array(nuclei)
R["n_syllables"] = len(nuclei)
R["speech_rate_syll_per_s"] = len(nuclei) / (ipus[-1][1] - ipus[0][0])
R["articulation_rate_syll_per_s"] = len(nuclei) / phon

# per IPU: syllables, mean syllable duration, final lengthening, nPVI of inter-nucleus intervals
def npvi(d):
    d = np.asarray(d, float)
    return 100 * np.mean(np.abs(np.diff(d)) / ((d[:-1] + d[1:]) / 2)) if len(d) >= 2 else np.nan
rows = []
for a, b in ipus:
    n = nuclei[(nuclei >= a) & (nuclei <= b)]
    if len(n) == 0: continue
    iv = np.diff(n)
    f = F[(tF >= a) & (tF <= b)]; tf = tF[(tF >= a) & (tF <= b)]
    vf = f > 0
    st = 12 * np.log2(f[vf] / 100.0) if vf.sum() else np.array([])
    slope = np.polyfit(tf[vf] - a, st, 1)[0] if vf.sum() >= 10 else np.nan
    rows.append({"a": a, "b": b, "dur": b - a, "syl": len(n), "msd": (b - a) / len(n), "iv": iv.tolist(),
                 "f0_st_median": float(np.median(st)) if len(st) else np.nan,
                 "f0_st_first": float(np.median(st[: max(1, len(st) // 5)])) if len(st) else np.nan,
                 "f0_st_last": float(np.median(st[-max(1, len(st) // 5):])) if len(st) else np.nan,
                 "decl_st_per_s": slope})
S = np.array([r["syl"] for r in rows])
R["ipu_syllables"] = {"median": float(np.median(S)), "mean": float(S.mean()), "q": np.percentile(S, [10, 25, 75, 90]).tolist(),
                      "n": len(S)}

# acoustic Menzerath: IPU length (syllables) -> mean syllable duration
g = collections.defaultdict(list)
for r in rows:
    if r["syl"] >= 2: g[min(r["syl"], 40)].append(r["msd"])
xs = np.array(sorted(k for k in g if len(g[k]) >= 8), float)
ys = np.array([np.mean(g[k]) for k in xs]); ws = np.array([len(g[k]) for k in xs], float)
rho, p = stats.spearmanr([r["syl"] for r in rows if r["syl"] >= 2], [r["msd"] for r in rows if r["syl"] >= 2])
bfit = np.polyfit(np.log(xs), np.log(ys), 1, w=np.sqrt(ws))
R["menzerath_acoustic"] = {"spearman": rho, "p": p, "loglog_slope_b": bfit[0], "points": np.column_stack([xs, ys, ws]).tolist()}

# phrase-final lengthening: last inter-nucleus interval vs. internal ones (IPUs with >= 5 nuclei)
last = [r["iv"][-1] for r in rows if len(r["iv"]) >= 4]
internal = [x for r in rows if len(r["iv"]) >= 4 for x in r["iv"][:-1]]
R["final_lengthening"] = {"last_mean": float(np.mean(last)), "internal_mean": float(np.mean(internal)),
                          "ratio": float(np.mean(last) / np.mean(internal)), "p": stats.mannwhitneyu(last, internal).pvalue}
# nPVI of inter-nucleus intervals vs within-IPU permutation
rng = np.random.default_rng(3)
ivs = [np.array(r["iv"]) for r in rows if len(r["iv"]) >= 3]
obs = np.mean([npvi(v) for v in ivs])
null = [np.mean([npvi(rng.permutation(v)) for v in ivs]) for _ in range(200)]
R["npvi_intervals"] = {"observed": obs, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                       "z": float((obs - np.mean(null)) / np.std(null)), "n_ipus": len(ivs)}

# ---------------------------------------------------------------- F0
vf = F[F > 0]; st_all = 12 * np.log2(vf / 100.0)
R["f0"] = {"median_hz": float(np.median(vf)), "p5_hz": float(np.percentile(vf, 5)), "p95_hz": float(np.percentile(vf, 95)),
           "range_st_5_95": float(np.percentile(st_all, 95) - np.percentile(st_all, 5)),
           "sd_st": float(np.std(st_all))}
decl = np.array([r["decl_st_per_s"] for r in rows if r["decl_st_per_s"] == r["decl_st_per_s"] and r["dur"] >= 1.0])
R["declination"] = {"median_st_per_s": float(np.median(decl)), "share_negative": float((decl < 0).mean()), "n": len(decl),
                    "wilcoxon_p": stats.wilcoxon(decl).pvalue}
fl = [(r["f0_st_first"], r["f0_st_last"]) for r in rows if r["dur"] >= 1.0 and r["f0_st_first"] == r["f0_st_first"]]
R["first_last_st"] = {"first_mean": float(np.mean([a for a, _ in fl])), "last_mean": float(np.mean([b for _, b in fl])),
                      "drop_st": float(np.mean([a - b for a, b in fl]))}
# reset: start of IPU k+1 minus end of IPU k
resets = [rows[k + 1]["f0_st_first"] - rows[k]["f0_st_last"] for k in range(len(rows) - 1)
          if rows[k]["f0_st_last"] == rows[k]["f0_st_last"] and rows[k + 1]["f0_st_first"] == rows[k + 1]["f0_st_first"]]
R["reset_st"] = {"mean": float(np.mean(resets)), "median": float(np.median(resets))}

# ---------------------------------------------------------------- DFA on IPU durations (in spoken order)
from rhythm_dfa import dfa
x = np.log(np.array([r["syl"] for r in rows], float))
scales = np.unique(np.round(np.logspace(np.log10(4), np.log10(len(x) / 8), 16)).astype(int))
Fo = dfa(x, scales); a_obs = np.polyfit(np.log(scales), np.log(Fo), 1)[0]
a_null = [np.polyfit(np.log(scales), np.log(dfa(rng.permutation(x), scales)), 1)[0] for _ in range(100)]
R["dfa_ipu_syllables"] = {"alpha": a_obs, "null_mean": float(np.mean(a_null)), "null_sd": float(np.std(a_null)),
                          "z": float((a_obs - np.mean(a_null)) / np.std(a_null)), "n": len(x)}
R["dfa_curve"] = {"scales": scales.tolist(), "F": Fo.tolist()}

# ---------------------------------------------------------------- comparison with the text corpus
U = load_units()
cl_syl = [sum(syllables(t, phonetic=True) for t in c if not t[0].isdigit())
          for u in U for s in u["sentences"] if s["kind"] != "address" for c in s["clauses"] if c]
cl_syl = np.array([c for c in cl_syl if c > 0])
R["text_clause_syllables"] = {"median": float(np.median(cl_syl)), "mean": float(cl_syl.mean()),
                              "q": np.percentile(cl_syl, [10, 25, 75, 90]).tolist()}
R["ipu_vs_clause_ks"] = stats.ks_2samp(S, cl_syl)._asdict()
# distribution fits of IPU syllables (Weibull shape), for comparison with the sentence-length law
c_, _, sc_ = stats.weibull_min.fit(S.astype(float), floc=0)
R["ipu_syllables_weibull"] = {"shape": c_, "scale": sc_}
# profile across tenths of the speech: F0, articulation and speech rate, phrase length
T0, T1 = rows[0]["a"], rows[-1]["b"]; prof = []
for k in range(10):
    a_, b_ = T0 + k * (T1 - T0) / 10, T0 + (k + 1) * (T1 - T0) / 10
    f = F[(tF >= a_) & (tF < b_)]; f = f[f > 0]
    rs = [r for r in rows if a_ <= r["a"] < b_ or (k == 9 and r["a"] >= a_)]
    syl = sum(r["syl"] for r in rs); dur = sum(r["dur"] for r in rs)
    prof.append({"decile": k, "f0_med": float(np.median(f)), "f0_p10": float(np.percentile(f, 10)),
                 "f0_p90": float(np.percentile(f, 90)), "artic": syl / dur, "speech_rate": syl / (b_ - a_),
                 "ipu_syl_mean": float(np.mean([r["syl"] for r in rs])), "n_ipu": len(rs)})
R["decile_profile"] = prof
save("acoustic.json", R)
json.dump(rows, open(os.path.join(RES, "acoustic_ipus.json"), "w"))
print(json.dumps({k: v for k, v in R.items() if k not in ("dfa_curve",)}, indent=1, default=str)[:5000])
