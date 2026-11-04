"""Shared prosodic segmentation used by acoustic.py (KPD recording) and video.py (broadcast clips).

segment(snd)   -> intensity/pitch tracks, inter-pausal units (IPUs) and syllable nuclei, with the same parameters as
                  Appendix C: silence threshold 25 dB below the 0.99 intensity quantile, pauses >= 0.25 s,
                  IPUs >= 0.10 s, nuclei = intensity peaks above threshold separated by a >= 2 dB dip and voiced
                  (de Jong & Wempe 2009).
ipu_rows(...)  -> per-IPU syllable count, duration, F0 statistics in semitones re 100 Hz, declination slope.
"""
import numpy as np
from scipy import signal

MINPAUSE = 0.25
MINSPEECH = 0.10
DIP_DB = 2.0
SIL_DB = 25.0


def tracks(snd, f0_floor=60, f0_ceiling=400):
    inten = snd.to_intensity(minimum_pitch=60, time_step=0.01)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=f0_floor, pitch_ceiling=f0_ceiling)
    return inten.xs(), inten.values[0], pitch.xs(), pitch.selected_array["frequency"]


def segment(snd, mask=None, thr=None, **kw):
    """mask: optional list of (a, b) spans; intensity frames outside them count as silence."""
    tI, I, tF, F = tracks(snd, **kw)
    inside = np.ones_like(I, bool)
    if mask is not None:
        inside = np.zeros_like(I, bool)
        for a, b in mask:
            inside |= (tI >= a) & (tI < b)
    if thr is None:
        thr = np.percentile(I[inside], 99) - SIL_DB
    speech = (I > thr) & inside
    runs, cur, start = [], speech[0], 0
    for i in range(1, len(speech)):
        if speech[i] != cur:
            runs.append((cur, tI[start], tI[i])); cur = speech[i]; start = i
    runs.append((cur, tI[start], tI[-1]))
    merged = []
    for s, a, b in runs:
        if not s and (b - a) < MINPAUSE and merged and merged[-1][0]:
            merged[-1] = (True, merged[-1][1], b); continue
        if merged and merged[-1][0] == s:
            merged[-1] = (s, merged[-1][1], b)
        else:
            merged.append((s, a, b))
    ipus = [(a, b) for s, a, b in merged if s and (b - a) >= MINSPEECH]

    def f0_at(t):
        i = min(max(np.searchsorted(tF, t), 0), len(F) - 1); return F[i]
    peaks, _ = signal.find_peaks(np.where(speech, I, -np.inf), height=thr)
    nuclei = []
    for p in peaks:
        if nuclei:
            j = np.searchsorted(tI, nuclei[-1])
            if I[j:p + 1].min() > min(I[p], I[j]) - DIP_DB:
                if I[p] > I[j]: nuclei[-1] = tI[p]
                continue
        if f0_at(tI[p]) > 0:
            nuclei.append(tI[p])
    return dict(tI=tI, I=I, tF=tF, F=F, thr=thr, ipus=ipus, nuclei=np.array(nuclei))


def ipu_rows(seg):
    tF, F, nuclei = seg["tF"], seg["F"], seg["nuclei"]
    rows = []
    for a, b in seg["ipus"]:
        n = nuclei[(nuclei >= a) & (nuclei <= b)]
        f = F[(tF >= a) & (tF <= b)]; tf = tF[(tF >= a) & (tF <= b)]
        vf = f > 0
        st = 12 * np.log2(f[vf] / 100.0) if vf.sum() else np.array([])
        slope = np.polyfit(tf[vf] - a, st, 1)[0] if vf.sum() >= 10 else np.nan
        rows.append({"a": float(a), "b": float(b), "dur": float(b - a), "syl": int(len(n)),
                     "iv": np.diff(n).tolist(),
                     "f0_hz_median": float(np.median(f[vf])) if vf.sum() else np.nan,
                     "f0_st_median": float(np.median(st)) if len(st) else np.nan,
                     "f0_st_first": float(np.median(st[: max(1, len(st) // 5)])) if len(st) else np.nan,
                     "f0_st_last": float(np.median(st[-max(1, len(st) // 5):])) if len(st) else np.nan,
                     "decl_st_per_s": float(slope)})
    return rows
