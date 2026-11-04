"""Appendix C, second part — three broadcast clips and two archival audio recordings supplied by the author, compared
with the KPD recording.

FN1     TV interview, 19 May 2000 (RTV Klan archive)                        Nano is party chairman, not PM
FN2     Parliament, 29 Jul 2002: programme of the government (session S18)   the only clip whose text is in the edition
FN3     Parliament, 3 Jun 2002: debate on the European Parliament resolution  Nano is party chairman, not PM
FN4     Parliament, 5 Nov 1999                                               audio track empty (0 packets): not analysed
FN1990  audio, 1990 (Historiani, YouTube): speech at a meeting, introduced by the chair      text not in the edition
FN1991  audio, March 1991 (Historiani, YouTube): Nano as Chairman of the Council of Ministers at a meeting of party
        and state cadres called by a district committee of the Party of Labour before the elections of 31 March 1991
        text not in the edition
(An earlier version also analysed a 29-s video clip of a 1991 address; it was withdrawn at the author's request.)

Steps
1. Speaker attribution. Silero VAD v5 (ONNX) marks speech; a pretrained GE2E speaker encoder (Resemblyzer weights, run in
   NumPy) embeds 1.6-s windows. Clips: k-means (k = 2) per clip; the cluster closer to the centroid of the KPD recording
   is Nano, checked against spans fixed from shot content, F0 and the automatic transcript. Archival recordings: the
   chair's introduction is located from the transcript; the rest is tested for a second voice (k = 2 within the span).
2. Segmentation of Nano's speech into inter-pausal units (IPUs) with the VAD (pauses >= 0.25 s, IPUs >= 0.10 s);
   syllable nuclei as in acoustic.py; F0 statistics in semitones re 100 Hz.
3. Calibration of the nucleus detector against the exact transcript of FN2 (printed text) and against automatic
   transcripts (Whisper large-v3-turbo, int8 ONNX via sherpa-onnx) of samples of the three long recordings.
4. FN2: alignment of each IPU with the printed text of S18, classification of pause sites.

Usage: python video.py   (expects the WAVs produced by run_all.sh in ../video, ../audio and ../audio90)
"""
import os, sys, json, re, difflib
import numpy as np, soundfile as sf, parselmouth
from sklearn.cluster import KMeans
from common import save, RES
from prep import syllables, tokens, split_clauses
from prosody import tracks, MINPAUSE, MINSPEECH, DIP_DB, SIL_DB
from scipy import signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
VID = os.environ.get("NANO_VIDEO", os.path.join(HERE, "..", "video"))
AUD = os.environ.get("NANO_AUDIO", os.path.join(HERE, "..", "audio"))
A90 = os.environ.get("NANO_AUDIO90", os.path.join(HERE, "..", "audio90"))
SR = 16000
rng = np.random.default_rng(11)

REC = {
    "KPD": dict(wav=os.path.join(AUD, "nano_seg.wav"), spans=None,
                label="Party council (KPD), 3 Dec 2001", short="KPD 2001"),
    "FN1": dict(wav=os.path.join(VID, "FN1.wav"), spans=[(42.8, 85.8), (105.6, 131.0)],
                label="TV interview, 19 May 2000", short="Interview 2000"),
    "FN2": dict(wav=os.path.join(VID, "FN2.wav"), spans=[(26.8, 47.55), (48.05, 72.2)],   # audio splice at ~47.7 s
                label="Parliament, government programme, 29 Jul 2002 (S18)", short="Parliament 2002 (S18)"),
    "FN3": dict(wav=os.path.join(VID, "FN3.wav"), spans=[(24.0, 82.9)],
                label="Parliament, debate, 3 Jun 2002", short="Parliament 2002 (debate)"),
    # archival audio: one span from the end of the chair's introduction (located in the automatic transcript: FN1990 "Shoku
    # Fatos Nano ka fjalën ..." ends at 8.3 s, Nano opens at 12.9 s; FN1991 the chair speaks until 93.2 s, Nano opens at
    # 96.5 s) to the end of the tape, which in both cases breaks off in mid-speech
    "FN1990": dict(wav=os.path.join(A90, "FN1990.wav"), spans=[(12.7, 749.2)], archive=True,
                   label="Party meeting (aktiv), 1990", short="Party aktiv 1990"),
    "FN1991": dict(wav=os.path.join(A90, "FN1991.wav"), spans=[(96.4, 3358.6)], archive=True,
                   label="Pre-election meeting of cadres, March 1991", short="Cadres meeting 1991"),
}
CLIPS = ("FN1", "FN2", "FN3")
ARCH = ("FN1990", "FN1991")
LONG = ("KPD",) + ARCH                      # syllables from the detector, calibrated on a transcribed sample


def load(path):
    w, sr = sf.read(path); assert sr == SR, path
    return w.astype(np.float32)


# ------------------------------------------------------------------ 1. VAD and speaker attribution
from vad_np import SileroVAD
from voice_encoder_np import VoiceEncoder
vad = SileroVAD(os.path.join(HERE, "silero_vad.onnx"))
enc = VoiceEncoder(os.path.join(HERE, "resemblyzer_pretrained.pt"))


def vad_mask(w):
    p, step = vad.probs(w)
    on, off, cur, m = 0.5, 0.35, False, np.zeros(len(p), bool)
    for i, v in enumerate(p):
        cur = (v >= on) if not cur else (v >= off)
        m[i] = cur
    return m, step


WIN, HOP = 1.6, 0.5


def window_embeddings(w, m, step, spans=None):
    T = len(w) / SR; ts = np.arange(0, T - WIN, HOP); keep = []
    tm = np.arange(len(m)) * step
    for t in ts:
        sel = (tm >= t) & (tm < t + WIN)
        if sel.any() and m[sel].mean() > 0.6: keep.append(t)
    out = []
    for i in range(0, len(keep), 256):          # batches keep the LSTM's gate tensors small on long recordings
        M = np.stack([enc.mel(enc.normalize(w[int(t * SR):int((t + WIN) * SR)]))[:160] for t in keep[i:i + 256]])
        out.append(enc.forward(M))
    return np.array(keep), np.concatenate(out)


R = {"recordings": {}, "diarisation": {}}
K = load(REC["KPD"]["wav"])
mK, stK = vad_mask(K)
tk, EK = window_embeddings(K[:int(600 * SR)], mK[:int(600 / stK)], stK)   # first 10 minutes as reference
ref = EK.mean(0); ref /= np.linalg.norm(ref)
# held-out check: windows from the remainder of the KPD recording against the reference
tk2, EK2 = window_embeddings(K[int(600 * SR):int(900 * SR)], mK[int(600 / stK):int(900 / stK)], stK)
R["diarisation"]["KPD_heldout_cos"] = {"median": float(np.median(EK2 @ ref)), "p5": float(np.percentile(EK2 @ ref, 5))}

for c in ARCH:
    # one speaker after the chair's introduction? two clusters inside the span, compared with the KPD reference;
    # the windows before the span (the chair) are compared with the centroid of the span
    w = load(REC[c]["wav"]); m, st = vad_mask(w)
    t, E = window_embeddings(w, m, st)
    sim = E @ ref; centre = t + WIN / 2
    a0, b0 = REC[c]["spans"][0]
    ins = (t >= a0) & (t + WIN <= b0); pre = t + WIN <= a0
    km = KMeans(2, n_init=10, random_state=0).fit(E[ins])
    cm = [sim[ins][km.labels_ == k].mean() for k in (0, 1)]
    cen = E[ins].mean(0); cen /= np.linalg.norm(cen)
    R["diarisation"][c] = {"n_windows": int(ins.sum()), "cluster_cos": [float(x) for x in cm],
        "single_speaker": bool(abs(cm[0] - cm[1]) < 0.05), "cluster_sizes": [int((km.labels_ == k).sum()) for k in (0, 1)],
        "cos_to_KPD_span": float(np.median(sim[ins])), "cos_to_KPD_span_p5": float(np.percentile(sim[ins], 5)),
        "cos_to_KPD_chair": float(np.median(sim[pre])),
        "cos_to_span_centroid_span": float(np.median(E[ins] @ cen)), "cos_to_span_centroid_chair": float(np.median(E[pre] @ cen)),
        "n_windows_chair": int(pre.sum()),
        "windows": [[float(a), int(b), float(s)] for a, b, s in zip(t, ins, sim)]}
    REC[c]["vad"] = (m, st)

for c in CLIPS:
    w = load(REC[c]["wav"]); m, st = vad_mask(w)
    t, E = window_embeddings(w, m, st)
    sim = E @ ref
    km = KMeans(2, n_init=10, random_state=0).fit(E)
    cm = [sim[km.labels_ == k].mean() for k in (0, 1)]
    nano_k = int(np.argmax(cm))
    # one speaker only if the two clusters are equally close to the reference (difference < 0.05)
    is_nano = km.labels_ == nano_k if abs(cm[0] - cm[1]) >= 0.05 else np.ones(len(t), bool)
    centre = t + WIN / 2
    inside = np.zeros(len(t), bool)
    for a, b in REC[c]["spans"]:
        inside |= (centre >= a) & (centre <= b)
    R["diarisation"][c] = {"cluster_cos": [float(x) for x in cm], "single_speaker": bool(abs(cm[0] - cm[1]) < 0.05),
        "n_windows": int(len(t)), "cos_to_KPD_nano_cluster": float(sim[is_nano].mean()),
        "cos_to_KPD_other_cluster": float(sim[~is_nano].mean()) if (~is_nano).any() else None,
        "share_windows_nano": float(is_nano.mean()),
        "agreement_with_spans": float((is_nano == inside).mean()),
        "nano_in_spans": float(is_nano[inside].mean()), "other_outside_spans": float((~is_nano[~inside]).mean()) if (~inside).any() else None,
        "windows": [[float(a), int(b), float(s)] for a, b, s in zip(t, is_nano, sim)]}
    REC[c]["vad"] = (m, st)
REC["KPD"]["vad"] = (mK, stK)


# ------------------------------------------------------------------ 2. IPUs, nuclei, F0
def ipus_from_vad(m, step, spans, T):
    spans = spans or [(0.0, T)]
    out = []
    for a, b in spans:
        i0, i1 = int(np.ceil(a / step)), int(np.floor(b / step))
        seg = m[i0:i1]; runs, cur, s0 = [], seg[0], 0
        for i in range(1, len(seg)):
            if seg[i] != cur: runs.append((cur, s0, i)); cur, s0 = seg[i], i
        runs.append((cur, s0, len(seg)))
        sp = [((i0 + x) * step, (i0 + y) * step) for v, x, y in runs if v]
        merged = []
        for x, y in sp:
            if merged and x - merged[-1][1] < MINPAUSE: merged[-1] = (merged[-1][0], y)
            else: merged.append((x, y))
        out.append([(x, y) for x, y in merged if y - x >= MINSPEECH])
    return out          # list per span


def analyse(name):
    cfg = REC[name]; snd = parselmouth.Sound(cfg["wav"]); w = snd.values[0]
    tI, I, tF, F = tracks(snd)
    m, st = cfg["vad"]
    per_span = ipus_from_vad(m, st, cfg["spans"], snd.duration)
    ipus = [x for s in per_span for x in s]
    inside = np.zeros_like(I, bool)
    for a, b in ipus: inside |= (tI >= a) & (tI < b)
    thr = np.percentile(I[inside], 99) - SIL_DB

    def f0_at(t):
        i = min(max(np.searchsorted(tF, t), 0), len(F) - 1); return F[i]
    peaks, _ = signal.find_peaks(np.where(inside & (I > thr), I, -np.inf), height=thr)
    nuc = []
    for p in peaks:
        if nuc:
            j = np.searchsorted(tI, nuc[-1])
            if I[j:p + 1].min() > min(I[p], I[j]) - DIP_DB:
                if I[p] > I[j]: nuc[-1] = tI[p]
                continue
        if f0_at(tI[p]) > 0: nuc.append(tI[p])
    nuc = np.array(nuc)
    rows = []
    for a, b in ipus:
        n = nuc[(nuc >= a) & (nuc <= b)]
        sel = (tF >= a) & (tF <= b) & (F > 0); f = F[sel]; tf = tF[sel]
        stv = 12 * np.log2(f / 100.0) if len(f) else np.array([])
        slope = np.polyfit(tf - a, stv, 1)[0] if len(f) >= 10 else np.nan
        rows.append({"a": a, "b": b, "dur": b - a, "syl": int(len(n)),
                     "f0_st_med": float(np.median(stv)) if len(stv) else np.nan,
                     "f0_first": float(np.median(stv[:max(1, len(stv) // 5)])) if len(stv) else np.nan,
                     "f0_last": float(np.median(stv[-max(1, len(stv) // 5):])) if len(stv) else np.nan,
                     "decl": float(slope)})
    pauses = [s[k + 1][0] - s[k][1] for s in per_span for k in range(len(s) - 1)]
    vf = np.concatenate([F[(tF >= a) & (tF <= b) & (F > 0)] for a, b in ipus])
    stv = 12 * np.log2(vf / 100.0)
    return dict(rows=rows, pauses=np.array(pauses), f0=vf, st=stv, nuclei=nuc, ipus=ipus, per_span=per_span, thr=thr)


def boot(x, fn=np.median, B=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return [np.nan, np.nan]
    bs = [fn(rng.choice(x, len(x))) for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ratio_boot(num, den, B=2000):
    num, den = np.asarray(num, float), np.asarray(den, float); n = len(num)
    bs = [num[i].sum() / den[i].sum() for i in (rng.integers(0, n, n) for _ in range(B))]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


A = {k: analyse(k) for k in REC}

# ------------------------------------------------------------------ 3. calibration with transcripts
import asr                                   # Whisper turbo via sherpa-onnx (asr.py)
CACHE = os.path.join(RES, "asr_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def transcribe(name, a, b):
    key = f"{name}:{a:.2f}-{b:.2f}"
    if key not in cache:
        w = load(REC[name]["wav"]); cache[key] = asr.transcribe(w, a, b)
        json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=0)
    return cache[key]


def transcribe_robust(name, a, b):
    """Whisper occasionally truncates or skips a stretch when a window edge moves by 0.1 s. Decode three windows with
    edges widened by 0, 0.1 and 0.2 s (all inside the surrounding pauses) and keep the decode of median length."""
    hyps = [transcribe(name, max(0.0, a - d), b + d) for d in (0.0, 0.1, 0.2)]
    return sorted(hyps, key=lambda h: nsyl(h))[1]


def degenerate(text):
    w = text.lower().split(); run = best = 1
    for x, y in zip(w, w[1:]):
        run = run + 1 if x == y else 1; best = max(best, run)
    return best >= 4


def nsyl(text, phonetic=True):
    return sum(syllables(t, phonetic=phonetic) for t in tokens(text) if not t[0].isdigit())


S18 = [s for s in json.load(open(os.path.join(HERE, "corpus.json"))) if s["date"] == "2002-07-29"][0]
P = S18["speeches"][0]["paragraphs"]
p9 = P[9]["text"]; p11 = P[11]["text"]
cut = p9.index("krimit të organizuar") + len("krimit të organizuar")
FN2_TEXT = p9[:cut] + ". " + p11                      # the broadcast keeps the first three se-clauses of ¶9, then ¶11
CAL = {}
for name, text in (("FN2", FN2_TEXT),):
    det = len(A[name]["nuclei"])
    CAL[name] = {"text_syllables_phonetic": nsyl(text, True), "text_syllables_orthographic": nsyl(text, False),
                 "detected_nuclei": det, "words": len([t for t in tokens(text) if not t[0].isdigit()])}
    CAL[name]["detection_ratio"] = det / CAL[name]["text_syllables_phonetic"]


def asr_sample(name, n=12, L=25.0):
    """n windows of about L s spread evenly over the analysed speech, edges snapped to pauses, automatic transcripts."""
    a_ = A[name]
    lo, hi = REC[name]["spans"][0] if REC[name]["spans"] else (0.0, parselmouth.Sound(REC[name]["wav"]).duration)
    starts = np.linspace(lo + 30, hi - 60, n)
    out = []
    for s0 in starts:
        # snap window edges to pauses so that no word is cut
        ip = [x for x in a_["ipus"] if x[0] >= s0][:200]
        a = ip[0][0] - 0.1; b = a
        for x in ip:
            if x[1] - a > L: break
            b = x[1] + 0.1
        txt = transcribe_robust(name, a, b)
        det = int(((a_["nuclei"] >= a) & (a_["nuclei"] <= b)).sum())
        phon = sum(min(y, b) - max(x, a) for x, y in a_["ipus"] if y > a and x < b)
        # a decode is unusable if the decoder looped (a word repeated 4+ times in a row) or stopped early
        # (fewer recognised syllables than half the detected nuclei)
        valid = not degenerate(txt) and nsyl(txt) >= 0.5 * det
        out.append({"a": a, "b": b, "text": txt, "asr_syllables": nsyl(txt), "detected": det, "phonation": phon, "valid": valid})
    kv = [k for k in out if k["valid"]]
    return {"windows": out, "n_valid": len(kv), "n_windows": len(out),
            "asr_syllables": int(sum(k["asr_syllables"] for k in kv)), "detected": int(sum(k["detected"] for k in kv)),
            "detection_ratio": sum(k["detected"] for k in kv) / sum(k["asr_syllables"] for k in kv),
            "ratio_ci": ratio_boot([k["detected"] for k in kv], [k["asr_syllables"] for k in kv])}


# KPD and the two archival recordings: 12 windows of 25 s each
for name in LONG:
    CAL[f"{name}_asr_sample"] = asr_sample(name)
# same check on the exact transcript, to measure the bias of the automatic syllable count itself
for name, spans in (("FN2", [(26.8, 47.55), (47.75, 72.2)]),):
    CAL[name]["asr_text"] = " ".join(transcribe_robust(name, a, b) for a, b in spans)
    CAL[name]["asr_syllables"] = nsyl(CAL[name]["asr_text"])
    CAL[name]["asr_over_exact"] = CAL[name]["asr_syllables"] / CAL[name]["text_syllables_phonetic"]
# automatic transcripts of the other two clips (Nano spans only)
for name, spans in (("FN1", [(42.8, 66.5), (66.5, 85.8), (105.6, 119.0), (119.0, 131.0)]),
                    ("FN3", [(24.0, 49.8), (49.8, 71.2), (71.2, 82.9)])):
    CAL[name] = {"asr_text": " ".join(transcribe_robust(name, a, b) for a, b in spans)}
    CAL[name]["asr_syllables"] = nsyl(CAL[name]["asr_text"]); CAL[name]["detected_nuclei"] = len(A[name]["nuclei"])
    CAL[name]["detection_ratio_vs_asr"] = CAL[name]["detected_nuclei"] / CAL[name]["asr_syllables"]
CAL["asr_bias"] = CAL["FN2"]["asr_over_exact"]          # the one exact transcript left after FN5 was withdrawn
R["calibration"] = CAL
BIAS = CAL["asr_bias"]

# ------------------------------------------------------------------ register comparison
for k, a in A.items():
    rows = a["rows"]; D = np.array([r["dur"] for r in rows]); S = np.array([r["syl"] for r in rows])
    phon = D.sum()
    # transcript-based syllable totals: exact for FN2, automatic for FN1/FN3, detector x own calibration for the long ones
    if k == "FN2": tsyl, src = CAL[k]["text_syllables_phonetic"], "exact transcript"
    elif k in ("FN1", "FN3"): tsyl, src = CAL[k]["asr_syllables"] / BIAS, "automatic transcript, bias-corrected"
    else: tsyl, src = len(a["nuclei"]) / CAL[f"{k}_asr_sample"]["detection_ratio"] / BIAS, "detector, calibrated on a transcribed sample"
    scale = tsyl / max(len(a["nuclei"]), 1)
    decl = np.array([r["decl"] for r in rows if r["dur"] >= 1.0 and np.isfinite(r["decl"])])
    R["recordings"][k] = {
        "label": REC[k]["label"], "short": REC[k]["short"], "spans": REC[k]["spans"],
        "speech_time_s": float(phon), "n_ipus": len(rows), "n_pauses": int(len(a["pauses"])),
        "pause_median": float(np.median(a["pauses"])) if len(a["pauses"]) else np.nan, "pause_median_ci": boot(a["pauses"]),
        "pause_mean": float(np.mean(a["pauses"])) if len(a["pauses"]) else np.nan,
        "ipu_dur_median": float(np.median(D)), "ipu_dur_ci": boot(D),
        "detected_nuclei": int(len(a["nuclei"])), "artic_detected": float(len(a["nuclei"]) / phon),
        "syllables_est": float(tsyl), "syllable_source": src, "artic_est": float(tsyl / phon),
        "artic_est_ci": [x * scale for x in ratio_boot(S, D)],
        "ipu_syl_est_median": float(np.median(S * scale)), "ipu_syl_est_ci": boot(S * scale),
        "f0_median_hz": float(np.median(a["f0"])), "f0_median_ci": [float(100 * 2 ** (x / 12)) for x in boot(a["st"])],
        "f0_range_st": float(np.percentile(a["st"], 95) - np.percentile(a["st"], 5)),
        "f0_sd_st": float(np.std(a["st"])),
        "decl_median": float(np.median(decl)) if len(decl) else np.nan, "decl_ci": boot(decl),
        "decl_share_negative": float((decl < 0).mean()) if len(decl) else np.nan, "decl_n": int(len(decl)),
        "pause_share_ge_1s": float((a["pauses"] >= 1).mean()) if len(a["pauses"]) else np.nan,
    }

# ------------------------------------------------------------------ 4. FN2: IPUs against the printed text
# Spoken reference = printed text, except one departure returned identically by two independent decodes (chunk-level and
# IPU-level) and confirmed by the author on listening to the broadcast: the adjective phrase "të patundur" is postposed after "institucional dhe politik".
REORDER = ("me një vullnet të patundur institucional dhe politik", "me një vullnet institucional dhe politik të patundur")
SPAN_TEXT = [p9[:cut] + ".", p11.replace(*REORDER)]
assert REORDER[1] in SPAN_TEXT[1]
a2 = A["FN2"]
FUNC = {"dhe", "e", "si", "që", "se", "për", "me", "në", "nga", "duke", "të", "a", "ose", "por", "ku"}


def sim(h, ref_):
    hyp = re.sub(r"[^\wëç ]", "", h.lower())
    return difflib.SequenceMatcher(None, hyp, ref_).ratio() * max(len(hyp), len(ref_))


def align(hyps, text):
    W = tokens(text); Wl = [x.lower() for x in W]; n, L = len(hyps), len(W)
    best = np.full((n + 1, L + 1), -1e9); arg = np.zeros((n + 1, L + 1), int); best[0, 0] = 0
    for i in range(1, n + 1):
        for k in range(i, L + 1):
            for j in range(max(i - 1, k - 40), k):
                if best[i - 1, j] < -1e8: continue
                v = best[i - 1, j] + sim(hyps[i - 1], " ".join(Wl[j:k]))
                if v > best[i, k]: best[i, k], arg[i, k] = v, j
    bnd = [L]
    for i in range(n, 0, -1): bnd.append(arg[i, bnd[-1]])
    bnd = bnd[::-1]
    pos, cur = [], 0
    for t in W:
        m_ = re.search(re.escape(t), text[cur:]); s_ = cur + m_.start(); pos.append((s_, s_ + len(t))); cur = s_ + len(t)

    def site(k):
        if k <= 0 or k >= L: return "edge"
        gap = text[pos[k - 1][1]:pos[k][0]]
        if re.search(r"[.!?]", gap): return "sentence end"
        if re.search(r"[,;:–—]", gap): return "comma"
        return "phrase boundary, unpunctuated" if Wl[k] in FUNC else "inside phrase"
    return W, bnd, site, L


al, sites, punct_total, punct_paused = [], [], 0, 0
for s_ipus, text in zip(a2["per_span"], SPAN_TEXT):
    hyps = [transcribe("FN2", max(0, x - 0.08), y + 0.08) for x, y in s_ipus]
    W, bnd, site, L = align(hyps, text)
    punct_total += sum(1 for k in range(1, L) if site(k) in ("sentence end", "comma"))
    for i, (x, y) in enumerate(s_ipus):
        seg = " ".join(W[bnd[i]:bnd[i + 1]])
        last = i == len(s_ipus) - 1
        st_ = site(bnd[i + 1]) if not last else "end of excerpt"
        row = next(r for r in a2["rows"] if abs(r["a"] - x) < 1e-9)
        al.append({"a": x, "b": y, "asr": hyps[i], "text": seg, "syl": nsyl(seg), "site_after": st_,
                   "f0_first": row["f0_first"], "f0_last": row["f0_last"], "f0_med": row["f0_st_med"], "decl": row["decl"],
                   "pause_after": None if last else s_ipus[i + 1][0] - y,
                   "artic": nsyl(seg) / (y - x)})
        if not last:
            sites.append(st_)
            punct_paused += st_ in ("sentence end", "comma")
clause_syl = [nsyl(c) for t in SPAN_TEXT for c in split_clauses(t)]
# pitch reset into each IPU (first fifth of IPU i+1 minus last fifth of IPU i), within a span; anaphoric 'se' openings
resets = []
for i in range(len(al) - 1):
    if al[i]["pause_after"] is None: continue
    resets.append({"into": i + 2, "opens_with_se": al[i + 1]["text"].lower().startswith("se "),
                   "reset_st": al[i + 1]["f0_first"] - al[i]["f0_last"]})
pa = np.array([r["pause_after"] for r in al if r["pause_after"] is not None])
ps = [r["site_after"] for r in al if r["pause_after"] is not None]
R["fn2_alignment"] = {
    "ipus": al, "n_pauses": len(sites), "pause_sites": {s: sites.count(s) for s in sorted(set(sites))},
    "pause_dur_by_site": {s: float(np.mean([p for p, q in zip(pa, ps) if q == s])) for s in set(ps)},
    "printed_punctuation_marks": punct_total, "punctuation_marks_realised_as_pause": int(punct_paused),
    "printed_clause_syllables": clause_syl, "ipu_syllables_exact": [r["syl"] for r in al],
    "spoken_reference": SPAN_TEXT, "departures": [{"printed": REORDER[0], "spoken": REORDER[1]}],
    "ipu_artic_cv": float(np.std([r["artic"] for r in al]) / np.mean([r["artic"] for r in al])),
    "resets": resets,
    "reset_se_mean": float(np.mean([r["reset_st"] for r in resets if r["opens_with_se"]])),
    "reset_other_mean": float(np.mean([r["reset_st"] for r in resets if not r["opens_with_se"]])),
    "ipu_syl_median": float(np.median([r["syl"] for r in al])), "clause_syl_median": float(np.median(clause_syl)),
    "artic_exact": float(sum(r["syl"] for r in al) / sum(r["b"] - r["a"] for r in al))}

# FN2: exact per-IPU syllable counts from the alignment replace the scaled detector counts
ex = [r["syl"] for r in al]
R["recordings"]["FN2"]["ipu_syl_est_median"] = float(np.median(ex)); R["recordings"]["FN2"]["ipu_syl_est_ci"] = boot(ex)
R["recordings"]["FN2"]["ipu_syllable_source"] = "exact (aligned printed text)"
# long recordings: widen the intervals by the uncertainty of the calibration ratio
for name in LONG:
    k = R["recordings"][name]; cr = CAL[f"{name}_asr_sample"]; r0, (rl, rh) = cr["detection_ratio"], cr["ratio_ci"]
    for key in ("artic_est_ci", "ipu_syl_est_ci"):
        k[key] = [k[key][0] * r0 / rh, k[key][1] * r0 / rl]
    k["ci_note"] = "includes the uncertainty of the calibration ratio"
# pitch level with intervals over IPUs (median of per-IPU median F0), less optimistic than intervals over frames
for name, a in A.items():
    med = np.array([r["f0_st_med"] for r in a["rows"] if np.isfinite(r["f0_st_med"])])
    R["recordings"][name]["f0_ipu_median_hz"] = float(100 * 2 ** (np.median(med) / 12))
    R["recordings"][name]["f0_ipu_median_ci_hz"] = [float(100 * 2 ** (x / 12)) for x in boot(med)]

# ------------------------------------------------------------------ 5. the long recordings: order in time (as in C.1)
# DFA of log IPU length in spoken order against shuffles (detector counts: a constant calibration factor does not change
# the exponent of a log series), and the profile of phrase length and pitch over tenths of the speech.
from rhythm_dfa import dfa
rng_long = np.random.default_rng(31)
R["long_recordings"] = {}
for name in LONG:
    a = A[name]; rows = [r for r in a["rows"] if r["syl"] > 0]
    scale = R["recordings"][name]["syllables_est"] / max(len(a["nuclei"]), 1)
    out = {"n_ipus_with_nuclei": len(rows), "n_ipus": len(a["rows"])}
    for key, vals in (("syl", [r["syl"] for r in rows]), ("dur", [r["dur"] for r in a["rows"]])):
        x = np.log(np.array(vals, float))
        scales = np.unique(np.round(np.logspace(np.log10(4), np.log10(len(x) / 8), 16)).astype(int))
        Fo = dfa(x, scales); a_obs = np.polyfit(np.log(scales), np.log(Fo), 1)[0]
        a_null = [np.polyfit(np.log(scales), np.log(dfa(rng_long.permutation(x), scales)), 1)[0] for _ in range(100)]
        out[f"dfa_{key}"] = {"alpha": float(a_obs), "null_mean": float(np.mean(a_null)), "null_sd": float(np.std(a_null)),
                             "z": float((a_obs - np.mean(a_null)) / np.std(a_null)), "n": int(len(x))}
    snd = parselmouth.Sound(REC[name]["wav"]); _, _, tF, F = tracks(snd)
    T0, T1 = a["rows"][0]["a"], a["rows"][-1]["b"]; prof = []
    for k in range(10):
        a_, b_ = T0 + k * (T1 - T0) / 10, T0 + (k + 1) * (T1 - T0) / 10
        rs = [r for r in a["rows"] if a_ <= r["a"] < b_ or (k == 9 and r["a"] >= a_)]
        f = np.concatenate([F[(tF >= r["a"]) & (tF <= r["b"]) & (F > 0)] for r in rs])
        prof.append({"decile": k, "n_ipu": len(rs), "ipu_syl_mean_est": float(np.mean([r["syl"] for r in rs]) * scale),
                     "f0_med": float(np.median(f)), "f0_p90": float(np.percentile(f, 90)),
                     "pause_share": float(1 - sum(r["dur"] for r in rs) / (b_ - a_)),
                     "artic_detected": float(sum(r["syl"] for r in rs) / sum(r["dur"] for r in rs))})
    out["decile_profile"] = prof
    m_ = [p["ipu_syl_mean_est"] for p in prof]; f_ = [p["f0_med"] for p in prof]
    out["first_tenth_longest"] = bool(m_[0] == max(m_)); out["last_tenth_highest_f0"] = bool(f_[-1] == max(f_))
    out["first_tenth_rank"] = int(1 + sum(v > m_[0] for v in m_)); out["last_tenth_f0_rank"] = int(1 + sum(v > f_[-1] for v in f_))
    # a drift in tape speed would move F0 and syllable rate in the same proportion: compare the two across tenths
    out["f0_ratio_last_first"] = prof[-1]["f0_med"] / prof[0]["f0_med"]
    out["artic_ratio_last_first"] = prof[-1]["artic_detected"] / prof[0]["artic_detected"]
    out["f0_spearman_tenth"] = float(stats.spearmanr(range(10), f_).statistic)
    out["speech_share"] = float(sum(r["dur"] for r in a["rows"]) / (T1 - T0))
    if name in ARCH:
        # mains hum in the non-speech frames: at the recording speed it lies at 50 Hz and its odd harmonics
        w = load(REC[name]["wav"]); m, st = REC[name]["vad"]
        keep = np.repeat(~m, int(round(st * SR)))[:len(w)]
        fr, P = signal.welch(w[:len(keep)][keep], SR, nperseg=2 ** 17); hum = {}
        for h in (50, 150):
            sel = (fr > h - 5) & (fr < h + 5); Ps = np.log(P[sel]); j = int(np.argmax(Ps))
            d = 0.5 * (Ps[j - 1] - Ps[j + 1]) / (Ps[j - 1] - 2 * Ps[j] + Ps[j + 1])
            hum[str(h)] = {"peak_hz": float(fr[sel][j] + d * (fr[1] - fr[0])),
                           "prominence_db": float(10 * np.log10(P[sel].max() / np.median(P[sel])))}
        out["mains_hum"] = hum
    R["long_recordings"][name] = out

# corrections to Appendix C.1 (intensity-based segmentation of acoustic.py), using the KPD calibration above
AC = json.load(open(os.path.join(RES, "acoustic.json"))); AR = json.load(open(os.path.join(RES, "acoustic_ipus.json")))
fac = 1 / CAL["KPD_asr_sample"]["detection_ratio"] / BIAS
U_ = json.load(open(os.path.join(HERE, "units.json")))
cl = np.array([c for c in (sum(syllables(t, phonetic=True) for t in c if not t[0].isdigit())
               for u in U_ for s in u["sentences"] if s["kind"] != "address" for c in s["clauses"] if c) if c > 0])
Sc = np.array([r["syl"] for r in AR]) * fac
R["c1_corrections"] = {"factor": fac, "speech_rate": AC["speech_rate_syll_per_s"] * fac,
                       "articulation_rate": AC["articulation_rate_syll_per_s"] * fac,
                       "ipu_syllables_median": float(np.median(Sc)),
                       "ks_ipu_vs_clause_corrected": float(stats.ks_2samp(Sc, cl).statistic),
                       "clause_syllables_median": float(np.median(cl)),
                       "kpd_vad_articulation_rate": R["recordings"]["KPD"]["artic_est"]}

save("video.json", R)
print(json.dumps({k: v for k, v in R["recordings"].items()}, indent=1, default=str)[:6000])
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "windows"} for k, v in R["diarisation"].items() if isinstance(v, dict)}, indent=1))
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("windows", "asr_text")} if isinstance(v, dict) else v)
                  for k, v in CAL.items()}, indent=1, default=str))
for r in al: print(f"{r['a']:.2f}-{r['b']:.2f} {r['syl']}syl [{r['site_after']}] pause {r['pause_after']}: {r['text']}  ||  {r['asr']}")
print({k: v for k, v in R["fn2_alignment"].items() if k not in ("ipus", "spoken_reference")})
