"""Section 7 — figures of speech as measurable structures.
Anaphora / epistrophe at sentence and clause level against permutation nulls, anaphoric run lengths
(tricolon), enumerations, antithetic and adversative constructions, interrogatives and exclamations,
pronominal stance, modality, forms of address, Jakobsonian local recurrence, alliteration, burstiness."""
import collections, re, math
import numpy as np
from scipy import stats
from common import *
from prep import syllables

U = load_units()
rng = np.random.default_rng(5)
R = {}
NW = {t: len(term_words(U, t)) for t in TERMS}; NW["all"] = sum(NW[t] for t in TERMS)
R["words_by_term"] = NW

def first_word(tokens):
    for t in tokens:
        if not t[0].isdigit(): return t.lower()
    return None
def last_word(tokens):
    for t in reversed(tokens):
        if not t[0].isdigit(): return t.lower()
    return None

# ---------------------------------------------------------------- sentence-level anaphora / epistrophe
sp_sents = [[s for s in u["sentences"] if s["kind"] != "address"] for u in U]
def adj_equal(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a is not None and a == b)
def sentence_level(fn):
    seqs = [[fn(s["tokens"]) for s in ss] for ss in sp_sents]
    obs = sum(adj_equal(x) for x in seqs)
    null = [sum(adj_equal(list(rng.permutation(np.array(x, dtype=object)))) for x in seqs) for _ in range(1000)]
    pairs = sum(max(len(x) - 1, 0) for x in seqs)
    return {"observed": obs, "pairs": pairs, "null_mean": np.mean(null), "null_sd": np.std(null),
            "ratio": obs / np.mean(null), "z": (obs - np.mean(null)) / np.std(null),
            "p_perm": (1 + sum(n >= obs for n in null)) / 1001}
R["anaphora_sentence"] = sentence_level(first_word)
R["epistrophe_sentence"] = sentence_level(last_word)
# words driving sentence anaphora
drv = collections.Counter()
for ss in sp_sents:
    fw = [first_word(s["tokens"]) for s in ss]
    for a, b in zip(fw, fw[1:]):
        if a and a == b: drv[a] += 1
R["anaphora_sentence_words"] = drv.most_common(15)

# ---------------------------------------------------------------- clause-level anaphora (inside the period)
SENTS = [s for ss in sp_sents for s in ss]
cl_first = [[first_word(c) for c in s["clauses"] if c] for s in SENTS]
cl_last = [[last_word(c) for c in s["clauses"] if c] for s in SENTS]
def clause_level(seqs, n_perm=500):
    obs = sum(adj_equal(x) for x in seqs)
    # null A: within-sentence permutation of clause order (conservative)
    nullA = [sum(adj_equal(list(rng.permutation(np.array(x, dtype=object)))) for x in seqs if len(x) > 1)
             for _ in range(200)]
    # null B: clause-initial words redistributed across the corpus, sentence structure kept
    # (separately for sentence-initial and non-initial clause positions)
    pos0 = np.array([x[0] for x in seqs if x], dtype=object)
    rest = np.array([w for x in seqs for w in x[1:]], dtype=object)
    lens = [len(x) for x in seqs if x]
    nullB = []
    for _ in range(n_perm):
        p0 = rng.permutation(pos0); pr = rng.permutation(rest); k = 0; tot = 0
        for i, n in enumerate(lens):
            seq = [p0[i]] + list(pr[k:k + n - 1]); k += n - 1
            tot += adj_equal(seq)
        nullB.append(tot)
    pairs = sum(max(len(x) - 1, 0) for x in seqs)
    return {"observed": obs, "pairs": pairs,
            "nullA_within_mean": np.mean(nullA), "nullA_within_sd": np.std(nullA),
            "zA": (obs - np.mean(nullA)) / np.std(nullA),
            "nullB_across_mean": np.mean(nullB), "nullB_across_sd": np.std(nullB),
            "ratioB": obs / np.mean(nullB), "zB": (obs - np.mean(nullB)) / np.std(nullB)}
R["anaphora_clause"] = clause_level(cl_first)
R["epistrophe_clause"] = clause_level(cl_last)
drv = collections.Counter()
for x in cl_first:
    for a, b in zip(x, x[1:]):
        if a and a == b: drv[a] += 1
R["anaphora_clause_words"] = drv.most_common(20)

# ---------------------------------------------------------------- run lengths (the tricolon question)
def runs(seqs):
    out = []
    for x in seqs:
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[j + 1] is not None and x[j + 1] == x[i]: j += 1
            if j > i: out.append(j - i + 1)
            i = j + 1
    return out
def run_profile(rl):
    c = collections.Counter(rl); n2 = sum(v for k, v in c.items() if k >= 2)
    n3 = sum(v for k, v in c.items() if k >= 3); n4 = sum(v for k, v in c.items() if k >= 4)
    n5 = sum(v for k, v in c.items() if k >= 5)
    return {"counts": dict(sorted(c.items())), "P_ge3_given_ge2": n3 / n2 if n2 else np.nan,
            "P_ge4_given_ge3": n4 / n3 if n3 else np.nan, "P_ge5_given_ge4": n5 / n4 if n4 else np.nan}
rl_clause = runs(cl_first); rl_sent = runs([[first_word(s["tokens"]) for s in ss] for ss in sp_sents])
R["runs_clause"] = run_profile(rl_clause); R["runs_sentence"] = run_profile(rl_sent)
# null run profile from the across-corpus clause-word redistribution
pos0 = np.array([x[0] for x in cl_first if x], dtype=object)
rest = np.array([w for x in cl_first for w in x[1:]], dtype=object); lens = [len(x) for x in cl_first if x]
null_runs = []
for _ in range(200):
    p0 = rng.permutation(pos0); pr = rng.permutation(rest); k = 0; seqs = []
    for i, n in enumerate(lens):
        seqs.append([p0[i]] + list(pr[k:k + n - 1])); k += n - 1
    null_runs.extend(runs(seqs))
R["runs_clause_null"] = run_profile(null_runs)
# test: continuation hazard at 3 lower than at 2 (runs 'close' at three)
c = collections.Counter(rl_clause)
n2 = sum(v for k, v in c.items() if k >= 2); n3 = sum(v for k, v in c.items() if k >= 3); n4 = sum(v for k, v in c.items() if k >= 4)
tab = np.array([[n3, n2 - n3], [n4, n3 - n4]])
R["runs_clause_hazard_test"] = {"table": tab, "fisher_p": stats.fisher_exact(tab)[1]}

# ---------------------------------------------------------------- enumerations: 'A, B dhe C' series
SUBORD = {"që", "i", "e", "të", "sepse", "nëse", "por", "ndërsa", "kur", "pasi", "megjithëse", "ndonëse", "duke", "si", "ku"}
def enumerations(sent_text):
    segs = [x.strip() for x in re.split(r",", sent_text)]
    out = []; i = 0
    while i < len(segs):
        j = i; members = 0
        while j < len(segs) and 1 <= len(segs[j].split()) <= 5:
            # final member pair joined by a coordinator
            m = re.search(r"^(.{1,60}?)\s+(?:dhe|apo|ose|si dhe|e)\s+(.{1,60})$", segs[j])
            if m and members >= 1 and len(m.group(1).split()) <= 5 and len(m.group(2).split()) <= 6:
                members += 2; j += 1; break
            members += 1; j += 1
        else:
            pass
        if members >= 3 and j > i and re.search(r"\s(?:dhe|apo|ose|e)\s", segs[j - 1]):
            out.append(members); i = j
        else:
            i += 1
    return out
en = [k for s in SENTS for k in enumerations(s["text"])]
ce = collections.Counter(min(k, 6) for k in en)
R["enumerations"] = {"n": len(en), "dist": dict(sorted(ce.items())), "share_three": ce[3] / len(en) if en else np.nan,
                     "per_1000_words": 1000 * len(en) / NW["all"]}

# ---------------------------------------------------------------- lexical markers per 1000 words, by term
def rate(pattern, flags=re.I):
    rx = re.compile(pattern, flags); out = {}
    for t in TERMS + ("all",):
        n = 0
        for u in U:
            if t != "all" and u["term"] != t: continue
            for s in u["sentences"]:
                n += len(rx.findall(s["text"].replace("’", "'")))
        out[t] = 1000 * n / NW[t]
    return out
MARK = {
    "antithesis_jo_vetem_por": r"\bjo vetëm\b[^.?!]{0,250}?\bpor\b",
    "antithesis_jo_por": r"\bjo\b(?! vetëm)[^.?!;]{1,90}?,?\s+por\b",
    "antithesis_as_as": r"\bas\b[^.?!]{1,90}?\bas\b",
    "antithesis_nga_njera_ane": r"\bnga njëra anë\b",
    "adversative_por": r"\bpor\b",
    "contrastive_ndersa_kurse": r"\b(ndërsa|kurse)\b",
    "concessive_megjithate": r"\b(megjithatë|megjithëse|ndonëse|sidoqoftë)\b",
    "pron_1sg": r"\b(unë|mua|im|ime|tim|time|imi|imja)\b",
    "verb_1sg_cognitive": r"\b(besoj|mendoj|theksoj|them|dua|kërkoj|konsideroj|vlerësoj|siguroj|përsëris|falënderoj|ftoj|kujtoj|jam|kam)\b",
    "pron_1pl": r"\b(ne|neve|ynë|jonë|tonë|tanë|sonë|tona|tanët|tonat|na)\b",
    "verb_1pl": r"\b(kemi|jemi|duam|themi|mund të \w+jmë)\b|\b\w{3,}(ojmë|jmë)\b",
    "pron_2pl": r"\b(ju|juve|juaj|tuaj|tuaja|tuajat)\b",
    # 2nd-person-plural present/imperative in -oni; nouns in -ioni (komisioni, informacioni ...) and eight other
    # nouns in -oni attested in the corpus are excluded
    "verb_2pl": r"\b(keni|jeni|doni|thoni)\b|\b(?!\w*ioni\b)(?!(?:anxhoni|betoni|harmoni|maqedoni|rajoni|sinkroni|"
                r"telefoni|uashingtoni)\b)\w{3,}oni\b",
    "deontic_duhet": r"\bduhet\b",
    "possibility_mund": r"\bmund\b",
    "future_do_te": r"\bdo t[ëa]\b|\bdo t'",
    "intensifier_shume": r"\b(shumë|tepër|jashtëzakonisht|veçanërisht|thellësisht)\b",
    "absolute_never_all": r"\b(kurrë|asnjëherë|asnjë|gjithmonë|gjithnjë|absolutisht)\b",
    # a name after zoti/zonja (title word matched in any case, so sentence-initial "Zoti" counts; titles of office excluded)
    "address_zoti_name": r"\b(?i:zoti|zonja|zotit|zonjës)\s+(?!(?:Kryetar|President)\w*\b)[A-ZËÇ]\w+",
    "address_honorific": r"\b(i nderuar|e nderuar|të nderuar)\b",
    "address_kolege": r"\bkolegë\b",
    "mention_berisha": r"\bBerish\w*\b",
    "europe_integration": r"\b(europ\w*|euroatlantik\w*|integrim\w*|NATO|BE-\w*)\b",
    "reform": r"\breform\w*\b",
    "democracy": r"\bdemokraci\w*|\bdemokratik\w*",
    "law_state": r"\bshtet\w* (ligjor|i së drejtës)|\bligj\w*",
}
R["markers_per_1000"] = {k: rate(v) for k, v in MARK.items()}
for k in ("address_zoti_name", "mention_berisha", "europe_integration"):
    R["markers_per_1000"][k] = rate(MARK[k], 0) if k != "europe_integration" else rate(MARK[k], re.I)
# chi-square test across terms for selected markers
def count_by_term(pattern, flags=re.I):
    rx = re.compile(pattern, flags)
    return [sum(len(rx.findall(s["text"])) for u in U if u["term"] == t for s in u["sentences"]) for t in TERMS]
R["marker_tests"] = {}
for k in MARK:
    cnt = count_by_term(MARK[k], 0 if k in ("address_zoti_name", "mention_berisha") else re.I)
    tab = np.array([cnt, [NW[t] - c for t, c in zip(TERMS, cnt)]])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    R["marker_tests"][k] = {"counts": cnt, "chi2": chi2, "p": p}
# stance ratio we/I
we = np.array(count_by_term(MARK["pron_1pl"])) + np.array(count_by_term(MARK["verb_1pl"]))
i_ = np.array(count_by_term(MARK["pron_1sg"])) + np.array(count_by_term(MARK["verb_1sg_cognitive"]))
R["we_I_ratio"] = dict(zip(TERMS, (we / i_).tolist()))

# sentence kinds per term
kinds = {t: collections.Counter(s["kind"] for u in U if u["term"] == t for s in u["sentences"]) for t in TERMS}
R["sentence_kinds"] = {t: dict(v) for t, v in kinds.items()}
R["question_per_1000_sent"] = {t: 1000 * kinds[t]["question"] / sum(kinds[t].values()) for t in TERMS}
R["question_marks_total"] = sum(s["text"].count("?") for u in U for s in u["sentences"])
excl = [s["text"] for u in U for s in u["sentences"] if s["kind"] == "exclamation"]
formula = re.compile(r"^(ju )?fale?m?i?nderit|^faleminderit|^falemnderit", re.I)
R["exclamations"] = {"total": len(excl), "formulaic_thanks": sum(1 for e in excl if formula.search(e.strip()))}
R["question_examples"] = [s["text"] for u in U for s in u["sentences"] if s["kind"] == "question"][:32]

# ---------------------------------------------------------------- Jakobsonian local recurrence
F = collections.Counter(all_words(U)); top = {w for w, _ in F.most_common(100)}
def local_recurrence(tokens, k=20):
    hits = tot = 0
    for i, w in enumerate(tokens):
        if w in top: continue
        tot += 1
        if w in tokens[max(0, i - k):i]: hits += 1
    return hits, tot
def recurrence_test(level, k=20, n_perm=100):
    obs_h = obs_t = 0; units = []
    for u in U:
        if level == "speech":
            blocks = [unit_words(u)]
        else:  # paragraph
            pars = collections.defaultdict(list)
            for s in u["sentences"]: pars[s["par"]].extend(words(s))
            blocks = list(pars.values())
        for b in blocks:
            h, t = local_recurrence(b, k); obs_h += h; obs_t += t; units.append(b)
    null = []
    for _ in range(n_perm):
        hh = 0
        for b in units:
            h, _t = local_recurrence(list(rng.permutation(np.array(b, dtype=object))), k); hh += h
        null.append(hh)
    return {"k": k, "observed_rate": obs_h / obs_t, "null_rate_mean": np.mean(null) / obs_t,
            "null_rate_sd": np.std(null) / obs_t, "ratio": obs_h / np.mean(null), "z": (obs_h - np.mean(null)) / np.std(null)}
R["recurrence_speech_shuffle"] = recurrence_test("speech", 20, 60)
R["recurrence_paragraph_shuffle"] = recurrence_test("paragraph", 20, 60)
R["recurrence_by_term"] = {}
for t in TERMS:
    hits = tot = 0
    for u in U:
        if u["term"] != t: continue
        h, tt = local_recurrence(unit_words(u)); hits += h; tot += tt
    R["recurrence_by_term"][t] = hits / tot

# ---------------------------------------------------------------- alliteration of adjacent polysyllables
DIG = ("dh", "gj", "ll", "nj", "rr", "sh", "th", "xh", "zh")
def onset(w):
    w = w.lower(); return w[:2] if w[:2] in DIG else w[:1]
def allit(tokens):
    ws = [t.lower() for t in tokens if not t[0].isdigit() and t.lower() not in top and syllables(t) >= 2]
    return sum(1 for a, b in zip(ws, ws[1:]) if onset(a) == onset(b) and onset(a) not in "aeëiouy"), max(len(ws) - 1, 0), ws
obs = 0; pairs = 0; bags = []
for s in SENTS:
    h, p, ws = allit(s["tokens"]); obs += h; pairs += p; bags.append(ws)
null = []
for _ in range(300):
    tot = 0
    for ws in bags:
        if len(ws) < 2: continue
        pw = list(rng.permutation(np.array(ws, dtype=object)))
        tot += sum(1 for a, b in zip(pw, pw[1:]) if onset(a) == onset(b) and onset(a) not in "aeëiouy")
    null.append(tot)
R["alliteration"] = {"observed": obs, "pairs": pairs, "null_mean": np.mean(null), "null_sd": np.std(null),
                     "ratio": obs / np.mean(null), "z": (obs - np.mean(null)) / np.std(null)}

# ---------------------------------------------------------------- burstiness of frequent content words
W = all_words(U); pos = collections.defaultdict(list)
for i, w in enumerate(W): pos[w].append(i)
cands = [w for w, f in F.most_common(600) if w not in top and f >= 40]
B = []
for w in cands:
    d = np.diff(pos[w]); mu, sd = d.mean(), d.std()
    B.append((w, F[w], (sd - mu) / (sd + mu)))
B.sort(key=lambda x: -x[2])
R["burstiness"] = {"n_words": len(B), "median_B": float(np.median([b for _, _, b in B])),
                   "most_bursty": B[:15], "least_bursty": B[-15:]}
fw = [(w, F[w]) for w, _ in F.most_common(30)]
Bf = []
for w, f in fw:
    d = np.diff(pos[w]); Bf.append((d.std() - d.mean()) / (d.std() + d.mean()))
R["burstiness_function_words_median"] = float(np.median(Bf))
# ---------------------------------------------------------------- additions (separate random stream, so earlier results are unchanged)
rng2 = np.random.default_rng(55)
from statsmodels.stats.multitest import multipletests
_keys = list(R["marker_tests"])
_adj = multipletests([R["marker_tests"][k]["p"] for k in _keys], method="fdr_bh")[1]
for k, a in zip(_keys, _adj): R["marker_tests"][k]["p_bh"] = float(a)
R["marker_tests_bh_family"] = len(_keys)        # Benjamini-Hochberg over all markers in MARK, not only those tabulated
TITLES = {"ministër", "kryeministër", "zëvendëskryeministër", "zëvendësministër", "ministre", "sekretar", "kryetar"}
def sentence_level_variant(fn, drop_first=(), mask_words=()):
    seqs = []
    for ss in sp_sents:
        ss = [s for s in ss if first_word(s["tokens"]) not in drop_first]
        seqs.append([None if fn(s["tokens"]) in mask_words else fn(s["tokens"]) for s in ss])
    obs = sum(adj_equal(x) for x in seqs)
    null = [sum(adj_equal(list(rng2.permutation(np.array(x, dtype=object)))) for x in seqs) for _ in range(1000)]
    return {"observed": obs, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "ratio": obs / np.mean(null), "z": (obs - np.mean(null)) / np.std(null)}
R["anaphora_sentence_no_titles"] = sentence_level_variant(first_word, drop_first=TITLES)
R["anaphora_sentence_no_titles_no_ne"] = sentence_level_variant(first_word, drop_first=TITLES, mask_words={"në"})
drv = collections.Counter()
for ss in sp_sents:
    lw = [last_word(s["tokens"]) for s in ss]
    for a, b in zip(lw, lw[1:]):
        if a and a == b: drv[a] += 1
R["epistrophe_sentence_words"] = drv.most_common(15)
drv = collections.Counter()
for x in cl_last:
    for a, b in zip(x, x[1:]):
        if a and a == b: drv[a] += 1
R["epistrophe_clause_words"] = drv.most_common(20)

save("figures.json", R)
print(json.dumps(R, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1, ensure_ascii=False)[:9000])
