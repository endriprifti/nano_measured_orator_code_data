"""Section 8 (language models) — cross-validated surprisal of every token in the corpus under
(i) an interpolated Kneser–Ney 5-gram model and (ii) a small decoder-only Transformer, both over a
SentencePiece BPE vocabulary trained on the training folds only.

Each speech unit is assigned to one of K folds; for each fold, the tokenizer and both models are
trained on the other folds and the held-out speeches are scored. Output: results/surprisal.json with,
for each speech unit, the per-word surprisal (bits, summed over sub-word pieces) under both models.
"""
import collections, json, math, os, sys, time, re
import numpy as np
import sentencepiece as spm
from common import load_units, RES, HERE

K = 5
VOCAB = 2000
U = load_units()
rng = np.random.default_rng(2026)

# ---------------------------------------------------------------- token streams (lowercased; numbers -> '#')
PUNCT = re.compile(r"[A-Za-zËëÇç]+(?:['\-][A-Za-zËëÇç]+)*'?|\d+(?:[.,]\d+)*|[,;:.!?]")
def stream(u):
    toks, isword = [], []
    for s in u["sentences"]:
        for m in PUNCT.finditer(s["text"].replace("’", "'")):
            t = m.group(0)
            if t[0].isdigit(): toks.append("#"); isword.append(False)
            elif t in ",;:.!?": toks.append(t); isword.append(False)
            else: toks.append(t.lower()); isword.append(True)
        toks.append("¶"); isword.append(False)          # sentence boundary symbol
    return toks, isword
STREAMS = [stream(u) for u in U]

# balanced fold assignment by word count
order = np.argsort([-len(s[0]) for s in STREAMS]); load = np.zeros(K); fold = np.zeros(len(U), int)
for i in order:
    k = int(np.argmin(load)); fold[i] = k; load[k] += len(STREAMS[i][0])

# ---------------------------------------------------------------- Kneser–Ney n-gram over BPE ids
class KN:
    def __init__(self, n, V):
        self.n, self.V = n, V
    def fit(self, seqs):
        n = self.n
        self.c = [collections.Counter() for _ in range(n + 1)]       # counts of k-grams (top order uses raw)
        self.cont = [collections.Counter() for _ in range(n + 1)]    # N1+(• g)
        for s in seqs:
            s = [0] * (n - 1) + s
            for i in range(n - 1, len(s)):
                for k in range(1, n + 1):
                    g = tuple(s[i - k + 1:i + 1]); self.c[k][g] += 1
        for k in range(2, n + 1):
            for g in self.c[k]: self.cont[k - 1][g[1:]] += 1
        # context statistics
        self.ctx_tot = [collections.Counter() for _ in range(n + 1)]   # sum over w of count(h,w) (raw for top, cont for lower)
        self.ctx_types = [collections.Counter() for _ in range(n + 1)] # N1+(h •)
        for k in range(1, n + 1):
            src = self.c[k] if k == n else self.cont[k]
            for g, v in src.items():
                self.ctx_tot[k][g[:-1]] += v; self.ctx_types[k][g[:-1]] += 1
        self.D = []
        for k in range(n + 1):
            if k == 0: self.D.append(0.75); continue
            src = self.c[k] if k == n else self.cont[k]
            cc = collections.Counter(src.values()); n1, n2 = cc[1], cc[2]
            self.D.append(n1 / (n1 + 2 * n2) if n1 + n2 > 0 else 0.75)
        return self
    def prob(self, h, w):
        """h: tuple of up to n-1 previous ids."""
        p = 1.0 / self.V
        for k in range(1, self.n + 1):
            ctx = h[len(h) - (k - 1):] if k > 1 else ()
            if k > 1 and len(ctx) < k - 1: break
            src = self.c[k] if k == self.n else self.cont[k]
            tot = self.ctx_tot[k].get(ctx, 0)
            if tot == 0: continue
            D = self.D[k]
            cnt = src.get(ctx + (w,), 0)
            p = max(cnt - D, 0) / tot + D * self.ctx_types[k][ctx] / tot * p
        return p

# ---------------------------------------------------------------- small GPT in JAX
import jax, jax.numpy as jnp, optax
CFG = dict(d=192, L=4, H=4, T=128, ff=768, drop=0.1)
def init_params(key, V, cfg=CFG):
    d, L, T, ff = cfg["d"], cfg["L"], cfg["T"], cfg["ff"]
    ks = jax.random.split(key, 4 + 4 * L); s = 0.02
    P = {"wte": jax.random.normal(ks[0], (V, d)) * s, "wpe": jax.random.normal(ks[1], (T, d)) * s,
         "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d), "blocks": []}
    for l in range(L):
        k = ks[4 + 4 * l: 8 + 4 * l]
        P["blocks"].append({"ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d), "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
                            "qkv": jax.random.normal(k[0], (d, 3 * d)) * s, "proj": jax.random.normal(k[1], (d, d)) * s / math.sqrt(2 * L),
                            "fc": jax.random.normal(k[2], (d, ff)) * s, "fc2": jax.random.normal(k[3], (ff, d)) * s / math.sqrt(2 * L)})
    return P
def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b
def forward(P, idx, key=None, train=False, cfg=CFG):
    B, T = idx.shape; d, H = cfg["d"], cfg["H"]
    x = P["wte"][idx] + P["wpe"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    def drop(x, k):
        if not train: return x
        keep = jax.random.bernoulli(k, 1 - cfg["drop"], x.shape); return jnp.where(keep, x / (1 - cfg["drop"]), 0)
    keys = jax.random.split(key, 2 * len(P["blocks"]) + 1) if train else [None] * (2 * len(P["blocks"]) + 1)
    for i, bl in enumerate(P["blocks"]):
        h = ln(x, bl["ln1_g"], bl["ln1_b"])
        q, k_, v = jnp.split(h @ bl["qkv"], 3, -1)
        q = q.reshape(B, T, H, d // H).transpose(0, 2, 1, 3); k_ = k_.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)
        att = (q @ k_.transpose(0, 1, 3, 2)) / math.sqrt(d // H)
        att = jnp.where(mask, att, -1e9); att = jax.nn.softmax(att, -1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + drop(y @ bl["proj"], keys[2 * i])
        h = ln(x, bl["ln2_g"], bl["ln2_b"])
        x = x + drop(jax.nn.gelu(h @ bl["fc"]) @ bl["fc2"], keys[2 * i + 1])
    x = ln(x, P["lnf_g"], P["lnf_b"])
    return x @ P["wte"].T                                   # tied output embedding
def loss_fn(P, xb, yb, key):
    logits = forward(P, xb, key, train=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
@jax.jit
def eval_logp(P, xb):
    return jax.nn.log_softmax(forward(P, xb, train=False), -1)

def train_gpt(train_ids, val_ids, V, max_epochs=14, B=32, seed=0, log=print):
    T = CFG["T"]; key = jax.random.PRNGKey(seed)
    P = init_params(key, V)
    data = np.array(train_ids, np.int32); vdat = np.array(val_ids, np.int32)
    steps_per_epoch = max(1, len(data) // (B * T))
    total = steps_per_epoch * max_epochs
    sched = optax.warmup_cosine_decay_schedule(0.0, 2e-3, min(200, total // 10), total, 1e-4)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.1))
    st = opt.init(P)
    @jax.jit
    def step(P, st, xb, yb, k):
        l, g = jax.value_and_grad(loss_fn)(P, xb, yb, k)
        up, st = opt.update(g, st, P); return optax.apply_updates(P, up), st, l
    def val_loss(P):
        n = (len(vdat) - 1) // T; ls = []
        for i in range(0, n, 16):
            xs = np.stack([vdat[j * T:(j + 1) * T] for j in range(i, min(n, i + 16))])
            ys = np.stack([vdat[j * T + 1:(j + 1) * T + 1] for j in range(i, min(n, i + 16))])
            lp = eval_logp(P, xs); ls.append(-np.take_along_axis(np.array(lp), ys[..., None], -1).mean())
        return float(np.mean(ls))
    best, bestP, bad = 1e9, P, 0; r = np.random.default_rng(seed)
    for ep in range(max_epochs):
        t0 = time.time(); tl = []
        for _ in range(steps_per_epoch):
            st_idx = r.integers(0, len(data) - T - 1, B)
            xb = np.stack([data[i:i + T] for i in st_idx]); yb = np.stack([data[i + 1:i + T + 1] for i in st_idx])
            key, sk = jax.random.split(key)
            P, st, l = step(P, st, xb, yb, sk); tl.append(float(l))
        vl = val_loss(P)
        log(f"    epoch {ep+1}: train {np.mean(tl):.3f} val {vl:.3f} nats/token  ({time.time()-t0:.0f}s)")
        if vl < best - 1e-3: best, bestP, bad = vl, P, 0
        else:
            bad += 1
            if bad >= 2: break
    return bestP, best

def score_gpt(P, ids, stride=64):
    """Per-position log2-prob for ids[1:], each position conditioned on >= T-stride tokens when available."""
    T = CFG["T"]; ids = np.array(ids, np.int32); n = len(ids); out = np.full(n, np.nan)
    starts = list(range(0, max(1, n - 1), stride))
    for s0 in starts:
        x = ids[s0:s0 + T]
        if len(x) < 2: break
        pad = T - len(x); xin = np.pad(x, (0, pad)) if pad else x
        lp = np.array(eval_logp(P, xin[None]))[0]
        tgt = ids[s0 + 1:s0 + T + 1]
        m = len(tgt)
        vals = lp[np.arange(m), tgt] / math.log(2)
        first = 0 if s0 == 0 else T - stride - 1
        for j in range(max(first, 0), m):
            if np.isnan(out[s0 + 1 + j]): out[s0 + 1 + j] = -vals[j]
        if s0 + T >= n: break
    return out

# ---------------------------------------------------------------- cross-validation loop
def main(models=("kn", "gpt")):
    tmp = os.path.join(HERE, "spm"); os.makedirs(tmp, exist_ok=True)
    results = {u["id"]: {} for u in U}
    meta = {"K": K, "vocab": VOCAB, "cfg": CFG, "folds": {}}
    logf = open(os.path.join(RES, "lm.log"), "a")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()
    for k in range(K):
        tr = [i for i in range(len(U)) if fold[i] != k]; te = [i for i in range(len(U)) if fold[i] == k]
        txt = os.path.join(tmp, f"train{k}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            for i in tr:
                toks = STREAMS[i][0]; line = []
                for t in toks:
                    if t == "¶": f.write(" ".join(line) + "\n"); line = []
                    else: line.append(t)
        spm.SentencePieceTrainer.train(input=txt, model_prefix=os.path.join(tmp, f"bpe{k}"), vocab_size=VOCAB,
                                       model_type="bpe", character_coverage=1.0, user_defined_symbols=["¶"],
                                       bos_id=-1, eos_id=-1, unk_id=1, pad_id=0, minloglevel=2)
        sp = spm.SentencePieceProcessor(model_file=os.path.join(tmp, f"bpe{k}.model"))
        def encode(i):
            ids, owner = [], []          # owner = index of the corpus token each piece belongs to
            for j, t in enumerate(STREAMS[i][0]):
                pcs = sp.encode(t); ids.extend(pcs); owner.extend([j] * len(pcs))
            return ids, owner
        enc = {i: encode(i) for i in range(len(U))}
        train_seqs = [enc[i][0] for i in tr]
        log(f"fold {k}: train units {len(tr)}, test units {len(te)}, train pieces {sum(map(len, train_seqs))}")
        fm = {}
        if "kn" in models:
            t0 = time.time(); kn = KN(5, VOCAB).fit(train_seqs); log(f"  KN fitted ({time.time()-t0:.0f}s), D={['%.2f' % d for d in kn.D]}")
            for i in te:
                ids, owner = enc[i]; s = [0] * 4 + ids; bits = []
                for j in range(4, len(s)): bits.append(-math.log2(kn.prob(tuple(s[j - 4:j]), s[j])))
                results[U[i]["id"]]["kn_piece_bits"] = bits
        if "gpt" in models:
            flat = [x for sq in train_seqs for x in sq]; cut = int(0.95 * len(flat))
            P, vbest = train_gpt(flat[:cut], flat[cut:], VOCAB, log=log, seed=k)
            fm["gpt_val_nats"] = vbest
            for i in te:
                ids, owner = enc[i]
                results[U[i]["id"]]["gpt_piece_bits"] = score_gpt(P, ids).tolist()
        for i in te:
            ids, owner = enc[i]
            results[U[i]["id"]]["owner"] = owner; results[U[i]["id"]]["n_pieces"] = len(ids)
        meta["folds"][k] = fm
        # aggregate fold-level bits per piece / per word
        for mname in [m for m in models]:
            key = f"{mname}_piece_bits"
            b = [x for i in te for x in results[U[i]["id"]][key] if x == x]
            log(f"  {mname}: mean {np.mean(b):.3f} bits/piece over {len(b)} pieces")
    # map to corpus tokens: surprisal of a token = sum of its pieces' surprisals
    out = {}
    for i, u in enumerate(U):
        r = results[u["id"]]; toks, isword = STREAMS[i]; owner = r["owner"]
        rec = {"tokens": toks, "isword": isword}
        for m in models:
            pb = r[f"{m}_piece_bits"]; tb = np.zeros(len(toks)); has = np.ones(len(toks), bool)
            for j, b in enumerate(pb):
                if b != b: has[owner[j]] = False; continue
                tb[owner[j]] += b
            tb[~has] = np.nan
            rec[m] = tb.tolist()
        rec["n_pieces"] = r["n_pieces"]; rec["fold"] = int(fold[i])
        out[u["id"]] = rec
    json.dump({"meta": meta, "units": out}, open(os.path.join(RES, "surprisal.json"), "w"))
    log("done")

if __name__ == "__main__":
    main(tuple(sys.argv[1:]) if len(sys.argv) > 1 else ("kn", "gpt"))
