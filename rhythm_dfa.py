import numpy as np
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
