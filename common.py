"""Shared loading utilities for the analysis scripts."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
TERMS = ("T1", "T2", "T3")
TERM_LABEL = {"T1": "1991", "T2": "1997–1998", "T3": "2002–2005"}

def load_units():
    U = json.load(open(os.path.join(HERE, "units.json"), encoding="utf-8"))
    U.sort(key=lambda u: (u["date"], u["id"]))
    return U

def is_word(t):
    return not t[0].isdigit()

def words(sent):
    return [t.lower() for t in sent["tokens"] if is_word(t)]

def unit_words(u):
    return [w for s in u["sentences"] for w in words(s)]

def term_words(U, term):
    return [w for u in U if u["term"] == term for w in unit_words(u)]

def all_words(U):
    return [w for u in U for w in unit_words(u)]

def save(name, obj):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(os.path.join(RES, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=conv)

# figure style: one quiet grey + one accent (blue); a single hue in shades for ordered series
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
GREY = "#9A9994"; DARK = "#3A3A38"; ACCENT = "#2a78d6"
TERM_COL = {"T1": "#8fb4e3", "T2": "#2a78d6", "T3": "#123f78"}
TERM_LS = {"T1": ":", "T2": "--", "T3": "-"}
TERM_MK = {"T1": "o", "T2": "s", "T3": "^"}   # shades of one hue: chronological order
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": DARK, "axes.linewidth": 0.6,
    "xtick.color": DARK, "ytick.color": DARK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
