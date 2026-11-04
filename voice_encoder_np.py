"""NumPy re-implementation of the Resemblyzer GE2E speaker encoder (Wan et al. 2018; Jemine 2019).

The pretrained weights ship inside the PyPI package `Resemblyzer` (resemblyzer/pretrained.pt, legacy torch
serialisation). PyTorch could not be installed in the analysis environment, so the checkpoint is read with a
custom unpickler and the 3-layer LSTM (40 -> 256) + linear 256 + ReLU + L2 norm is run in NumPy.
Pre-processing follows resemblyzer.audio: volume normalisation to -30 dBFS (increase only), 40-band mel power
spectrogram (25 ms window, 10 ms step), partial utterances of 160 frames at 1.3 partials per second.
Long-silence trimming with webrtcvad is replaced by the caller passing speech-only samples.
"""
import pickle, struct, collections, io
import numpy as np
import librosa

SR = 16000
DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32}


class _Ref:
    def __init__(self, key, dtype, size): self.key, self.dtype, self.size = key, dtype, size


class _Lazy:
    def __init__(self, ref, offset, size, stride): self.ref, self.offset, self.size, self.stride = ref, offset, size, stride


def _rebuild(storage, offset, size, stride, *args):
    return _Lazy(storage, offset, tuple(size), tuple(stride))


class _Unp(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name.startswith("_rebuild_tensor"): return _rebuild
        if mod == "torch" and name.endswith("Storage"): return name
        if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        _, stype, key, location, size, view = pid
        return _Ref(key, DT[stype], size)


def _load(path):
    f = open(path, "rb")
    for _ in range(3): pickle.load(f)
    obj = _Unp(f).load()
    keys = pickle.load(f)
    refs = {}
    def collect(x):
        if isinstance(x, _Lazy): refs[x.ref.key] = x.ref
        elif isinstance(x, dict):
            for v in x.values(): collect(v)
    collect(obj)
    data = {}
    for k in keys:
        n = struct.unpack("<q", f.read(8))[0]
        dt = np.dtype(refs[k].dtype) if k in refs else np.dtype(np.float32)
        data[k] = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    def mat(x):
        if isinstance(x, _Lazy):
            buf = data[x.ref.key]; it = buf.itemsize
            return np.lib.stride_tricks.as_strided(buf[x.offset:], shape=x.size,
                                                   strides=tuple(s * it for s in x.stride)).astype(np.float32)
        if isinstance(x, dict): return {k: mat(v) for k, v in x.items()}
        return x
    return mat(obj)


def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class VoiceEncoder:
    def __init__(self, path):
        ck = _load(path)
        W = ck["model_state"]
        self.layers = []
        for l in range(3):
            self.layers.append((W[f"lstm.weight_ih_l{l}"], W[f"lstm.weight_hh_l{l}"],
                                W[f"lstm.bias_ih_l{l}"] + W[f"lstm.bias_hh_l{l}"]))
        self.lin_w, self.lin_b = W["linear.weight"], W["linear.bias"]

    def forward(self, mels):                     # mels: (B, T, 40)
        x = mels.astype(np.float32)
        B, T, _ = x.shape
        for Wih, Whh, b in self.layers:
            H = Whh.shape[1]
            h = np.zeros((B, H), np.float32); c = np.zeros((B, H), np.float32)
            pre = x @ Wih.T + b                  # (B, T, 4H); gate order i, f, g, o (PyTorch)
            out = np.empty((B, T, H), np.float32)
            for t in range(T):
                g = pre[:, t] + h @ Whh.T
                i, f, gg, o = _sig(g[:, :H]), _sig(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]), _sig(g[:, 3 * H:])
                c = f * c + i * gg; h = o * np.tanh(c); out[:, t] = h
            x = out
        e = np.maximum(h @ self.lin_w.T + self.lin_b, 0)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    @staticmethod
    def mel(wav):
        m = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=400, hop_length=160, n_mels=40)
        return m.astype(np.float32).T

    @staticmethod
    def normalize(wav, target=-30.0):
        rms = np.sqrt(np.mean((wav * 32767) ** 2)); d = target - 20 * np.log10(rms / 32767 + 1e-12)
        return wav * 10 ** (d / 20) if d > 0 else wav

    def embed(self, wav, rate=1.3, min_coverage=0.75, return_partials=False):
        wav = self.normalize(wav.astype(np.float32))
        spf = 160; n_frames = int(np.ceil((len(wav) + 1) / spf)); step = int(round((SR / rate) / spf))
        slices = [(i, i + 160) for i in range(0, max(1, n_frames - 160 + step + 1), step)]
        last = slices[-1]; cov = (len(wav) - last[0] * spf) / (160 * spf)
        if cov < min_coverage and len(slices) > 1: slices = slices[:-1]
        need = slices[-1][1] * spf
        if need >= len(wav): wav = np.pad(wav, (0, need - len(wav)))
        M = self.mel(wav)
        mels = np.stack([M[a:b] for a, b in slices])
        P = self.forward(mels)
        e = P.mean(0); e /= np.linalg.norm(e)
        return (e, P, slices) if return_partials else e
