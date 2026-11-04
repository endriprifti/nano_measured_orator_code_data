"""Silero VAD v5 (ONNX model shipped in the PyPI package `silero-vad`) run with onnxruntime, no PyTorch.
Returns per-32-ms-frame speech probabilities for a 16 kHz mono waveform, and speech spans after hysteresis."""
import numpy as np
import onnxruntime as ort

class SileroVAD:
    def __init__(self, path):
        o = ort.SessionOptions(); o.inter_op_num_threads = 1; o.intra_op_num_threads = 1
        self.s = ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])

    def probs(self, wav, sr=16000):
        wav = wav.astype(np.float32); N = 512; C = 64
        state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros((1, C), np.float32); out = []
        for i in range(0, len(wav) - N + 1, N):
            x = np.concatenate([ctx, wav[None, i:i + N]], axis=1)
            p, state = self.s.run(None, {"input": x, "state": state, "sr": np.array(sr, dtype=np.int64)})
            ctx = x[:, -C:]; out.append(float(p[0, 0]))
        return np.array(out), N / sr

    def spans(self, wav, sr=16000, on=0.5, off=0.35, min_speech=0.25, min_sil=0.10):
        p, step = self.probs(wav, sr)
        sp, cur, a = [], False, 0
        for i, v in enumerate(p):
            if not cur and v >= on: cur, a = True, i
            elif cur and v < off: cur = False; sp.append([a * step, i * step])
        if cur: sp.append([a * step, len(p) * step])
        merged = []
        for a, b in sp:
            if merged and a - merged[-1][1] < min_sil: merged[-1][1] = b
            else: merged.append([a, b])
        return [(a, b) for a, b in merged if b - a >= min_speech], p, step
