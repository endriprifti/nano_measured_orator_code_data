"""Whisper large-v3-turbo (int8 ONNX, sherpa-onnx release 'asr-models') used only to locate broadcast excerpts in
the printed text. Usage: python asr.py file.wav start end [start end ...]  -> prints one hypothesis per span."""
import os, sys, numpy as np, soundfile as sf, sherpa_onnx
M = os.environ.get("NANO_WHISPER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                                                "sherpa-onnx-whisper-turbo")) + "/"
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M + "turbo-encoder.int8.onnx", decoder=M + "turbo-decoder.int8.onnx",
                                                 tokens=M + "turbo-tokens.txt", language="sq", task="transcribe",
                                                 num_threads=2, tail_paddings=800)
def transcribe(wav, a, b):
    s = rec.create_stream(); s.accept_waveform(16000, wav[int(a * 16000):int(b * 16000)].astype(np.float32))
    rec.decode_stream(s); return s.result.text
if __name__ == "__main__":
    w, sr = sf.read(sys.argv[1]); spans = list(map(float, sys.argv[2:]))
    for a, b in zip(spans[::2], spans[1::2]):
        print(f"[{a:.1f}-{b:.1f}] {transcribe(w, a, b)}", flush=True)
