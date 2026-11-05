"""Whisper large-v3-turbo (int8 ONNX, sherpa-onnx release 'asr-models'), used to establish the setting of the tapes,
to locate broadcast excerpts in the printed text and to count syllables.
Usage: python asr.py file.wav start end [start end ...]  -> prints one hypothesis per span.

Each segment is followed by PAD seconds of silence before decoding. Without it the decoder drops the last one to two
seconds of speech (found against the author's exact transcripts of two windows of the 1990 and 1991 tapes: the
unpadded decode of a 24.7-s window of FN1990 lost its final 30 syllables). A segment plus padding must fit in one
30-s Whisper window."""
import os, sys, numpy as np, soundfile as sf, sherpa_onnx
M = os.environ.get("NANO_WHISPER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                                                "sherpa-onnx-whisper-turbo")) + "/"
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M + "turbo-encoder.int8.onnx", decoder=M + "turbo-decoder.int8.onnx",
                                                 tokens=M + "turbo-tokens.txt", language="sq", task="transcribe",
                                                 num_threads=2, tail_paddings=800)
PAD = 5.0
MAXSEG = 30.0 - PAD - 0.1


def transcribe(wav, a, b):
    seg = wav[int(a * 16000):int(b * 16000)].astype(np.float32)
    if len(seg) / 16000 > MAXSEG:
        raise ValueError(f"segment of {len(seg) / 16000:.1f} s is too long for one padded Whisper window")
    s = rec.create_stream(); s.accept_waveform(16000, np.concatenate([seg, np.zeros(int(PAD * 16000), np.float32)]))
    rec.decode_stream(s); return s.result.text


if __name__ == "__main__":
    w, sr = sf.read(sys.argv[1]); spans = list(map(float, sys.argv[2:]))
    for a, b in zip(spans[::2], spans[1::2]):
        print(f"[{a:.1f}-{b:.1f}] {transcribe(w, a, b)}", flush=True)
