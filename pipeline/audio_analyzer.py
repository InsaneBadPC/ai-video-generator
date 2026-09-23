"""Audio Analyzer — z mp3/textu písně vytvoří audio_map.json.

Lehký a odolný: NEPOUŽÍVÁ librosa/numba (nemůžou se nainstalovat na
Termux/aarch64/python3.14). Dekóduje pomocí ffmpeg (mono 22.05 kHz),
BPM spočítá numpy autokorelací, energii po ~1s blocích, slova přes
faster-whisper (CPU, int8).

Výstup:
  {
    "duration_s": float,
    "bpm": float,
    "beats_s": [float, ...],
    "sections": [{ "start_s", "end_s", "label", "mean_energy" }],
    "segments": [{ "start_s", "end_s", "energy" }],   # ~1s grid
    "words": [{ "start_s", "end_s", "word" }],
    "source": {...}
  }
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 22050


def decode_to_float(audio_path: str) -> np.ndarray:
    """ffmpeg -> mono f32 PCM (SR=22050). Bezpečné, bez temp souboru (stdout)."""
    cmd = ["ffmpeg", "-v", "error", "-i", audio_path, "-f", "f32le",
           "-ac", "1", "-ar", str(SR), "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _autocorr_bpm(y: np.ndarray, sr: int = SR) -> float:
    """BPM přes normalizovanou autokorelaci v okně 60–200 BPM."""
    if len(y) < sr * 8:
        return 0.0
    win = y[: sr * 30]                       # max 30s okno
    win = win - win.mean()
    if not np.any(win):
        return 0.0
    win = win / (np.abs(win).max() + 1e-9)
    min_lag = int(sr * 60 / 200)             # 200 BPM → nejkratší
    max_lag = int(sr * 60 / 60)              # 60 BPM → nejdelší
    n = len(win)
    best_lag, best_score = 0, -np.inf
    # subsample: počítáme každý 4. lag kvůli rychlosti (sr=22050 → 220 lags)
    for lag in range(min_lag, min(max_lag, n // 2), 4):
        a = win[: n - lag]
        b = win[lag:]
        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if score > best_score:
            best_score, best_lag = score, lag
    if best_lag <= 0:
        return 0.0
    bpm = 60.0 * sr / best_lag
    # sjednocení do rozsahu 60–200 (oktávové)
    while bpm > 200:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    return float(bpm)


def _rough_second_grid(y: np.ndarray, duration: float) -> tuple[list, float]:
    """Energie na 1s bloky a globální práh."""
    n = int(np.ceil(duration))
    seg, energies = [], []
    frame = SR  # 1s
    for i in range(n):
        chunk = y[i * frame:(i + 1) * frame]
        if chunk.size == 0:
            continue
        e = float((chunk.astype(np.float64) ** 2).mean())
        energies.append(e)
        seg.append({"start_s": i, "end_s": min(i + 1, duration), "energy": round(e, 6)})
    thr = float(np.mean(energies)) if energies else 0.0
    return seg, thr


def _sections(segments: list[dict], thr: float, duration: float) -> list[dict]:
    if not segments:
        return [{"start_s": 0, "end_s": duration, "label": "whole", "mean_energy": 0.0}]
    out, cur = [], None
    for s in segments:
        label = "chorus" if s["energy"] >= thr else "verse"
        if cur and cur["label"] == label:
            cur["end_s"] = s["end_s"]
            cur["n"] += 1
            cur["energy_sum"] += s["energy"]
        else:
            cur = {"start_s": s["start_s"], "end_s": s["end_s"], "label": label, "n": 1, "energy_sum": s["energy"]}
            out.append(cur)
    for c in out:
        c["mean_energy"] = round(c["energy_sum"] / c["n"], 6)
        c.pop("n", None)
        c.pop("energy_sum", None)
    return out


def _transcribe(audio_path: str) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = model.transcribe(audio_path, language="cs", word_timestamps=True)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append({"start_s": round(w.start, 3), "end_s": round(w.end, 3), "word": w.word})
        return words
    except Exception as e:
        sys.stderr.write(f"[whisper] {e}\n")
        return []


def _align_lyrics_proportional(lyrics: str, sections: list[dict], duration: float) -> list[dict]:
    """Fallback bez whisper: rozvrhne slova textu proporcionálně mezi sekce.
    Nahrazuje chybějící word timestamps čistě v Pythonu."""
    tokens = [w for w in lyrics.split() if any(c.isalnum() for c in w)]
    if not tokens or len(tokens) < 4:
        return []
    speech = [s for s in sections if s["mean_energy"] > 0]
    if not speech:
        return []
    span = {"start": speech[0]["start_s"], "end": speech[-1]["end_s"]}
    usable = max(span["end"] - span["start"], 1.0)
    out = []
    for i, tok in enumerate(tokens):
        t = span["start"] + (i + 0.5) * usable / len(tokens)
        out.append({"start_s": round(t, 3), "end_s": round(min(t + 0.4, duration), 3), "word": tok})
    return out


def analyze_audio(audio_path: str, lyrics_path: str | None) -> dict:
    y = decode_to_float(audio_path)
    duration = float(len(y) / SR)

    bpm = round(_autocorr_bpm(y), 2)
    seg, thr = _rough_second_grid(y, duration)
    sections = _sections(seg, thr, duration)
    words = _transcribe(audio_path)

    lyrics = ""
    if lyrics_path and Path(lyrics_path).exists():
        lyrics = Path(lyrics_path).read_text(encoding="utf-8", errors="replace")[:8000]

    words = words or _align_lyrics_proportional(lyrics, sections, duration)

    return {
        "duration_s": round(duration, 3),
        "bpm": bpm,
        "sections": sections,
        "segments": seg,
        "words": words,
        "source": {
            "audio": audio_path,
            "lyrics_path": lyrics_path,
            "lyrics_excerpt": lyrics[:400],
            "counts": {"segments": len(seg), "words": len(words)},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lyrics", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = analyze_audio(a.audio, a.lyrics)
    out = a.out or (Path(a.audio).stem + ".audio_map.json")
    Path(out).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK -> {out}  (dur={d['duration_s']}s bpm={d['bpm']} segments={len(d['segments'])} words={len(d['words'])})")


if __name__ == "__main__":
    main()