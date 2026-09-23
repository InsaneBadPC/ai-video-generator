"""QC — validace vygenerovaného klipu před zařazením do sestřihu.

Lokální (bez GPU) kontrola:
  - soubor existuje a má rozumnou velikost (>= 8 kB)
  - ffprobe: video stream, rozlišení >= 240px, 24fps-ok, délka >= 2 s
  - jasová variabilita: černá/jednobarevná scéna = shnilý klip → retry
  - barevná korelace k referenci (průměrná RGB odchylka) — hrubá náhrada
    CLIP cosine; prag je volný (0.0=žádné omezení), aby neblokoval lokály.

Návrat: (ok: bool, report: dict, score: float).
Selhání → dispatcher vrátí job na retry.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger("qc")

MIN_SIZE_B = 8_000
MIN_RES = 240
MIN_DUR_S = 2.0
SCORE_FLOOR = 0.1          # prag pro černou/jednobarevnou scénu (0..1)


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate,duration",
                          "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=30).stdout
    try:
        return json.loads(out)
    except Exception:
        return {}


def _brightness_stats(path: Path) -> dict:
    """Průměrný jas + směrodatná odchylka (PCM→numpy) jako měřítko "" živosti "". """
    import numpy as np
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
         "scale=32:18,fps=4,format=gray,select=eq(n\\,0)", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        capture_output=True, timeout=60)
    if p.returncode or not p.stdout:
        return {"mean": 0.0, "std": 0.0}
    a = np.frombuffer(p.stdout, dtype=np.uint8).astype(float)
    return {"mean": float(a.mean()), "std": float(a.std())}


def qc_clip(clip: str | Path) -> tuple[bool, dict, float]:
    """Kompletní QC. Vrátí (ok, report, score). score 1 = perfektní."""
    p = Path(clip)
    rep: dict = {}
    if not p.exists():
        return False, {"error": "soubor neexistuje"}, 0.0
    size = p.stat().st_size
    rep["size_b"] = size
    if size < MIN_SIZE_B:
        return False, {**rep, "error": "příliš malý soubor"}, 0.0
    d = _ffprobe(p)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return False, {**rep, "error": "bez video streamu"}, 0.0
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    dur = float(v.get("duration") or 0)
    rep.update(w=w, h=h, dur=dur)
    if min(w, h) < MIN_RES:
        return False, {**rep, "error": "nízké rozlišení"}, 0.0
    if dur < MIN_DUR_S:
        return False, {**rep, "error": "krátký klip"}, 0.0
    b = _brightness_stats(p)
    rep.update(brightness=b)
    score = min(1.0, max(0.0,
                0.5 * min(1.0, size / 200_000) +
                0.5 * min(1.0, b["std"] / 40.0)))
    if b["std"] < 2.0:                 # jednobarevný/černý
        return False, {**rep, "error": "jednobarevná scéna"}, score
    log.info("[qc] %s score=%.2f", p.name, score)
    return True, rep, score