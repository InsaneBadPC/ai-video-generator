"""Loop Library — lokální (bez GPU) zdroj klipů a stylových referencí.

Knihovna: /storage/emulated/0/Pictures/Track Images (32 obrázků + 1 loop video).
Role v pipeline:
  tier 0  → Loop klipy (mp4) použít přímo jako video scény.
  tier 1  → Obrázky 16:9/landscape → Ken Burns klip (zoompan v ffmpeg).
  tier 2  → Obrázky jako stylová reference → přidat do promptu (FLUX/Wan).
Žádná GPU spotřeba → ušetří ZeroGPU kvótu pro zbytek.

Indexace je deterministická: pořadí výběru = stabilní hashed podle nálady
scény, ne opakovat stejný soubor ve videu.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("looplib")

VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v", ".avi"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_DIR = "/storage/emulated/0/Pictures/Track Images"
INDEX_FILE = "state/loop_library.json"


@dataclass
class Media:
    path: Path
    kind: str               # "loop" | "ref_img"
    width: int
    height: int
    ratio: float            # w/h
    duration_s: float = 0.0
    mood: str = "neutral"   # dark | bright | warm | cold | neutral
    hash: str = ""

    @property
    def landscape(self) -> bool:
        return self.ratio >= 1.2

    @property
    def portrait(self) -> bool:
        return self.ratio <= 0.85


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=width,height,duration", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30).stdout
    try:
        d = json.loads(out)
        st = next((s for s in d.get("streams", []) if s.get("width")), d.get("streams", [{}])[0])
        dur = float(d.get("format", {}).get("duration") or st.get("duration") or 0)
        return {"w": int(st.get("width") or 0), "h": int(st.get("height") or 0), "dur": dur}
    except Exception:
        return {"w": 0, "h": 0, "dur": 0}


def _brightness_heuristic(path: Path) -> str:
    """Odhad nálady obrázku z průměrného jasu a dominující barvy."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((48, 48))
        px = list(im.getdata())
        n = len(px) or 1
        avg = tuple(sum(c[i] for c in px) / n for i in range(3))
        lum = 0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2]
        r, g, b = avg
        if lum < 70:
            return "dark"
        if r > 1.2 * b and r > g:
            return "warm"
        if b > 1.2 * r and b > g:
            return "cold"
        return "bright"
    except Exception:
        return "neutral"


def index_library(dir_path: str = None, force: bool = False) -> list[Media]:
    """Naskenuje adresář a nacache index do state/loop_library.json."""
    Path("state").mkdir(parents=True, exist_ok=True)
    cache = Path(INDEX_FILE)
    if cache.exists() and not force:
        try:
            d = json.loads(cache.read_text())
            if d.get("dir") == (dir_path or DEFAULT_DIR) and time.time() - d.get("ts", 0) < 7200:
                return [Media(Path(m["path"]), m["kind"], m["width"], m["height"],
                              m["ratio"], m.get("duration_s", 0), m.get("mood", "neutral"),
                              m.get("hash", "")) for m in d["media"]]
        except Exception:
            pass
    base = Path(dir_path or DEFAULT_DIR)
    if not base.exists():
        log.warning("[looplib] knihovna %s neexistuje", base)
        return []
    media = []
    for f in sorted(base.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in VIDEO_EXT:
            p = _probe(f)
            media.append(Media(f, "loop", p["w"], p["h"],
                               p["w"] / p["h"] if p["h"] else 1.0, p["dur"],
                               "neutral",
                               hashlib.md5(f.name.encode()).hexdigest()[:10]))
        elif ext in IMG_EXT:
            p = _probe(f)
            media.append(Media(f, "ref_img", p["w"], p["h"],
                               p["w"] / p["h"] if p["h"] else 1.0, 0.0,
                               _brightness_heuristic(f),
                               hashlib.md5(f.name.encode()).hexdigest()[:10]))
    # re-probe druhý průchod: mood jen pro obrázky (u videí necháme neutral)
    for m in media:
        if m.kind == "ref_img":
            m.mood = _brightness_heuristic(m.path)
    cache.write_text(json.dumps({
        "dir": str(base), "ts": time.time(), "media": [{
            "path": str(m.path), "kind": m.kind, "width": m.width, "height": m.height,
            "ratio": m.ratio, "duration_s": m.duration_s, "mood": m.mood, "hash": m.hash}
            for m in media]}, ensure_ascii=False))
    log.info("[looplib] index %d medií (%d loopů, %d obrázků)", len(media),
             sum(1 for m in media if m.kind == "loop"),
             sum(1 for m in media if m.kind == "ref_img"))
    return media


def select_scene_media(scene_label: str = "verse", mood_hint: str = "neutral",
                       used_hashes: set = None, library: list[Media] = None,
                       prefer_loops: bool = True) -> Media | None:
    """Vybere medií pro scénu: 1) loop (libovolná orientace), 2) landscape obrázek.
    Deterministic via hash(scene_label) seed na uspořádání."""
    library = library or index_library()
    used = used_hashes or set()
    seed = hashlib.sha1(scene_label.encode()).hexdigest()
    seed_i = int(seed[:8], 16)

    def key(m):
        # priorita: loop > landscape obrázek > cokoliv; nepoužité > použité; stable hash
        prio = 0
        if m.kind == "loop":
            prio = 0
        elif m.landscape:
            prio = 1
        elif not m.portrait:
            prio = 2
        else:
            prio = 3
        if m.hash in used:
            prio += 10
        return (prio, (seed_i - int(m.hash, 16)) % 10000, m.path.name)

    cands = sorted(library, key=key)
    if not cands:
        return None
    # najdi první nepoužitý
    for m in cands:
        if m.hash not in used:
            return m
    return cands[0]


def style_reference_for(scene_label: str, library: list[Media] = None) -> str | None:
    """Cesta k obrázku pro stylovou referenci (do promptu engine→IMG gen)."""
    lib = library or index_library()
    imgs = [m for m in lib if m.kind == "ref_img" and m.landscape]
    if not imgs:
        return None
    m = select_scene_media(scene_label, library=lib, prefer_loops=False)
    return str(m.path) if m and m.kind == "ref_img" else str(imgs[0].path)