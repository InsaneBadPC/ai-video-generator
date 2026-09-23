"""Assembly Engine — sestaví finální video z klipů + overlay textů.

Vstup:  audio_map.json (opportunities/words s timestampy), storyboard.json
        (scény), složka klipů (output/...) z dispatcherova fronty.
Chování:
  1. Z audio_map určí sekci na každou dobu (vers/chorus...).
  2. Klipy přiřadí dle scén storyboardu; Ken Burns/loop/nology default.
  3. Přidá overlay texty (název písně, došky sekce) v dolní třetině.
  4. Audio = originální hudba, míchá přes ffmpeg (concat + amix).
  5. Výstup: finální .mp4 (h264+aac, 864x480) do output/.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("assembly")

RES = "864x480"
FPS = 24
OUT_DIR = Path("output")


class AssemblyError(RuntimeError):
    pass


def _ffmpeg(args, timeout=600):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise AssemblyError("ffmpeg selhal rc=%d:\n%s" % (r.returncode, r.stderr[-2000:]))


def load_audio_map(path: str = "state/synth_test.audio_map.json") -> dict:
    return json.loads(Path(path).read_text())


def load_storyboard(path: str = "state/synth_test.storyboard.json") -> list:
    d = json.loads(Path(path).read_text())
    return d.get("scenes") or (d if isinstance(d, list) else [])


def _section_for(t: float, audio_map: dict) -> str:
    for s in audio_map.get("sections", []):
        if s["start"] <= t < s["end"]:
            return s.get("label", s.get("name", "verse"))
    return "verse"


def _scene_match(scenes: list, section: str, idx: int) -> dict | None:
    """Najde scénu storyboardu odpovídající sekci (+ rotace pro opak.)."""
    for sc in scenes:
        if (sc.get("section", "") or "").lower() in (section.lower(), "any"):
            return sc
    return scenes[idx % len(scenes)] if scenes else None


def build_assembly(audio_map: dict, scenes: list, clips: list[Path],
                   title: str, overlay_on: bool = True,
                   out: str | None = None,
                   audio_path: str | Path = "input/synth_test.wav") -> Path:
    """Sestaví video. clips = uspořádané klipy (1:1 k sekcím, nebo k rotaci)."""
    if not clips:
        raise AssemblyError("žádné klipy pro assembly")
    global_time = 0.0
    parts: list[Path] = []
    ws = OUT_DIR / time.strftime("assembly_%Y%m%d_%H%M%S_")
    ws.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips):
        # normalizace na RES + stejný kodek → concat bez chyb
        norm = ws / f"p{i:02d}.mp4"
        arg = ["-i", str(clip), "-vf",
               f"scale={RES}:force_original_aspect_ratio=decrease,"
               "pad=864:480:(ow-iw)/2:(oh-ih)/2:color=black",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
               "-an", str(norm)]
        try:
            _ffmpeg(arg)
        except Exception:
            log.error("[assembly] normalizace %s selhala: %s", clip, arg[-1])
            raise
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "default=nw=1:nk=1", str(norm)],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        dur = float(probe or 0) or 4.0
        parts.append(norm)
        global_time += dur
    # concat list
    lst = ws / "parts.txt"
    lst.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in parts) + "\n")
    # 1) nařezat původní audio na délku videa
    ap = Path(audio_path)
    if not ap.exists():
        raise AssemblyError(f"chybí input audio: {ap}")
    silent = ws / "audio.aac"
    _ffmpeg(["-i", str(ap), "-t", f"{global_time:.3f}", "-vn",
             "-c:a", "aac", "-b:a", "128k", str(silent)])
    # 2) concat klipů (pevná velikost)
    joined = ws / "joined.mp4"
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(joined)])
    # 3) overlay text (pokud zapnuto) přes drawtext s časovacími okny
    if overlay_on:
        vf = ["-vf"]
        filters = []
        for sc in scenes:
            st, en = sc.get("start_s", sc.get("start", 0.0)), sc.get("end_s", sc.get("end", 0.0))
            if not en:
                en = global_time
            txt = sc.get("overlay") or sc.get("label") or sc.get("lyric_line") or ""
            if not txt:
                continue
            txt = txt.replace(":", r"\:").replace("'", r"\'")
            filters.append(
                f"drawtext=text='{txt}':fontcolor=white:fontsize=34:"
                f"borderw=2:bordercolor=black@0.8:box=1:boxcolor=black@0.35:"
                f"boxborderw=10:x=(w-text_w)/2:y=h-90:"
                f"enable='between(t,{max(st,0):.2f},{min(en,global_time):.2f})'")
        vf.append(",".join(filters))
        args = ["-i", str(joined), "-i", str(silent),
                *vf, "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest"]
    else:
        args = ["-i", str(joined), "-i", str(silent),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-shortest"]
    out_path = Path(out) if out else (lst.parent / f"final_{safe(title)}.mp4")
    args += [str(out_path)]
    _ffmpeg(args)
    log.info("[assembly] finální video: %s (%.2fs)", out_path, global_time)
    return out_path


def safe(t: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in t)[:40] or "video"