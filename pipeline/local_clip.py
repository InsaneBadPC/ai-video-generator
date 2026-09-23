"""Lokální klipy bez GPU — tier 0/1 video backendu.

Tier 0: loop video z knihovny → oříznout na požadovanou délku (h264+aac).
Tier 1: landscape obrázek z knihovny → Ken Burns (zoompan) klip přes ffmpeg.
Výstup: standardní mp4 (h264), fps 24, rozlišen dle --res.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("localclip")

FPS = 24
RES = "864x480"
OUT_ROOT = Path("output")


def _run_ffmpeg(args, timeout=180) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        log.warning("[localclip] ffmpeg rc=%d: %s", r.returncode, r.stderr[-600:])
        raise RuntimeError("ffmpeg selhal: " + r.stderr[-300:])


def _out(seed: int, kind: str) -> Path:
    d = Path(time.strftime("output/clip_%Y%m%d_%H%M%S_"))
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"s{seed}_{kind}.mp4")


def loop_clip(video: str | Path, seed: int, seconds: int) -> Path:
    """Tier 0: použije hotový loop klip, natáhne/zkrátí na délku."""
    src = Path(video)
    dst = _out(seed, "loop")
    args = ["-stream_loop", "-1", "-i", str(src), "-t", str(seconds)]
    if src.suffix.lower() == ".webm":
        args += ["-c:v", "libvpx-vp9", "-c:a", "libopus"]
    else:
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 "-vf", (f"scale={RES}:force_original_aspect_ratio=decrease,"
                         "pad=864:480:(ow-iw)/2:(oh-ih)/2:color=black"),
                 "-r", str(FPS)]
    args += [str(dst)]
    _run_ffmpeg(args)
    log.info("[localclip] loop %s -> %s (%ds)", src.name, dst, seconds)
    return dst


def ken_burns(image: str | Path, seed: int, seconds: int,
              res: str = RES, reverse: bool = False) -> Path:
    """Tier 1: obrázek → pomalý zoom (Ken Burns) klip. Zoom dovnitř/vně."""
    src = Path(image)
    dst = _out(seed, "kb")
    total = int(FPS * seconds)
    z = ("min(1+0.001*(on),1.15)" if reverse
         else "min(1.15-0.001*(on),1.15)")
    vf = (f"scale=1920:1080:force_original_aspect_ratio=increase,"
          f"crop=1920:1080,"
          f"zoompan=z='{z}':d={total}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
          f"s={res}:fps={FPS}")
    args = ["-loop", "1", "-i", str(src), "-t", str(seconds),
            "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(FPS), str(dst)]
    _run_ffmpeg(args)
    log.info("[localclip] ken burns %s -> %s (%ds)", src.name, dst, seconds)
    return dst


def needs_interpolate_first(src: Path) -> bool:
    """Pro obrázky s EXIF/zvláštním rozlišením se zoompan zpomalí; helper."""
    return False


def make_local_clip(media_type: str, media_path: str | Path, seed: int,
                    seconds: int) -> Path:
    """Rychlý dispatcher: loop → dstřih, jinak Ken Burns."""
    if media_type == "loop":
        return loop_clip(media_path, seed, seconds)
    return ken_burns(media_path, seed, seconds)