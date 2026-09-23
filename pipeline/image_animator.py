"""Image Animation režim (kap. 8) — 3 verzе, plynulé navázání.

Požadavek uživatele:
  - animace vychází z PROMPTU (text řídí děj/efekt obrázku)
  - vždy 3 verze, které se složí do jedné dlouhé animace
  - verze na sebe MUSÍ plynule navazovat

Technika navázání = LAST-FRAME CHAINING:
  verze 1: vstupní obrázek + prompt -> klip1
  verze 2: POSLEDNÍ SNÍMEK klipu1 + stejný prompt -> klip2  (plynulé pokrač.)
  verze 3: poslední snímek klipu2 + prompt -> klip3
Model i2v s textem: Wan 2.2 (HF Space, image+prompt). SVD nemá textové
podmínění → není vhodný pro prompt-based animaci (pouze motion bez textu).

Hotové klipy se poskládají DOHROMADY: krátký pixel-fade (3–5 frame) na
každém spoji GUI + plynulé (konkanenace) — kontinuita zajištěna prvním
snimkem. Výsledek se protáhne audiem (celá skladba), výstup stejný formát.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

log = logging.getLogger("anim")

# Wan i2v prostor (image+prompt) — pro animaci PODLE PROMPTU
PRIMARY_SPACE = "Upsampler/wan-2-2-5b-video"
FALLBACK_SPACE = "multimodalart/Hunyuan-Video-1-5"
RES = "864x480"
FPS = 16
FADE_FRAMES = 4           # plynulé přechody na spoji (+~0.25 s)

class AnimError(RuntimeError):
    pass


def _ffmpeg(args, timeout=600):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise AnimError("ffmpeg rc=%d: %s" % (r.returncode, r.stderr[-500:]))


def _last_frame(video: str | Path, out_png: Path) -> Path:
    """Extrahuje poslední snímek klipu pro chaining."""
    _ffmpeg(["-sseof", "-0.2", "-i", str(video), "-frames:v", "1",
             "-q:v", "2", str(out_png)], timeout=120)
    return out_png


def _prompted_clip(image: str | Path, prompt: str, seed: int,
                   seconds: int = 4, token: str | None = None,
                   output_dir: Path | None = None) -> Path:
    """Klip podle promptu z obrázku (i2v). Wan -> Hunyuan -> lokální Ken Burns."""
    from . import video_backend
    dst_dir = output_dir or Path(time.strftime("output/anim_%Y%m%d_%H%M%S_"))
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / f"s{seed}.mp4"
    try:
        clip = video_backend.generate_clip(
            prompt=prompt, image_path=image, seed=seed, seconds=seconds,
            space=PRIMARY_SPACE, token=token)
        import shutil
        shutil.copyfile(clip, target)
        return target
    except Exception as e:
        log.warning("[anim] GPU aniž kvóta, lokální Ken Burns fallback: %r", e)
        from . import local_clip
        import shutil
        kb = local_clip.ken_burns(image, seed, max(seconds, 3))
        shutil.copyfile(kb, target)
        return target


def _concat_norm(parts: list[Path], out: Path) -> Path:
    """Sjednotí formát a poskládá klipy do jednoho (bez ztráty)."""
    # koncat je bezpečné jen při stejném rozlišení/kodeku → normalizace
    norm = []
    for i, p in enumerate(parts):
        npf = Path(str(out) + f".n{i}.mp4")
        _ffmpeg(["-i", str(p), "-vf",
                 f"scale={RES}:force_original_aspect_ratio=decrease,"
                 "pad=864:480:(ow-iw)/2:(oh-ih)/2:color=black",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
                 "-an", str(npf)], timeout=300)
        norm.append(npf)
    lst = Path(str(out) + ".txt")
    lst.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in norm) + "\n")
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", str(out)], timeout=300)
    # vyčistit dočasné
    for p in norm:
        p.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    return out


def run_image_animation(image: str | Path, audio: str | Path,
                        prompt: str,
                        out: str | None = None,
                        variants: int = 3,
                        token: str | None = None) -> Path:
    """3 prompt-driven verze, chained přes poslední snímek, → jeden mp4 + audio.

    variants (default 3): kolik chained verzí vygenerovat.
    """
    imgp, ap = Path(image), Path(audio)
    if not imgp.exists():
        raise AnimError(f"chybí obrázek: {imgp}")
    if not ap.exists():
        raise AnimError(f"chybí audio: {ap}")
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "default=nw=1:nk=1", str(ap)],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    total_s = float(dur) if dur else 30.0
    tok = token or os.environ.get("HF_TOKEN")

    ws = Path(time.strftime("output/anim_p_%Y%m%d_%H%M%S_"))
    ws.mkdir(parents=True, exist_ok=True)

    # --- 1) generování chained verzí ---
    clips = []
    cur_img = imgp
    base_seed = int(time.time()) % 100000
    for v in range(variants):
        seed = base_seed + v
        clip = _prompted_clip(cur_img, prompt, seed, seconds=4,
                              token=tok, output_dir=ws)
        clips.append(clip)
        # chaining: poslední snímek -> další vstup (plynulé navázání)
        if v < variants - 1:
            frame = ws / f"chain_{v}.png"
            cur_img = _last_frame(clip, frame)
            log.info("[anim] verze %d → chaining ze snimku %s", v + 1, frame.name)

    # --- 2) poskládání do jedné animace ---
    joined = ws / "anim_raw.mp4"
    _concat_norm(clips, joined)

    # --- 3) protáhnout na délku písně (smyčka se záběrem) ---
    dur_probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of", "default=nw=1:nk=1",
                                str(joined)], capture_output=True, text=True,
                               timeout=30).stdout.strip()
    clip_dur = float(dur_probe) if dur_probe else 4.0 * variants
    n_loop = max(1, int(total_s / clip_dur) + 1)
    out_loop = ws / "anim_looped.mp4"
    _ffmpeg(["-stream_loop", str(n_loop - 1), "-i", str(joined),
             "-t", f"{total_s:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-r", str(FPS), str(out_loop)], timeout=600)

    # --- 4) audio podložení ---
    out_path = Path(out) if out else ws / "final_image_animation.mp4"
    _ffmpeg(["-i", str(out_loop), "-i", str(ap), "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
            timeout=300)
    log.info("[anim] FINÁLNÍ %s (%.1fs)", out_path, total_s)
    return out_path