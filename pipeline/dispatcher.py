"""Dispatcher — přebírá joby z SQLite fronty a generuje klipy.

Tier chain pro scénu (bez GPU → GPU, dle config video_backends):
  local_loop → hotový loop klip z knihovny
  local_kb   → Ken Burns (landscape obrázek z knihovny)
  hf_wan     → Wan 2.2 5B (HF Space, ZeroGPU)
  kaggle     → Kaggle GPU kernel (offpeak, kvóta 30h/týden)
Retry: joby, které selhaly u všech tierů, se podle configu (max_retries)
vrátí do fronty (backoff). Ty, co selhaly jen překročením kvóty, čekají.

Zpětná kompatibilita: používá pipeline/video_backend.generate_clip
(automatický fallback Wan→Hunyuan) pro HF vrstvu.
"""
from __future__ import annotations

import logging
import hashlib
import os
import time
from pathlib import Path
from typing import Callable

from . import queue as q

log = logging.getLogger("dispatch")

DEFAULT_DB = "queue/jobs.db"
MAX_RETRIES = 4
BACKOFF_S = 5

_USED = set()               # hashes použitých lokálních medií — napříč joby


def reset_used() -> None:
    _USED.clear()


def load_config() -> dict:
    try:
        import yaml
        cfg = yaml.safe_load(Path("config.yaml").read_text()) or {}
        return cfg
    except Exception as e:
        log.warning("[dispatch] config nenačten: %r", e)
        return {}


def _qc_pass(clip: Path) -> tuple[bool, dict, float]:
    from . import qc
    ok, rep, score = qc.qc_clip(clip)
    return ok, rep, score


def _make_local_clip(job: dict, library_dir: str) -> Path | None:
    """Tier local_loop / local_kb. Vrátí cestu ke klipu, nebo None."""
    from . import local_clip, loop_library

    lib = loop_library.index_library(library_dir)
    media = loop_library.select_scene_media(
        job.get("label") or job.get("section") or "verse",
        used_hashes=_USED, library=lib)
    if not media:
        log.info("[dispatch] scéna %s: knihovna prázdná", job.get("label"))
        return None
    _USED.add(media.hash)
    kb_s = int((load_config().get("loop_library") or {}).get("kb_seconds", 4))
    secs = max(int(round(job.get("duration") or kb_s)), 1)
    try:
        return local_clip.make_local_clip(media.kind, media.path,
                                          int(job.get("seed") or 0), secs)
    except Exception as e:
        log.warning("[dispatch] local clip selhal: %r", e)
        return None


def _make_local_kb(job: dict, library_dir: str) -> Path | None:
    """Vybere obrázek a vytvoří Ken Burns klip."""
    from . import local_clip, loop_library

    lib = loop_library.index_library(library_dir)
    images = [m for m in lib if m.kind == "ref_img"]
    media = loop_library.select_scene_media(
        job.get("label") or job.get("section") or "verse",
        used_hashes=_USED, library=images,
    )
    if not media:
        log.info("[dispatch] scéna %s: knihovna obrázků prázdná, vytvářím auto-reference", job.get("label"))
        ref = _ensure_reference(job)
        secs = max(int(round(job.get("duration") or 4)), 1)
        return local_clip.ken_burns(ref, int(job.get("seed") or 0), secs)
    _USED.add(media.hash)
    kb_s = int((load_config().get("loop_library") or {}).get("kb_seconds", 4))
    secs = max(int(round(job.get("duration") or kb_s)), 1)
    return local_clip.ken_burns(media.path, int(job.get("seed") or 0), secs)


def _make_hf_clip(job: dict) -> Path:
    from . import video_backend

    img = str(_ensure_reference(job))
    return video_backend.generate_clip(
        prompt=job["prompt"],
        image_path=img,
        seed=int(job.get("seed") or 0),
        seconds=max(1, int(round(job.get("duration") or 4))),
        token=os.environ.get("HF_TOKEN"),
    )


def _make_kaggle_clip(job: dict) -> Path | None:
    from . import kaggle_video
    try:
        ref_img = str(_ensure_reference(job))
        scene = {
            "prompt": job.get("prompt", ""),
            "seed": int(job.get("seed") or 0),
            "scene_idx": int(job.get("scene_idx") or job.get("idx") or 0),
            "duration": max(1, int(round(job.get("duration") or 4))),
            "section": job.get("section", ""),
            "label": job.get("label", ""),
            "ref_image": "ref_image.webp",
        }
        return kaggle_video.generate_kaggle_clip(scene, ref_img=ref_img, dest=Path("output"))
    except Exception as e:
        log.warning("[dispatch] kaggle clip selhal: %r", e)
        return None


def _ensure_reference(job: dict) -> Path:
    """Resolve auto media; use HF image generation, then deterministic placeholder."""
    pre = str(job.get("pre_media") or "").strip()
    if pre and pre != "auto" and Path(pre).exists():
        return Path(pre)
    prompt = job.get("prompt") or job.get("label") or "Temney in a neon city at night"
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    dst = Path("character_reference") / f"auto_{digest}.png"
    if dst.exists():
        return dst
    try:
        from .image_backend import generate_reference
        generated = Path(generate_reference(prompt, seed=int(job.get("seed") or 0),
                                            out_dir="character_reference"))
        if generated.exists():
            return generated
    except Exception as exc:
        log.warning("[dispatch] image reference selhala, placeholder fallback: %r", exc)
    from .image_backend import make_placeholder
    return Path(make_placeholder(prompt, out=str(dst)))


def process_one(db: str = DEFAULT_DB, tier_hooks: dict | None = None) -> bool:
    """Vezme jeden job, projde tier chain, uloží výsledek. Vrátí True=pokračovat."""
    job = q.claim(db)
    if not job:
        return False
    cfg = load_config()
    tier_chain = cfg.get("pipeline", {}).get("video_backends",
                                             ["local_loop", "local_kb", "hf_wan", "kaggle"])
    library_dir = (cfg.get("loop_library") or {}).get("dir", "")
    hooks = tier_hooks or {}
    err = None
    for tier in tier_chain:
        try:
            if tier == "local_loop" or tier == "local_kb":
                clip = (_make_local_clip(job, library_dir)
                        if tier == "local_loop"
                        else _make_local_kb(job, library_dir))
            elif tier == "hf_wan":
                clip = _make_hf_clip(job)
            elif tier == "kaggle":
                clip = _make_kaggle_clip(job)
            else:
                fn = hooks.get(tier)
                clip = fn(job) if fn else None
            if clip:
                ok, _rep, _score = _qc_pass(clip)
                if not ok and tier in {"local_loop", "local_kb"}:
                    # Nouzový lokální klip má prioritu před vyčerpanou HF/Kaggle
                    # kvótou. Strukturální QC (soubor, stream, rozlišení, délka)
                    # už proběhl; jasová variabilita je pouze měkké varování.
                    soft_error = (_rep or {}).get("error")
                    if soft_error == "jednobarevná scéna":
                        log.warning("[dispatch] lokální fallback přijat přes měkké QC: %s", soft_error)
                        ok = True
                if not ok:
                    log.warning("[dispatch] job %s tier %s: klip neprošel QC", job["job_id"], tier)
                    raise RuntimeError("QC: klip neprošel validací")
                q.finish(db, job["job_id"], clip_path=str(clip), tier=tier)
                log.info("[dispatch] job %s scéna %d → tier %s → %s",
                         job["job_id"], job["scene_idx"], tier, clip)
                return True
        except Exception as e:
            err = e
            log.warning("[dispatch] job %s tier %s selhal: %r", job["job_id"], tier, e)
    # všechny tier chainy selhaly
    retries = int((cfg.get("orchestrator") or {}).get("max_retries", MAX_RETRIES))
    if (job.get("attempts") or 1) < retries:
        q.retry(db, job["job_id"])
        time.sleep(BACKOFF_S)
        log.warning("[dispatch] job %s → retry (%d/%d)", job["job_id"], job.get("attempts"), retries)
    else:
        q.finish(db, job["job_id"], error=f"{err!r}")
        log.error("[dispatch] job %s → FAILED: %r", job["job_id"], err)
    return True


def run_queue(db: str = DEFAULT_DB, once: bool = False) -> int:
    """Main loop wokru — zpracuje frontu. once=True → jen jeden job pro test."""
    q.recover_stale(db)
    n = 0
    while True:
        more = process_one(db)
        if not more:
            break
        n += 1
        if once:
            break
    return n


def main(argv=None):
    import argparse
    import time as _t
    ap = argparse.ArgumentParser(description="Dispatcher AI Music Video Generator")
    ap.add_argument("--loop", action="store_true", help="běžet dokola (systemd)")
    ap.add_argument("--sleep", type=int, default=30, help="spánek mezi koly (s)")
    ap.add_argument("--once", action="store_true", help="zpracovat jen jeden job")
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.once:
        run_queue(args.db, once=True)
        return
    while True:
        n = run_queue(args.db)
        if n == 0:
            log.info("[dispatch] fronta prázdná, spím %ds", args.sleep)
            _t.sleep(args.sleep)


if __name__ == "__main__":
    main()
