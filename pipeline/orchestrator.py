"""Orchestrator — spustí celý pipeline E2E: audio → storyboard → klipy → video.

Použití (E2E test):
  python3 -m pipeline.orchestrator --audio input/synth_test.wav --title "Synth Test"

Kroky:
  1. audio_analyzer → state/{name}.audio_map.json
  2. storyboard_generator (LLM) → state/{name}.storyboard.json
  3. queue: pro každou scénu rozřezat do chunku po CHUNK_S (lokální klipy),
     přidat job (label = "section+idx" pro rotaci medií)
  4. dispatcher.run_queue → klipy (lokální loop/Ken Burns, fallback GPU)
  5. assembly.build_assembly → output/final_{title}.mp4
  6. notify (Telegram, pokud enabled)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from pathlib import Path

from . import assembly
from . import dispatcher
from . import queue as q
from . import storyboard_generator
from .audio_analyzer import analyze_audio

log = logging.getLogger("orchestrator")

CHUNK_S = 4
CHUNK_GRACE = 0.5          # tolerance zkrácení poslední scény
DEFAULT_QUEUE_DB = "queue/e2e.db"


def _storyboard_for(name: str, force: bool = False) -> dict:
    out = Path(f"state/{name}.storyboard.json")
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    amap = Path(f"state/{name}.audio_map.json")
    sb = storyboard_generator.generate_storyboard(str(amap),
                                                  str(Path(f"input/{name}.txt")))
    out.write_text(json.dumps(sb, ensure_ascii=False, indent=2), encoding="utf-8")
    return sb


def plan_jobs(sb: dict, name: str, db: str, chunk_s: int = CHUNK_S) -> int:
    """Rozřeže scény na chunky (lokální klip délky) a naplní frontu."""
    scenes = sb["scenes"]
    seed = int(time.time()) % 10000
    added = 0
    for sc in scenes:
        st, en = sc["start_s"], sc["end_s"]
        label = sc.get("label") or sc.get("section") or "verse"
        n_chunks = max(1, math.ceil((en - st) / chunk_s))
        for k in range(n_chunks):
            cst = st + k * chunk_s
            cend = min(cst + chunk_s, en)
            # prompt pro GPU fallback: video_prompt scény (+ časové okno)
            prompt = sc.get("video_prompt") or sc.get("image_prompt") or sc.get("lyric_line") or label
            q.add_scene_job(db, name, int(sc.get("scene") or added), label,
                            f"{label}@{int(cst)}-{int(cend)}",
                            prompt, seed=seed + added,
                            pre_media="auto", duration=max(0.1, cend - cst))
            added += 1
    return added


def run_e2e(audio: str, title: str, force_sb: bool = False,
            db: str = DEFAULT_QUEUE_DB, chunk_s: int = CHUNK_S) -> Path:
    ap = Path(audio)
    name = ap.stem
    # 1) audio analýza
    amap = analyze_audio(str(ap), str(Path(f"input/{name}.txt")) if Path(f"input/{name}.txt").exists() else None)
    Path(f"state/{name}.audio_map.json").write_text(
        json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("audio_map -> state/%s.audio_map.json", name)
    # 2) storyboard
    sb = _storyboard_for(name, force=force_sb)
    # 3) fronta (čistý start pro E2E)
    Path(db).unlink(missing_ok=True)
    added = plan_jobs(sb, name, db, chunk_s)
    log.info("do fronty přidáno %d chunků", added)
    # 4) dispatcher
    n = dispatcher.run_queue(db)
    log.info("dispatcher zpracoval %d jobů", n)
    stats = q.stats(db)
    if stats.get("done", 0) < added:
        log.warning("nedokončeno: %s (z %d)", stats, added)
    # 5) assembly
    rows = []
    with _conn(db) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT scene_idx, clip_path, tier FROM jobs WHERE status='done' "
            "ORDER BY scene_idx, created").fetchall()
    clips = []
    seen = set()
    for r in rows:
        p = Path(r["clip_path"]).resolve()
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            clips.append(p)
    if not clips:
        raise RuntimeError("nebyl vygenerován žádný klip pro assembly")
    log.info("assembly z %d klipů", len(clips))
    final = assembly.build_assembly(amap, sb["scenes"], clips, title=title,
                                    overlay_on=True, audio_path=str(ap))
    return final


def _conn(db: str):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="E2E AI Music Video Generator")
    ap.add_argument("--audio", required=True, help="cesta k mp3/wav")
    ap.add_argument("--title", default="Temney")
    ap.add_argument("--force-storyboard", action="store_true")
    ap.add_argument("--db", default=DEFAULT_QUEUE_DB)
    ap.add_argument("--chunk-s", type=int, default=CHUNK_S)
    ap.add_argument("--mode", choices=["full_scenes", "image_animation"],
                    default=None, help="přepíše mode z config.yaml")
    ap.add_argument("--image", default=None,
                    help="vstupní obrázek pro image_animation (cover)")
    ap.add_argument("--prompt", default=None,
                    help="prompt pro animaci obrazu (image_animation)")
    a = ap.parse_args(argv)
    cfg = _load_cfg()
    mode = a.mode or (cfg.get("pipeline") or {}).get("mode", "full_scenes")
    if mode == "image_animation":
        image = a.image or "character_reference/temney_flux_ref.webp"
        prompt = a.prompt or _default_anim_prompt(a.title, cfg)
        final = _run_anim_e2e(a.audio, image, prompt, a.title)
    else:
        final = run_e2e(a.audio, a.title, a.force_storyboard, a.db, a.chunk_s)
    print(f"\nFINAL VYSTUP: {final} ({final.stat().st_size} B)")
    try:
        from . import storage_r2
        storage_cfg = cfg.get("storage") or {}
        if (storage_cfg.get("r2") or storage_r2.enabled()) and storage_r2.enabled():
            key = storage_r2.upload_file(final, f"videos/{final.name}")
            log.info("R2 upload dokončen: %s", key)
    except Exception as exc:
        log.warning("R2 upload přeskočen, lokální výstup zůstává: %r", exc)
    try:
        from . import notify
        if notify.enabled():
            notify.notify(f"✅ Video hotové: {a.title}\n{final}")
    except Exception:
        pass
    return final


def _load_cfg() -> dict:
    try:
        import yaml
        return yaml.safe_load(Path("config.yaml").read_text()) or {}
    except Exception:
        return {}


def _default_anim_prompt(title: str, cfg: dict) -> str:
    """Výchozí living-image prompt, pokud uživatel nezadá vlastní."""
    return (f"{title} — cover art slowly comes alive: gentle neon light "
            "flicker, drifting fog, subtle camera drift, cinematic glow, "
            "no text, no watermark")


def _run_anim_e2e(audio: str, image: str, prompt: str, title: str) -> Path:
    """Image Animation režim (kap. 8): 1 obrázek + prompt → 3 klipy → 1 video."""
    from . import image_animator
    from .queue import add_scene_job, init
    db = DEFAULT_QUEUE_DB
    Path(db).unlink(missing_ok=True)
    init(db)
    job_id = add_scene_job(db, Path(audio).stem, 0, "image_animation",
                           "cover", prompt, seed=1, job_type="image_animation")
    try:
        final = image_animator.run_image_animation(image, audio, prompt)
    except Exception as exc:
        q.finish(db, job_id, error=repr(exc))
        raise
    q.finish(db, job_id, clip_path=str(final), tier="image_animation")
    return final


if __name__ == "__main__":
    main()
