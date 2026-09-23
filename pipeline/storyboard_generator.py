"""Storyboard generátor → storyboard.json.

Vstup: audio_map.json (krok 2), text písně, charakter Temney, parametry.
Výstup: seznam scén, každá:
  {
    "scene": 0,
    "start_s": .., "end_s": .., "label": "verse|chorus",
    "lyric_lines": [..],
    "image_prompt": "..",         # pro FLUX character
    "video_prompt": "..",         # pro Wan/LTX
    "overlay_text": "..|null",    # dle config.cc_tier.text_overlay
  }
Scény = spojené sousední sekce ze sekcí audio_map, oříznuté na max délku.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .llm_client import chat_json

log = logging.getLogger("storyboard")

CHARACTER_BIBLE = """Temney — tajemná postava z ghetta zničená bolestí a osamělostí.
Vzhled: mladý muž ~20 let, tmavé krátké vlasy, vyhublý, propadlé tváře, unavené pronikavé
oči, hadry/staré obnošené černé oblečení, stopy špíny, drobné jizvy, přívěšek na krku.
Aura: melancholie, zlom, naděje uprostřed trosek. Město vždy noční, neon, beton, patina.
Pro umělce: VŽDY je Temney ústřední postavou, nikdy ho nenahrazuj jinou tváří."""

_SYSTEM = (
    "Jsi režisér hudebních videí pro charakterní postavu TEMNEY (tajemný mladík z ghetta "
    "zničený bolestí). Generuješ storyboard JSON pro noční městské sny: beton, neon, déšť, "
    "patina. Jazyk promptů: anglicky. Striktní formát níže, žádný komentář mimo JSON."
)


def _build_prompt(audio_map: dict, lyrics: str, tempo: str | None) -> str:
    sections = audio_map["sections"]
    bpm = audio_map.get("bpm", 0)
    dur = audio_map.get("duration_s", 0)
    scenes_meta = ", ".join(f'{{"start_s":{s["start_s"]},"end_s":{s["end_s"]},"label":"{s["label"]}"}}'
                            for s in sections[:8])
    return f"""{_SYSTEM}

TEXT PÍSNĚ:
{lyrics[:4000]}

HUDBA: {bpm:.0f} BPM, {dur:.0f}s, struktura ČAS PODLE ENERGIÍ (label verse=klidná, chorus=energická):
[{scenes_meta}]

CHARAKTER: {CHARACTER_BIBLE}

POZADÍ: Temney vystupuje každou scénu v nočním městě. Konzistentní styl celého videa.
MAX: {len(sections[:8])} scén, každá scéna délka odpovídá jejímu časovému oknu.

Odpověz JEN JSON pole scén. Pro každou scénu:
  {{
    "start_s": float, "end_s": float,
    "label": "verse"|"chorus",
    "lyric_line": "přesný řádek textu, který ve scéně zazní (nebo \"\")",
    "image_prompt": "detailní prompt na obrázek scény — Temney v nočním městě, styl, světlo, úhel, barvy",
    "video_prompt": "krátký pohybový prompt — kamera, pohyb Temney/scény (např. slow dolly in)",
    "overlay_text": "1–3 slova nálady scény pro titulkek (anglicky, styl neon; \"\" suppress)"
  }}
Nezahrnuj scény kratší než 1.5s. Drž {dur:.0f} vteřin součtem start/end v pořadí.""" 


def generate_storyboard(audio_map_path: str, lyrics: str | None = None,
                        out_path: str | None = None) -> dict:
    amap = json.loads(Path(audio_map_path).read_text(encoding="utf-8"))
    text = lyrics or amap.get("source", {}).get("lyrics_excerpt") or ""
    tempo = f"{amap.get('bpm', 0):.0f} BPM" if amap.get("bpm") else None

    prompt = _build_prompt(amap, text, tempo)
    try:
        scenes = chat_json(prompt, max_tokens=4096)
    except Exception as exc:
        log.warning("[storyboard] LLM nedostupné, používám deterministický fallback: %r", exc)
        scenes = _fallback_scenes(amap, text)
    # normalizace: výsledek může být list nebo {"scenes": [...]}
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes") or scenes.get("storyboard") or []
    for i, sc in enumerate(scenes):
        sc.setdefault("scene", i)
    out = {"character": "Temney", "bpm": amap.get("bpm", 0),
           "duration_s": amap.get("duration_s", 0), "scenes": scenes,
           "source_audio_map": audio_map_path}
    if out_path:
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Uložen storyboard -> %s (%d scén)", out_path, len(scenes))
    return out


def _fallback_scenes(audio_map: dict, lyrics: str) -> list[dict]:
    """Vytvoří validní storyboard i bez externího LLM/API."""
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    result = []
    for i, section in enumerate(audio_map.get("sections") or []):
        start, end = float(section.get("start_s", 0)), float(section.get("end_s", 0))
        if end - start < 0.1:
            continue
        label = section.get("label", "verse")
        lyric = lines[i % len(lines)] if lines else ""
        motion = "slow dolly in" if label == "verse" else "handheld neon pulse"
        result.append({
            "scene": i, "start_s": start, "end_s": end, "label": label,
            "lyric_line": lyric, "image_prompt":
            "Temney in a rain-soaked neon city at night, cinematic gritty portrait, "
            "consistent character, moody blue and magenta lighting",
            "video_prompt": f"{motion}, drifting rain, subtle natural movement",
            "overlay_text": label.title(),
        })
    return result or [{
        "scene": 0, "start_s": 0, "end_s": float(audio_map.get("duration_s", 1)),
        "label": "whole", "lyric_line": "", "image_prompt": CHARACTER_BIBLE,
        "video_prompt": "slow cinematic camera movement", "overlay_text": "",
    }]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_storyboard(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                        sys.argv[3] if len(sys.argv) > 3 else None)
