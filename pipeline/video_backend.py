"""Video backend — adapter na HF Space (Wan 2.2 / LTX image→video).

Primární: Upsampler/wan-2-2-5b-video (Wan 2.2 5B, ZeroGPU, free, bez gatingu).
Fallback: multimodalart/Hunyuan-Video-1-5 (i2v).
Rozhraní vyměnitelné — endpoint se zjistí přes view_api + mapping dle typu
parametrů (Image → referenční obrázek, str → prompt, numerické → sane defaults).
ZeroGPU kvóta: anon ~3 min GPU/den; s HF_TOKEN limit volnější.
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("video")

PRIMARY = "Upsampler/wan-2-2-5b-video"
FALLBACK = "multimodalart/Hunyuan-Video-1-5"
DEFAULT_SEC = 4
DEFAULT_RES = (512, 512)
FALLBACK_SEC = 2          # Hunyuan kratší klipy (kvóta)


class VideoError(RuntimeError):
    pass


def _num_default(pname: str) -> int:
    p = pname.lower()
    if "duration" in p or "seconds" in p or "length" in p or "time" in p:
        return DEFAULT_SEC
    if "width" in p:
        return DEFAULT_RES[0]
    if "height" in p:
        return DEFAULT_RES[1]
    if "step" in p or "inference" in p:
        return 4
    if "guidance" in p:
        return 5
    return 0


def _call_gradio(client, image_path: Path, prompt: str, seed: int,
                 seconds: int | None = None) -> Path:
    """Zjistí endpoint, naplní parametry dle komponent a spustí predict."""
    from gradio_client import handle_file
    try:
        api = client.view_api(return_format="dict")
        eps = api.get("named_endpoints") or {}
    except Exception as e:
        raise VideoError(f"view_api selhal: {e!r}") from e
    if not eps:
        raise VideoError("Space nemá pojmenované endpointy")

    def is_image(p):
        c = str(p.get("component") or "").lower()
        t = str((p.get("type") or {}).get("title") or p.get("type") or "").lower()
        return "image" in c or "imagedata" in t

    def is_str(p):
        t = p.get("type")
        if isinstance(t, dict):
            t = t.get("type")  # 'string' | 'number' ...
        return "string" in str(t).lower()

    def is_num(p):
        t = p.get("type")
        if isinstance(t, dict):
            t = t.get("type")
        return "number" in str(t).lower() or "integer" in str(t).lower()

    def comp_has_image(p):
        return is_image(p)

    # vyber endpoint s image+prompt
    chosen = None
    for n, ep in eps.items():
        ps = ep.get("parameters", [])
        if any(is_image(x) for x in ps) and any(is_str(x) for x in ps):
            chosen = (n, ep)
            break
    if not chosen:
        chosen = (next(iter(eps)), eps[next(iter(eps))])
    name, ep = chosen
    params = ep.get("parameters", [])

    kwargs = {}
    seen_image, seen_prompt = False, False
    for p in params:
        pn = p.get("parameter_name") or p.get("label")
        if not pn:
            continue
        plow = str(pn).lower()
        if is_image(p):
            kwargs[pn] = handle_file(str(image_path))
            seen_image = True
        elif is_str(p):
            if "negative" in plow or "neg_" in plow:
                kwargs[pn] = ""
            elif not seen_prompt:
                kwargs[pn] = prompt
                seen_prompt = True
        elif is_num(p):
            if "seed" in plow:
                kwargs[pn] = int(seed)
            elif any(k in plow for k in ("duration", "second", "length", "frames")):
                kwargs[pn] = seconds or DEFAULT_SEC
            elif "step" in plow:
                kwargs[pn] = 4
    # randomize_seed je bool — vypnout, ať seed platí
    for p in params:
        pn = p.get("parameter_name") or p.get("label")
        if pn and "random" in str(pn).lower() and "seed" in str(pn).lower():
            kwargs[pn] = False
            break
    if not (seen_image and seen_prompt):
        raise VideoError(f"Space {getattr(client, '_src', '?')}: nenašel jsem image+prompt (endpoint {name})")
    log.info("[video] predict %s (image+prompt, seed=%s)", name, seed)
    r = client.predict(**kwargs, api_name=name)
    if isinstance(r, (tuple, list)):
        r = r[0]
    if isinstance(r, dict):
        r = r.get("video") or r.get("path") or r.get("url")
    return _store_video(r, seed)


def _store_video(result, seed: int) -> Path:
    if isinstance(result, dict):
        p = result.get("path") or result.get("url")
    else:
        p = str(result)
    if not p:
        raise VideoError(f"prázdný video výstup: {result!r}")
    dst = Path(time.strftime("output/clip_%Y%m%d_%H%M%S_")) / ("s%d.mp4" % seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if p.startswith(("http://", "https://")):
        urllib.request.urlretrieve(p, dst)
    elif Path(p).exists():
        shutil.copyfile(p, dst)
    elif p.startswith("data:"):
        dst.write_bytes(base64.b64decode(p.split(",", 1)[1]))
    else:
        raise VideoError(f"Nelze stáhnout video: {p}")
    log.info("[video] klip %d B -> %s", dst.stat().st_size, dst)
    return dst


def generate_clip(prompt: str, image_path: str | Path, seed: int = 0,
                  seconds: int | None = None, space: str | None = None,
                  token: str | None = None, timeout: int = 300) -> Path:
    """Vygeneruje klip (image+prompt → mp4). Sek __falls back__ chainem spaceů."""
    from gradio_client import Client
    imgp = Path(image_path)
    if not imgp.exists():
        raise VideoError(f"referenční obrázek neexistuje: {imgp}")
    tok = token or os.environ.get("HF_TOKEN")
    space = space or os.environ.get("VIDEO_SPACE") or PRIMARY
    chain = [space] + [x for x in (PRIMARY, FALLBACK) if x != space]
    last_err = None
    for sp in chain:
        try:
            log.info("[video] pokus %s dur=%s", sp, seconds or DEFAULT_SEC)
            client = Client(sp, verbose=False, token=tok)
            return _call_gradio(client, imgp, prompt, seed, seconds=seconds)
        except Exception as e:
            last_err = e
            log.warning("[video] %s selhal: %r", sp, e)
    raise VideoError(f"Všechny video backends selhaly; poslední: {last_err}")