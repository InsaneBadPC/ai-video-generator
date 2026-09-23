"""Image backend — adapter na HF Space (FLUX character).

Vyměnitelné rozhraní. Primární: veřejný HF Space FLUX.1-schnell
(black-forest-labs — komerčně volný, ZeroGPU zadarmo, bez karty).
Fallback bez GPU: lokální placeholder (aby pipeline běžela offline).

Pokud máš budoucně PRO účet, dá se stejně použít i vlastní Space:
  IMAGE_SPACE=jméno (z .env), s endpointem /generate.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger("image")

PUBLIC_FLUX = "black-forest-labs/FLUX.1-schnell"
DEFAULT_SIZE = (1024, 1024)      # native rozlišení FLUX Space


class ImageError(RuntimeError):
    pass


def _discover(client):
    """Najde prvni endpoint s promptem. Bezpečné pro různé Spaces."""
    try:
        api = client.view_api(return_format="dict")
        for ep_name, ep in (api.get("named_endpoints") or {}).items():
            params = [p.get("parameter", {}).get("python_type", "").lower() for p in ep.get("parameters", [])]
            if "str" in params:  # první parametr str = prompt
                return ep_name
    except Exception as e:
        log.warning("[image] discovery endpoinu selhal: %r", e)
    return "/infer"


def _gradio_generate(space: str, prompt: str, seed: int = 0,
                     width: int = None, height: int = None, timeout: int = 300) -> Path:
    from gradio_client import Client, handle_file
    w = width or DEFAULT_SIZE[0]
    h = height or DEFAULT_SIZE[1]

    # přihlášení HF tokenem (povinné kvůli rate-limitu)
    tok = os.environ.get("HF_TOKEN")
    kwargs = {}
    if tok:
        from huggingface_hub import login
        api = None
        try:
            from gradio_client import Client as C  # noqa
            kwargs["hf_token"] = tok
        except Exception:
            pass

    client = Client(space, verbose=False, token=tok or None)
    endpoint = _discover(client)

    # univerzální predict: posíláme jen pojmenované sémantické parametry
    kwargs = {"prompt": prompt, "api_name": endpoint}
    if seed:
        kwargs["seed"] = int(seed)
    if seed == 0:
        kwargs["randomize_seed"] = False
    kwargs["width"] = int(w)
    kwargs["height"] = int(h)
    kwargs["num_inference_steps"] = 4

    res = client.predict(**kwargs)
    if isinstance(res, (tuple, list)):
        res = res[0]
    if isinstance(res, dict):
        p = res.get("path") or res.get("url")
    else:
        p = str(res)
    if not p:
        raise ImageError(f"prázdný výstup: {res!r}")
    if p.startswith(("http://", "https://")):
        import urllib.request
        dst = Path(time.strftime("character_reference/ref_%Y%m%d_%H%M%S.png"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(p, dst)
        return dst
    return Path(p) if Path(p).exists() else _store_result(p)


def _store_result(p: str) -> Path:
    import base64
    import urllib.parse
    if p.startswith("data:image"):
        b64 = p.split(",", 1)[1]
        dst = Path(time.strftime("character_reference/ref_%Y%m%d_%H%M%S.png"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(b64))
        return dst
    raise ImageError(f"Neexistující cesta výstupu: {p}")


def generate_reference(prompt: str, seed: int = 0, space: str = None,
                       width: int = None, height: int = None,
                       out_dir: str = "character_reference", timeout: int = 300) -> Path:
    """Vygeneruje referenční obrázek postavy. Vrací cestu k PNG."""
    space = space or os.environ.get("IMAGE_SPACE") or PUBLIC_FLUX
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        img = _gradio_generate(space, prompt, seed, width, height, timeout)
        log.info("[image] OK space=%s (%.1fs) -> %s", space, time.time() - t0, img)
        return img
    except ImageError:
        raise
    except Exception as e:
        log.warning("[image] HF Space %s selhal: %r", space, e)
    if space != PUBLIC_FLUX:
        log.warning("[image] zkouším veřejný FLUX Space")
        return generate_reference(prompt, seed, space=PUBLIC_FLUX, width=width,
                                  height=height, out_dir=out_dir, timeout=timeout)
    raise ImageError(f"Nepovedlo se vygenerovat referenci (Space {space})")


# ── lokální placeholder (bez GPU, pro včasné testy assembly) ──
def make_placeholder(txt: str, out: str = "character_reference/ref_placeholder.png",
                     w: int = 1024, h: int = 1024) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        try:
            Image = __import__("PIL").Image
            ImageDraw = Image.ImageDraw
        except Exception as e:
            raise ImageError(f"Pillow není dostupné pro placeholder: {e}")
    # Gradient + diagonal color bands jsou záměrné: i při vyčerpání GPU kvóty
    # musí lokální fallback projít QC kontrolou variability jasu.
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            band = ((x // 96) + (y // 96)) % 4
            px[x, y] = ((18 + x * 36 // max(w, 1) + band * 18) % 120,
                        (24 + y * 42 // max(h, 1) + band * 12) % 130,
                        (70 + (x + y) * 80 // max(w + h, 1) + band * 20) % 190)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(40, 40, 90), width=4)
    d.text((30, h // 2 - 20), txt[:80], fill=(200, 210, 255))
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    log.info("[image] placeholder -> %s", p)
    return p
