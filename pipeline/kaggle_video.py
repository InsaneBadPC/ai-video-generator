"""Kaggle video backend — GPU zdroj #2 (offload pro HF Spaces).

Princip: vygeneruje Kaggle notebook-zdroj (kernel) s modelem Wan/LTX,
pushne zadání scény jako JSON (vstup), spustí kernel přes API s GPU
(accelerator), čeká na dokončení (poll), stáhne výstupní klip.

Týdenní kvóta 30 h (Always Free) — sleduje se v queue (dispatcher).
Ne automaticky spouští notebook při každé scéně: HF Space je primární,
Kaggle = náhrada, když HF ZeroGPU kvóta vyprchá.

Nový Kaggle API: KAGGLE_API_TOKEN (Bearer KGAT_) přes env, ne kaggle.json.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("kaggle")

KAGGLE_USER_SLUG = "petrinsane"         # ověřeno přes kaggle config view
KERNEL_ID_PREFIX = "sc_"                # sc = songcraft video

# Sloty notebooků: Always Free default, vyžaduje secrets HF_TOKEN atd.
DEFAULT_ACCEL = "GPU"                   # P100/T4 (free tier)
DEFAULT_LANG = "python"
DATASET_SLUG = "sc-flow-scene"
KERNEL_SLUG = "sc-chunk"


class KaggleError(RuntimeError):
    pass


def _env():
    return {**os.environ, "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_KEY", "")}


def _run(args, timeout=120):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                       env=_env(), timeout=timeout)
    if r.returncode:
        log.warning("[kaggle] %s rc=%d stderr=%s", "kaggle " + " ".join(args), r.returncode, r.stderr[:250])
        raise KaggleError(r.stderr[:400])
    return r.stdout


def check_access() -> dict:
    """Je token platný a máme přístup ke kernelům? Vrátí user info."""
    out = _run(["kernels", "list", "--page-size", "1"])
    return {"ok": True, "sample": out.splitlines()[1] if len(out.splitlines()) > 1 else ""}


def weekly_quota_used_h() -> float:
    """HRUBÝ odhad čerpání týdenní kvóty: součet trvání session z API.
    Kaggle API neposkytuje quotu přímo; dispatcher sleduje runtime v queue."""
    return 0.0


def _dataset_meta(slug: str) -> dict:
    return {
        "title": slug,
        "id": f"{KAGGLE_USER_SLUG}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "resources": [],
    }


def _ensure_dataset(slug: str = DATASET_SLUG):
    """Dataset existuje? Pokud ne, vytvoř ho."""
    try:
        _run(["datasets", "status", f"{KAGGLE_USER_SLUG}/{slug}"])
        log.info("[kaggle] dataset %s exists", slug)
        return
    except KaggleError:
        pass
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "README.md").write_text("# sc-flow-scene", encoding="utf-8")
        (td / "dataset-metadata.json").write_text(json.dumps(_dataset_meta(slug), indent=2), encoding="utf-8")
        _run(["datasets", "create", "-p", str(td), "-u"], timeout=180)
        log.info("[kaggle] dataset %s created", slug)


def upload_dataset_version(scene: dict, ref_img_path: str = None, slug: str = DATASET_SLUG):
    """Nahraje novou verzi datasetu s scene.json + ref obrázkem."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dataset-metadata.json").write_text(json.dumps(_dataset_meta(slug), indent=2), encoding="utf-8")
        (td / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        if ref_img_path and Path(ref_img_path).exists():
            import shutil
            shutil.copy(ref_img_path, str(td / "ref_image.webp"))
        _run(["datasets", "version", "-p", str(td), "-m", f"scene {scene.get('scene_idx', 0)}"], timeout=180)
        log.info("[kaggle] dataset %s/%s version uploaded", KAGGLE_USER_SLUG, slug)


def _kernel_metadata(name: str) -> dict:
    return {
        "id": f"{KAGGLE_USER_SLUG}/{name}",
        "title": name,
        "code_file": f"{name}.py",
        "language": DEFAULT_LANG,
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def push_kernel(name: str, code: str, title: str = None,
                accelerator: str = DEFAULT_ACCEL, language: str = DEFAULT_LANG):
    """Pushne notebook do Kaggle a spustí ho."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / f"{name}.py").write_text(code, encoding="utf-8")
        (td / "kernel-metadata.json").write_text(json.dumps(_kernel_metadata(name), indent=2), encoding="utf-8")
        _run(["kernels", "push", "-p", str(td), "--accelerator", accelerator], timeout=300)
        log.info("[kaggle] kernel %s/%s pushed", KAGGLE_USER_SLUG, name)


def _wait_kernel(kernel: str, timeout: int = 1800, poll_interval: int = 30) -> str:
    """Poll status until kernel dokončí, vrátí finální status."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            out = _run(["kernels", "status", kernel])
            status = out.strip().lower()
            log.info("[kaggle] kernel %s status: %s", kernel, status)
            if "error" in status or "failed" in status:
                raise KaggleError(f"kernel {kernel} failed: {status}")
            if "complete" in status or "done" in status:
                return status
            time.sleep(poll_interval)
        except KaggleError:
            time.sleep(poll_interval)
    raise KaggleError(f"kernel {kernel} timed out after {timeout}s")


def _scene_code(scene: dict, ref_img: str = None) -> str:
    """Kernel kód s VLOŽENÝMI vstupy (scene + base64 obrázek) — bez datasetu."""
    import base64 as _b64
    data = {"scene": scene}
    if ref_img and Path(ref_img).exists():
        data["ref_b64"] = _b64.b64encode(Path(ref_img).read_bytes()).decode()
        data["ref_ext"] = Path(ref_img).suffix.lstrip(".") or "png"
    payload = json.dumps(data)
    code = (
        KERNEL_BODY
        .replace("__PAYLOAD__", json.dumps(payload))
    )
    return code


KERNEL_BODY = '''import json, base64, io, os, sys
import numpy as np
from PIL import Image

payload = json.loads(__PAYLOAD__)
sc = payload["scene"]
prompt = sc.get("prompt", "")
seed = int(sc.get("seed", 0))
secs = int(sc.get("duration", 4))

if payload.get("ref_b64"):
    ext = payload.get("ref_ext", "png")
    img = Image.open(io.BytesIO(base64.b64decode(payload["ref_b64"]))).convert("RGB")
else:
    raise SystemExit("Nikdy nebyl posláný žádný ref obrázek")
img = img.resize((576, 320))

import torch, gc, subprocess, importlib.metadata
from diffusers.utils import export_to_video
import os
os.makedirs("/kaggle/output", exist_ok=True)

# Repo multimodalart/stable-video-diffusion je staré → starý název pipeline
try:
    from diffusers import StableVideoDiffusionPipeline as PIPE
except ImportError:
    from diffusers import StableVideoDiffusionImg2VidPipeline as PIPE

print("Loading SVD...", flush=True)
pipe = PIPE.from_pretrained(
    "vdo/stable-video-diffusion-img2vid-xt-1-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

print(f"Generating {secs}s ...", flush=True)
frames = pipe(image=img, num_frames=25, num_inference_steps=25,
              negative_prompt="", guidance_scale=1.0,
              generator=torch.Generator("cuda").manual_seed(seed)).frames[0]

out_path = "/kaggle/output/clip.mp4"
export_to_video(frames, out_path, fps=8)
print(f"Saved {out_path}", flush=True)
'''

KERNEL_CODE = _scene_code({"scene_idx": 0, "prompt": "smoke", "seed": 0, "duration": 4},
                          "character_reference/temney_flux_ref.webp")


def generate_kaggle_clip(scene: dict, ref_img: str = None, dest: Path = Path("output")) -> Path:
    """Full flow: kernel (SVD i2v, vstupy v kódu) → poll → pull output.
    Vrací lokální cestu ke klipu nebo vyvolává KaggleError.
    """
    code = _scene_code(scene, ref_img)
    push_kernel(KERNEL_SLUG, code, title="sc-chunk")
    kernel = f"{KAGGLE_USER_SLUG}/{KERNEL_SLUG}"
    _wait_kernel(kernel, timeout=2400)
    return pull_output(kernel, dest)


def push_scene(kernel: str, scene: dict, ref_img: str):
    """Plná Kaggle GPU integrace po propojení s dispatcherem."""
    pass


def list_kernels() -> list[str]:
    out = _run(["kernels", "list", "--mine"])
    return [l.strip().split()[0] for l in out.splitlines()[1:] if l.strip()]


def get_status(kernel: str) -> str:
    out = _run(["kernels", "status", kernel])
    return out.strip()


def pull_output(kernel: str, dest: Path = Path("output")) -> Path:
    """Stáhne výstupní soubory kernelu; vrátí první .mp4."""
    out_dir = dest / f"kg_{kernel.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(["kernels", "output", kernel, "-p", str(out_dir)], timeout=180)
    mp4s = list(out_dir.rglob("*.mp4"))
    if not mp4s:
        raise KaggleError(f"kernel {kernel} nevrátil žádný mp4")
    return mp4s[0]