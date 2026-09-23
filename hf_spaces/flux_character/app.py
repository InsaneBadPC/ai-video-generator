"""HF Space #1 — FLUX.1-schnell character reference image (ZeroGPU).

Vstup: (prompt:str, seed:int, width:int, height:int)
Výstup: PIL image jako PNG.

Nakládá model jednou (lazy), vrací 904x1024-подobné rozlišení v násobcích 16.
FLUX.1-schnell = Apache-2.0, komerčně volný, 4 inference steps.
"""
from __future__ import annotations

import os
import time

import gradio as gr
import torch
from PIL import Image
from diffusers import FluxPipeline

MODEL = "black-forest-labs/FLUX.1-schnell"
_pipe = None


def load():
    global _pipe
    if _pipe is None:
        t0 = time.time()
        _pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        _pipe.enable_model_cpu_offload()
        _pipe.vae.enable_slicing()
        print(f"[flux] model loaded in {time.time()-t0:.1f}s", flush=True)
    return _pipe


def generate(prompt: str, seed: int = 0, width: int = 896, height: int = 896):
    pipe = load()
    g = torch.Generator("cpu").manual_seed(seed if seed else 42)
    w, h = _nearest16(width), _nearest16(height)
    t0 = time.time()
    img = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
               generator=g, width=w, height=h, max_sequence_length=256).images[0]
    print(f"[flux] render {w}x{h} in {time.time()-t0:.1f}s", flush=True)
    return (img, f"OK {w}x{h} seed={seed if seed else 42} in {time.time()-t0:.0f}s")


def _nearest16(v: int) -> int:
    v = max(512, min(int(v), 1152))
    return (v // 16) * 16


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, value="Portrait of Temney, gaunt young man in dark clothes, moody city night, neon, cinematic"),
        gr.Number(label="Seed", value=0, precision=0),
        gr.Number(label="Width", value=896, precision=0),
        gr.Number(label="Height", value=896, precision=0),
    ],
    outputs=[gr.Image(type="pil", label="Reference"), gr.Textbox(label="Info")],
    title="FLUX Character — Temney",
    description="Generates a consistent character reference image (16:9 or square).",
)

if __name__ == "__main__":
    demo.launch(max_threads=2)