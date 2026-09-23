"""LLM klient — unifikované OpenAI-kompatibilní volání s fallbacky.

Backends (z config.llm):
  1. openrouter (začíná sk-or-)
  2. groq (začíná gsk_)
  3. gemini (začíná AQ. a je OpenAI-kompatibilní przez
     https://generativelanguage.googleapis.com/v1beta/openai)
Při chybě/rate-limit/décommit přepne na další backend, jakmile vyčerpá
retries. Vrací čistý text odpovědi.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("llm")

BACKEND_SPEC = {
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash",
        "env": "OPENROUTER_API_KEY",
        "headers_extra": {"HTTP-Referer": "https://songcraft.studio", "X-Title": "AI-Music-Video"},
    },
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "env": "GROQ_API_KEY",
    },
    "gemini": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "env": "GOOGLE_AI_STUDIO_KEY",
    },
}


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, payload: dict, timeout: int = 60) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise LLMError(f"HTTP {e.code} {e.reason}: {body}") from e


def _parse_affordable(msg: str) -> int | None:
    """Z hlášky OpenRouter '...can only afford N...' vytáhne N."""
    import re
    m = re.search(r"only afford (\d+)", msg)
    return int(m.group(1)) if m else None


def _is_retryable(e: LLMError) -> bool:
    """Vrátí True pro dočasné chyby: 5xx, 429, 402 in-flight budget."""
    m = str(e)
    return ("HTTP 500" in m or "HTTP 502" in m or "HTTP 503" in m or
            "HTTP 429" in m or "in_flight_budget_exhausted" in m)


def _headers(key: str, backend: str, spec: dict) -> dict:
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    h.update(spec.get("headers_extra", {}))
    return h


def _endpoint(spec: dict) -> str:
    return f"{spec['base']}/chat/completions"


def chat(prompt: str, max_tokens: int = 4096, timeout: int = 90) -> str:
    """Zkusí backends v pořadí (gemini→openrouter→groq); 1. úspěch vyhraje."""
    ordered = ["gemini", "openrouter", "groq"]
    last_err = None
    for name in ordered:
        spec = BACKEND_SPEC[name]
        key = os.environ.get(spec["env"])
        if not key:
            log.warning("[llm] %s: chybí klíč %s", name, spec["env"])
            continue
        payload = {
            "model": spec["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            tries = 0
            while True:
                tries += 1
                try:
                    t0 = time.time()
                    out = _post(_endpoint(spec), _headers(key, name, spec), payload, timeout=timeout)
                    log.info("[llm] %s OK (%.1fs, %d tok)", name, time.time() - t0, len(out))
                    return out
                except LLMError as e:
                    if tries >= 3:
                        raise
                    afford = _parse_affordable(str(e))
                    if afford and afford >= 64 and "402" in str(e):
                        payload["max_tokens"] = min(payload["max_tokens"], afford)
                        log.info("[llm] %s: downsizing max_tokens na %d (402 limit)", name, payload["max_tokens"])
                        continue
                    if _is_retryable(e):
                        time.sleep(2 * tries)
                        log.info("[llm] %s: retry %d/3 (%s)", name, tries, str(e)[:80])
                        continue
                    raise
        except LLMError as e:
            last_err = e
            log.warning("[llm] %s selhal: %s → fallback", name, e)
        except Exception as e:
            last_err = e
            log.warning("[llm] %s vyjímka: %r → fallback", name, e)
    raise LLMError(f"Všechny LLM backends selhaly; poslední: {last_err}")


def chat_json(prompt: str, max_tokens: int = 4096, timeout: int = 90) -> dict:
    """chat + robustní parsování JSON (fence ```json, extra data na konci)."""
    raw = chat(prompt, max_tokens=max_tokens, timeout=timeout)
    if "```" in raw:
        parts = [p for p in raw.split("```") if p.strip() and ("{" in p or "[" in p)]
        if parts:
            raw = parts[-1]
    def _try(s):
        try:
            return json.loads(s)
        except Exception:
            return None
    # přímý pokus a pak nejdelší vyvážený úsek
    for candidate in (raw, _first_balanced(raw, "[]"), _first_balanced(raw, "{}"),
                      _first_balanced(raw[raw.find("["):], "[]")):
        if candidate is None:
            continue
        obj = _try(candidate)
        if obj is not None:
            return obj
    raise LLMError("LLM nevrátil parsovatelný JSON") from None


def _first_balanced(s: str, pair: str) -> str | None:
    '''Vrátí první vyvážený úsek začínající op (a končící close).'''
    op, cl = pair[0], pair[1]
    start = s.find(op)
    if start == -1:
        return None
    depth = 0
    instr = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == op:
            depth += 1
        elif c == cl:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _maxtok(budget_frac: float) -> int:
    return max(512, int(48000 * budget_frac))