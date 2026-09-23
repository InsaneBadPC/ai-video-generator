"""Telegram notifikace — schvalovací uzly + oznámení dokončení/chyb.

Bez TELEGRAM_CHAT_ID v .env se modul auto-vypne (enabled=False) — neblokuje
pipeline. Posílá: start, storyboard hotov, chyby, finální video odkaz.
Volá API přes urllib (žádná závislost), timeout 15s.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("tg")

API = "https://api.telegram.org/bot{token}/{method}"


def _token():
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()


def _chat_id():
    return os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def enabled() -> bool:
    cfg = Path("config.yaml")
    if cfg.exists():
        try:
            import yaml
            if not (yaml.safe_load(cfg.read_text()) or {}).get("telegram", {}).get("enabled", False):
                return False
        except Exception:
            pass
    return bool(_token() and _chat_id())


def _post(method: str, data: dict) -> bool:
    token = _token()
    if not token:
        return False
    try:
        req = urllib.request.Request(
            API.format(token=token, method=method),
            data=urllib.parse.urlencode(data).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = r.status == 200
            if not ok:
                log.warning("[tg] %s rc=%s", method, r.status)
            return ok
    except Exception as e:
        log.warning("[tg] %s selhalo: %r", method, e)
        return False


def notify(text: str, parse: str = "HTML") -> bool:
    if not enabled():
        return False
    return _post("sendMessage", {"chat_id": _chat_id(), "text": text, "parse_mode": parse})


def notify_file(path: str | Path, caption: str = "") -> bool:
    """Multipart upload videa/obrázku (mp4) — ruční multipart, bez httpx."""
    import uuid as _uuid
    p = Path(path)
    if not p.exists():
        return False
    token = _token()
    boundary = "----TG" + _uuid.uuid4().hex
    body = bytearray()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; "
             f'name="chat_id"\r\n\r\n{_chat_id()}\r\n').encode()
    if caption:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="caption"\r\n\r\n{caption}\r\n').encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; "
             f"filename=\"{p.name}\"\r\nContent-Type: video/mp4\r\n\r\n").encode()
    body += p.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    if not token:
        return False
    try:
        req = urllib.request.Request(
            API.format(token=token, method="sendVideo"),
            data=bytes(body))
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except Exception as e:
        log.warning("[tg] upload selhal: %r", e)
        return False