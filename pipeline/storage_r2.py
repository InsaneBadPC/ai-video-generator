"""Volitelné úložiště Cloudflare R2 přes S3 kompatibilní API.

R2 je explicitně opt-in přes STORAGE_R2_ENABLED nebo config.yaml. Chybějící
credentials nikdy nevypisujeme do logu a neblokují lokální pipeline.
"""
from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path

log = logging.getLogger("storage_r2")


class StorageError(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get("STORAGE_R2_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def _client():
    if not enabled():
        raise StorageError("R2 je vypnuté (STORAGE_R2_ENABLED není true)")
    required = ["CLOUDFLARE_R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
                "CLOUDFLARE_R2_BUCKET", "CLOUDFLARE_R2_ENDPOINT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise StorageError("Chybí R2 konfigurace: " + ", ".join(missing))
    try:
        import boto3
    except ImportError as exc:
        raise StorageError("Chybí závislost boto3") from exc
    return boto3.client(
        "s3",
        endpoint_url=os.environ["CLOUDFLARE_R2_ENDPOINT"],
        aws_access_key_id=os.environ["CLOUDFLARE_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CLOUDFLARE_R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_file(path: str | Path, key: str | None = None) -> str:
    """Nahraje soubor a vrátí jeho R2 object key."""
    src = Path(path)
    if not src.exists() or not src.is_file():
        raise StorageError(f"Soubor neexistuje: {src}")
    object_key = key or src.name
    client = _client()
    extra = {}
    content_type, _ = mimetypes.guess_type(src.name)
    if content_type:
        extra["ContentType"] = content_type
    client.upload_file(str(src), os.environ["CLOUDFLARE_R2_BUCKET"], object_key,
                       ExtraArgs=extra or None)
    log.info("[r2] nahráno %s jako %s", src.name, object_key)
    return object_key


def upload_json(path: str | Path, key: str | None = None) -> str:
    return upload_file(path, key or (Path(path).stem + ".json"))


def signed_url(key: str, expires: int = 3600) -> str:
    """Vytvoří dočasný URL pro náhled; URL se neloguje."""
    client = _client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["CLOUDFLARE_R2_BUCKET"], "Key": key},
        ExpiresIn=expires,
    )
