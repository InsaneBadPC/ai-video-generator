"""Job Queue (SQLite) — trvalá fronta scénových úkolů.

Stavy: queued → claimed → running → done | failed
Každý job nese: song_id, scene_idx, sekce, prompt, media předvolba,
zpětné cesty (clip path, video source tier, seed). Dispatcher (queue_worker)
přebírá z fronty a volá scénový generátor.

SQLite = atomické claim (UPDATE ... WHERE status='queued' AND job_id=?)
→ bez duplikátů při vícenásobném workeru. Retry: attempts++, počet řídí config.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path

log = logging.getLogger("queue")

DEFAULT_DB = "queue/jobs.db"


def _conn(db: str) -> sqlite3.Connection:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init(db: str = DEFAULT_DB) -> None:
    with _conn(db) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                song_id TEXT NOT NULL,
                scene_idx INTEGER NOT NULL,
                section TEXT,
                label TEXT,
                prompt TEXT,
                seed INTEGER,
                pre_media TEXT,          -- 'auto' | path | 'local_loop' ...
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INTEGER DEFAULT 0,
                tier TEXT,               -- local_loop | local_kb | hf_wan | kaggle | svd
                clip_path TEXT,
                error TEXT,
                job_type TEXT DEFAULT 'full_scene',   -- full_scene | image_animation
                created REAL, started REAL, finished REAL
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_status ON jobs(status)")


def add_scene_job(db: str, song_id: str, scene_idx: int, section: str,
                  label: str, prompt: str, seed: int = 0,
                  pre_media: str = "auto", job_type: str = "full_scene") -> str:
    init(db)
    job_id = str(uuid.uuid4())[:12]
    with _conn(db) as c:
        c.execute("""INSERT INTO jobs (job_id,song_id,scene_idx,section,label,prompt,
                     seed,pre_media,status,created,job_type) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                  """, (job_id, song_id, scene_idx, section, label, prompt, seed,
                        pre_media, "queued", time.time(), job_type))
    log.info("[queue] job %s scene %d (%s) přidán", job_id, scene_idx, job_type)
    return job_id


def claim(db: str, worker: str = "") -> dict | None:
    """Atomicky převezme jeden queued job."""
    init(db)
    with _conn(db) as c:
        cur = c.execute("SELECT job_id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        cur = c.execute("UPDATE jobs SET status='running', started=?, attempts=attempts+1 "
                        "WHERE job_id=? AND status='queued'", (time.time(), row["job_id"]))
        if cur.rowcount != 1:
            return None
        cur = c.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],))
        return dict(cur.fetchone())


def finish(db: str, job_id: str, clip_path: str = "", tier: str = "",
           error: str = "") -> None:
    with _conn(db) as c:
        if error:
            c.execute("UPDATE jobs SET status='failed', error=?, finished=? "
                      "WHERE job_id=?", (error, time.time(), job_id))
        else:
            c.execute("UPDATE jobs SET status='done', clip_path=?, tier=?, finished=? "
                      "WHERE job_id=?", (clip_path, tier, time.time(), job_id))


def retry(db: str, job_id: str) -> None:
    with _conn(db) as c:
        c.execute("UPDATE jobs SET status='queued', error=NULL WHERE job_id=?", (job_id,))


def pending_count(db: str = DEFAULT_DB) -> int:
    init(db)
    with _conn(db) as c:
        return c.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' OR status='running'").fetchone()[0]


def stats(db: str = DEFAULT_DB) -> dict:
    init(db)
    with _conn(db) as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}