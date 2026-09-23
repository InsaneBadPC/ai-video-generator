from __future__ import annotations

import hmac
import os
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("QUEUE_DB", ROOT / "queue/jobs.db"))
STATIC = Path(__file__).resolve().parent / "static"
security = HTTPBasic()
app = FastAPI(title="AI Music Video Generator Dashboard", version="1.0")


class Action(BaseModel):
    action: str


def auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    expected_user = os.environ.get("DASHBOARD_USER", "")
    expected_password = os.environ.get("DASHBOARD_PASSWORD", "")
    user_ok = hmac.compare_digest(credentials.username, expected_user) if expected_user else False
    pass_ok = hmac.compare_digest(credentials.password, expected_password) if expected_password else False
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=30000")
    return c


@app.get("/api/health")
def health():
    return {"ok": True, "db": str(DB_PATH)}


@app.get("/api/stats")
def stats(_user: str = Depends(auth)):
    with conn() as c:
        rows = c.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


@app.get("/api/resources")
def resources(_user: str = Depends(auth)):
    names = ["OPENROUTER_API_KEY", "GROQ_API_KEY", "GOOGLE_AI_STUDIO_KEY",
             "HF_TOKEN", "KAGGLE_KEY", "CLOUDFLARE_R2_ACCESS_KEY_ID",
             "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    return {name: bool(os.environ.get(name)) for name in names}


@app.get("/api/logs")
def logs(_user: str = Depends(auth)):
    path = Path(os.environ.get("WORKER_LOG", ROOT / "logs" / "worker.log"))
    if not path.exists():
        return {"path": str(path), "lines": []}
    return {"path": str(path), "lines": path.read_text(errors="replace").splitlines()[-80:]}


@app.get("/api/jobs")
def jobs(_user: str = Depends(auth)):
    with conn() as c:
        rows = c.execute(
            "SELECT job_id,song_id,scene_idx,section,label,status,attempts,tier,clip_path,error,created,started,finished "
            "FROM jobs ORDER BY created DESC LIMIT 200"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        if item.get("clip_path"):
            item["clip_path"] = Path(item["clip_path"]).name
        result.append(item)
    return result


@app.post("/api/jobs/{job_id}/action")
def action(job_id: str, body: Action, _user: str = Depends(auth)):
    if body.action not in {"approve", "reject", "retry"}:
        raise HTTPException(400, "Unknown action")
    if body.action in {"approve", "retry"}:
        status_value, error = "queued", None
    else:
        status_value, error = "failed", "rejected from dashboard"
    with conn() as c:
        cur = c.execute("UPDATE jobs SET status=?, error=?, finished=NULL WHERE job_id=?",
                        (status_value, error, job_id))
        if cur.rowcount != 1:
            raise HTTPException(404, "Job not found")
    return {"ok": True, "job_id": job_id, "status": status_value}


@app.get("/", include_in_schema=False)
def index(_user: str = Depends(auth)):
    return FileResponse(STATIC / "index.html")


@app.get("/static/{path:path}", include_in_schema=False)
def static_file(path: str, _user: str = Depends(auth)):
    target = (STATIC / path).resolve()
    if STATIC not in target.parents or not target.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(target)
