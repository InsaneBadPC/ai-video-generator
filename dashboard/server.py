from __future__ import annotations

import hmac
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("QUEUE_DB", ROOT / "queue/jobs.db"))
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "state" / "runs"
security = HTTPBasic(auto_error=False)
app = FastAPI(title="AI Music Video Generator Dashboard", version="1.0")


class Action(BaseModel):
    action: str


def _valid_user(username: str, password: str) -> bool:
    expected_user = os.environ.get("DASHBOARD_USER", "")
    expected_password = os.environ.get("DASHBOARD_PASSWORD", "")
    user_ok = hmac.compare_digest(username, expected_user) if expected_user else False
    pass_ok = hmac.compare_digest(password, expected_password) if expected_password else False
    return user_ok and pass_ok


def _session_token() -> str:
    return hmac.new(os.environ.get("DASHBOARD_PASSWORD", "").encode(),
                    os.environ.get("DASHBOARD_USER", "").encode(), "sha256").hexdigest()


def auth(request: Request, credentials: HTTPBasicCredentials | None = Depends(security)) -> str:
    if hmac.compare_digest(request.cookies.get("video_session", ""), _session_token()):
        return os.environ.get("DASHBOARD_USER", "")
    user_ok = bool(credentials) and _valid_user(credentials.username, credentials.password)
    if not user_ok:
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


@app.post("/api/generate")
async def generate(request: Request, _user: str = Depends(auth)):
    # Starlette 1.7 defaults each multipart part to 1 MiB; full songs exceed that.
    form = await request.form(max_files=2, max_fields=3, max_part_size=50 * 1024 * 1024)
    try:
        audio = form.get("audio")
        image = form.get("image")
        prompt = form.get("prompt", "")
        title = form.get("title", "Temney")
        mode = form.get("mode", "full_scenes")
        if not isinstance(audio, UploadFile) or not audio.filename:
            raise HTTPException(400, "Vyber MP3 nebo jiný audio soubor z telefonu")
        if image is not None and not isinstance(image, UploadFile):
            raise HTTPException(400, "Neplatný soubor obrázku")
        if not all(isinstance(value, str) for value in (prompt, title, mode)):
            raise HTTPException(400, "Neplatné textové pole")
        if mode not in {"full_scenes", "image_animation"}:
            raise HTTPException(400, "Neplatný režim")
        if mode == "image_animation" and (not image or not image.filename):
            raise HTTPException(400, "Pro image_animation je potřeba obrázek")

        run_id = uuid.uuid4().hex[:12]
        RUNS.mkdir(parents=True, exist_ok=True)
        audio_path = ROOT / "input" / f"{run_id}_{Path(audio.filename).name}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with audio_path.open("wb") as out:
            shutil.copyfileobj(audio.file, out)
        image_path = None
        if image and image.filename:
            image_path = ROOT / "character_reference" / f"{run_id}_{Path(image.filename).name}"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            with image_path.open("wb") as out:
                shutil.copyfileobj(image.file, out)
        if prompt:
            (ROOT / "input" / f"{run_id}_{Path(audio.filename).stem}.txt").write_text(prompt, encoding="utf-8")
        log_path = ROOT / "logs" / f"run_{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "pipeline.orchestrator", "--mode", mode,
               "--audio", str(audio_path.relative_to(ROOT)), "--title", title]
        if image_path:
            cmd += ["--image", str(image_path.relative_to(ROOT))]
        if prompt:
            cmd += ["--prompt", prompt]
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        record = {"run_id": run_id, "pid": proc.pid, "status": "running", "mode": mode,
                  "title": title, "prompt": prompt, "audio": str(audio_path.name),
                  "image": image_path.name if image_path else None, "log": str(log_path),
                  "created": time.time()}
        (RUNS / f"{run_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return record
    finally:
        await form.close()


@app.get("/api/runs")
def runs(_user: str = Depends(auth)):
    result = []
    for path in sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        try:
            item = json.loads(path.read_text())
            log_file = Path(item.get("log", ""))
            lines = log_file.read_text(errors="replace").splitlines() if log_file.exists() else []
            if item.get("status") == "running":
                try:
                    stat = Path(f"/proc/{int(item['pid'])}/stat").read_text().split()
                    if len(stat) < 3 or stat[2] == "Z":
                        raise OSError("process finished")
                except OSError:
                    item["status"] = "finished"
                    final = next((x.split("FINAL VYSTUP: ", 1)[1].split(" (", 1)[0]
                                  for x in reversed(lines) if "FINAL VYSTUP: " in x), None)
                    if final:
                        item["output"] = final
                    path.write_text(json.dumps(item, ensure_ascii=False, indent=2))
            if item.get("output") or item.get("status") == "finished":
                item["progress"] = 100
                item["download"] = f"/api/runs/{item['run_id']}/download"
            else:
                stages = sum(1 for line in lines if "localclip" in line or "verze " in line)
                item["progress"] = min(95, 10 + stages * 20)
            item["message"] = "Hotovo" if item.get("progress") == 100 else "Generuji video…"
            result.append(item)
        except Exception:
            continue
    return result


@app.get("/api/runs/{run_id}/download")
def download_run(run_id: str, _user: str = Depends(auth)):
    record_path = RUNS / f"{run_id}.json"
    if not record_path.is_file():
        raise HTTPException(404, "Run nenalezen")
    item = json.loads(record_path.read_text())
    output = Path(item.get("output", ""))
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if ROOT not in output.parents or not output.is_file():
        raise HTTPException(404, "Výstup ještě není hotový")
    return FileResponse(output, media_type="video/mp4", filename=output.name)


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
def index(request: Request):
    if hmac.compare_digest(request.cookies.get("video_session", ""), _session_token()):
        return FileResponse(STATIC / "index.html")
    return HTMLResponse("""<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>
    <title>Video Agent přihlášení</title><style>body{font-family:system-ui;background:#10131a;color:#eef1f7;display:grid;place-items:center;min-height:100vh}form{background:#1b2130;padding:24px;border-radius:14px;display:grid;gap:12px;width:min(340px,85vw)}input,button{font-size:1rem;padding:12px;border-radius:8px;border:0}button{background:#3c82f6;color:white}</style>
    <form method=post action=/login><h2>AI Music Video Generator</h2><input name=username placeholder='Uživatel' autocomplete=username required><input name=password type=password placeholder='Heslo' autocomplete=current-password required><button>Přihlásit</button></form>""")


@app.post("/login", include_in_schema=False)
def login(username: str = Form(...), password: str = Form(...)):
    if not _valid_user(username, password):
        return HTMLResponse("Neplatné přihlášení", status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("video_session", _session_token(), httponly=True, secure=True,
                        samesite="lax", max_age=86400)
    return response


@app.get("/static/{path:path}", include_in_schema=False)
def static_file(path: str, _user: str = Depends(auth)):
    target = (STATIC / path).resolve()
    if STATIC not in target.parents or not target.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(target)
