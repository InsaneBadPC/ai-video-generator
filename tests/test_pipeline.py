import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import queue
from pipeline.audio_analyzer import analyze_audio
from pipeline.image_animator import run_image_animation


def test_queue_recovers_stale_running_job(tmp_path):
    db = str(tmp_path / "jobs.db")
    job_id = queue.add_scene_job(db, "song", 0, "verse", "cover", "test", job_type="image_animation")
    job = queue.claim(db)
    assert job["job_id"] == job_id
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE jobs SET started=? WHERE job_id=?", (time.time() - 3600, job_id))
    assert queue.recover_stale(db, stale_after=60) == 1
    assert queue.stats(db)["queued"] == 1


def test_local_image_animation_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    image = tmp_path / "cover.png"
    audio = tmp_path / "song.wav"
    output = tmp_path / "final.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24",
        "-frames:v", "1", str(image),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:a", "pcm_s16le", str(audio),
    ], check=True)
    amap = analyze_audio(str(audio), None)
    assert 2.5 < amap["duration_s"] < 3.5
    result = run_image_animation(str(image), str(audio), "subtle cinematic motion", variants=1, out=str(output))
    assert result == output
    assert output.exists() and output.stat().st_size > 10_000
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(output),
    ], capture_output=True, text=True, check=True)
    assert 2.5 <= float(probe.stdout.strip()) <= 3.5
