"""Run LLM-generated Python code inside an isolated Docker container."""

import os
import subprocess
import uuid
from pathlib import Path

IMAGE = "agent-sandbox"
ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4000


def run_code(code: str, data_file: str) -> dict:
    """Execute code with the CSV mounted read-only at /data/data.csv.

    Charts must be saved under /out. Returns stdout/stderr and the files created.
    """
    run_id = uuid.uuid4().hex[:8]
    run_dir = WORKSPACE / f"run_{run_id}"
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True)
    (run_dir / "script.py").write_text(code, encoding="utf-8")

    name = f"sandbox-{run_id}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",  # no internet access
        "--memory",
        "512m",
        "--cpus",
        "1",  # resource limits
        "--pids-limit",
        "128",
        "--cap-drop",
        "ALL",
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{run_dir / 'script.py'}:/work/script.py:ro",
        "-v",
        f"{Path(data_file).resolve()}:/data/data.csv:ro",
        "-v",
        f"{out_dir}:/out",
        IMAGE,
        "python",
        "/work/script.py",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
        stdout, stderr, ok = proc.stdout, proc.stderr, proc.returncode == 0
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        stdout, stderr, ok = "", f"Timed out after {TIMEOUT_SECONDS}s", False

    return {
        "ok": ok,
        "stdout": stdout[:MAX_OUTPUT_CHARS],
        "stderr": stderr[-MAX_OUTPUT_CHARS:],
        "files": sorted(p.name for p in out_dir.iterdir()),
        "out_dir": str(out_dir),
    }
