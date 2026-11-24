"""Small local DAG runner: no shell, atomic checkpoints, hash-verified resume."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import psutil

from evaluation.report import write_json
from evaluation.resources import ResourceMonitor, terminate_tree

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(folder):
    folder = Path(folder).resolve()
    hashes = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError("Stage artifacts cannot be symlinks or escape their attempt directory")
        if path.is_file():
            hashes[path.relative_to(folder).as_posix()] = sha256(path)
    return hashes


def input_hashes(paths):
    hashes = {}
    for item in paths:
        path = Path(item).resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    hashes[str(child.resolve())] = sha256(child)
        elif path.is_file():
            hashes[str(path)] = sha256(path)
        else:
            raise ValueError(f"Pipeline input does not exist: {path}")
    return hashes


def code_fingerprint():
    paths = []
    for folder in ("stage_a", "reconstruction", "agents", "scene_schema", "blender", "evaluation", "pipeline", "scripts"):
        paths.extend(sorted((ROOT/folder).glob("*.py")))
    paths.extend(sorted((ROOT/"config").glob("*schema*.json")))
    paths.append(ROOT/"requirements.txt")
    return input_hashes(paths)


def runtime_versions():
    packages = {}
    for name in ("numpy", "scipy", "open3d", "Pillow", "imageio-ffmpeg", "pyarrow", "jsonschema", "psutil", "torch", "torchvision", "transformers", "lpips"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(executable=sys.executable, python=sys.version, packages=packages)


@contextmanager
def run_lock(output):
    lock = output/".pipeline.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        previous = json.loads(lock.read_text(encoding="utf-8"))
        try:
            live = psutil.Process(previous["pid"])
            if abs(live.create_time()-previous["created"]) < 0.01:
                raise ValueError("Pipeline output is locked by a live process")
        except psutil.NoSuchProcess:
            pass
        # Exact stale lock only; no attempt data is removed.
        lock.unlink()
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), stream)
    try:
        yield
    finally:
        lock.unlink()


def validate_steps(steps):
    if not isinstance(steps, list) or not steps:
        raise ValueError("Pipeline requires a nonempty stage list")
    seen = set()
    allowed = {"id", "depends_on", "command", "inputs", "outputs", "timeout_seconds", "retries", "retry_exit_codes", "always_run"}
    for step in steps:
        if set(step)-allowed:
            raise ValueError("Unknown stage option")
        identity = step.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", identity) or identity in seen:
            raise ValueError("Unique safe stage IDs required")
        dependencies = step.get("depends_on", [])
        if not isinstance(dependencies, list) or not set(dependencies) <= seen:
            raise ValueError("Stages must be topologically ordered with existing dependencies")
        seen.add(identity)
        if not isinstance(step.get("command"), list) or not step["command"] or any(not isinstance(arg, str) for arg in step["command"]):
            raise ValueError("Stage command must be a nonempty argv list, not shell text")
        outputs = step.get("outputs")
        if not isinstance(outputs, list) or not outputs or any(not isinstance(p, str) or not p.strip() or Path(p).anchor or p == "." or ".." in Path(p).parts for p in outputs):
            raise ValueError("Required outputs must be nonempty relative artifact paths")
        for field, minimum, maximum in [("timeout_seconds", 1, 86400), ("retries", 0, 10)]:
            value = step.get(field, 3600 if field == "timeout_seconds" else 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum or (field == "retries" and not isinstance(value, int)):
                raise ValueError(f"Invalid stage {field}")
        codes = step.get("retry_exit_codes", [1, 124])
        if not isinstance(codes, list) or any(isinstance(code, bool) or not isinstance(code, int) or code == 0 for code in codes):
            raise ValueError("Retry exit codes must be nonzero integers")
        if not isinstance(step.get("always_run", False), bool):
            raise ValueError("always_run must be boolean")


def _cached(entry, fingerprint, expected, output):
    if entry.get("status") not in ("completed", "cached") or entry.get("fingerprint") != fingerprint:
        return False
    hashes = entry.get("artifact_hashes")
    folder = Path(entry.get("output_directory", "")).resolve()
    if not folder.is_relative_to(output/"stages") or not folder.is_dir() or not isinstance(hashes, dict) or not hashes or not set(expected) <= set(hashes):
        return False
    return file_hashes(folder) == hashes


def run_steps(steps, output, *, settings=None, cache=True, resume=True):
    validate_steps(steps)
    output = Path(output).resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("Pipeline needs a dedicated output directory, not the workspace or its ancestor")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output/"pipeline_state.json"
    if not state_path.exists() and any(p.name != ".pipeline.lock" for p in output.iterdir()):
        raise ValueError("Refusing unmanaged nonempty pipeline output directory")
    with run_lock(output):
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(schema="pipeline.checkpoint/1", stages={})
        if state.get("schema") != "pipeline.checkpoint/1" or not isinstance(state.get("stages"), dict):
            raise ValueError("Pipeline checkpoint is not owned/valid")
        implementation, runtime = code_fingerprint(), runtime_versions()
        state["started_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        state["status"] = "running"
        current = {}
        write_json(state_path, state)
        for step in steps:
            identity = step["id"]
            stage_start = time.perf_counter()
            old = state["stages"].get(identity, {})
            history = old.get("attempts", [])
            for attempt in history:
                if attempt.get("status") == "running":
                    attempt.update(status="interrupted", error="Previous process stopped before checkpoint commit")
            deps = {key: current[key] for key in step.get("depends_on", [])}
            entry = dict(id=identity, status="pending", attempts=history, current_run_wall_seconds=0.)
            current[identity] = entry
            state["stages"][identity] = entry
            if not step.get("always_run") and any(d["status"] not in ("completed", "cached") for d in deps.values()):
                entry.update(status="skipped", error="Upstream stage failed/blocked; no synthetic or heuristic fallback")
                write_json(state_path, state)
                continue
            try:
                fingerprint_data = dict(step=step, settings=settings, inputs=input_hashes(step.get("inputs", [])), implementation=implementation,
                    dependencies={key: dict(fingerprint=d.get("fingerprint"), artifacts=d.get("artifact_hashes"), status="success" if d["status"] in ("completed", "cached") else d["status"]) for key, d in deps.items()},
                    runtime=runtime)
                fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
                entry["fingerprint"] = fingerprint
                entry["input_hashes"] = fingerprint_data["inputs"]
                entry["runtime"] = runtime
                if cache and resume and _cached(old, fingerprint, step["outputs"], output):
                    entry.update(output_directory=old["output_directory"], artifact_hashes=old["artifact_hashes"], status="cached", cache_verified=True)
                    entry["current_run_wall_seconds"] = time.perf_counter()-stage_start
                    print(f"Stage {identity}: hash-verified cache", flush=True)
                    write_json(state_path, state)
                    continue
                for retry in range(step.get("retries", 0)+1):
                    number = max((a["number"] for a in history), default=0)+1
                    workspace = output/"stages"/identity/f"attempt_{number:04d}"
                    # Crash-created folders without a state entry are retained.
                    while workspace.exists():
                        number += 1
                        workspace = output/"stages"/identity/f"attempt_{number:04d}"
                    artifacts = workspace/"artifacts"
                    artifacts.mkdir(parents=True)
                    entry["last_attempt_output_directory"] = str(artifacts)
                    context_path = workspace/"context.json"
                    write_json(context_path, dict(settings=settings or {}, stages={key: value for key, value in current.items() if key != identity}))
                    replacements = {"{output}": str(artifacts), "{python}": sys.executable, "{root}": str(ROOT), "{context}": str(context_path)}
                    replacements.update({"{dep:"+key+"}": value.get("output_directory", "") for key, value in deps.items()})
                    command = []
                    for arg in step["command"]:
                        for key, value in replacements.items():
                            arg = arg.replace(key, value)
                        command.append(arg)
                    attempt = dict(number=number, status="running", command=command, artifact_directory=str(artifacts), log=str(workspace/"stage.log"), started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
                    history.append(attempt)
                    entry["status"] = "running"
                    write_json(state_path, state)
                    print(f"Stage {identity}: attempt {number}", flush=True)
                    process = None
                    monitor = None
                    try:
                        with (workspace/"stage.log").open("w", encoding="utf-8") as log:
                            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, shell=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                            monitor = ResourceMonitor(process.pid).start()
                            started, last_update = time.perf_counter(), time.perf_counter()
                            while process.poll() is None:
                                if time.perf_counter()-started > step.get("timeout_seconds", 3600):
                                    terminate_tree(process)
                                    raise TimeoutError("Stage process-tree timeout")
                                if time.perf_counter()-last_update >= 30:
                                    print(f"Stage {identity}: still running; log {workspace/'stage.log'}", flush=True)
                                    last_update = time.perf_counter()
                                time.sleep(.1)
                        code = process.returncode
                        if code == 0:
                            hashes = file_hashes(artifacts)
                            if not set(step["outputs"]) <= set(hashes) or any((artifacts/name).stat().st_size == 0 for name in step["outputs"]):
                                raise ValueError("Stage exited successfully but required artifacts are missing")
                            attempt.update(status="completed", exit_code=0)
                            entry.update(status="completed", output_directory=str(artifacts), artifact_hashes=hashes)
                            entry.pop("error", None)
                        else:
                            attempt.update(status="blocked" if code == 2 else "failed", exit_code=code, error=f"Stage exit {code}; see {workspace/'stage.log'}")
                    except (OSError, ValueError, TimeoutError) as error:
                        attempt.update(status="failed", exit_code=124 if isinstance(error, TimeoutError) else 1, error=str(error))
                    except BaseException:
                        if process is not None and process.poll() is None:
                            terminate_tree(process)
                        attempt.update(status="interrupted", error="Pipeline interrupted")
                        entry.update(status="interrupted")
                        write_json(state_path, state)
                        raise
                    finally:
                        if monitor is not None:
                            attempt["resources"] = monitor.finish()
                        attempt["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
                        write_json(state_path, state)
                    if attempt["status"] == "completed":
                        break
                    entry.update(status=attempt["status"], error=attempt.get("error", ""))
                    write_json(state_path, state)
                    if attempt.get("exit_code") not in step.get("retry_exit_codes", [1, 124]) or retry == step.get("retries", 0):
                        break
                    print(f"Stage {identity}: retrying after exit {attempt['exit_code']}; failed attempt retained", flush=True)
                entry["current_run_wall_seconds"] = time.perf_counter()-stage_start
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry.update(status="failed", error=str(error), current_run_wall_seconds=time.perf_counter()-stage_start)
            write_json(state_path, state)
        state["status"] = "completed" if all(s["status"] in ("completed", "cached") for s in current.values()) else "blocked" if any(s["status"] == "blocked" for s in current.values()) else "failed"
        state["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write_json(state_path, state)
        write_json(output/"pipeline_report.json", dict(schema="pipeline.run/1", status=state["status"], started_at=state["started_at"], ended_at=state["ended_at"], stages=list(current.values())))
        return dict(status=state["status"], stages=list(current.values()))
