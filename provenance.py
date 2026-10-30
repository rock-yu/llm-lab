"""Small, dependency-free records shared by training and evaluation."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGES = ("torch", "python-dateutil", "transformers", "peft", "accelerate")


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def decision_snapshot(path):
    text = Path(path).read_text()
    if not text.strip():
        raise ValueError("the decision document must not be empty")
    return {**file_identity(path), "text": text}


def make_manifest(args, data_files, device, started, **details):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None  # optional dependency not installed
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True)
    hardware = {"machine": platform.machine(), "processor": platform.processor(),
                "platform": platform.platform(), "device": str(device)}
    torch = sys.modules.get("torch")
    if torch is not None:
        hardware["torch_threads"] = torch.get_num_threads()
        if str(device).startswith("cuda"):
            hardware["accelerator"] = torch.cuda.get_device_name(device)
    return {
        "schema_version": 1,
        "configuration": vars(args),
        "datasets": [file_identity(p) for p in data_files],
        "source": {
            "commit": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "python_files": [file_identity(p) for p in sorted(ROOT.glob("*.py"))],
        },
        "python": sys.version,
        "dependencies": versions,
        "hardware": hardware,
        "elapsed_seconds": time.perf_counter() - started,
        **details,
    }


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
