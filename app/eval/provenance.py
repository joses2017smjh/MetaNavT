"""What produced a neural result: model ids and revisions, device, library versions.

Filesystem-only: the HuggingFace revision is read from the hub cache's
refs/main file (never from the network), so a results file can name the exact
weights that scored it and a reader can pin them.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for env in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def hf_revision(model_name: str) -> str | None:
    """Commit hash of the cached snapshot for `model_name`, or None if it is not cached."""
    slug = "models--" + model_name.replace("/", "--")
    for root in hf_cache_roots():
        ref = root / slug / "refs" / "main"
        try:
            if ref.exists():
                return ref.read_text().strip() or None
        except OSError:
            continue
        snapshots = root / slug / "snapshots"
        try:
            if snapshots.is_dir():
                names = sorted(p.name for p in snapshots.iterdir())
                if names:
                    return names[0]
        except OSError:
            continue
    return None


def model_provenance(model_name: str | None, role: str) -> dict:
    if not model_name or model_name in {"hash", "tfidf", "overlap", "none"}:
        return {"role": role, "model": model_name, "revision": None, "neural": False}
    name = model_name.split(":", 1)[1] if model_name.startswith("st:") else model_name
    return {"role": role, "model": name, "revision": hf_revision(name), "neural": True}


def device_info(requested: str = "auto") -> dict:
    """torch / CUDA facts for the results header. Works without torch installed."""
    info: dict = {
        "requested": requested,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
        use_cuda = cuda and requested in {"auto", "cuda"}
        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": cuda,
                "device": "cuda" if use_cuda else "cpu",
                "device_name": torch.cuda.get_device_name(0) if use_cuda else platform.processor() or "cpu",
                "threads": torch.get_num_threads(),
            }
        )
    except Exception:  # torch missing: hash / tfidf only
        info.update({"torch": None, "cuda_available": False, "device": "cpu", "device_name": "cpu", "threads": None})
    try:
        import sentence_transformers  # type: ignore

        info["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        info["sentence_transformers"] = None
    return info


def command_line() -> str:
    """The command that produced a results file, with the env flags that change it."""
    flags = {k: os.environ[k] for k in ("BGE_ALLOW_DOWNLOAD", "HF_HOME", "CUDA_VISIBLE_DEVICES") if k in os.environ}
    prefix = " ".join(f"{k}={v}" for k, v in flags.items())
    return (prefix + " " if prefix else "") + "python -m " + " ".join([_module_name(sys.argv[0]), *sys.argv[1:]])


def _module_name(argv0: str) -> str:
    p = Path(argv0)
    if p.suffix == ".py":
        parts = list(p.with_suffix("").parts)
        if "app" in parts:
            return ".".join(parts[parts.index("app"):])
    return p.name


def git_sha(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"
