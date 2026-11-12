"""What produced a result: the command, model ids and revisions, device, library versions.

Filesystem-only: the HuggingFace revision is read from the hub cache's
refs/main file (never from the network), so a results file can name the exact
weights that scored it and a reader can pin them. A model served by Ollama is
named by a tag, which can point at other weights after a pull; its revision is
the manifest digest the server's /api/tags reports for that tag
(ollama_model_provenance), asked of the server that ran the model.

Results files are published with the repository, and this module is where
their writers take the command and the machine description from, so neither
carries a host name, a user name or an absolute path:

- command_line() keeps `python -m <module>` and every argument. An absolute
  path (an argument, `--flag=/path`, or the value of an environment flag such
  as HF_HOME) becomes repository-relative when it is inside the repository and
  `<path>/<file name>` when it is not; the user name and the host name become
  `<user>` and `<host>` wherever they appear.
- device_info() does not record the host name at all: it is dropped, not
  hashed (a hash of a guessable cluster node name would not hide it). The
  platform, Python version, CPU count and device name describe the machine.
- scrub() applies the same rules to any other value a writer records, such as
  a service's /health block or a path field. scrub_names() replaces only the
  user and host names, for text that must otherwise stay as written (the
  injection harness's tool results, where a path is corpus text or a model's).
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH_PLACEHOLDER, USER_PLACEHOLDER, HOST_PLACEHOLDER = "<path>", "<user>", "<host>"
# The environment flags that change a result, recorded in front of the command.
COMMAND_ENV_FLAGS = ("BGE_ALLOW_DOWNLOAD", "HF_HOME", "CUDA_VISIBLE_DEVICES")
# Account names that identify no one (CI runners, containers); replacing them would only mangle arguments.
GENERIC_ACCOUNTS = frozenset({"root", "runner", "user", "admin", "ubuntu", "nobody", "jovyan"})
_SEGMENT = r"[^\s/'\",;:=()\[\]{}<>]"
# An absolute (or home-relative) path inside a longer string. Not the "//host" or "/route" of a URL:
# the character before the slash may not be a word character, ".", ":", "/", "~" or "-".
_ABS_PATH_RE = re.compile(rf"(?<![\w.:/~<>-])(?:~/|/){_SEGMENT}+(?:/{_SEGMENT}*)*")


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


def ollama_base_url(base_url: str | None = None) -> str:
    return (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")


def _ollama_get(route: str, base_url: str | None, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(ollama_base_url(base_url) + route, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except (OSError, ValueError):  # not reachable, refused, HTTP error, not JSON
        return {}
    return payload if isinstance(payload, dict) else {}


def ollama_tags(base_url: str | None = None, timeout: float = 10.0) -> dict[str, dict]:
    """The Ollama server's /api/tags entries keyed by their name and model fields; {} when it does not answer."""
    out: dict[str, dict] = {}
    for entry in _ollama_get("/api/tags", base_url, timeout).get("models") or []:
        for key in (entry.get("name"), entry.get("model")):
            if key:
                out.setdefault(key, entry)
    return out


def ollama_version(base_url: str | None = None, timeout: float = 10.0) -> str | None:
    return _ollama_get("/api/version", base_url, timeout).get("version")


def ollama_tag_entry(model_name: str, tags: Mapping[str, Mapping]) -> Mapping:
    """The /api/tags entry Ollama resolves `model_name` to ({} when it is not listed); a bare name means name:latest."""
    return tags.get(model_name) or ({} if ":" in model_name else tags.get(f"{model_name}:latest")) or {}


def ollama_model_provenance(model_name: str, role: str, tags: Mapping[str, Mapping]) -> dict:
    """The model block of a model Ollama served: `revision` is the digest of the manifest the tag resolved to
    (None when the server does not list the tag), so the weights are pinned, not only named."""
    entry = ollama_tag_entry(model_name, tags)
    details = entry.get("details") or {}
    return {"role": role, "model": model_name, "revision": entry.get("digest"), "neural": True, "backend": "ollama",
            "family": details.get("family"), "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"), "size_bytes": entry.get("size")}


def device_info(requested: str = "auto") -> dict:
    """torch / CUDA facts for the results header. Works without torch installed."""
    info: dict = {
        "requested": requested,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
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


def model_device(model) -> str | None:
    """Where a loaded model's weights are (SentenceTransformer / CrossEncoder expose .device)."""
    for attr in ("device", "_target_device"):
        dev = getattr(model, attr, None)
        if dev is not None:
            return str(dev)
    inner = getattr(model, "model", None)  # CrossEncoder wraps a transformers model
    if inner is not None and getattr(inner, "device", None) is not None:
        return str(inner.device)
    return None


def command_line() -> str:
    """The command that produced a results file, with the env flags that change it, sanitized (see the module docstring)."""
    return sanitize_command(sys.argv, os.environ)


def sanitize_command(argv: Sequence[str], env: Mapping[str, str], *, root: Path | None = None) -> str:
    """`FLAG=value ... python -m <module> <args>` with every path, user name and host name made publishable."""
    flags = [f"{k}={scrub(str(env[k]), root=root)}" for k in COMMAND_ENV_FLAGS if k in env]
    module = _module_name(argv[0]) if argv else "-"
    return " ".join([*flags, "python", "-m", module, *(scrub(str(a), root=root) for a in argv[1:])])


def sanitize_path(value: str, *, root: Path | None = None) -> str:
    """An absolute or ~/ path as it may be published: relative to the repository when it is inside it
    (a symlinked prefix counts), else `<path>/<file name>`. Any other value only loses user and host names."""
    text = str(value)
    if not text.startswith(("/", "~/")):
        return _scrub_names(text)
    root = Path(root) if root is not None else REPO_ROOT
    path = Path(os.path.expanduser(text))
    for candidate in (path, path.resolve()):
        for base in (root, root.resolve()):
            try:
                return candidate.relative_to(base).as_posix()
            except ValueError:
                continue
    name = _scrub_names(path.name)
    return f"{PATH_PLACEHOLDER}/{name}" if name else PATH_PLACEHOLDER


def scrub(value: Any, *, root: Path | None = None) -> Any:
    """`value` with every absolute path (sanitize_path), user name and host name replaced, in strings at any depth."""
    if isinstance(value, str):
        return _scrub_names(_ABS_PATH_RE.sub(lambda m: sanitize_path(m.group(0), root=root), value))
    if isinstance(value, dict):
        return {k: scrub(v, root=root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v, root=root) for v in value]
    return value


def scrub_names(value: Any) -> Any:
    """`value` with every user name and host name replaced, in strings at any depth, and paths left as written:
    for text that must stay as a program or a model saw it apart from those names (scrub() rewrites every path)."""
    if isinstance(value, str):
        return _scrub_names(value)
    if isinstance(value, dict):
        return {k: scrub_names(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_names(v) for v in value]
    return value


def user_names() -> set[str]:
    """The names of the account running this process (environment, passwd, home directory), generic ones left out."""
    names = {os.environ.get(k, "") for k in ("USER", "LOGNAME", "USERNAME")}
    for get in (getpass.getuser, lambda: Path.home().name):
        try:
            names.add(get())
        except Exception:
            pass
    return {n for n in names if len(n) >= 3 and n.lower() not in GENERIC_ACCOUNTS}


def host_names() -> set[str]:
    """This machine's names: the node name as the OS reports it and its first label."""
    names: set[str] = set()
    for get in (platform.node, socket.gethostname):
        try:
            name = get()
        except Exception:
            continue
        if name:
            names.update({name, name.split(".")[0]})
    return {n for n in names if len(n) >= 3 and n.lower() != "localhost"}


def _scrub_names(text: str) -> str:
    for name in sorted(host_names(), key=len, reverse=True):  # the full name before its first label
        text = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", HOST_PLACEHOLDER, text)
    for name in sorted(user_names(), key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", USER_PLACEHOLDER, text)
    return text


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
