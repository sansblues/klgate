"""A vLLM build to launch servers from: a commit installed from vLLM's per-commit wheels,
or a Python environment you built yourself.

A commit without a wheel, such as a pull request's, installs from the nearest earlier
commit on main that has one, with the commit's Python files laid over it. That covers
changes to Python only; a commit that changes compiled code has to be built by hand.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

BuildsDir = Path(os.environ.get("KLGATE_BUILDS", Path.home() / ".cache" / "klgate" / "builds"))
WheelIndex = "https://wheels.vllm.ai/{sha}/vllm/"
Api = "https://api.github.com/repos/vllm-project/vllm/"
Raw = "https://raw.githubusercontent.com/vllm-project/vllm/{sha}/{path}"
Python = "3.12"
Sha = re.compile(r"[0-9a-f]{40}")
Timeout = 60  # seconds per request
Ancestors = 30  # commits to look back through for one with a wheel
# Files a commit may change without changing what a server computes.
Ignored = re.compile(r"(tests|docs|examples|benchmarks|tools|docker|recipes|\.[^/]+)/|[^/]*\.md$|\.[^/]*$|(DCO|LICENSE|mkdocs\.yaml)$")
# Files that can be laid over an installed wheel.
Overlaid = re.compile(r"vllm/.*\.(py|json|jinja|yaml|yml|txt)$")
# Packages whose version is likeliest to change what a server computes.
Numeric = re.compile(r"torch|triton|cuda|nvidia|flashinfer|numpy|transformers|tokenizers|tvm|tilelang|numba|safetensors|compressed")


class Build(NamedTuple):
    name: str  # names the directory its runs are kept in
    python: Path


def get(spec: str, packages: Sequence[str] = ()) -> Build:
    """The build spec names: a Python environment's directory or interpreter, or a vLLM
    tag, branch or commit to install."""
    path = Path(spec).expanduser()
    if path.exists():
        python = path / "bin" / "python" if path.is_dir() else path
        if not python.exists():
            raise ValueError(f"{path} holds no bin/python: pass a Python environment with vLLM installed")
        # An environment can change in place, so its runs are never reused.
        digest = hashlib.sha256(f"{python.resolve()} {time.time_ns()}".encode()).hexdigest()
        return Build(f"local-{digest[:12]}", python)
    sha = resolve(spec)
    return Build(sha[:12], install(sha, packages))


def resolve(ref: str) -> str:
    """The full commit sha of a vLLM tag, branch or commit, short or full."""
    if Sha.fullmatch(ref):
        return ref
    try:
        sha = fetch(Api + "commits/" + urllib.parse.quote(ref, safe=""), accept="application/vnd.github.sha")
    except urllib.error.HTTPError as e:
        raise ValueError(
            f"can't find {ref!r} in vLLM's GitHub repo ({e.code}): pass a tag like v0.20.0, a branch, a commit sha, "
            "or the path of a Python environment"
        ) from None
    if not Sha.fullmatch(sha):
        raise ValueError(f"GitHub answered {sha[:80]!r} for {ref!r}, not a commit sha")
    return sha


def pull(number: int) -> tuple[str, str]:
    """A pull request's merge base with main and its head commit."""
    try:
        head = json.loads(fetch(f"{Api}pulls/{number}"))["head"]["sha"]
    except urllib.error.HTTPError as e:
        raise ValueError(f"can't find pull request {number} in vLLM's GitHub repo ({e.code})") from None
    return merge_base(head), head


def install(sha: str, packages: Sequence[str] = ()) -> Path:
    """The python of the build's venv, installing the build first if needed, then any
    packages it needs pinned, such as a transformers that knows a newer model."""
    if not Sha.fullmatch(sha):
        raise ValueError(f"need a full 40-character commit sha, got {sha!r}")
    # Pinned packages get their own venv, so a plain build stays plain.
    venv = BuildsDir / "-".join([sha, *packages])
    python = venv / "bin" / "python"
    if not (venv / ".done").exists():
        base, files = source(sha)
        wheel = find_wheel(base)
        # Its dependencies as they were on the day of the commit, so that two builds
        # installed months apart don't differ by what was released in between.
        day = json.loads(fetch(f"{Api}commits/{base}"))["commit"]["committer"]["date"]
        subprocess.run(["uv", "venv", "-q", "--clear", "--python", Python, str(venv)], check=True)
        # Not quiet: a build is gigabytes of downloads, and uv says why one fails.
        pip = ["uv", "pip", "install", "--python", str(python), "--exclude-newer", day, wheel]
        if subprocess.run(pip).returncode:
            raise ValueError(f"couldn't install {sha[:12]}: uv's message is above. Rerun to try again")
        overlay(python, sha, files)
        (venv / ".done").write_text(f"{wheel}\n" + "".join(f"{f['status']} {f['filename']}\n" for f in files))
    if packages:
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), *packages], check=True)
    return python


def source(sha: str) -> tuple[str, list[dict]]:
    """The commit whose wheel to install for sha, and the files to lay over it."""
    if has_wheel(sha):
        return sha, []
    start = merge_base(sha)
    older = json.loads(fetch(f"{Api}commits?sha={start}&per_page={Ancestors}"))
    base = next((c["sha"] for c in older if has_wheel(c["sha"])), None)
    if base is None:
        raise ValueError(
            f"vLLM has no prebuilt wheel for {sha[:12]} or the {Ancestors} commits on main before it: "
            "build it yourself and pass the path of its Python environment"
        )
    files = changed(base, sha)
    if compiled := [f["filename"] for f in files if not Overlaid.match(f["filename"])]:
        raise ValueError(
            f"{sha[:12]} has no prebuilt wheel, and it changes more than Python since {base[:12]}, the nearest "
            f"commit with one ({', '.join(compiled[:5])}): build it yourself and pass the path of its Python "
            "environment"
        )
    return base, files


def changed(base: str, sha: str) -> list[dict]:
    """The files that differ between two commits and can change what a server computes."""
    files, page = [], 1
    while True:
        batch = json.loads(fetch(f"{Api}compare/{base}...{sha}?per_page=100&page={page}"))["files"]
        files += batch
        if len(batch) < 100:
            break
        page += 1
    if len(files) >= 300:  # GitHub lists no more
        raise ValueError(f"{sha[:12]} changes too many files since {base[:12]} to lay over its wheel")
    return [f for f in files if not Ignored.match(f["filename"])]


def overlay(python: Path, sha: str, files: list[dict]) -> None:
    """Make the installed vLLM's files those of sha."""
    if not files:
        return
    query = "import sysconfig; print(sysconfig.get_paths()['platlib'])"
    site = Path(subprocess.run([python, "-c", query], check=True, capture_output=True, text=True).stdout.strip())
    for f in files:
        if f["status"] in ("removed", "renamed"):
            (site / f.get("previous_filename", f["filename"])).unlink(missing_ok=True)
        if f["status"] != "removed":
            path = site / f["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            url = Raw.format(sha=sha, path=urllib.parse.quote(f["filename"]))
            new = urllib.request.urlopen(url, timeout=Timeout).read()
            # uv hardlinks a venv's files to its cache, which other builds share:
            # writing in place would change them all.
            path.unlink(missing_ok=True)
            path.write_bytes(new)


def differences(reference: Path, candidate: Path) -> list[str]:
    """The installed packages that differ between two environments, vLLM aside, those
    likeliest to change outputs first."""

    def packages(python: Path) -> dict[str, str]:
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(python)], capture_output=True, text=True)
        pairs = (re.split(r"==| @ ", line, maxsplit=1) for line in freeze.stdout.splitlines() if line.strip())
        return {p[0].lower(): p[1] if len(p) > 1 else "" for p in pairs}

    a, b = packages(reference), packages(candidate)
    return [
        f"{name} {a.get(name, 'absent')} → {b.get(name, 'absent')}"
        for name in sorted(a.keys() | b.keys(), key=lambda name: (not Numeric.search(name), name))
        if name != "vllm" and a.get(name) != b.get(name)
    ]


def merge_base(sha: str) -> str:
    try:
        return json.loads(fetch(f"{Api}compare/main...{sha}"))["merge_base_commit"]["sha"]
    except urllib.error.HTTPError as e:
        raise ValueError(f"can't find {sha[:12]} in vLLM's GitHub repo ({e.code})") from None


def has_wheel(sha: str) -> bool:
    try:
        return bool(wheels(sha))
    except urllib.error.HTTPError:
        return False


def find_wheel(sha: str) -> str:
    names = wheels(sha)
    if not names:
        raise ValueError(f"no {platform.machine()} wheel listed at {WheelIndex.format(sha=sha)}")
    return urllib.parse.urljoin(WheelIndex.format(sha=sha), names[0])


def wheels(sha: str) -> list[str]:
    html = urllib.request.urlopen(WheelIndex.format(sha=sha), timeout=Timeout).read().decode()
    return re.findall(rf'href="([^"]*{platform.machine()}\.whl)"', html)


def fetch(url: str, accept: str = "application/vnd.github+json") -> str:
    headers = {"Accept": accept}
    if token := os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=Timeout).read().decode().strip()
