"""Does a candidate vLLM build change a model's outputs beyond rerun noise?

Installs both builds, then records runs of each, alternating, every run on a fresh
server with empty caches, and compares them with klgate.drift.
"""

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import requests

from klgate import build, drift

ARGS = "--max-model-len 2048 --max-num-seqs 32"
# FlashInfer's sampler JIT-compiles with nvcc, which many machines lack.
ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
RUNS = 5
STARTUP = 1800  # seconds a server may take to start: big models load slowly


def main() -> None:
    ap = argparse.ArgumentParser(prog="klgate check", description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", help="vLLM tag, branch or commit")
    ap.add_argument("candidate", help="vLLM tag, branch or commit")
    ap.add_argument("--model", required=True)
    ap.add_argument("--args", default=ARGS, help=f"server arguments (default: {ARGS})")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="server environment")
    ap.add_argument("--runs", type=int, default=RUNS, help=f"runs per build, 4 to 8 (default: {RUNS})")
    ap.add_argument("--out", type=Path, default=Path("runs"), help="where runs are kept (default: runs)")
    a = ap.parse_args()
    if not 4 <= a.runs <= drift.MAX_RUNS:
        ap.error(f"--runs must be 4 to {drift.MAX_RUNS}")
    if bad := [kv for kv in a.env if "=" not in kv]:
        ap.error(f"--env takes K=V, not {bad[0]}")
    env = ENV | dict(kv.split("=", 1) for kv in a.env)
    try:
        result = check(build.resolve(a.reference), build.resolve(a.candidate), a.model, shlex.split(a.args), env,
                       a.runs, a.out)
    except ValueError as e:
        raise SystemExit(e) from None
    print(drift.render(result))


def check(reference: str, candidate: str, model: str, args: list[str], env: dict, runs: int, out: Path,
          packages: Sequence[str] = ()) -> dict:
    """Compare `runs` runs of each build, recording only the ones missing."""
    shas = (reference, candidate)
    pythons = [build.install(sha, packages) for sha in shas]
    key = hashlib.sha256(json.dumps([model, args, env, list(packages)]).encode()).hexdigest()[:8]
    dirs = [out / f"{sha[:12]}-{key}" for sha in shas]
    # Alternating builds spreads any drift in the machine over both.
    for i in range(runs):
        for python, d in zip(pythons, dirs):
            if len(list(d.glob("*.npz"))) <= i:
                record(python, model, args, env, d)
    return drift.compare(*([drift.load(p) for p in sorted(d.glob("*.npz"))[:runs]] for d in dirs))


def record(python: Path, model: str, args: list[str], env: dict, out: Path) -> None:
    """One run, on a new server with empty caches."""
    seed = secrets.randbelow(2**31)
    path = out / f"{seed}.npz"
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    cache = tempfile.mkdtemp(prefix="klgate-")
    with open(path.with_suffix(".log"), "w") as log:
        server = subprocess.Popen(
            [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port),
             "--disable-uvicorn-access-log", *args],
            stdout=log, stderr=subprocess.STDOUT,
            # The build's bin holds tools its kernels may call, such as ninja.
            env=os.environ | {"PYTHONPATH": "", "PATH": f"{python.parent}:{os.environ['PATH']}"} | env
            | drift.fresh_caches(cache))
    try:
        url = f"http://127.0.0.1:{port}"
        wait(server, url, path.with_suffix(".log"))
        run = asyncio.run(drift.record(url, seed, {"server_args": args, "env": env}))
        drift.save(run, path)
        print(f"recorded {path}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(120)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(cache, ignore_errors=True)


def wait(server: subprocess.Popen, url: str, log: Path) -> None:
    start = time.time()
    while time.time() - start < STARTUP:
        if server.poll() is not None:
            raise SystemExit(f"the server died; see {log}. Rerun to resume: recorded runs are kept")
        try:
            if requests.get(f"{url}/health", timeout=2).ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise SystemExit(f"the server didn't start in {STARTUP} s; see {log}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
