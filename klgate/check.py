"""Does a candidate vLLM build change a model's outputs beyond rerun noise?

Installs both builds, then records runs of each, alternating, every run on a fresh
server with empty caches, and compares them with klgate.drift. It starts with 2 runs of
each build and records more only while the builds differ, so builds that are the same
cost 4 server launches and a change costs 14.
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
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import requests

from klgate import build, drift

ARGS = "--max-model-len 2048 --max-num-seqs 32"
# FlashInfer's sampler JIT-compiles with nvcc, which many machines lack.
ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
STARTUP = 1800  # seconds a server may take to start: big models load slowly
EXIT = {"same": 0, "changed": 1}  # 2 is an error


def main() -> None:
    ap = argparse.ArgumentParser(prog="klgate check", description=__doc__.split("\n\n")[0])
    ap.add_argument("builds", nargs="*", metavar="build",
                    help="the reference, then the candidate: a vLLM tag, branch or commit, or the path of a "
                         "Python environment with vLLM installed")
    ap.add_argument("--pr", type=int, help="a vLLM pull request to check against its merge base, instead of two builds")
    ap.add_argument("--model", required=True)
    ap.add_argument("--args", default=ARGS, help=f"server arguments (default: {ARGS})")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="server environment")
    ap.add_argument("--out", type=Path, default=Path("runs"), help="where runs are kept (default: runs)")
    ap.add_argument("--full", action="store_true",
                    help="record 7 runs of each build even if they read the same sooner: the most sensitive check")
    ap.add_argument("--json", type=Path, help="also write the result here")
    a = ap.parse_args()
    if (a.pr is None) == (len(a.builds) != 2):
        ap.error("pass a reference and a candidate build, or --pr")
    if bad := [kv for kv in a.env if "=" not in kv]:
        ap.error(f"--env takes K=V, not {bad[0]}")
    env = ENV | dict(kv.split("=", 1) for kv in a.env)
    try:
        reference, candidate = a.builds or build.pull(a.pr)
        result = check(reference, candidate, a.model, shlex.split(a.args), env, a.out, full=a.full)
    except Exception as e:  # the exit code is the verdict, so no failure may exit 1
        print(e if isinstance(e, ValueError) else repr(e), file=sys.stderr)
        raise SystemExit(2) from None
    print(drift.render(result))
    if differ := result["packages"]:
        more = f" and {len(differ) - 5} more" if len(differ) > 5 else ""
        print(f"The builds' other packages differ too, which can change outputs: {', '.join(differ[:5])}{more}")
    if a.json:
        a.json.write_text(json.dumps(drift.jsonable(result), indent=2) + "\n")
    raise SystemExit(EXIT[result["verdict"]])


def check(reference: str, candidate: str, model: str, args: list[str], env: dict, out: Path,
          packages: Sequence[str] = (), full: bool = False) -> dict:
    """Compare the builds stage by stage, recording only the runs missing, until they
    read the same or changed; with full, through the last stage whatever they read."""
    builds = [build.get(spec, packages) for spec in (reference, candidate)]
    key = hashlib.sha256(json.dumps([model, args, env, list(packages)]).encode()).hexdigest()[:8]
    dirs = [out / f"{b.name}-{key}" for b in builds]
    for stage in drift.STAGES:
        # Alternating builds spreads any drift in the machine over both.
        for i in range(stage):
            for b, d in zip(builds, dirs):
                if len(list(d.glob("*.npz"))) <= i:
                    record(b.python, model, args, env, d)
        result = drift.compare(*([drift.load(p) for p in sorted(d.glob("*.npz"))[:stage]] for d in dirs))
        if result["verdict"] != "more" and not full:
            break
    return result | {"packages": build.differences(*(b.python for b in builds))}


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
        print(f"recorded {path}", file=sys.stderr, flush=True)
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
            raise ValueError(f"the server died; see {log}. Rerun to resume: recorded runs are kept")
        try:
            if requests.get(f"{url}/health", timeout=2).ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise ValueError(f"the server didn't start in {STARTUP} s; see {log}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
