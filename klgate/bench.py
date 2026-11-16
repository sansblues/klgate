"""Replay vLLM bug fixes, and pairs of commits with identical engine code, through
klgate check.

Each bench/entries/<id>.toml names a reference build (the fix, or one of two
identical-code commits), a candidate build (the buggy parent, or the other commit), a
model and server settings. Its kind is "known" (a bug klgate was built on), "unseen" (a
bug picked by a rule fixed before running) or "same" (identical engine code).

Usage:
  klgate bench                             # every result in data/, decided again; CPU, seconds
  klgate bench bench/entries/<id>.toml     # replay one entry here, adding to data/<id>.json
"""

import argparse
import json
import shlex
import tomllib
from pathlib import Path

import numpy as np

from klgate import check, drift

ROOT = Path(__file__).resolve().parent.parent
ENTRIES, DATA = ROOT / "bench" / "entries", ROOT / "data"
KINDS = {
    "known": "Bugs klgate was built on",
    "unseen": "Bugs it never saw, picked by a rule fixed before running",
    "same": "Identical engine code: any CHANGED is a false alarm",
}


def main() -> None:
    ap = argparse.ArgumentParser(prog="klgate bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("entry", nargs="?", type=Path,
                    help="bench/entries/<id>.toml to replay here; omit to decide every result in data/ again")
    a = ap.parse_args()
    print(replay(a.entry) if a.entry else table())


def replay(path: Path) -> str:
    e = tomllib.loads(path.read_text())
    args, env = shlex.split(e["args"]), check.ENV | e.get("env", {})
    result = check.check(e["reference"], e["candidate"], e["model"], args, env, check.RUNS,
                         ROOT / "runs", e.get("packages", []))
    out = DATA / f"{e['id']}.json"
    replays = json.loads(out.read_text()) if out.exists() else []
    if all(r["matrices"] != result["matrices"] for r in replays):  # not the same runs again
        replays.insert(0, {"gpu": drift.gpu_name(), "setup": {"server_args": args, "env": env},
                           "matrices": result["matrices"]})
        out.write_text(json.dumps(replays, indent=1) + "\n")
    return drift.render(result)


def table() -> str:
    entries = [tomllib.loads(p.read_text()) for p in sorted(ENTRIES.glob("*.toml"))]
    lines = []
    for kind, title in KINDS.items():
        lines += ["", title]
        for e in (e for e in entries if e["kind"] == kind):
            for r in json.loads((DATA / f"{e['id']}.json").read_text()):
                matrices = {part: np.array(m) for part, m in r["matrices"].items()}
                d = drift.decide(matrices, len(matrices["probes"]) // 2)
                part = max(d["ratios"], key=d["ratios"].get)
                lines.append(
                    f"  {e['id']:20} {r['gpu'].removeprefix('NVIDIA '):18} "
                    f"{'CHANGED' if d['changed'] else 'same':8} {times(d['score']):>9} {part:8} "
                    f"needed {times(d['needed']):>6}  p {d['p']:.3f}  {r.get('note', '')}".rstrip())
    return "\n".join(lines[1:])


def times(x: float) -> str:
    return f"{x:,.2f}×" if np.isfinite(x) else "∞×"


if __name__ == "__main__":
    main()
