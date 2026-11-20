"""Replay vLLM bug fixes, and pairs of commits with identical engine code, through
klgate check.

Each bench/entries/<id>.toml names a reference build (the fix, or one of two
identical-code commits), a candidate build (the buggy parent, or the other commit), a
model and server settings. Its kind is "known" (a bug klgate was built on), "unseen" (a
bug picked by a rule fixed before running) or "same" (identical engine code).

A replay records every stage's runs, so the table can say what a check would read
whichever of them it happened to record first.

Usage:
  klgate bench                             # every result in data/, decided again; CPU, seconds
  klgate bench bench/entries/<id>.toml     # replay one entry here, adding to data/<id>.json
"""

import argparse
import itertools
import json
import shlex
import sys
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
WORDS = {"same": "same", "changed": "CHANGED", "more": "differs"}


def main() -> None:
    ap = argparse.ArgumentParser(prog="klgate bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("entry", nargs="?", type=Path,
                    help="bench/entries/<id>.toml to replay here; omit to decide every result in data/ again")
    a = ap.parse_args()
    try:
        print(replay(a.entry) if a.entry else table())
    except ValueError as e:
        print(e, file=sys.stderr)
        raise SystemExit(2) from None


def replay(path: Path) -> str:
    e = tomllib.loads(path.read_text())
    args, env = shlex.split(e["args"]), check.ENV | e.get("env", {})
    result = check.check(e["reference"], e["candidate"], e["model"], args, env, ROOT / "runs",
                         e.get("packages", []), full=True)
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
                n = len(matrices["probes"]) // 2
                d = drift.decide(matrices, n)
                reached, launches = staged(matrices)
                part = max(d["ratios"], key=d["ratios"].get)
                lines.append(
                    f"  {e['id']:20} {r['gpu'].removeprefix('NVIDIA '):18} {n} runs  "
                    f"{WORDS[d['verdict']]:8} {times(d['score']):>9} {part:8} needed {times(d['needed']):>6}  "
                    f"reached in {100 * reached:3.0f} % of orders, {launches:4.1f} launches  {r.get('note', '')}".rstrip())
    return "\n".join(lines[1:])


def staged(matrices: dict[str, np.ndarray]) -> tuple[float, float]:
    """How often a staged check gets as far as all of these runs, over every choice of
    the runs it records first, and the server launches it takes on average."""
    n = len(matrices["probes"]) // 2
    stages = [s for s in drift.STAGES if s < n]

    def differ(ref: tuple[int, ...], cand: tuple[int, ...]) -> bool:
        keep = [*ref, *(n + i for i in cand)]
        sub = {part: m[np.ix_(keep, keep)] for part, m in matrices.items()}
        return drift.decide(sub, len(ref))["verdict"] != "same"

    def walk(stage: int, ref: tuple[int, ...], cand: tuple[int, ...]) -> tuple[float, float]:
        """From runs ref and cand at stages[stage]: the chance of reaching the end, and
        the launches spent."""
        size = stages[stage] if stage < len(stages) else n
        groups = [[g for g in itertools.combinations(range(n), size) if set(have) <= set(g)] for have in (ref, cand)]
        reached = launches = 0.0
        for r, c in itertools.product(*groups):
            if size == n or not differ(r, c):
                reached, launches = reached + (size == n), launches + 2 * size
            else:
                a, b = walk(stage + 1, r, c)
                reached, launches = reached + a, launches + b
        count = len(groups[0]) * len(groups[1])
        return reached / count, launches / count

    return walk(0, (), ())


def times(x: float) -> str:
    return f"{x:,.2f}×" if np.isfinite(x) else "∞×"


if __name__ == "__main__":
    main()
