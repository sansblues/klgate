"""klgate check <reference> <candidate> --model NAME   compare two vLLM builds
klgate bench [bench/entries/<id>.toml]                decide the replays again, or replay one"""

import sys

from klgate import bench, check


def main() -> None:
    commands = {"check": check.main, "bench": bench.main}
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in commands:
        print(__doc__)
        raise SystemExit(name not in ("-h", "--help"))
    del sys.argv[1]
    commands[name]()


if __name__ == "__main__":
    main()
