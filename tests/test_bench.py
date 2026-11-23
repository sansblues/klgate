import numpy as np

from klgate import bench


def test_the_bench_reads_as_the_readme_says():
    caught = {
        section.splitlines()[0]: {
            line.split()[0]
            for line in section.splitlines()[1:]
            if " differs " in line or " CHANGED " in line
        }
        for section in bench.table().split("\n\n")
    }
    known, unseen, same = (
        caught[bench.KINDS[kind]] for kind in ("known", "unseen", "same")
    )
    assert len(known) == 5 and len(unseen) == 5 and not same


def test_a_staged_check_costs_3_launches_or_9():
    rng = np.random.default_rng(0)
    same = {"probes": 1 + rng.random((9, 9))}
    changed = {"probes": same["probes"].copy()}
    changed["probes"][:5, 5:] += 9
    assert bench.staged(changed, 5) == (1.0, 9.0)
    found, launches = bench.staged(same, 5)
    assert found < 0.02 and 3 <= launches < 6
    # A replay with more runs than the stages ask for: they decide on as many as they
    # ask for, and cost that.
    big = {"probes": 1 + rng.random((14, 14))}
    big["probes"][:10, 10:] += 9
    assert bench.staged(big, 10) == (1.0, 9.0)
