from klgate import bench


def test_the_bench_reads_as_the_readme_says():
    caught = {
        section.splitlines()[0]: {line.split()[0] for line in section.splitlines()[1:] if " differs " in line or " CHANGED " in line}
        for section in bench.table().split("\n\n")
    }
    known, unseen, same = (caught[bench.KINDS[kind]] for kind in ("known", "unseen", "same"))
    assert len(known) == 6 and len(unseen) == 6 and not same
