from klgate import bench


def test_the_bench_reads_as_the_readme_says():
    known, unseen, same = (
        {line.split()[0] for line in section.splitlines()[1:] if " differs " in line or " CHANGED " in line}
        for section in bench.table().split("\n\n")
    )
    assert len(known) == 6 and len(unseen) == 6 and not same
