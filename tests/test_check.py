"""The staging, the exit codes and the builds, with fake servers. CPU only."""

import math
import sys

import numpy as np
import pytest

from klgate import build, check, drift
from tests.test_drift import fake_run, world


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Two builds whose runs come from logits: the candidate's shifted by `shift`."""
    rng = np.random.default_rng(0)
    probes, traffic = world(rng)
    change = rng.normal(size=probes.shape)
    state = {"shift": 0.0, "launches": 0, "order": []}

    def record(python, model, args, env, out):
        state["launches"] += 1
        state["order"].append(python.name)
        seed = state["launches"]
        shift = 0.0 if python.name == "reference" else state["shift"]
        noisy = probes + shift * change + 0.01 * rng.normal(size=probes.shape)
        out.mkdir(parents=True, exist_ok=True)
        drift.save(fake_run(noisy, traffic + 0.01 * rng.normal(size=traffic.shape), seed), out / f"{seed:03}.npz")

    monkeypatch.setattr(check, "record", record)
    monkeypatch.setattr(build, "get", lambda spec, packages=(): build.Build(spec, tmp_path / spec))
    monkeypatch.setattr(build, "differences", lambda a, b: [])
    return state


def test_builds_that_are_the_same_cost_three_launches(fake, tmp_path):
    result = check.check("reference", "candidate", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("same", (2, 1), 3)
    assert fake["order"] == ["reference", "candidate", "reference"]


def test_a_change_takes_every_stage(fake, tmp_path):
    fake["shift"] = 0.1
    result = check.check("reference", "candidate", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("changed", (10, 4), 14)
    assert result["chance"] == 1 / math.comb(14, 4) <= drift.CHANCE
    # The candidate's runs lie between the reference's.
    assert "".join(name[0] for name in fake["order"]) == "rcr" "rcr" "rrcrrcrr"


def test_a_recorded_reference_costs_the_next_candidate_one_launch(fake, tmp_path):
    check.check("reference", "candidate", "m", [], {}, tmp_path, full=True)
    assert fake["launches"] == 14
    result = check.check("reference", "other", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("same", (10, 1), 15)
    fake["shift"] = 0.1
    result = check.check("reference", "shifted", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("changed", (10, 4), 19)


def test_a_check_resumes_from_the_runs_recorded(fake, tmp_path):
    check.check("reference", "candidate", "m", [], {}, tmp_path)
    assert fake["launches"] == 3
    fake["shift"] = 0.1
    # The reference holds 2 runs: a changed candidate records the other 8 and its own 4.
    result = check.check("reference", "shifted", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("changed", (10, 4), 15)
    assert "".join(name[0] for name in fake["order"][3:]) == "s" "rsr" "rrsrrsrr"


def test_the_exit_code_is_the_verdict(fake, tmp_path, monkeypatch, capsys):
    def run(*argv) -> int:
        monkeypatch.setattr(sys, "argv", ["klgate check", *argv, "--model", "m", "--out", str(tmp_path)])
        with pytest.raises(SystemExit) as e:
            check.main()
        return e.value.code

    assert run("reference", "candidate", "--json", str(tmp_path / "r.json")) == 0
    assert "Verdict: same" in capsys.readouterr().out and (tmp_path / "r.json").exists()
    fake["shift"] = 0.1
    assert run("reference", "shifted") == 1
    assert "Verdict: CHANGED" in capsys.readouterr().out
    monkeypatch.setattr(build, "get", lambda spec, packages=(): (_ for _ in ()).throw(RuntimeError("boom")))
    assert run("reference", "candidate") == 2
    assert run("only-one") == 2  # argparse


def test_a_commit_without_a_wheel_installs_from_the_nearest_one_with_its_python_laid_over(monkeypatch):
    head, base, old = "a" * 40, "b" * 40, "c" * 40
    files = [
        {"filename": "vllm/x.py", "status": "modified"},
        {"filename": "tests/test_x.py", "status": "added"},
        {"filename": "docs/x.md", "status": "modified"},
    ]
    monkeypatch.setattr(build, "has_wheel", lambda sha: sha == old)
    monkeypatch.setattr(build, "merge_base", lambda sha: base)

    def fetch(url, accept=""):
        if "commits?sha=" in url:
            return f'[{{"sha": "{base}"}}, {{"sha": "{old}"}}]'
        assert f"compare/{old}...{head}" in url
        return __import__("json").dumps({"files": files})

    monkeypatch.setattr(build, "fetch", fetch)
    assert build.source(head) == (old, files[:1])
    assert build.source(old) == (old, [])
    files.append({"filename": "csrc/attention.cu", "status": "modified"})
    with pytest.raises(ValueError, match="changes more than Python since cccccccccccc.*csrc/attention.cu.*build it yourself"):
        build.source(head)


def test_a_path_is_a_build_of_its_own_each_time(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").touch()
    a, b = build.get(str(tmp_path)), build.get(str(tmp_path / "bin" / "python"))
    assert a.python == b.python == tmp_path / "bin" / "python" and a.name != b.name
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no bin/python"):
        build.get(str(tmp_path / "empty"))


def test_laying_files_over_a_build_leaves_other_builds_alone(tmp_path, monkeypatch):
    # uv hardlinks every venv's files to one cache.
    cache, site = tmp_path / "cache.py", tmp_path / "site" / "vllm"
    site.mkdir(parents=True)
    cache.write_text("old")
    (site / "x.py").hardlink_to(cache)
    (site / "gone.py").write_text("old")

    class Reply:
        def read(self):
            return b"new"

    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": str(tmp_path / "site")})())
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url, timeout: Reply())
    build.overlay(tmp_path / "python", "a" * 40, [
        {"filename": "vllm/x.py", "status": "modified"},
        {"filename": "vllm/new/y.py", "status": "added"},
        {"filename": "vllm/gone.py", "status": "removed"},
    ])
    assert (site / "x.py").read_text() == "new" and cache.read_text() == "old"
    assert (site / "new" / "y.py").read_text() == "new" and not (site / "gone.py").exists()
