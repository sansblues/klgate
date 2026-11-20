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
    state = {"shift": 0.0, "launches": 0}

    def record(python, model, args, env, out):
        state["launches"] += 1
        seed = state["launches"]
        shift = 0.0 if python.name == "reference" else state["shift"]
        noisy = probes + shift * change + 0.01 * rng.normal(size=probes.shape)
        out.mkdir(parents=True, exist_ok=True)
        drift.save(fake_run(noisy, traffic + 0.01 * rng.normal(size=traffic.shape), seed), out / f"{seed:03}.npz")

    monkeypatch.setattr(check, "record", record)
    monkeypatch.setattr(build, "get", lambda spec, packages=(): build.Build(spec, tmp_path / spec))
    monkeypatch.setattr(build, "differences", lambda a, b: [])
    return state


def test_builds_that_are_the_same_cost_four_launches(fake, tmp_path):
    result = check.check("reference", "candidate", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("same", 2, 4)


def test_a_change_takes_every_stage(fake, tmp_path):
    fake["shift"] = 0.1
    result = check.check("reference", "candidate", "m", [], {}, tmp_path)
    assert (result["verdict"], result["runs"], fake["launches"]) == ("changed", 7, 14)
    assert result["chance"] == 2 / math.comb(14, 7) <= drift.CHANCE


def test_recorded_runs_are_reused(fake, tmp_path):
    check.check("reference", "candidate", "m", [], {}, tmp_path, full=True)
    assert fake["launches"] == 14
    check.check("reference", "other", "m", [], {}, tmp_path)
    assert fake["launches"] == 16  # the reference's runs were kept


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
