# klgate

Does a new vLLM build change your model's outputs?

Some vLLM bugs change what a model computes without breaking anything you'd notice: a value
rounded to bf16 too early, a stale KV block reused. Answers barely move, accuracy stays
within noise, every test passes, and the bug ships. vLLM's own GSM8K check, which fails on
an 8-point accuracy drop, caught only 1 of the 7 such bugs klgate was built on.

Comparing outputs directly doesn't work either, because a build doesn't reproduce itself:
batches form differently each run, and every fresh server compiles and tunes its kernels
its own way. So klgate asks a narrower question: does the new build differ from the old one
by more than the old one differs from itself? It runs each build a few times, compares
their token probabilities, and answers CHANGED or same, with no thresholds to tune.

## Usage

Check a pull request against the commit it branched from:

```sh
uv run klgate check --pr 67789 --model Qwen/Qwen3-0.6B
```

Or any two builds, the reference first:

```sh
uv run klgate check v0.33.0 main --model Qwen/Qwen3-0.6B
```

A build is a vLLM tag, branch or commit, or the path of a Python environment you installed
vLLM into yourself. Pass the server arguments (`--args`) and environment (`--env K=V`) you
serve with, since a check covers only the code they run.

EXAMPLE

The exit code is 0 for same, 1 for CHANGED and 2 for an error, and `--json` writes the
result to a file.

### What it costs

Every run needs a fresh server, so the cost is server launches, about two minutes each for
a small model:

| The builds | Launches |
|---|---|
| read the same | 4, sometimes 10 |
| changed | 14 |
| read the same, and the reference was checked before | 2, sometimes 5 |

klgate starts with 2 runs of each build and records more only while they differ. Runs are
kept in `runs/`, so a reference build is recorded once however many candidates you compare
with it, and an interrupted check resumes. `--full` records all 14 whatever the first
runs read: the most sensitive check.

Commits install from vLLM's per-commit wheels into `~/.cache/klgate/builds`, about 9 GB
each. A commit without a wheel, such as a pull request's, installs from the nearest earlier
commit that has one with its Python files laid over it; if it changes compiled code, build
it yourself and pass its environment's path.

## How it works

Each run sends three kinds of requests:

- **Probes**: 32 chunks of WikiText-2 scored with prompt logprobs, as llama.cpp's
  `--kl-divergence` does. Both builds see the same tokens, so a kernel that rounds
  differently shows even when no answer changes.
- **Decode**: the first half of each chunk continued greedily, one request at a time, so
  decode kernels run without batching.
- **Traffic**: 250 GSM8K questions behind a shared prefix, more than fit in one batch. This
  runs what serving runs: prefix caching, chunked prefill, CUDA graphs, KV block reuse.

For each kind, klgate divides the mean KL between the builds by the mean KL between reruns of
the same build. That's about 1 when nothing changed; the largest of the three is the score.

To decide, it tries every way to split the runs into two groups. If the builds were the
same, the real split would look like any other, so it would rarely score highest. When it
does, the builds differ, and klgate records more runs to rule out chance:

| Runs of each build | Identical builds differ | Then |
|---|---|---|
| 2 | 1 time in 3 | same: stop. Differ: 3 more |
| 5 | 1 time in 126 | same: stop. Differ: 2 more |
| 7 | 1 time in 1,716 | same, or CHANGED |

So identical builds read CHANGED less than once in 1,716 checks, and a "same" says how
large a change would have been caught.

## Does it work

`uv run klgate bench` decides every replay again from its saved results in `data/`:

RESULTS

To replay one: `uv run klgate bench bench/entries/<id>.toml`.

## Limits

- It finds changes, not bugs. An intended numerics change reads CHANGED too; the output
  then says how large it is and where it shows most.
- It misses a change smaller than the one needed, and a bug that strikes too few runs.
  `--full` helps.
- It compares two builds serving one configuration, so a pull request that adds a
  configuration, such as a new KV cache dtype, has nothing to be compared with.
- It also reads CHANGED when the two builds' other packages differ, and says so.
- It needs Linux on x86_64 and an NVIDIA GPU.
