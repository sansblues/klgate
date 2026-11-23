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

For example, this replays the bug fixed in
[#42379](https://github.com/vllm-project/vllm/pull/42379), with the fix as the reference
and its buggy parent as the candidate:

```sh
uv run klgate check 124fac10 e9499996 --model Qwen/Qwen3-0.6B \
  --args "--enforce-eager --gpu-memory-utilization 0.55 --max-model-len 2048 --max-num-seqs 32"
```

On a laptop's RTX 3050 Ti, under WSL, which also needs `--env VLLM_USE_V2_MODEL_RUNNER=0`, it
compared 9 runs and printed:

```text
First runs: probe KL 0.001111, NLL -0.036 %, same top token 97.99 %, 48.0 % of outputs differ
Across builds over rerun noise, 5 reference and 4 candidate runs: probes ∞×, decode ∞×, traffic 1.13×
Verdict: CHANGED, ∞× is above the 3.20× needed; identical builds read this way 1 in 126 times
Where: probes, 32 of 32 chunks differ more across builds than between reruns; most, chunk 21, at position 404 (token ids, probabilities)
  reference: 22235 (0.546), 6755 (0.147), 6569 (0.026)
  candidate: 22235 (0.463), 6755 (0.181), 6569 (0.032)
```

The first line says how far apart the builds' first runs are, the second how each kind of
request scores, and the last where the change shows most. Two commits with identical engine
code, c0056b19 and e9499996, differed by chance on their first 3 runs, and after 6 read
`same within noise, 0.98× is not above the 1.32× needed`.

The exit code is 0 for same, 1 for CHANGED and 2 for an error, and `--json` writes the
result to a file.

### What it costs

Every run needs a fresh server, so the cost is server launches, about two minutes each for
a small model on a datacenter GPU:

| The builds | Launches |
|---|---|
| read the same | 3, sometimes 6: about 4 on average |
| changed | 9 |

klgate starts with 2 runs of the reference and 1 of the candidate, and records more only
while they differ. Runs are kept in `runs/`, so an interrupted check resumes, and a
reference recorded for one candidate counts for the next on the same machine. `--full`
records all 9 whatever the first runs read, for a change the first candidate run happens
to hide.

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

To decide, it tries every way to split the runs into a reference group and a candidate
group. If the builds were the same, the real split would look like any other, so it would
rarely score highest. When it does, the builds differ, and klgate records more runs to
rule out chance:

| Reference runs | Candidate runs | Identical builds differ | Then |
|---|---|---|---|
| 2 | 1 | 1 time in 3 | same: stop. Differ: 3 more |
| 4 | 2 | 1 time in 15 | same: stop. Differ: 3 more |
| 5 | 4 | 1 time in 126 | same, or CHANGED |

So identical builds read CHANGED less than once in 100 checks, on any model, settings or
GPU, and a "same" says how large a change would have been caught. Most builds checked are
the same, so most checks stop after 3 or 6 launches.

## Does it work

`uv run klgate bench` decides every replay again from its saved results in `data/`:

| Replays | Result |
|---|---|
| 7 known bugs, used to design klgate | 5 found |
| 8 bugs it never saw, picked by a rule fixed beforehand ([how](bench/unseen.md)) | 5 found |
| 1 bug raised in vLLM's [RFC #61698](https://github.com/vllm-project/vllm/issues/61698) as out of reach, on 2 models | missed |
| 8 replays of 2 commit pairs with identical engine code, on 4 kinds of GPU | no false alarm, 3.6 launches each |

Each replay holds at least 5 runs of the reference and 4 of the candidate, so every one is
decided with all the runs a check would use. Some of the earlier ones were recorded before the decode requests were
added, and are decided on probes and traffic alone.

For each replay the table also says what a check costs on average, over every choice of the
runs it might have recorded, and how often it would have flagged the change: every time
for 8 of the 10 found, and 92 to 98 % of the time for #30915 and #36580. Those two show in
traffic alone, and now and then the first candidate run's probes match one reference run's
bit for bit and not the other's, which makes that other run look like the odd one.

Two misses come close: #41574 scores 1.37× against the 1.40× needed, because its model's
reruns disagree a thousand times more than usual, and #53877 scores 2.87× against 3.14×,
because one of its 5 reference runs sits farther from the other four than the bug moves
the candidate.

Of the others, #32118 needs memory pressure this workload doesn't create, #41873 doesn't
trigger under it, and #64279 struck only 3 of the 5 buggy runs. #67789 needs a chunked
prefill to share a step with a longer fresh prompt, which these requests rarely if ever
produce: traffic's prompts all hit one cached prefix, and the probes are all one length.
Prompts of mixed lengths were tried for it and drowned in the noise of its quantized KV
cache, where reruns of the fixed build disagreed a thousand times more than with a plain
one. A test that asks the model for a fact from the start of its prompt catches that bug.

To replay one: `uv run klgate bench bench/entries/<id>.toml`.

## Limits

- It finds changes, not bugs. An intended numerics change reads CHANGED too; the output
  then says how large it is and where it shows most.
- It misses a change smaller than the one needed, and a bug that strikes too few runs.
- A reference run that landed apart from the others, by however little, can hide a change
  from the first candidate runs. `--full` compares all 9 runs, which outweigh it.
- It compares probabilities, so a build that only cuts outputs short reads the same.
- It compares two builds serving one configuration, so a pull request that adds a
  configuration, such as a new KV cache dtype, has nothing to be compared with.
- It also reads CHANGED when the two builds' other packages differ, and says so.
- Comparing with a reference recorded earlier assumes the machine hasn't changed since.
  That is untested beyond one sitting.
- It needs Linux on x86_64 and an NVIDIA GPU.
