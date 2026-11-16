# klgate

Does a new vLLM build change your model's outputs?

Some vLLM bugs change what a model computes without breaking anything you'd notice: a value
rounded to bf16 too early, a stale KV block reused. Answers barely move, accuracy stays
within noise, every test passes, and the bug ships. vLLM's own GSM8K check, which fails on
an 8-point accuracy drop, caught only 1 of the 7 such bugs replayed here.

Comparing outputs directly doesn't work either, because a build doesn't reproduce itself:
batches form differently each run, and every fresh server compiles and tunes its kernels
its own way. So klgate asks a narrower question: does the new build differ from the old one
by more than the old one differs from itself? It runs each build several times, compares
their token probabilities, and answers CHANGED or same, with no thresholds to tune.

## Usage

```sh
uv run klgate check <reference> <candidate> --model <model>
```

The builds are vLLM commits, branches or tags. For example, this replays the bug fixed in
[#53877](https://github.com/vllm-project/vllm/pull/53877), with the fix as the reference
and its buggy parent as the candidate:

```sh
uv run klgate check 56058fd5 87b9b5b8 --model Qwen/Qwen3.5-0.8B \
  --args "--enforce-eager --gpu-memory-utilization 0.6 --max-model-len 2048 --max-num-seqs 32 --limit-mm-per-prompt '{\"image\": 0, \"video\": 0}'"
```

On an A40 it printed:

```text
First runs: probe KL 0.000183, same top token 99.54 %, 30.4 % of outputs differ
Across builds over rerun noise: probes 0.99×, decode 3.59×, traffic 0.98×
Verdict: CHANGED, 3.59× is above the 1.25× needed (p = 0.0040 over 252 splits, alpha 0.05)
```

The bug rounded a value to bf16 in the decode kernel, so only decode shows it.

Pass the server arguments (`--args`) and environment (`--env K=V`) you serve with, since a
check covers only the code they run. Each build runs 5 times (`--runs`, 4 to 8), each time
on a fresh server with empty caches; that takes about half an hour for a small model on an
A40, mostly server startups. Builds install from vLLM's per-commit wheels into
`~/.cache/klgate/builds`, about 9 GB each. Runs are kept in `runs/`, so rerunning resumes.

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
same, the real split would look like any other, so it reads CHANGED only when the real
split scores above all but 5 % of them. Unchanged code reads CHANGED at most 5 % of the time,
and a "same" says how large a change would have been caught.

## Does it work

`uv run klgate bench` decides every replay again from its saved results in `data/`:

| Replays | Result |
|---|---|
| 7 known bugs, used to design klgate | 6 caught |
| 8 bugs it never saw, picked by a rule fixed beforehand ([how](bench/unseen.md)) | 6 caught |
| 6 replays of 2 commit pairs with identical engine code | no false alarm |

Of the misses, #32118 needs memory pressure this workload doesn't create, #41873 doesn't
trigger under it, and #64279 struck only 3 of the 5 buggy runs. Among the catches, #60047
changed no output token at all, and #41574 and #54048 stood out although reruns of their
models disagree a thousand times more than usual.

To replay one: `uv run klgate bench bench/entries/<id>.toml`.

## Limits

- It finds changes, not bugs. An intended numerics change reads CHANGED too.
- It misses a change smaller than the one needed, and a bug that strikes too few runs.
  More runs help.
- It needs Linux on x86_64 and an NVIDIA GPU.

klgate's eval is the same code as the one proposed for vLLM's eval catalog in
[RFC #61698](https://github.com/vllm-project/vllm/issues/61698).
