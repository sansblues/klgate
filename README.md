# klgate

Catch silent numeric changes between two vLLM builds.

A kernel or cache change can shift a model's outputs without failing a test: a value
rounded to bf16 too early, a stale KV block. vLLM's GSM8K gate fails only on an 8-point
accuracy drop; of the 7 known bugs replayed here, it caught only #45361. klgate runs both builds several times and asks whether the candidate differs
from the reference by more than reruns differ from each other.

## Check a build

This replays [#53877](https://github.com/vllm-project/vllm/pull/53877): the fix is the
reference, its buggy parent the candidate.

```sh
uv run klgate check 56058fd5 87b9b5b8 --model Qwen/Qwen3.5-0.8B \
  --args "--enforce-eager --gpu-memory-utilization 0.6 --max-model-len 2048 --max-num-seqs 32 --limit-mm-per-prompt '{\"image\": 0, \"video\": 0}'"
```

On an A40 this printed:

```text
First runs: probe KL 0.000183, same top token 99.54 %, 30.4 % of outputs differ
Across builds over rerun noise: probes 0.99×, decode 3.59×, traffic 0.98×
Verdict: CHANGED, 3.59× is above the 1.25× needed (p = 0.0040 over 252 splits, alpha 0.05)
```

The bug rounded beta to bf16 in the decode kernel, so only decode sees it.

- Builds are vLLM commits, branches or tags with a per-commit wheel. They install into
  `~/.cache/klgate/builds` (`KLGATE_BUILDS` moves them), about 9 GB each.
- Each build runs 5 times (`--runs`, 4 to 8), alternating, every run on a fresh server with
  empty compile and autotune caches, because a fresh server can land in a different numeric
  state, as far from another as a real bug. Runs are kept in `runs/`, so rerunning resumes.
- Pass the server arguments (`--args`) and environment (`--env K=V`) you serve with: a check
  covers only the code they run.
- A check of a small model takes about half an hour on an A40, mostly server startups.

## What it measures

Each run sends three kinds of requests:

- **Probes**: 32 chunks of 1,024 WikiText-2 tokens, scored with prompt logprobs on their last
  512 positions, as llama.cpp's `--kl-divergence` does.
- **Decode**: the first half of each chunk, continued greedily for 128 tokens, one request
  at a time, so decode kernels run without batching.
- **Traffic**: 250 GSM8K questions behind a shared 5-shot prefix, generated greedily, more
  than fit in one batch. This runs prefix caching, chunked prefill, CUDA-graph padding and
  KV block reuse, as serving does.

Generated outputs are compared up to and including their first differing token. KL is taken over the top
20 tokens, a lower bound on the true KL.

## How it decides

Each part scores the mean KL across builds over the mean KL between reruns: about 1× when
nothing changed. The score is the largest of the three.

If the builds were the same, the real split of the runs into reference and candidate would
be no more extreme than any other split. So klgate scores every split, and reads CHANGED
when the real one beats all but 5 % of them. Unchanged code then reads CHANGED at most 5 %
of the time, with no thresholds to tune, and a "same" tells you how large a change it would
have caught.

## How well it works

`uv run klgate bench` decides every replay again from its KL matrices in `data/`:

| Replays | Result |
|---|---|
| 7 known bugs, which shaped the design | 6 caught |
| 8 bugs it never saw, picked by a rule fixed before running ([how](bench/unseen.md)) | 6 caught |
| 6 replays of 2 commit pairs with identical engine code: A40 and L4, eager and compiled, two machines | no false alarm |

The misses: #32118 needs memory pressure, #41873 doesn't trigger under this workload, and
#64279 struck only 3 of 5 buggy runs. #37810 turned out inert with its model, so it isn't
counted. #60047 reads CHANGED on a KL of 1e-10 that changes no output token, because batch
invariance makes its reruns bit-identical. #41574 and #54048 read CHANGED where reruns of one
build differ about 1,000 times more than usual. To replay a bug: `uv run klgate bench
bench/entries/<id>.toml`.

## Limits

- It finds changes, not bugs: an intended numerics change reads CHANGED too.
- A change below the score needed is missed, and so is a bug that strikes too few runs:
  #64279 struck 3 of 5. More runs help.
- It needs Linux on x86_64 with an NVIDIA GPU.

`klgate/drift.py` is the same code as the logprob drift eval proposed for vLLM's eval
catalog ([RFC #61698](https://github.com/vllm-project/vllm/issues/61698)).
