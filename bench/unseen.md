# Bugs klgate never saw

klgate was designed around 7 known bugs, so catching them proves little. To test it fairly,
two more sets of bugs were picked by rules written before any of them was replayed, each
with a prediction. This file keeps those rules, the predictions and every result, misses
included.

Of the 8 that could be scored, 6 were caught. #41873 was predicted not to trigger under
klgate's workload, and #64279 struck only 3 of its 5 buggy runs.

## Set 1

The rule: every mined fix that shows under greedy decoding, fits one GPU, has wheels for
both commits and needs only server settings. That left 5 of 14; the other 9 need LoRA, sleep
mode, speculative decoding or special checkpoints. Of the 5, #48010's buggy build crashes at
startup with today's packages, #18957 predates the API the eval uses, and #47574 needs FP8 KV
on Blackwell.

| PR | What changes | Result | Score (needed) | p |
|---|---|---|---|---|
| #60156 | FP8 softmax weights underflow in Triton attention (L4) | CHANGED | probes 2.79× (1.33×) | 0.008 |
| #41873 | Stale `is_prefilling` in padded Mamba CUDA-graph rows | same, as predicted: doesn't trigger here | 0.98× (1.29×) | 0.79 |

## Set 2

### Results

The rule and predictions below were written before any of these runs.

| PR | Predicted | Result | Score (needed) | p |
|---|---|---|---|---|
| #33919 | caught, probes | CHANGED | probes 8.86×, decode ∞× (1.70×) | 0.008 |
| #60047 | unsure | CHANGED | decode ∞× (8.15×) | 0.008 |
| #54048 | caught, probes | CHANGED | decode 2.03× (1.27×) | 0.004 |
| #25936 | caught, probes | CHANGED | traffic 47.8× (1.75×) | 0.004 |
| #41574 | caught, probes | CHANGED | decode 1.37× (1.27×) | 0.008 |
| #64279 | caught, gross | missed | traffic 1.17× (1.19×) | 0.079 |
| #37810 | caught, decode | not a test: the bug is inert here | decode 1.33× (1.33×) | 0.44 |

- #60047: batch invariance makes each build's reruns bit-identical, so a decode KL of ~1e-10
  across builds reads as changed, though no output token changes.
- #54048: this 4-bit community quant is noisy (probe KL between reruns of one build ~0.17,
  outputs repetitive in both builds), yet the change still reads at 2× that noise in decode
  (probes 1.40×, traffic 1.30×). Predicted probes; decode decided.
- #41574: with `--quantization fp8` on an A40, reruns of one build differ by probe KL
  ~0.22, yet the GELU change reads at 1.37× that noise in decode (1.27× needed).
  Predicted probes (1.16×); decode decided.
- #64279: probes and decode are bit-identical across builds; the bug shows only in batched
  generation, and only in 3 of the 5 buggy launches (the other 2 read like the fixed build).
  The prediction "gross" was wrong. Not rerun with more launches: that would tune on this set.
- #37810: every `A_log` value in the Qwen3.5-0.8B checkpoint is exactly representable in bf16,
  so holding it in bf16 changes nothing: 8 of the 10 runs decode bit-identically, and the
  other two, both candidate runs, match each other. The triage that called it eligible was
  wrong.

Of the 7 that ran, #37810 was inert; 5 of the other 6 were caught.

### The rule, written 2026-11-15 before any replay

Every vLLM commit on `main` from 2025-08-15 to 2026-11-11 whose subject mentions a fix and
a numeric word (accuracy, precision, incorrect, wrong, garbage, NaN, corrupt, numeric,
rounding, overflow, underflow, mismatch, gibberish, correctness, bf16, fp32, fp16, dtype,
stale, off-by-one), minus the 212 fixes screened earlier, minus subjects about
hardware or features the eval can't run (ROCm, XPU, CPU, multi-GPU, LoRA, speculative
decoding, Blackwell/Hopper-only kernels, multimodal, offloading, frontend/API). 134 commits,
plus #37054 from the earlier list.

Eligible bugs meet all of:

1. The fix changes what a model computes (not crash, validation, perf or logging only).
2. A public model on one A40 (48 GB) or one L4 (24 GB) reaches the changed code.
3. Server arguments and environment alone trigger it, with the eval's own requests.
4. Fix and parent merged on or after 2025-08-15, with wheels for both.

Eligibility and a prediction (which part sees it, caught or not, why) are written per bug
below before its replay. Every eligible bug is replayed, up to the 10 most recent; each
result is reported, misses included.

### Candidates and predictions

Triage read every diff: 10 of the 135 are eligible. #47309 met every rule but the 4th (no
wheels). #37810 was confirmed eligible once the checkpoint's `A_log` proved to be stored in
fp32.
Predictions are mine, written 2026-11-15 before any replay.

| PR | What changes | Model, GPU, settings | Predicted part | Caught? |
|---|---|---|---|---|
| #64279 | Whole-graph compile hands vLLM ops non-eager strides, corrupting the KV cache | Qwen/Qwen3-0.6B, L4, `--model-impl transformers --compilation-config '{"mode": 2, "backend": "inductor"}'` | all | yes, high: gross |
| #60047 | fp32 LM-head reduction order under batch invariance (~1e-6 relative) | Qwen/Qwen3-0.6B, L4, `--hf-overrides '{"head_dtype": "float32"}'`, `VLLM_BATCH_INVARIANT=1` | probes, decode | unsure: caught only if batch-invariant reruns are bit-identical |
| #54048 | Gemma4 MoE router logits rounded to bf16 before fp32 | cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit, A40, `--dtype bfloat16` | probes | yes, medium: occasional expert flips |
| #42782 | Explicit `--kv-cache-dtype bfloat16` ignored: KV stored in fp8, other backend | nm-testing/TinyLlama-1.1B-Chat-v1.0-kvcache-fp8-tensor, L4, `--kv-cache-dtype bfloat16` | all | yes, high: gross |
| #41574 | Gemma4 MoE experts use exact GELU instead of tanh GELU | google/gemma-4-26B-A4B-it, A40, `--quantization fp8 --limit-mm-per-prompt '{"image": 0}'` | probes | yes, medium: small but in every layer; may not load |
| #37810 | Qwen3.5 GDN `A_log` held in bf16 instead of fp32 | Qwen/Qwen3.5-0.8B, A40, `--limit-mm-per-prompt '{"image": 0, "video": 0}'` | decode (probes unsure) | yes, medium: like #53877, but in prefill too |
| #37054 | FP8 MLA KV reads V with the wrong scale (2× with computed scales) | RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8, L4, `--kv-cache-dtype fp8 --calculate-kv-scales` | decode, traffic | yes, high: gross |
| #33919 | DeepSeek-V2 routing (n_group 1, softmax) in bf16 instead of fp32 | deepseek-ai/DeepSeek-V2-Lite, A40, `--trust-remote-code` | probes | yes, high: every token's routing weights |
| #29867 | Static FP8 scaling overflows when building MLA decode weights (up to ~6%) | RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8, L4 | decode, traffic | yes, high |
| #25936 | ERNIE-4.5 MoE router and bias in bf16 instead of fp32 | baidu/ERNIE-4.5-21B-A3B-PT, A40, `--quantization fp8` | probes | yes, medium: expert flips; may not fit |

Four of these (#64279, #42782, #37054, #29867) are gross: any output check should catch
them. The informative ones are the other six.

### Changes made before the runs

- L4 was out of stock, so the L4 rows run on an RTX 4090: also Ada (sm89), also 24 GB.
- #54048 also gets `--limit-mm-per-prompt '{"image": 0}'` and the two DeepSeek-Coder rows
  `--trust-remote-code`: neither changes what the text model computes.
- Every row also runs with `--max-model-len 2048 --max-num-seqs 32`, as every replay does.

### Changes made after the runs began

- #42782's buggy build crashed at startup: FlashInfer's JIT found no `ninja`, because the
  replay harness didn't put the build's own `bin` on `PATH`. The harness was fixed; the replay
  was not rerun, for budget.
- #37054 and #29867 were not run, for budget (both predicted gross).
- #54048, #25936 and #41574 first failed at startup on package versions resolved today, not
  on the eval: Transformers too old for Gemma 4 (#41574), too new for the ERNIE tokenizer
  (#25936), and Gemma 4 profiling video input (#54048, and #41574 on its second try).
  Both builds of each pair got the same Transformers (5.7.0 for #41574, 4.57.6 for #25936),
  and #54048 and #41574 also get `--limit-mm-per-prompt '{"image": 0, "video": 0}'`. None of
  this changes the text model's numerics, and both builds of a pair stay identical outside
  the commit.
