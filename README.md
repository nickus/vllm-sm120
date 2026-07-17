# vllm-sm120

Patches and packaging to run **vLLM** with **native NVFP4 MoE kernels on consumer
Blackwell (SM120: RTX 5090 / RTX 6000 Blackwell)**, targeting
**Qwen3.6-35B-A3B** (hybrid GatedDeltaNet + softmax-attention MoE) on a single
32 GB RTX 5090.

## Why this repo exists

vLLM ships CUTLASS NVFP4 kernels compiled for SM120 — `cutlass_scaled_mm_supports_fp4(120)`
returns `True` and `nvfp4_scaled_mm_sm120_kernels.cu` / `nvfp4_blockwise_moe_kernel.cu`
are built into the wheels. But the Python **backend-selection logic** only matches
the datacenter Blackwell family (SM100), so on SM120:

- dense NVFP4 layers silently fall back to **Marlin weight-only** ("This may degrade
  performance for compute-heavy workloads"), and
- **NVFP4 MoE hard-fails**: `No NvFp4 MoE backend supports the deployment configuration`
  (vllm-project/vllm#35065, closed as not planned; tracked in #31085).

The fix is (mostly) Python: teach the quantization backend selection
(`mxfp4.py` / modelopt NVFP4 paths) that SM120 has the same kernels as SM100.
That means we can **overlay-patch the official Docker image** instead of doing a
full CUDA source build.

## Layout

```
patches/     # overlay .py patches against a pinned vllm/vllm-openai image
Dockerfile   # FROM vllm/vllm-openai:<pinned> + COPY overlay patches
```

## Baseline numbers (what we must beat)

Current prod stack: llama.cpp b9737, Qwen3.6-35B-A3B UD-Q4_K_XL, RTX 5090,
`-c 262144 -np 4 --kv-unified`:

| metric | llama.cpp (measured) |
|---|---|
| single-stream decode | ~182–187 tok/s |
| aggregate, 10 parallel reqs | ~380 tok/s (peak ~528) |
| prefix reuse | only via `-ctxcp` checkpoints (~87× prefill on prefix-extension); `--cache-reuse` disabled for hybrid-SSM arch |

Reference points for vLLM on RTX 5090 (community-reported):

| config | single-stream | notes |
|---|---|---|
| GPTQ-Int4 + `gptq_marlin`, 131K ctx, fp8 KV | **194–197 tok/s** | 31.3 GB VRAM, prefix caching on (align) |
| NVFP4 (RedHatAI, ~17 GB) via Marlin fallback | ~105 tok/s solo / 160 agg | `--max-num-seqs 2`, 96K ctx — unpatched SM120 path |
| FP8 official checkpoint | n/a | ~35 GB weights, does not fit 32 GB |

Goal: native SM120 NVFP4 MoE ≥ GPTQ-Marlin single-stream, with better prefill,
plus vLLM continuous batching + align-mode prefix caching
(`--enable-prefix-caching --mamba-cache-mode align`, merged 2026-01, PR #30877)
for agent-swarm aggregate throughput.

## Status

- [ ] Pin base image (`vllm/vllm-openai` ≥ 0.19.1) and reproduce the SM120 MoE hard-fail
- [ ] Overlay patch: accept SM120 in NVFP4/MXFP4 backend selection
- [ ] Verify native CUTLASS NVFP4 MoE path is taken (no Marlin fallback warning)
- [ ] Benchmark vs GPTQ-Marlin and llama.cpp baselines
- [ ] Upstream the patch to vllm-project/vllm

## License

Patches are derived from [vLLM](https://github.com/vllm-project/vllm) — Apache-2.0.
