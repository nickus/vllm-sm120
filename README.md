# vllm-sm120

Make **NVFP4 checkpoints work correctly on consumer Blackwell (SM120: RTX 5090,
RTX PRO Blackwell GeForce-class)** with stock **vLLM v0.20.0** — via two small
Python overlay files, no CUDA build, no custom wheels.

Reference deployment: **Qwen3.6-35B-A3B** (hybrid GatedDeltaNet + softmax
attention MoE, 256 experts) on a single 32 GB RTX 5090.

> 🤖 **Want this on a different vLLM version / GPU / model?** See
> [AGENTS.md](AGENTS.md) — it's written so you can hand it to Claude Code (or
> any coding agent) and say *"adapt this patch to my setup"*.

## What the patch actually does (v0.20.0 reframe)

The folklore ("SM120 NVFP4 hard-fails / silently falls back to Marlin",
vllm#31085, vllm#35065) describes vLLM **before** the SM120 enablement PRs.
By v0.20.0, the **native CUTLASS NVFP4 kernels already accept SM120**:

- dense: `CutlassNvFp4LinearKernel.is_supported()` → `cutlass_fp4_supported()` ✅
- MoE: `CutlassExpertsFp4._supports_current_device` lists `family(120)` ✅

The **residual bug is selection ordering**: both selectors try **FlashInfer**
NVFP4 paths first, and FlashInfer claims SM120 via a `>=` capability test
(`has_device_capability(100)`) while being **broken on 12.x** — `mm_fp4` GEMM
fails (flashinfer#2577) and the grouped-GEMM MoE produces garbage / illegal
instructions (cutlass#3096). Field reports on RTX 5090 match.

So the patch **narrows FlashInfer instead of widening CUTLASS** — two files:

| file | change |
|---|---|
| `patches/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py` | `FlashInferCutlassNvFp4LinearKernel.is_supported()` declines `family(120)` → falls through to `CutlassNvFp4LinearKernel` (`nvfp4_scaled_mm_sm120` kernel) |
| `patches/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` | `FlashInferExperts._supports_quant_scheme()` declines SM120 **only for the NVFP4 scheme** → MoE oracle lands on `CutlassExpertsFp4` (`nvfp4_blockwise_moe_kernel`); Marlin/emulation remain as further fallbacks |

Details, before/after hunks, and residual risks: [PATCH_NOTES.md](PATCH_NOTES.md).

## Checkpoint choice matters (measured 2026-07-18)

- ❌ `nvidia/Qwen3.6-35B-A3B-NVFP4` — **modelopt_mixed** (FP8 linears + FP4):
  the MoE experts resolve to the *Unquantized* backend, get materialized in
  bf16, and OOM a 32 GB card during load (measured: >30.3 GiB and climbing).
- ✅ `RedHatAI/Qwen3.6-35B-A3B-NVFP4` — uniform NVFP4 (25.0 GB), maps the MoE
  onto the proper FP4 method. Let vLLM auto-detect (`--quantization` forced to
  `modelopt_fp4` gets overridden by the checkpoint config anyway).
- Note: NVFP4 checkpoints for this model are **~23–25 GB, not 17 GB** — about
  the same as GPTQ-Int4 (24.4 GB). The win to look for is native FP4 tensor-core
  GEMMs in compute-bound phases (prefill, wide-batch decode), not VRAM.

## Two ways to deploy

**A. Docker overlay** (this repo's [Dockerfile](Dockerfile)): copies
`patches/vllm/**` over site-packages in `vllm/vllm-openai:v0.20.0`.

**B. Kubernetes ConfigMap subPath mounts** (no image build at all): put the two
files in a ConfigMap and mount each one over its site-packages path, e.g.

```yaml
volumeMounts:
  - name: sm120-patch
    mountPath: /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py
    subPath: nvfp4_linear_flashinfer.py
  - name: sm120-patch
    mountPath: /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
    subPath: flashinfer_cutlass_moe.py
```

(Confirm the dist-packages path with
`python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))'` in the
container — it's Python-minor-version-specific.)

## Verification gate — do not skip

cutlass#3096: on some CUDA-12.x builds the NVFP4 grouped-GEMM can **silently
return zeros** on sm_120. After deploying, before trusting the server:

1. Startup log must show the CUTLASS FP4 path (no `Unquantized MoE backend`,
   no `Marlin` for the MoE, no FlashInfer NVFP4 selection).
2. Run real prompts ("The capital of France is", short code task) and check the
   completions are coherent — not empty, not repeated tokens, not gibberish.
3. Only then benchmark.

## Reference numbers (RTX 5090, 32 GB, vLLM v0.20.0, 131K ctx, fp8 KV, align-mode prefix caching)

| config | solo decode | prefill (97K-tok doc) | aggregate decode (32-way) |
|---|---|---|---|
| llama.cpp b9737 Q4_K_XL (old stack) | ~182–187 tok/s | ~1,050 tok/s | ~380 tok/s (4 slots) |
| vLLM GPTQ-Int4 `gptq_marlin` | 182.3 tok/s | 11,550 tok/s | **2,760 tok/s** |
| vLLM NVFP4 + this patch | *trial in progress* | *trial in progress* | *trial in progress* |

## Status

- [x] Patch written against v0.20.0 (`patches/`), py_compile-clean, diff-verified
- [x] Deployed to the reference cluster via ConfigMap overlay
- [ ] NVFP4 serving verified end-to-end (correctness gate) — in progress
- [ ] Benchmarked vs GPTQ-Marlin
- [ ] Upstream: turn the narrowing checks into a proper fix in vllm-project/vllm

## License

Patches are derived from [vLLM](https://github.com/vllm-project/vllm) — Apache-2.0.
