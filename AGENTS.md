# AGENTS.md — adapting this patch with a coding agent

This file is written for AI coding agents (Claude Code, etc.). If you're a
human: clone the repo, open your agent in it, and say something like
*"adapt this patch to vLLM v0.2X.Y for my RTX 5090"* — the instructions below
give the agent everything it needs.

## What this repo is

Two-file Python overlay for **vLLM v0.20.0** that makes NVFP4 checkpoints use
the **native CUTLASS FP4 kernels on SM120** (consumer Blackwell, e.g. RTX 5090)
instead of FlashInfer paths that claim SM120 support but are broken on it.
Full rationale: `README.md`; exact hunks: `PATCH_NOTES.md`.

The patch **narrows** FlashInfer eligibility; it does **not** widen any
capability check. That direction is the key design decision — the CUTLASS
backends already accept SM120 in v0.20.0+.

## Task: re-target the patch to a different vLLM version

1. **Get the exact source your user runs.** `git clone --depth 1 --branch
   <tag> https://github.com/vllm-project/vllm` — or better, read the files
   straight out of the running container (`kubectl exec` / `docker exec`),
   since images occasionally diverge from tags.
2. **Check whether the patch is still needed.** Grep upstream for the bug
   first — if FlashInfer's NVFP4 `is_supported()` / `_supports_quant_scheme()`
   already excludes capability-12.x (or ranks below CUTLASS for SM120), the
   patch is obsolete: tell the user, don't patch.
3. **Locate the two selection points** (paths as of v0.20.0; they move between
   versions — grep, don't trust paths):
   - dense: `FlashInferCutlassNvFp4LinearKernel.is_supported()` in
     `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`
   - MoE: `FlashInferExperts._supports_quant_scheme()` in
     `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`
   Useful greps: `is_device_capability_family`, `has_device_capability(100)`,
   `cutlass_fp4_supported`, `No NvFp4 MoE backend`, `nvfp4`.
4. **Re-apply the same two semantic changes** to the new version's code:
   decline SM120 in FlashInfer's NVFP4 dense kernel; decline SM120 in
   FlashInfer's MoE expert class **for the NVFP4 scheme only** (leave
   mxfp4/mxfp8 and genuinely SM100-only kernels — trtllm, cutedsl — alone).
   Copy the COMPLETE modified files into `patches/vllm/<same relative path>`;
   keep every other byte identical to upstream (diff-verify).
5. **Confirm the fallback exists**: the next backend in each selector's order
   must accept SM120 (`CutlassNvFp4LinearKernel` / `CutlassExpertsFp4` or
   their successors). If CUTLASS does NOT accept SM120 in that version, this
   patch alone cannot help — say so instead of widening checks blindly.
6. **Update `Dockerfile`'s `ARG VLLM_VERSION`** and, if the user deploys via
   Kubernetes ConfigMap subPath (see README), remind them the
   `dist-packages/pythonX.Y` mount paths are version-specific.
7. **Verify**: `python3 -m py_compile` both files; then the runtime gate below.

## Runtime verification gate (never skip, never soften)

NVFP4 on SM120 has a known **silent-garbage** failure mode (cutlass#3096:
grouped-GEMM can return zeros on CUDA-12.x builds). After deploying:

1. Startup log must select the CUTLASS FP4 path — red flags: `Unquantized MoE
   backend` (checkpoint format problem → OOM), `Marlin` chosen for the MoE
   (patch not engaged), any FlashInfer NVFP4 selection (patch not mounted).
2. Send real prompts; completions must be coherent (not empty / repeated
   tokens / gibberish). A model that loads and serves 200 OK can still be
   returning garbage — check the actual text.
3. Only then benchmark.

## Known traps (learned the hard way — save your user the time)

- **Checkpoint format ≠ checkpoint format.** `nvidia/...-NVFP4` checkpoints can
  be `modelopt_mixed` (FP8 linears): vLLM then routes the MoE to the
  *Unquantized* backend → bf16 expert materialization → OOM on 32 GB. Uniform
  NVFP4 checkpoints (e.g. RedHatAI's) work. Don't force `--quantization`; the
  checkpoint config overrides it anyway.
- **NVFP4 ≈ GPTQ-Int4 in size** for this class of model (~23–25 GB). The win is
  FP4 tensor-core compute, not VRAM.
- **First boot is slow** (checkpoint download + torch.compile). On Kubernetes,
  set the startup probe budget in hours and persist `VLLM_CACHE_ROOT`.
- **Hybrid GDN models**: keep `--mamba-cache-mode align` for prefix caching, and
  keep `--max-num-batched-tokens` far from the mamba block size (exact-block
  budgets serialize the scheduler).

## Repo layout contract

- `patches/vllm/**` — complete overlay files mirroring the vllm package tree
  (never partial diffs; the Dockerfile and ConfigMap flows copy whole files).
- `PATCH_NOTES.md` — per-file before/after hunks + risks; update it whenever
  `patches/` changes, including the vLLM version it targets.
- `Dockerfile` — overlay build; `ARG VLLM_VERSION` must match the patch target.
- Commit style: imperative subject, explain the *why* in the body.
