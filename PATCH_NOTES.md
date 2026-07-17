# PATCH_NOTES — NVFP4 on consumer Blackwell (SM120) for vLLM v0.20.0

Target: **vLLM v0.20.0** (git tag `v0.20.0`, commit
`88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`).
Scope: **pure-Python overlay only** — the NVFP4 CUDA kernels already ship
compiled in the wheels (`cutlass_scaled_mm_supports_fp4(120) == True`,
`nvfp4_scaled_mm_sm120_kernels.cu` and `nvfp4_blockwise_moe_kernel.cu` are built).

---

## TL;DR — what changed relative to the original task premise

The task premise (from issues #35065 / #31085 / #33416) is that SM120 **hard-fails**
NVFP4 MoE (`No NvFp4 MoE backend supports the deployment configuration`) and
silently falls back to Marlin for dense layers. **That premise describes vLLM
BEFORE the SM120 enablement PRs landed.**

By **v0.20.0 the capability-gating fix is already merged** (lineage of
PR #24968 "Enable MOE support for SM_120"). Verified directly in the checked-out
source:

| Path | Gate in v0.20.0 | SM120 (12.0)? |
|---|---|---|
| Dense NVFP4 CUTLASS — `CutlassNvFp4LinearKernel.is_supported` (`kernels/linear/nvfp4/cutlass.py`) | `cutlass_fp4_supported()` (queries the compiled kernel capability directly) | **Accepted** |
| MoE NVFP4 CUTLASS — `CutlassExpertsFp4._supports_current_device` (`fused_moe/experts/cutlass_moe.py:682‑689`) | `is_device_capability_family(100) or (110) or (120)` | **Accepted** |
| MoE FlashInfer‑CUTLASS — `FlashInferExperts._supports_current_device` (`fused_moe/flashinfer_cutlass_moe.py:124‑136`) | `... or is_device_capability_family(120)` | **Accepted** |

So on v0.20.0 the NVFP4 MoE no longer hard-fails and dense no longer silently
drops to Marlin — the SM120 CUTLASS backends are reachable **out of the box**.

### The actual residual problem on v0.20.0

The remaining gap is **selection *ordering*, not exclusion**. Both selectors try
**FlashInfer NVFP4 paths *before* the vLLM-native CUTLASS path**, and on the
stock `vllm/vllm-openai` image (which bundles FlashInfer) those FlashInfer paths
*claim* SM120 support but are **broken on capability 12.x**:

- FlashInfer NVFP4 `mm_fp4` GEMM fails on SM120 — every FlashInfer backend
  (cutlass / cudnn / trtllm) errors or returns zeros
  (flashinfer-ai/flashinfer#2577).
- FlashInfer/CUTLASS grouped-GEMM NVFP4 MoE produces **garbage output / illegal
  instruction / CUDA-graph-replay crashes** on SM120
  (NVIDIA/cutlass#3096; nvidia/Qwen3.6-35B-A3B-NVFP4 HF discussion #9).

Dense selection order (`kernels/linear/__init__.py:260`):
`FlashInferCutlass → Cutlass → Marlin → …` — FlashInferCutlass wins on SM120.
MoE selection order (`fused_moe/oracle/nvfp4.py:161`):
`FLASHINFER_TRTLLM → FLASHINFER_CUTEDSL(×2) → FLASHINFER_CUTLASS → VLLM_CUTLASS → MARLIN`
— the three CuteDSL/TRTLLM entries are correctly `family(100)`-gated and skip
SM120, but `FLASHINFER_CUTLASS` wins on SM120 before `VLLM_CUTLASS` is reached.

**These two overlays deterministically route SM120 NVFP4 (dense + MoE) onto the
vLLM-native CUTLASS backends**, exactly as the task requests
("route SM120 onto the CUTLASS nvfp4 scaled_mm and the CUTLASS/blockwise nvfp4
MoE backend, NOT onto FlashInfer-only paths"). They are **narrowing** edits
(remove SM120 from FlashInfer eligibility); nothing SM100-only is widened.

> Note on direction: the task anticipated *widening* SM100→SM120 checks. On
> v0.20.0 that widening is already upstream, so the correct minimal edit is the
> mirror image — *narrowing* the FlashInfer paths that are ordered ahead of the
> (already-SM120-capable) vLLM CUTLASS path and are empirically broken on SM120.

---

## Patched files

### 1. `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py` — dense NVFP4

Class `FlashInferCutlassNvFp4LinearKernel`, method `is_supported()`.

**Before**
```python
        if (
            cutlass_fp4_supported()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            return True, None
        return False, "FlashInfer + >=sm_100 required"
```

**After**
```python
        if (
            cutlass_fp4_supported()
            and current_platform.has_device_capability(100)
            # [vllm-sm120 overlay] Exclude consumer Blackwell SM120 ...
            and not current_platform.is_device_capability_family(120)
            and has_flashinfer()
        ):
            return True, None
        return False, "FlashInfer + >=sm_100 (excluding sm_120) required"
```

**Why.** `has_device_capability(100)` is a `>=` test, so it returns `True` on
SM120 (120 ≥ 100). With FlashInfer present, this FlashInfer kernel is first in
the registry and wins on SM120 — but its FP4 GEMM is broken there (#2577).
Declining SM120 makes the loop fall through to the very next entry,
`CutlassNvFp4LinearKernel` (`is_supported == cutlass_fp4_supported()`, `True` on
SM120), which dispatches the dedicated `nvfp4_scaled_mm_sm120` CUTLASS kernel.
Only capability 12.x is affected; SM90/SM100/SM110 behaviour is unchanged.
`current_platform` is already imported in this module.

### 2. `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` — NVFP4 MoE

Class `FlashInferExperts`, staticmethod `_supports_quant_scheme()`.

**Before**
```python
            # nvfp4, wmxfp4amxfp8 on 10.0+
            or (
                scheme
                in [
                    (kMxfp4Static, kMxfp8Dynamic),
                    (kNvfp4Static, kNvfp4Dynamic),
                ]
                and p.has_device_capability(100)
            )
        )
```

**After**
```python
            # wmxfp4amxfp8 on 10.0+
            or (
                scheme == (kMxfp4Static, kMxfp8Dynamic)
                and p.has_device_capability(100)
            )
            # nvfp4 on 10.0+, EXCEPT consumer Blackwell SM120. ...
            or (
                scheme == (kNvfp4Static, kNvfp4Dynamic)
                and p.has_device_capability(100)
                and not p.is_device_capability_family(120)
            )
        )
```

**Why.** The original combined the `wmxfp4amxfp8` and `nvfp4` schemes into one
`has_device_capability(100)` branch. The overlay splits them so **only the NVFP4
scheme declines SM120**; the `wmxfp4amxfp8` branch is byte-for-byte equivalent in
effect (unchanged eligibility). On SM120 the NvFp4 MoE oracle
(`oracle/nvfp4.py:select_nvfp4_moe_backend`) then skips `FLASHINFER_CUTLASS` and
selects the next backend, `VLLM_CUTLASS` → `CutlassExpertsFp4` (which already
accepts `family(120)`), i.e. the native `nvfp4_blockwise_moe_kernel.cu` grouped
GEMM. `MARLIN` and `EMULATION` remain as further fallbacks, so the path can never
regress to the "no backend" hard-fail. `FlashInferExperts._supports_current_device`
is left untouched, so FP8 / block-FP8 MoE on SM120 via FlashInfer is unaffected.

Both files were copied verbatim from v0.20.0 and edited in place; `diff` against
the upstream source shows only the hunks above, and both compile
(`python -m py_compile`).

---

## What is intentionally NOT changed

- **`fused_moe/experts/cutlass_moe.py` (`CutlassExpertsFp4`)** — already accepts
  `family(120)` in v0.20.0. No edit needed; this is the backend we route *to*.
- **`kernels/linear/nvfp4/cutlass.py` (`CutlassNvFp4LinearKernel`)** — already
  accepts SM120 via `cutlass_fp4_supported()`. This is the dense backend we route
  *to*.
- **Genuinely SM100-only FlashInfer MoE kernels** — `TrtLlmNvFp4Experts*`,
  `FlashInferCuteDSL(Batched)Experts` are gated `is_device_capability_family(100)`
  and already skip SM120. Left as-is per the task ("if a check guards a
  genuinely-SM100-only kernel, leave it").
- **MXFP4** (`quantization/mxfp4.py`, `CutlassExpertsMxfp4`, mxfp4 oracle) — out
  of scope. Target checkpoints (Qwen3.6-35B-A3B-NVFP4, Nemotron-NVFP4) are NVFP4,
  not MXFP4. `CutlassExpertsMxfp4` is still `family(100)`-only on v0.20.0; SM120
  MXFP4 MoE remains unsupported and is not addressed here.

---

## Residual risks (must validate on real SM120 hardware)

1. **Native CUTLASS NVFP4 MoE grouped GEMM may still produce garbage on SM120
   depending on how the wheel was compiled.** NVIDIA/cutlass#3096 reports the
   NVFP4 grouped-GEMM template yields correct results on sm_120 **only when built
   for the full-feature target `compute_120f` (requires CUDA 13.0)**; a
   CUDA 12.8 build (as in issue #35065's environment) can silently return zeros.
   This overlay selects the right *backend*, but cannot fix a mis-compiled
   *kernel* — that needs the wheel/base-image to be built with CUDA 13 /
   `-gencode ...=compute_120f`. **Verify MoE numerical correctness (not just "it
   runs") before trusting output.** The dense `nvfp4_scaled_mm_sm120` kernel is a
   dedicated SM120 kernel and is the lower-risk of the two.

2. **FlashInfer env overrides still reach the broken paths.** If a user sets
   `VLLM_USE_FLASHINFER_MOE_FP4=1`, `VLLM_FLASHINFER_MOE_BACKEND=...`,
   `VLLM_NVFP4_GEMM_BACKEND=flashinfer-*`, or `moe_backend=flashinfer_cutlass`,
   the oracle honours the explicit request and this overlay does not override it.
   For SM120, leave those unset (or use `cutlass`).

3. **fp8 KV cache interaction is unvalidated on SM120 with the CUTLASS NVFP4
   path.** The community GPTQ-Marlin baseline used fp8 KV; the CUTLASS NVFP4 MoE +
   fp8-KV combination on sm_120 has not been validated here. Test
   `--kv-cache-dtype fp8` separately.

4. **CUDA-graph capture / replay.** Several SM120 crash reports are during
   CUDA-graph replay. If instability appears, test with `--enforce-eager` to
   isolate graph-capture issues from kernel issues.

5. **Not exercised in this environment.** These edits were validated by source
   analysis, diff review and `py_compile` only — there is no SM120 GPU here to run
   an end-to-end generation. Treat as "ready to test", not "verified working".

---

## Confidence that the MoE path engages on SM120

**High confidence the vLLM-native CUTLASS MoE backend is *selected* on SM120**
(that is what the Python overlay controls): with FlashInfer's NVFP4 MoE declined
for capability 12.x, `select_nvfp4_moe_backend` deterministically reaches
`VLLM_CUTLASS` / `CutlassExpertsFp4`, whose `_supports_current_device` already
returns `True` for `family(120)`. The remaining `is_supported_config` predicates
(activation, quant scheme `kNvfp4Static×kNvfp4Dynamic`, EP size == 1, standard
activation format) are satisfied by the standard single-GPU NVFP4 deployment, and
MARLIN/EMULATION remain as guaranteed fallbacks so the hard-fail cannot recur.

**Moderate confidence the selected kernel *computes correctly*** — that depends on
the wheel being built with CUDA 13 / `compute_120f` (risk #1). This is outside the
reach of a pure-Python overlay and must be confirmed on hardware.
