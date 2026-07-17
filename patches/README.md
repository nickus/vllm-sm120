# patches/

Overlay files mirroring the `vllm` package tree. Anything under `patches/vllm/`
is copied over the installed package in the Docker image.

Planned first patch (see vllm-project/vllm#31085): in the NVFP4/MXFP4
quantization backend selection, treat SM120 (`capability (12, 0)`) as
kernel-compatible with the SM100 family so the already-compiled CUTLASS
NVFP4 MoE kernels are selected instead of hard-failing / falling back to Marlin.
