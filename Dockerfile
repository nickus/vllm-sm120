# Overlay-patched vLLM for consumer Blackwell (SM120: RTX 5090).
# The NVFP4 SM120 CUDA kernels already ship compiled in the official wheels;
# we only override the Python backend-selection modules that refuse SM120.
#
# Pin the exact upstream image before patching — overlay paths are
# version-specific (site-packages layout can move between releases).
ARG VLLM_VERSION=v0.20.0
FROM vllm/vllm-openai:${VLLM_VERSION}

# Overlay patched modules on top of the installed vllm package.
# patches/ mirrors the vllm package tree, e.g.:
#   patches/vllm/model_executor/layers/quantization/mxfp4.py
COPY patches/ /tmp/patches/
RUN set -eu; \
    SITE=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))'); \
    if [ -d /tmp/patches/vllm ]; then cp -rv /tmp/patches/vllm/. "$SITE"/; fi; \
    rm -rf /tmp/patches
