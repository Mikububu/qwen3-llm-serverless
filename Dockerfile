# ── Qwen3-30B-A3B abliterated-erotic AWQ on RunPod Serverless ──────────────
#
# KEY DECISIONS (based on research):
#   1. vLLM 0.10.2 — last version with V0 engine (removed in 0.11.0)
#      AND has Qwen3 MoE AWQ fix (broken in 0.8.x, fixed in 0.9.0+)
#   2. Sync LLM class + VLLM_USE_V1=0 — only way to skip EngineCore subprocess
#      (AsyncLLMEngine ignores VLLM_USE_V1=0 — hardcoded to V1)
#   3. /dev/shm remount in entrypoint — safety net in case V0 still needs it
#   4. Lazy model loading — heartbeat must start before model download
# ────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# ── vLLM 0.10.2 — the sweet spot ──────────────────────────────────────────
# 0.8.x: Qwen3 MoE AWQ broken (missing packed_modules_mapping)
# 0.10.2: has the fix + V0 engine still available
# 0.11.0+: V0 engine removed, VLLM_USE_V1=0 no longer works
RUN pip install --no-cache-dir "vllm==0.10.2"

# ── Python deps ────────────────────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    runpod \
    huggingface-hub \
    "transformers>=4.51.0,<5.0.0" \
    tokenizers

# ── Copy handler + entrypoint ──────────────────────────────────────────────
COPY handler.py /handler.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Engine config ──────────────────────────────────────────────────────────
ENV MODEL_ID=forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4
ENV MAX_MODEL_LEN=16384
ENV GPU_MEMORY_UTILIZATION=0.92
ENV QUANTIZATION=awq
ENV DTYPE=half
ENV ENFORCE_EAGER=1
ENV TRUST_REMOTE_CODE=1

# CRITICAL: Force V0 engine — no EngineCore subprocess, no /dev/shm
ENV VLLM_USE_V1=0
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# RunPod / HuggingFace cache
ENV BASE_PATH="/runpod-volume"
ENV HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub"
ENV HF_HOME="${BASE_PATH}/huggingface-cache/hub"
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTHONPATH="/"

# Quiet logs
ENV RAY_METRICS_EXPORT_ENABLED=0
ENV RAY_DISABLE_USAGE_STATS=1
ENV TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["/entrypoint.sh"]
