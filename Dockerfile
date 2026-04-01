# ── Build from the same base as runpod/worker-v1-vllm:v2.14.0 ──────────────
# but swap vLLM 0.16.0 → 0.18.0 for AWQ MoE (Qwen3-30B-A3B) support.
#
# WHY the old custom image crashed:
#   handler.py called `LLM(model=...)` at module level, which blocks Python
#   for minutes while downloading / loading the model. That means
#   `runpod.serverless.start()` never executes, the heartbeat ping never
#   starts, and RunPod marks the worker "unhealthy" after ~60 s.
#
#   The official worker-vllm uses AsyncLLMEngine (non-blocking) and calls
#   `runpod.serverless.start()` immediately so the heartbeat fires first.
# ────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# ── vLLM 0.18.0 (the whole point) ──────────────────────────────────────────
# pytorch/pytorch base already has Python, pip, CUDA, cuDNN, PyTorch
RUN pip install --no-cache-dir "vllm==0.18.0"

# ── Python deps (mirrors worker-vllm v2.14.0 builder/requirements.txt) ─────
RUN python3 -m pip install --no-cache-dir \
    ray \
    pandas \
    pyarrow \
    runpod \
    huggingface-hub \
    packaging \
    "typing-extensions>=4.8.0" \
    pydantic \
    pydantic-settings \
    hf-transfer \
    "transformers>=4.57.0,<5.0.0" \
    "bitsandbytes>=0.45.0" \
    kernels \
    torch-c-dlpack-ext

# ── Copy the handler ───────────────────────────────────────────────────────
COPY handler.py /handler.py

# ── Model + engine config via env vars ──────────────────────────────────────
# The handler reads these at startup.
ENV MODEL_ID=forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.92
ENV QUANTIZATION=awq
ENV DTYPE=half
ENV ENFORCE_EAGER=1
ENV KV_CACHE_DTYPE=fp8
ENV TRUST_REMOTE_CODE=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# RunPod cache paths (same as official worker)
ENV BASE_PATH="/runpod-volume"
ENV HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub"
ENV HF_HOME="${BASE_PATH}/huggingface-cache/hub"
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTHONPATH="/"

# Suppress noisy Ray metrics
ENV RAY_METRICS_EXPORT_ENABLED=0
ENV RAY_DISABLE_USAGE_STATS=1
ENV TOKENIZERS_PARALLELISM=false
ENV RAYON_NUM_THREADS=4

CMD ["python3", "-u", "/handler.py"]
