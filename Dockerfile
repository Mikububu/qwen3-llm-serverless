FROM runpod/worker-v1-vllm:v2.14.0

# Upgrade vLLM to 0.18.0 for AWQ MoE support (v2.14.0 ships 0.16.0)
RUN pip install --no-cache-dir --upgrade vllm==0.18.0

# Model and config — the base image's handler reads these env vars
ENV MODEL_NAME=forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.92
ENV QUANTIZATION=awq
ENV DTYPE=half
ENV ENFORCE_EAGER=1
ENV KV_CACHE_DTYPE=fp8
ENV TRUST_REMOTE_CODE=1
