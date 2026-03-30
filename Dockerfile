FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install vLLM 0.18.0 (required for GPTQ-Marlin MoE on Qwen3)
# Pin transformers to avoid Python 3.11 dataclass incompatibility
RUN pip install --no-cache-dir \
    vllm==0.18.0 \
    runpod \
    huggingface_hub

COPY handler.py .

ENV MODEL_ID=Qwen/Qwen3-30B-A3B-GPTQ-Int4
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.90

CMD ["python", "-u", "handler.py"]
