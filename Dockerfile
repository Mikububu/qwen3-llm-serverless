FROM runpod/worker-v1-vllm:v2.14.0

# Override the default handler with our custom one that uses vLLM offline mode
# The base image already has vLLM, runpod, and all dependencies
# We just need to upgrade vLLM to 0.18.0 for AWQ MoE support

RUN pip install --no-cache-dir --upgrade vllm==0.18.0

WORKDIR /app
COPY handler.py .

ENV MODEL_ID=forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.92
ENV QUANTIZATION=awq

CMD ["python", "-u", "handler.py"]
