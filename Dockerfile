FROM vllm/vllm-openai:v0.18.0

WORKDIR /app

# Add RunPod handler
RUN pip install --no-cache-dir runpod

COPY handler.py .

ENV MODEL_ID=Qwen/Qwen3-30B-A3B-GPTQ-Int4
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.90

CMD ["python", "-u", "handler.py"]
