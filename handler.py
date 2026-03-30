"""
RunPod Serverless Handler for Qwen3-30B-A3B abliterated-erotic AWQ-Int4
OpenAI-compatible chat completions via vLLM
"""
import os
import time
import runpod
from vllm import LLM, SamplingParams

# Model: downloads from HuggingFace on cold start, RunPod caches on host
MODEL_ID = os.environ.get("MODEL_ID", "forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
QUANTIZATION = os.environ.get("QUANTIZATION", "awq")

print(f"Loading {MODEL_ID} with vLLM (quantization={QUANTIZATION}, max_model_len={MAX_MODEL_LEN})...")
llm = LLM(
    model=MODEL_ID,
    quantization=QUANTIZATION,
    dtype="float16",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEM_UTIL,
    enforce_eager=True,
    trust_remote_code=True,
    kv_cache_dtype="fp8",
)
tokenizer = llm.get_tokenizer()
print(f"Model loaded. max_model_len={MAX_MODEL_LEN}")


def handler(job):
    inp = job["input"]

    # Support both direct params and OpenAI-envelope format
    if "openai_input" in inp:
        params = inp["openai_input"]
    else:
        params = inp

    messages = params.get("messages", [])
    if not messages:
        return {"error": "No messages provided"}

    max_tokens = min(params.get("max_tokens", 4096), 20000)
    temperature = params.get("temperature", 0.8)
    top_p = params.get("top_p", 0.95)
    top_k = params.get("top_k", 20)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    start = time.time()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.time() - start

    text = outputs[0].outputs[0].text
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)

    return {
        "id": f"chatcmpl-{job['id'][:16]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": outputs[0].outputs[0].finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "elapsed_ms": int(elapsed * 1000),
    }


runpod.serverless.start({"handler": handler})
