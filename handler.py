"""
RunPod Serverless Handler for Qwen3-30B-A3B abliterated-erotic AWQ-Int4

Uses AsyncLLMEngine (not LLM class) — same pattern as the official
RunPod worker-vllm. AsyncLLMEngine doesn't spawn EngineCore subprocess,
avoiding /dev/shm issues in Docker containers.
"""
import os
import sys
import asyncio
import traceback
import multiprocessing
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
QUANTIZATION = os.environ.get("QUANTIZATION", "awq")
DTYPE = os.environ.get("DTYPE", "half")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") in ("1", "true", "True")
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "fp8")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") in ("1", "true", "True")

# ── Engine (initialized at startup, NOT lazily) ───────────────────────────
# AsyncLLMEngine doesn't block — it returns quickly and loads in background.
# This is how the official RunPod worker does it.
engine = None
tokenizer = None


def _init_engine():
    global engine, tokenizer
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from transformers import AutoTokenizer

    log.info(f"Initializing AsyncLLMEngine for {MODEL_ID}...")

    args = AsyncEngineArgs(
        model=MODEL_ID,
        quantization=QUANTIZATION,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=TRUST_REMOTE_CODE,
        kv_cache_dtype=KV_CACHE_DTYPE,
    )

    engine = AsyncLLMEngine.from_engine_args(args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
    log.info("AsyncLLMEngine initialized.")


# ── Handler ────────────────────────────────────────────────────────────────
async def handler(job):
    from vllm import SamplingParams
    from vllm.utils import random_uuid

    inp = job["input"]
    if "openai_input" in inp:
        params = inp["openai_input"]
    else:
        params = inp

    messages = params.get("messages", [])
    if not messages:
        yield {"error": "No messages provided"}
        return

    max_tokens = min(params.get("max_tokens", 4096), 20000)
    temperature = params.get("temperature", 0.8)
    top_p = params.get("top_p", 0.95)
    top_k = params.get("top_k", 20)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling, request_id)

    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for request_output in results_generator:
        prompt_tokens = len(request_output.prompt_token_ids)
        for output in request_output.outputs:
            full_text = output.text
            completion_tokens = len(output.token_ids)

    # Strip <think> tags
    import re
    clean_text = re.sub(r'<think>[\s\S]*?</think>\s*', '', full_text).strip()

    yield {
        "id": f"chatcmpl-{request_id[:16]}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": clean_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    try:
        _init_engine()
    except Exception as e:
        log.error(f"Engine init failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    log.info("Starting RunPod serverless worker...")
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
    })
