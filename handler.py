"""
RunPod Serverless Handler for Qwen3-30B-A3B abliterated-erotic AWQ-Int4
OpenAI-compatible chat completions via vLLM AsyncLLMEngine.

CRITICAL DESIGN NOTE:
  The model MUST NOT load at module level (before runpod.serverless.start()).
  RunPod's heartbeat ping starts inside runpod.serverless.start(). If the
  main thread is blocked downloading/loading a 30B model for minutes before
  that call, the heartbeat never fires and RunPod kills the worker as
  "unhealthy" after ~60 seconds.

  Solution: call runpod.serverless.start() immediately. Load the engine
  lazily on the first request (or in a background thread that doesn't block
  the main thread).
"""

import os
import sys
import time
import traceback
import multiprocessing
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

# ── Config from env vars ────────────────────────────────────────────────────
MODEL_ID = os.environ.get(
    "MODEL_ID", "forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4"
)
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
QUANTIZATION = os.environ.get("QUANTIZATION", "awq")
DTYPE = os.environ.get("DTYPE", "half")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") in ("1", "true", "True")
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "fp8")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") in ("1", "true", "True")


# ── Lazy engine singleton ───────────────────────────────────────────────────
_engine = None
_tokenizer = None


def _load_engine():
    """Synchronous model load — called once, inside the async handler."""
    from vllm import LLM

    log.info(f"Loading {MODEL_ID} with vLLM "
             f"(quant={QUANTIZATION}, max_len={MAX_MODEL_LEN}) ...")
    start = time.time()
    llm = LLM(
        model=MODEL_ID,
        quantization=QUANTIZATION,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=TRUST_REMOTE_CODE,
        kv_cache_dtype=KV_CACHE_DTYPE,
    )
    tokenizer = llm.get_tokenizer()
    elapsed = time.time() - start
    log.info(f"Model loaded in {elapsed:.1f}s  max_model_len={MAX_MODEL_LEN}")
    return llm, tokenizer


def _get_engine():
    """Return (llm, tokenizer), loading on first call."""
    global _engine, _tokenizer
    if _engine is None:
        _engine, _tokenizer = _load_engine()
    return _engine, _tokenizer


# ── Handler ─────────────────────────────────────────────────────────────────
def handler(job):
    """Process a single chat-completion request."""
    from vllm import SamplingParams

    try:
        llm, tokenizer = _get_engine()
    except Exception as e:
        log.error(f"Engine init failed: {e}\n{traceback.format_exc()}")
        # CUDA errors → worker is broken, let RunPod spin a fresh one
        if "CUDA" in str(e) or "cuda" in str(e):
            sys.exit(1)
        return {"error": f"Engine initialization failed: {e}"}

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


# ── Entry point ─────────────────────────────────────────────────────────────
# Guard against vLLM spawning sub-processes that re-import this file.
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    log.info("Starting RunPod serverless worker (model loads on first request) ...")
    runpod.serverless.start({"handler": handler})
