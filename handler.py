"""
RunPod Serverless Handler for uncensored LLM (AWQ-Int4)

CRITICAL: Uses synchronous LLM class (not AsyncLLMEngine).
- AsyncLLMEngine is hardcoded to V1 engine in vLLM >= 0.8 and IGNORES VLLM_USE_V1=0
- Only the sync LLM class respects VLLM_USE_V1=0, giving us V0 engine
- V0 engine has no EngineCore subprocess, no /dev/shm requirement
- Lazy loading: model loads on first request so heartbeat starts first
"""
import os
import sys
import re
import time
import threading
import traceback
import multiprocessing

# MUST be set before any vLLM import
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import runpod
from runpod import RunPodLogger

log = RunPodLogger()

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "forbiddenmichael/Qwen3-30B-A3B-abliterated-erotic-AWQ-Int4")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
QUANTIZATION = os.environ.get("QUANTIZATION", "awq")
DTYPE = os.environ.get("DTYPE", "half")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") in ("1", "true", "True")
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") in ("1", "true", "True")

# ── Lazy engine singleton ───────────────────────────────────────────────────
_llm = None
_tokenizer = None
_load_lock = threading.Lock()
_load_error = None


def _load_engine():
    """Load model — called once, lazily on first request."""
    global _llm, _tokenizer, _load_error

    from vllm import LLM
    from transformers import AutoTokenizer

    log.info(f"Loading {MODEL_ID} with vLLM V0 engine "
             f"(quant={QUANTIZATION}, max_len={MAX_MODEL_LEN}, "
             f"gpu_util={GPU_MEM_UTIL}) ...")
    start = time.time()

    _llm = LLM(
        model=MODEL_ID,
        quantization=QUANTIZATION,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)

    elapsed = time.time() - start
    log.info(f"Model loaded in {elapsed:.1f}s")


def _get_engine():
    """Return (llm, tokenizer), loading on first call."""
    global _llm, _tokenizer, _load_error
    if _llm is not None:
        return _llm, _tokenizer

    with _load_lock:
        if _llm is not None:
            return _llm, _tokenizer
        if _load_error:
            raise _load_error
        try:
            _load_engine()
        except Exception as e:
            _load_error = e
            raise
    return _llm, _tokenizer


# ── Handler (synchronous — V0 engine, no subprocess) ──────────────────────
def handler(job):
    from vllm import SamplingParams

    try:
        llm, tokenizer = _get_engine()
    except Exception as e:
        log.error(f"Engine init failed: {e}\n{traceback.format_exc()}")
        if "CUDA" in str(e) or "cuda" in str(e):
            sys.exit(1)
        return {"error": f"Engine initialization failed: {e}"}

    inp = job["input"]
    params = inp.get("openai_input", inp)
    messages = params.get("messages", [])
    if not messages:
        return {"error": "No messages provided"}

    max_tokens = min(params.get("max_tokens", 4096), 20000)
    temperature = params.get("temperature", 0.8)
    top_p = params.get("top_p", 0.95)
    top_k = params.get("top_k", 20)
    repetition_penalty = params.get("repetition_penalty", 1.15)

    # Qwen3 supports enable_thinking; Qwen2.5 does not
    template_kwargs = {}
    if "qwen3" in MODEL_ID.lower() or "Qwen3" in MODEL_ID:
        template_kwargs["enable_thinking"] = params.get("enable_thinking", True)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **template_kwargs,
    )

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    start = time.time()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.time() - start

    text = outputs[0].outputs[0].text
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)

    # Strip <think> reasoning tags if present (safety net)
    clean_text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    clean_text = re.sub(r'<think>[\s\S]*$', '', clean_text).strip()

    return {
        "id": f"chatcmpl-{job['id'][:16]}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": clean_text},
            "finish_reason": outputs[0].outputs[0].finish_reason or "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "elapsed_ms": int(elapsed * 1000),
    }


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    log.info("Starting RunPod serverless worker (V0 engine, model loads on first request) ...")
    runpod.serverless.start({"handler": handler})
