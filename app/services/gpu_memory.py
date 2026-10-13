import logging

logger = logging.getLogger(__name__)

def release_ollama_vram():
    """
    Explicit lifecycle helper to unload Ollama models from GPU VRAM.
    This does NOT execute any OCR or inference request; it merely sends keep_alive: 0
    to ensure Ollama does not hoard VRAM when local PyTorch vision models execute.
    """
    try:
        import httpx
        for model in ["gemma3:latest", "gemma3:270m", "phi3:latest"]:
            try:
                httpx.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": model, "keep_alive": 0},
                    timeout=1.0
                )
            except Exception:
                pass
        logger.info("[VRAM] Sent keep_alive=0 to Ollama to release GPU memory.")
    except Exception as e:
        logger.debug(f"[VRAM] Ollama memory release check skipped: {e}")

def release_cuda_cache():
    """Releases cached CUDA memory in PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
