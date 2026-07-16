import os
from typing import Any, Tuple

import torch


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: transformers. Install it in your active Python environment, e.g.\n"
            "  python -m pip install -U transformers huggingface_hub accelerate safetensors sentencepiece\n"
            "Then rerun the pipeline."
        ) from e
    return AutoModelForCausalLM, AutoTokenizer


def _load_tokenizer_with_fallback(AutoTokenizer: Any, model_id: str, cache_dir_arg: Any, trust_remote_code: bool) -> Any:
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir_arg,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        message = str(exc)
        tokenizer_backend_error = "Couldn't instantiate the backend tokenizer"
        if tokenizer_backend_error not in message:
            raise
        # Some newer Qwen releases only load cleanly through the slow tokenizer path
        # in this environment; retrying here keeps the extraction pipeline model-agnostic.
        return AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir_arg,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def pick_device(explicit_device: str) -> str:
    if explicit_device:
        return explicit_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, explicit_dtype: str) -> torch.dtype:
    if explicit_dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if explicit_dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {explicit_dtype}. Use one of {sorted(mapping)}")
        return mapping[explicit_dtype]
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str, cache_dir: str = "", trust_remote_code: bool = False) -> Any:
    _, AutoTokenizer = _require_transformers()
    cache_dir_arg = cache_dir or None
    tokenizer = _load_tokenizer_with_fallback(AutoTokenizer, model_id, cache_dir_arg, trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_local_model(
    model_id: str, device: str, dtype: torch.dtype, cache_dir: str = "", trust_remote_code: bool = False
) -> Tuple[object, object]:
    AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    cache_dir_arg = cache_dir or None
    tokenizer = _load_tokenizer_with_fallback(AutoTokenizer, model_id, cache_dir_arg, trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir_arg, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()
    return model, tokenizer


def env_default_model_id() -> str:
    return os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
