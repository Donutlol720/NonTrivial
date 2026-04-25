import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import InferenceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_model import env_default_model_id, load_local_model, load_tokenizer, pick_device, pick_dtype


LEGACY_HF_INFERENCE_BASE_URL = "https://api-inference.huggingface.co"


def read_prompt_sets(path: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def get_example(items: List[Dict[str, object]], example_id: str) -> Dict[str, object]:
    if not items:
        raise ValueError("prompt_sets.jsonl is empty")
    if not example_id:
        return items[0]
    for item in items:
        if item.get("example_id") == example_id:
            return item
    raise ValueError(f"example_id not found: {example_id}")


def build_prompt_text(tokenizer, user_prompt: str, system_prompt: str) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts: List[str] = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


@torch.inference_mode()
def generate_local(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    input_len = int(inputs["input_ids"].shape[-1])
    gen_ids = output_ids[input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_remote(
    model_id: str,
    prompt_text: str,
    token: str,
    provider: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    provider_arg = provider if provider else None
    token_arg = token or None
    client = InferenceClient(model=model_id, provider=provider_arg, token=token_arg)
    try:
        text = client.text_generation(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return text.strip()
    except Exception as e:
        msg = str(e)
        if "404" not in msg and "Not Found" not in msg:
            raise

    legacy_client = InferenceClient(model=model_id, token=token_arg, base_url=LEGACY_HF_INFERENCE_BASE_URL)
    text = legacy_client.text_generation(
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return text.strip()


def debug_one(
    model_id: str,
    backend: str,
    provider: str,
    device: str,
    dtype: str,
    cache_dir: str,
    system: str,
    example_id: str,
    prompt_override: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    hf_token: str,
) -> Tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "prompts" / "prompt_sets.jsonl"
    tokenizer = load_tokenizer(model_id, cache_dir=cache_dir)

    if prompt_override:
        user_prompt = prompt_override
        meta = {"example_id": "", "family_id": "", "domain": "", "prompt_type": "", "label": "", "notes": ""}
    else:
        items = read_prompt_sets(dataset_path)
        ex = get_example(items, example_id=example_id)
        user_prompt = str(ex["prompt"])
        meta = {
            "example_id": str(ex.get("example_id", "")),
            "family_id": str(ex.get("family_id", "")),
            "domain": str(ex.get("domain", "")),
            "prompt_type": str(ex.get("prompt_type", "")),
            "label": str(ex.get("label", "")),
            "notes": str(ex.get("notes", "")),
        }

    prompt_text = build_prompt_text(tokenizer, user_prompt=user_prompt, system_prompt=system)
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    token_count = int(input_ids.shape[-1])

    model_max = getattr(tokenizer, "model_max_length", None)
    now = datetime.now(timezone.utc).isoformat()

    print(json.dumps({"ts": now, "model_id": model_id, "backend": backend, "provider": provider, **meta}, ensure_ascii=False))
    print(json.dumps({"prompt_token_count": token_count, "model_max_length": model_max}, ensure_ascii=False))
    print("----- PROMPT_TEXT_BEGIN -----")
    print(prompt_text)
    print("----- PROMPT_TEXT_END -----")

    if backend == "remote":
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or ""
        if not token:
            token = getpass.getpass("Hugging Face token (input hidden): ").strip()
        response = generate_remote(
            model_id=model_id,
            prompt_text=prompt_text,
            token=token,
            provider=provider,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return prompt_text, response

    resolved_device = pick_device(device)
    resolved_dtype = pick_dtype(resolved_device, dtype)
    model, tokenizer2 = load_local_model(model_id, device=resolved_device, dtype=resolved_dtype, cache_dir=cache_dir)
    response = generate_local(
        model=model,
        tokenizer=tokenizer2,
        prompt_text=prompt_text,
        device=resolved_device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return prompt_text, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default_model_id())
    parser.add_argument("--backend", choices=["local", "remote"], default=os.environ.get("QWEN_BACKEND", "remote"))
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER", "auto"))
    parser.add_argument("--device", default=os.environ.get("QWEN_DEVICE", ""))
    parser.add_argument("--dtype", default=os.environ.get("QWEN_DTYPE", ""))
    parser.add_argument("--cache-dir", default=os.environ.get("QWEN_CACHE_DIR", ""))
    parser.add_argument("--system", default=os.environ.get("QWEN_SYSTEM", ""))
    parser.add_argument("--example-id", default=os.environ.get("QWEN_EXAMPLE_ID", ""))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    _, response = debug_one(
        model_id=args.model,
        backend=args.backend,
        provider=args.provider,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        system=args.system,
        example_id=args.example_id,
        prompt_override=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        hf_token=args.hf_token,
    )

    print("----- RESPONSE_BEGIN -----")
    print(response)
    print("----- RESPONSE_END -----")


if __name__ == "__main__":
    main()
