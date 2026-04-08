import argparse
import os
import shutil
import sys

import torch
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


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
        }
        if explicit_dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {explicit_dtype}. Use one of {sorted(mapping)}")
        return mapping[explicit_dtype]
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def _ensure_free_space(path: str, min_free_gib: int) -> None:
    total, used, free = shutil.disk_usage(path)
    min_free = min_free_gib * (1024**3)
    if free < min_free:
        raise RuntimeError(
            f"Not enough free disk space at {path} (free={free / (1024**3):.2f} GiB, need>={min_free_gib} GiB). "
            f"Free space, or set --cache-dir to a drive with more space, or use --backend remote."
        )


def load(model_id: str, device: str, dtype: torch.dtype, cache_dir: str):
    cache_dir_arg = cache_dir or None
    if cache_dir:
        _ensure_free_space(cache_dir, min_free_gib=12)
    else:
        _ensure_free_space(os.path.expanduser("~"), min_free_gib=12)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir_arg)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir_arg)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, messages) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    messages,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
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


def generate_remote(model_id: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model=model_id, token=token)
    messages = [{"role": "user", "content": prompt}]

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        resp = client.chat.completions.create(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return resp.choices[0].message["content"].strip()

    text = client.text_generation(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--backend", choices=["local", "remote"], default=os.environ.get("QWEN_BACKEND", "local"))
    parser.add_argument("--device", default=os.environ.get("QWEN_DEVICE", ""))
    parser.add_argument("--dtype", default=os.environ.get("QWEN_DTYPE", ""))
    parser.add_argument("--cache-dir", default=os.environ.get("QWEN_CACHE_DIR", ""))
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if args.backend == "remote":
        try:
            text = generate_remote(
                model_id=args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(text)
            return
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)

    try:
        model, tokenizer = load(args.model, device=device, dtype=dtype, cache_dir=args.cache_dir)
        messages = [{"role": "user", "content": args.prompt}]
        text = generate(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(text)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
