import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_model import env_default_model_id, load_local_model, load_tokenizer, pick_device, pick_dtype


def _ensure_free_space(path: str, min_free_gib: int) -> None:
    total, used, free = shutil.disk_usage(path)
    min_free = min_free_gib * (1024**3)
    if free < min_free:
        raise RuntimeError(
            f"Not enough free disk space at {path} (free={free / (1024**3):.2f} GiB, need>={min_free_gib} GiB)."
        )


def read_prompt_sets(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def get_example(items: List[Dict[str, Any]], example_id: str) -> Dict[str, Any]:
    for item in items:
        if item.get("example_id") == example_id:
            return item
    raise ValueError(f"example_id not found: {example_id}")


def build_prompt_text(tokenizer: Any, user_prompt: str, system_prompt: str) -> str:
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
def extract_last_token_vectors(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: str,
) -> Tuple[List[int], int, int, torch.Tensor]:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    token_count = int(input_ids.shape[-1])
    token_index = token_count - 1

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states. Ensure output_hidden_states=True is supported.")

    vectors: List[torch.Tensor] = []
    for hs in hidden_states:
        vec = hs[0, token_index, :].detach().to("cpu", dtype=torch.float32)
        vectors.append(vec)
    stacked = torch.stack(vectors, dim=0)

    return input_ids[0].detach().to("cpu").tolist(), token_count, token_index, stacked


@torch.inference_mode()
def generate_response(
    model: Any,
    tokenizer: Any,
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


def load_existing_activations(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_single_example(
    out_path: Path,
    record: Dict[str, Any],
) -> None:
    container = load_existing_activations(out_path)
    if "schema_version" not in container:
        container["schema_version"] = 1
    if "examples" not in container or not isinstance(container.get("examples"), dict):
        container["examples"] = {}
    container["examples"][record["example_id"]] = record
    torch.save(container, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=env_default_model_id())
    parser.add_argument("--device", default=os.environ.get("QWEN_DEVICE", ""))
    parser.add_argument("--dtype", default=os.environ.get("QWEN_DTYPE", ""))
    parser.add_argument("--cache-dir", default=os.environ.get("QWEN_CACHE_DIR", ""))
    parser.add_argument("--system", default=os.environ.get("QWEN_SYSTEM", ""))
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--out", default=os.environ.get("QWEN_LAYER_STATES", "activations/layer_states.pt"))
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "prompts" / "prompt_sets.jsonl"
    out_path = (repo_root / args.out).resolve() if not os.path.isabs(args.out) else Path(args.out).resolve()

    items = read_prompt_sets(dataset_path)
    ex = get_example(items, example_id=args.example_id)

    cache_dir = args.cache_dir
    if cache_dir:
        _ensure_free_space(cache_dir, min_free_gib=12)
    else:
        _ensure_free_space(str(Path.home()), min_free_gib=12)

    resolved_device = pick_device(args.device)
    resolved_dtype = pick_dtype(resolved_device, args.dtype)
    model, tokenizer = load_local_model(args.model, device=resolved_device, dtype=resolved_dtype, cache_dir=cache_dir)
    tokenizer2 = load_tokenizer(args.model, cache_dir=cache_dir)

    user_prompt = str(ex["prompt"])
    prompt_text = build_prompt_text(tokenizer2, user_prompt=user_prompt, system_prompt=args.system)

    token_ids, token_count, token_index, vectors = extract_last_token_vectors(
        model=model, tokenizer=tokenizer2, prompt_text=prompt_text, device=resolved_device
    )

    token_texts: Optional[List[str]] = None
    try:
        token_texts = tokenizer2.convert_ids_to_tokens(token_ids)
    except Exception:
        token_texts = None

    response: Optional[str] = None
    if args.generate:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            device=resolved_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    now = datetime.now(timezone.utc).isoformat()
    record: Dict[str, Any] = {
        "example_id": str(ex.get("example_id", "")),
        "family_id": str(ex.get("family_id", "")),
        "domain": str(ex.get("domain", "")),
        "prompt_type": str(ex.get("prompt_type", "")),
        "label": str(ex.get("label", "")),
        "notes": str(ex.get("notes", "")),
        "ground_truth": str(ex.get("ground_truth", "")),
        "model_id": str(args.model),
        "device": resolved_device,
        "dtype": str(resolved_dtype).replace("torch.", ""),
        "system_prompt": args.system,
        "prompt": user_prompt,
        "full_prompt": prompt_text,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "token_count": token_count,
        "token_index": token_index,
        "vector_rule": "last_prompt_token",
        "layer_vectors": vectors,
        "layer_count": int(vectors.shape[0]),
        "hidden_dim": int(vectors.shape[1]),
        "generated_response": response,
        "created_at": now,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_single_example(out_path, record)

    loaded = load_existing_activations(out_path)
    ex2 = loaded.get("examples", {}).get(record["example_id"])
    if not ex2:
        raise RuntimeError("Failed to reload saved example")
    vec = ex2["layer_vectors"]
    if not isinstance(vec, torch.Tensor):
        raise RuntimeError("Saved vectors did not reload as a torch.Tensor")

    print(
        json.dumps(
            {
                "saved_to": str(out_path),
                "example_id": record["example_id"],
                "token_count": record["token_count"],
                "token_index": record["token_index"],
                "layer_vectors_shape": list(vec.shape),
            }
        )
    )


if __name__ == "__main__":
    main()
