import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.extract_activations as ea
from src.load_model import env_default_model_id, load_local_model, load_tokenizer, pick_device, pick_dtype


def select_items(items, example_ids):
    if not example_ids:
        return items
    wanted = set(example_ids)
    selected = [item for item in items if str(item.get("example_id", "")) in wanted]
    found = {str(item.get("example_id", "")) for item in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"example_id not found: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="activations/full_per_example")
    parser.add_argument("--example-id", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    items = select_items(ea.read_prompt_sets(repo_root / "prompts" / "prompt_sets.jsonl"), args.example_id)
    out_dir = (repo_root / args.out_dir).resolve() if not os.path.isabs(args.out_dir) else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = env_default_model_id()
    cache_dir = os.environ.get("QWEN_CACHE_DIR", "")
    system_prompt = os.environ.get("QWEN_SYSTEM", "")
    device = pick_device(os.environ.get("QWEN_DEVICE", ""))
    dtype = pick_dtype(device, os.environ.get("QWEN_DTYPE", ""))

    model, tokenizer_for_generate = load_local_model(model_id, device=device, dtype=dtype, cache_dir=cache_dir)
    tokenizer = load_tokenizer(model_id, cache_dir=cache_dir)

    print(
        json.dumps(
            {
                "examples_total": len(items),
                "model_id": model_id,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "out_dir": str(out_dir),
                "skip_existing": args.skip_existing,
                "max_new_tokens": args.max_new_tokens,
            }
        )
    )

    for i, ex in enumerate(items, start=1):
        eid = str(ex["example_id"])
        out_path = out_dir / f"{eid}.pt"
        if args.skip_existing and out_path.exists():
            print(json.dumps({"skipped": i, "example_id": eid, "reason": "exists"}, ensure_ascii=False))
            continue

        prompt_text = ea.build_prompt_text(tokenizer, user_prompt=str(ex["prompt"]), system_prompt=system_prompt)
        token_ids, token_count, token_index, vectors = ea.extract_last_token_vectors(
            model=model, tokenizer=tokenizer, prompt_text=prompt_text, device=device
        )

        try:
            token_texts = tokenizer.convert_ids_to_tokens(token_ids)
        except Exception:
            token_texts = None

        response = ea.generate_response(
            model=model,
            tokenizer=tokenizer_for_generate,
            prompt_text=prompt_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        record = {
            "example_id": eid,
            "family_id": str(ex.get("family_id", "")),
            "domain": str(ex.get("domain", "")),
            "prompt_type": str(ex.get("prompt_type", "")),
            "label": str(ex.get("label", "")),
            "notes": str(ex.get("notes", "")),
            "ground_truth": str(ex.get("ground_truth", "")),
            "model_id": str(model_id),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "system_prompt": system_prompt,
            "prompt": str(ex["prompt"]),
            "full_prompt": prompt_text,
            "token_ids": token_ids,
            "token_texts": token_texts,
            "token_count": int(token_count),
            "token_index": int(token_index),
            "vector_rule": "last_prompt_token",
            "layer_vectors": vectors,
            "layer_count": int(vectors.shape[0]),
            "hidden_dim": int(vectors.shape[1]),
            "generated_response": response,
            "created_at": None,
        }

        torch.save({"schema_version": 1, "examples": {eid: record}}, out_path)
        print(
            json.dumps(
                {
                    "saved": i,
                    "example_id": eid,
                    "token_count": int(token_count),
                    "response_chars": len(response),
                    "shape": [int(vectors.shape[0]), int(vectors.shape[1])],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
