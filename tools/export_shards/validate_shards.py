"""Validate exported ONNX shards against the reference HF model.

Runs greedy decoding both ways for the same prompt and asserts the generated
token sequences are identical. This is the go/no-go gate before wiring the
shards into the browser runtime.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEG_INF = -1e9


def build_causal_mask(seq: int, past: int) -> np.ndarray:
    total = past + seq
    mask = np.zeros((1, 1, seq, total), dtype=np.float32)
    for i in range(seq):
        for j in range(total):
            if j > past + i:
                mask[0, 0, i, j] = NEG_INF
    return mask


class ShardSession:
    def __init__(self, path: str, meta: dict, num_kv: int, head_dim: int):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.meta = meta
        self.n_local = meta["numLocalLayers"]
        self.num_kv = num_kv
        self.head_dim = head_dim
        self.reset()

    def reset(self) -> None:
        self.past = [
            np.zeros((1, self.num_kv, 0, self.head_dim), dtype=np.float32)
            for _ in range(2 * self.n_local)
        ]

    def run(self, x: np.ndarray, position_ids: np.ndarray, seq: int) -> np.ndarray:
        past_len = self.past[0].shape[2]
        feeds = {
            self.meta["inputName"]: x,
            "position_ids": position_ids,
            "causal_mask": build_causal_mask(seq, past_len),
        }
        for li in range(self.n_local):
            feeds[f"past_k_{li}"] = self.past[2 * li]
            feeds[f"past_v_{li}"] = self.past[2 * li + 1]

        out_names = [self.meta["outputName"]]
        for li in range(self.n_local):
            out_names.append(f"present_k_{li}")
            out_names.append(f"present_v_{li}")

        results = self.sess.run(out_names, feeds)
        primary = results[0]
        self.past = results[1:]
        return primary


def run_sharded(manifest: dict, out_dir: str, input_ids: list[int], max_new: int) -> list[int]:
    num_kv = manifest["numKeyValueHeads"]
    head_dim = manifest["headDim"]
    sessions = [
        ShardSession(os.path.join(out_dir, s["file"]), s, num_kv, head_dim)
        for s in manifest["shards"]
    ]

    generated: list[int] = []
    tokens = list(input_ids)
    pos = 0

    def step(token_window: list[int]) -> int:
        nonlocal pos
        seq = len(token_window)
        position_ids = np.arange(pos, pos + seq, dtype=np.int64)[None, :]
        x: np.ndarray = np.array([token_window], dtype=np.int64)
        for i, sess in enumerate(sessions):
            x = sess.run(x, position_ids, seq)
        pos += seq
        # x is logits [1, seq, vocab]; take last position.
        next_id = int(np.argmax(x[0, -1]))
        return next_id

    # Prefill with the full prompt, then decode one token at a time.
    nxt = step(tokens)
    generated.append(nxt)
    for _ in range(max_new - 1):
        nxt = step([nxt])
        generated.append(nxt)
    return generated


def run_reference(model, input_ids: list[int], max_new: int) -> list[int]:
    with torch.no_grad():
        out = model.generate(
            torch.tensor([input_ids], dtype=torch.long),
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
        )
    return out[0, len(input_ids):].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    with open(os.path.join(args.shards_dir, "shards.json")) as f:
        manifest = json.load(f)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    input_ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()

    ref = run_reference(model, input_ids, args.max_new)
    shard = run_sharded(manifest, args.shards_dir, input_ids, args.max_new)

    print("reference:", ref)
    print("sharded:  ", shard)
    if ref == shard:
        print("PASS: sharded greedy decode matches the reference model.")
    else:
        first = next((i for i in range(min(len(ref), len(shard))) if ref[i] != shard[i]), None)
        raise SystemExit(f"FAIL: token sequences diverge at index {first}.")


if __name__ == "__main__":
    main()
