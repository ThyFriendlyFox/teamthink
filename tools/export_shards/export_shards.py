"""Export a Llama-family causal LM into per-shard ONNX subgraphs for
pipeline-parallel inference in the browser.

Each shard runs a contiguous slice of decoder layers and owns the KV cache for
just those layers. The graph boundaries are:

  shard 0      : embed_tokens -> layers[0:a]            (input: input_ids)
  shard i      : layers[a:b]                            (input: hidden_states)
  shard K-1    : layers[b:N] -> final norm -> lm_head   (output: logits)

All shards share a common I/O contract so the browser runner is uniform:

  inputs:
    input_ids      [B, S]            (first shard only)
    hidden_states  [B, S, H]         (non-first shards)
    position_ids   [B, S]            (int64)
    causal_mask    [B, 1, S, T]      (float additive, T = past_len + S)
    past_k_<l>     [B, KVH, P, D]    (one per local layer, P = past_len)
    past_v_<l>     [B, KVH, P, D]
  outputs:
    hidden_states  [B, S, H]         (non-last shards)
    logits         [B, S, V]         (last shard only)
    present_k_<l>  [B, KVH, T, D]
    present_v_<l>  [B, KVH, T, D]

This re-implements only the attention core and layer plumbing; all weights
(projections, norms, embeddings, rotary) are reused directly from the loaded
Hugging Face model, so numerics match the reference implementation.

Supported architectures: Llama 3.x, SmolLM2, TinyLlama (LlamaForCausalLM) and
Qwen2/Qwen2.5 (attention q/k/v bias is auto-detected). Gemma-family models use a
different normalization/scaling scheme and are intentionally not supported here.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ShardRange:
    index: int
    start: int
    end: int
    is_first: bool
    is_last: bool


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [B, S, D] -> [B, 1, S, D] to broadcast over heads.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class ShardModule(nn.Module):
    """A contiguous slice of decoder layers, reusing weights from `hf_model`."""

    def __init__(self, hf_model, rng: ShardRange):
        super().__init__()
        cfg = hf_model.config
        self.rng = rng
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.scaling = self.head_dim ** -0.5

        base = hf_model.model
        self.rotary_emb = base.rotary_emb
        if rng.is_first:
            self.embed_tokens = base.embed_tokens
        self.layers = nn.ModuleList(
            [base.layers[i] for i in range(rng.start, rng.end)]
        )
        if rng.is_last:
            self.norm = base.norm
            self.lm_head = hf_model.lm_head

    def _attn(self, layer, hidden, cos, sin, past_k, past_v, mask):
        attn = layer.self_attn
        b, s, _ = hidden.shape
        q = attn.q_proj(hidden).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Prepend the cached keys/values for this shard's layer.
        k = torch.cat((past_k, k), dim=2)
        v = torch.cat((past_v, v), dim=2)
        present_k, present_v = k, v

        # Grouped-query attention: repeat kv heads to match query heads.
        if self.num_kv_heads != self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = scores + mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = attn.o_proj(out)
        return out, present_k, present_v

    def forward(self, x, position_ids, causal_mask, *past_kv):
        # past_kv is a flat list: [past_k_0, past_v_0, past_k_1, past_v_1, ...]
        hidden = self.embed_tokens(x) if self.rng.is_first else x
        cos, sin = self.rotary_emb(hidden, position_ids)

        presents: list[torch.Tensor] = []
        for li, layer in enumerate(self.layers):
            past_k = past_kv[2 * li]
            past_v = past_kv[2 * li + 1]
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attn_out, present_k, present_v = self._attn(
                layer, normed, cos, sin, past_k, past_v, causal_mask
            )
            hidden = residual + attn_out
            residual = hidden
            normed = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(normed)
            presents.append(present_k)
            presents.append(present_v)

        if self.rng.is_last:
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
            return (logits, *presents)
        return (hidden, *presents)


def plan_ranges(num_layers: int, num_shards: int) -> list[ShardRange]:
    base = num_layers // num_shards
    rem = num_layers % num_shards
    ranges: list[ShardRange] = []
    start = 0
    for i in range(num_shards):
        count = base + (1 if i < rem else 0)
        end = start + count
        ranges.append(
            ShardRange(
                index=i,
                start=start,
                end=end,
                is_first=(i == 0),
                is_last=(i == num_shards - 1),
            )
        )
        start = end
    return ranges


def export_shard(
    shard: ShardModule,
    rng: ShardRange,
    cfg,
    out_dir: str,
) -> dict:
    n_local = rng.end - rng.start
    num_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    b, s, past = 1, 4, 2  # dummy shapes; axes 1..3 are dynamic
    total = past + s

    if rng.is_first:
        x = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)
        x_name = "input_ids"
    else:
        x = torch.randn(b, s, cfg.hidden_size)
        x_name = "hidden_states"

    position_ids = torch.arange(past, past + s, dtype=torch.long).unsqueeze(0)
    causal_mask = torch.zeros(b, 1, s, total)

    past_kv = []
    for _ in range(n_local):
        past_kv.append(torch.randn(b, num_kv, past, head_dim))
        past_kv.append(torch.randn(b, num_kv, past, head_dim))

    input_names = [x_name, "position_ids", "causal_mask"]
    for li in range(n_local):
        input_names.append(f"past_k_{li}")
        input_names.append(f"past_v_{li}")

    out_name = "logits" if rng.is_last else "hidden_states_out"
    output_names = [out_name]
    for li in range(n_local):
        output_names.append(f"present_k_{li}")
        output_names.append(f"present_v_{li}")

    dynamic_axes: dict[str, dict[int, str]] = {
        x_name: {0: "batch", 1: "seq"},
        "position_ids": {0: "batch", 1: "seq"},
        "causal_mask": {0: "batch", 2: "seq", 3: "total"},
        out_name: {0: "batch", 1: "seq"},
    }
    for li in range(n_local):
        dynamic_axes[f"past_k_{li}"] = {0: "batch", 2: "past"}
        dynamic_axes[f"past_v_{li}"] = {0: "batch", 2: "past"}
        dynamic_axes[f"present_k_{li}"] = {0: "batch", 2: "total"}
        dynamic_axes[f"present_v_{li}"] = {0: "batch", 2: "total"}

    file_name = f"shard_{rng.index}.onnx"
    path = os.path.join(out_dir, file_name)
    args = (x, position_ids, causal_mask, *past_kv)

    torch.onnx.export(
        shard,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    return {
        "index": rng.index,
        "file": file_name,
        "layerStart": rng.start,
        "layerEnd": rng.end,
        "isFirst": rng.is_first,
        "isLast": rng.is_last,
        "inputName": x_name,
        "outputName": out_name,
        "numLocalLayers": n_local,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a model into ONNX shards.")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model-id", default=None, help="id used in the manifest")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config

    arch = (cfg.architectures or ["?"])[0]
    if "gemma" in arch.lower() or cfg.model_type.lower().startswith("gemma"):
        raise SystemExit(
            f"Architecture {arch} is not supported by this exporter (Gemma uses a "
            "different norm/scaling scheme)."
        )

    num_layers = cfg.num_hidden_layers
    if args.shards > num_layers:
        raise SystemExit(f"--shards {args.shards} exceeds layer count {num_layers}")

    ranges = plan_ranges(num_layers, args.shards)
    print(f"{num_layers} layers -> {args.shards} shards: "
          + ", ".join(f"[{r.start},{r.end})" for r in ranges))

    shard_meta = []
    with torch.no_grad():
        for rng in ranges:
            print(f"Exporting shard {rng.index} layers [{rng.start},{rng.end}) ...")
            shard = ShardModule(model, rng).eval()
            shard_meta.append(export_shard(shard, rng, cfg, args.out))

    manifest = {
        "modelId": args.model_id or args.model,
        "source": args.model,
        "architecture": arch,
        "numLayers": num_layers,
        "hiddenSize": cfg.hidden_size,
        "numAttentionHeads": cfg.num_attention_heads,
        "numKeyValueHeads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "headDim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "vocabSize": cfg.vocab_size,
        "ropeTheta": getattr(cfg, "rope_theta", 10000.0),
        "dtype": "fp32",
        "eosTokenId": cfg.eos_token_id,
        "shards": shard_meta,
    }
    with open(os.path.join(args.out, "shards.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.save_pretrained(args.out)
    except Exception as e:  # noqa: BLE001
        print(f"(tokenizer not saved: {e})")

    print(f"Done. Manifest + {len(shard_meta)} shards written to {args.out}")


if __name__ == "__main__":
    main()
