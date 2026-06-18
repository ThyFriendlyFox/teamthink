# Shard export tool (DEPRECATED)

> Superseded by the generic in-browser WebGPU executor
> (`lib/engine/webgpu`, `lib/engine/shard/webgpu-shard-runner.ts`). Models are
> now partitioned client-side at session time and each peer range-fetches its
> layer slice straight from Hugging Face `safetensors` — no offline export and
> no `shardManifestUrl`. This tool and its `shards.json` format are retained for
> reference only.

Splits a Llama-family causal LM into per-shard ONNX subgraphs for
pipeline-parallel inference across browsers. Each shard runs a contiguous slice
of decoder layers and owns the KV cache for just those layers.

This is an offline, one-time step run on a workstation (not part of the Next.js
build or the browser runtime).

## Setup

```bash
cd tools/export_shards
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Export

```bash
python export_shards.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --shards 2 \
  --model-id llama-3.2-1b-sharded \
  --out ./out/llama-3.2-1b
```

Produces `shard_0.onnx ... shard_{K-1}.onnx`, a `shards.json` manifest, and the
tokenizer files.

## Validate (go/no-go gate)

Greedy decode must match the reference model token-for-token:

```bash
python validate_shards.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --shards-dir ./out/llama-3.2-1b \
  --prompt "The capital of France is" \
  --max-new 16
```

A `PASS` line means the shard boundaries, KV I/O, and RoPE handling are correct.
Do not proceed to wiring the shards into the app until this passes.

## Hosting

Upload the contents of the `--out` directory (the `.onnx` files plus
`shards.json`) to a static host (Hugging Face Hub repo, Cloudflare R2, or any
CDN). Point a model's `shardManifestUrl` in `lib/config.ts` at the hosted
`shards.json`. Each peer downloads only the single shard it is assigned, so
total download per peer is roughly `model_size / K`.

## I/O contract

Every shard shares one contract so the browser runner is uniform:

| name | shape | notes |
| --- | --- | --- |
| `input_ids` | `[B, S]` | first shard only |
| `hidden_states` | `[B, S, H]` | non-first shards |
| `position_ids` | `[B, S]` | int64 |
| `causal_mask` | `[B, 1, S, T]` | additive float, `T = past_len + S` |
| `past_k_<l>` / `past_v_<l>` | `[B, KVH, P, D]` | one pair per local layer |
| `hidden_states_out` | `[B, S, H]` | non-last shards |
| `logits` | `[B, S, V]` | last shard only |
| `present_k_<l>` / `present_v_<l>` | `[B, KVH, T, D]` | one pair per local layer |

## Supported architectures

Llama 3.x, SmolLM2, TinyLlama (`LlamaForCausalLM`), and Qwen2/Qwen2.5 (attention
q/k/v bias auto-detected via the reused projections). Gemma-family models are
rejected (different normalization/scaling).
