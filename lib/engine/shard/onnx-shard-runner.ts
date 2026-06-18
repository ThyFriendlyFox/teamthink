import * as ort from "onnxruntime-web/webgpu";
import type { ShardManifest, ShardMeta } from "@/lib/engine/shard/manifest";

/**
 * Runs a single model shard (a contiguous slice of decoder layers) in an
 * onnxruntime-web WebGPU session and owns the KV cache for just those layers.
 *
 * All shards advance in lockstep over the same token stream, so each shard
 * derives its own `position_ids` and causal mask from its local KV length
 * rather than receiving them over the wire — only the hidden state (or the
 * token ids for the first shard) crosses the network.
 */

const NEG_INF = -1e9;

// onnxruntime-web ships its wasm/jsep artifacts separately; point at the CDN
// build matching the installed version so we don't have to bundle them.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

export interface TensorLike {
  dims: number[];
  /** float32 for hidden states / logits, BigInt64 for input ids. */
  data: Float32Array | BigInt64Array;
}

export class OnnxShardRunner {
  private session: ort.InferenceSession | null = null;
  private readonly meta: ShardMeta;
  private past: ort.Tensor[] = [];

  constructor(
    private readonly manifest: ShardManifest,
    readonly shardIndex: number,
  ) {
    this.meta = manifest.shards[shardIndex];
  }

  get isFirst(): boolean {
    return this.meta.isFirst;
  }
  get isLast(): boolean {
    return this.meta.isLast;
  }

  async load(
    baseUrl: string,
    onProgress?: (frac: number, text: string) => void,
  ): Promise<void> {
    const url = joinUrl(baseUrl, this.meta.file);
    const bytes = await fetchWithProgress(url, onProgress);
    this.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    });
    this.reset();
  }

  reset(): void {
    const { numKeyValueHeads, headDim } = this.manifest;
    this.past = [];
    for (let i = 0; i < this.meta.numLocalLayers * 2; i++) {
      this.past.push(
        new ort.Tensor("float32", new Float32Array(0), [
          1,
          numKeyValueHeads,
          0,
          headDim,
        ]),
      );
    }
  }

  private pastLen(): number {
    return this.past.length ? (this.past[0].dims[2] as number) : 0;
  }

  /**
   * Run this shard for a window of `seq` positions. Returns the hidden state
   * (non-last shards) or logits (last shard). Advances the local KV cache.
   */
  async run(primary: TensorLike): Promise<TensorLike> {
    if (!this.session) throw new Error("shard not loaded");
    const seq = primary.dims[1];
    const past = this.pastLen();
    const total = past + seq;

    const positionIds = new BigInt64Array(seq);
    for (let i = 0; i < seq; i++) positionIds[i] = BigInt(past + i);

    const mask = new Float32Array(seq * total);
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < total; j++) {
        mask[i * total + j] = j > past + i ? NEG_INF : 0;
      }
    }

    const feeds: Record<string, ort.Tensor> = {
      [this.meta.inputName]: this.meta.isFirst
        ? new ort.Tensor("int64", primary.data as BigInt64Array, primary.dims)
        : new ort.Tensor("float32", primary.data as Float32Array, primary.dims),
      position_ids: new ort.Tensor("int64", positionIds, [1, seq]),
      causal_mask: new ort.Tensor("float32", mask, [1, 1, seq, total]),
    };
    for (let li = 0; li < this.meta.numLocalLayers; li++) {
      feeds[`past_k_${li}`] = this.past[2 * li];
      feeds[`past_v_${li}`] = this.past[2 * li + 1];
    }

    const results = await this.session.run(feeds);
    const out = results[this.meta.outputName];
    const newPast: ort.Tensor[] = [];
    for (let li = 0; li < this.meta.numLocalLayers; li++) {
      newPast.push(results[`present_k_${li}`]);
      newPast.push(results[`present_v_${li}`]);
    }
    this.past = newPast;

    return { dims: out.dims as number[], data: out.data as Float32Array };
  }

  dispose(): void {
    this.session?.release?.();
    this.session = null;
    this.past = [];
  }
}

function joinUrl(base: string, file: string): string {
  return base.endsWith("/") ? base + file : `${base}/${file}`;
}

async function fetchWithProgress(
  url: string,
  onProgress?: (frac: number, text: string) => void,
): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(`failed to fetch shard ${url}: ${res.status}`);
  }
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) {
      const frac = total > 0 ? received / total : 0;
      onProgress(frac, `downloading shard ${(received / 1e6).toFixed(1)}MB`);
    }
  }
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    buf.set(c, offset);
    offset += c.length;
  }
  return buf;
}
