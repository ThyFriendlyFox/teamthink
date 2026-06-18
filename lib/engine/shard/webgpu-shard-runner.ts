/**
 * Runs a contiguous slice of decoder layers entirely in WebGPU, reading its
 * weights directly from the model's `safetensors` via HTTP Range requests (no
 * offline export, no per-model preparation). Owns the KV cache for just its
 * layers; only hidden states (or token ids, for the head) cross the network.
 *
 * Drop-in for the legacy OnnxShardRunner: same load -> run -> reset -> dispose
 * lifecycle and the same TensorLike in/out contract.
 */

import { computeInvFreq, type ArchDescriptor } from "@/lib/engine/hf/config";
import {
  fetchTensorF32,
  type SafetensorsIndex,
  type TensorEntry,
} from "@/lib/engine/hf/safetensors";
import { getGpu, Gpu } from "@/lib/engine/webgpu/device";
import { rmsnorm, softcap } from "@/lib/engine/webgpu/ops";
import {
  ChunkedMatrix,
  embedGather,
  forwardLayer,
  type ForwardCtx,
  type LayerKV,
  type LayerWeights,
} from "@/lib/engine/webgpu/transformer";
import type {
  ShardRange,
  TensorLike,
} from "@/lib/engine/shard/model-descriptor";

const MAX_CONTEXT = 2048;

export class WebGPUShardRunner {
  private g!: Gpu;
  private layers: LayerWeights[] = [];
  private kvs: LayerKV[] = [];
  private invFreq!: GPUBuffer;
  private finalNorm: GPUBuffer | null = null;
  private lmHead: ChunkedMatrix | null = null;
  /** CPU embedding matrix [vocab, hidden] for first-shard token lookup. */
  private embed: Float32Array | null = null;
  private cap: number;
  private pastLen = 0;

  constructor(
    private readonly desc: ArchDescriptor,
    private readonly index: SafetensorsIndex,
    readonly range: ShardRange,
  ) {
    this.cap = Math.min(desc.maxPositionEmbeddings || MAX_CONTEXT, MAX_CONTEXT);
  }

  get isFirst(): boolean {
    return this.range.isFirst;
  }
  get isLast(): boolean {
    return this.range.isLast;
  }

  async load(onProgress?: (frac: number, text: string) => void): Promise<void> {
    this.g = await getGpu();
    const d = this.desc;
    const repo = d.repo;

    // Count the tensors we will fetch for a coarse progress fraction.
    const names = this.tensorPlan();
    let done = 0;
    const total = names.length;
    const get = async (name: string): Promise<Float32Array> => {
      const entry = this.entry(name);
      const data = await fetchTensorF32(repo, entry);
      done++;
      onProgress?.(done / total, `fetching ${name} (${done}/${total})`);
      return data;
    };

    this.invFreq = this.g.upload(computeInvFreq(d));

    // First shard: keep the embedding matrix on the CPU for fast row gather.
    if (this.range.isFirst) {
      this.embed = await get("model.embed_tokens.weight");
    }

    for (let i = this.range.layerStart; i < this.range.layerEnd; i++) {
      this.layers.push(await this.loadLayer(i, get));
      this.kvs.push({
        k: this.g.storage(d.numKeyValueHeads * this.cap * d.headDim * 4),
        v: this.g.storage(d.numKeyValueHeads * this.cap * d.headDim * 4),
      });
    }

    // Last shard: final norm + lm_head (tied to embeddings when applicable).
    if (this.range.isLast) {
      this.finalNorm = this.g.upload(await get("model.norm.weight"));
      let head: Float32Array;
      if (d.tieWordEmbeddings) {
        head = this.embed ?? (await get("model.embed_tokens.weight"));
      } else {
        head = await get("lm_head.weight");
      }
      this.lmHead = ChunkedMatrix.fromF32(this.g, head, d.vocabSize, d.hiddenSize);
    }
  }

  reset(): void {
    this.pastLen = 0;
  }

  async run(primary: TensorLike): Promise<TensorLike> {
    const d = this.desc;
    const H = d.hiddenSize;
    const seq = primary.dims[1];

    let hidden: GPUBuffer;
    if (this.range.isFirst) {
      const tokens = Array.from(primary.data as BigInt64Array, (n) => Number(n));
      const emb = embedGather(
        this.embed!,
        tokens,
        H,
        d.traits.embeddingScale ?? 1,
      );
      hidden = this.g.upload(emb);
    } else {
      hidden = this.g.upload(primary.data as Float32Array);
    }

    const positions = new Int32Array(seq);
    for (let i = 0; i < seq; i++) positions[i] = this.pastLen + i;
    const posBuf = this.g.upload(positions);

    const ctx: ForwardCtx = { desc: d, cap: this.cap, invFreq: this.invFreq };
    for (let li = 0; li < this.layers.length; li++) {
      forwardLayer(
        this.g,
        ctx,
        this.layers[li],
        this.kvs[li],
        hidden,
        posBuf,
        seq,
        this.pastLen,
      );
    }
    this.pastLen += seq;

    if (this.range.isLast) {
      const normed = this.g.storage(seq * H * 4);
      const pass = this.g.beginPass();
      rmsnorm(this.g, pass, hidden, this.finalNorm!, normed, seq, H, d.traits.normOffset, d.rmsNormEps);
      this.g.submit(pass);

      const lastRow = this.g.storage(H * 4);
      this.g.copy(normed, (seq - 1) * H * 4, lastRow, 0, H * 4);

      const logits = this.g.storage(d.vocabSize * 4);
      this.lmHead!.matmulInto(this.g, lastRow, logits, 1);

      if (d.traits.finalLogitSoftcap) {
        const cp = this.g.beginPass();
        softcap(this.g, cp, logits, d.vocabSize, d.traits.finalLogitSoftcap);
        this.g.submit(cp);
      }

      const data = await this.g.read(logits, d.vocabSize * 4);
      hidden.destroy();
      posBuf.destroy();
      normed.destroy();
      lastRow.destroy();
      logits.destroy();
      return { dims: [1, 1, d.vocabSize], data };
    }

    const data = await this.g.read(hidden, seq * H * 4);
    hidden.destroy();
    posBuf.destroy();
    return { dims: [1, seq, H], data };
  }

  dispose(): void {
    for (const w of this.layers) {
      for (const b of [
        w.inputLn,
        w.qW,
        w.qB,
        w.kW,
        w.kB,
        w.vW,
        w.vB,
        w.oW,
        w.qNorm,
        w.kNorm,
        w.postAttnLn,
        w.preFfLn,
        w.postFfLn,
        w.gateW,
        w.upW,
        w.downW,
      ]) {
        b?.destroy();
      }
    }
    for (const kv of this.kvs) {
      kv.k.destroy();
      kv.v.destroy();
    }
    this.invFreq?.destroy();
    this.finalNorm?.destroy();
    this.lmHead?.destroy();
    this.layers = [];
    this.kvs = [];
    this.embed = null;
  }

  // --- weight loading -------------------------------------------------------

  private async loadLayer(
    i: number,
    get: (name: string) => Promise<Float32Array>,
  ): Promise<LayerWeights> {
    const d = this.desc;
    const t = d.traits;
    const base = `model.layers.${i}.`;
    const g = this.g;
    const qDim = d.numAttentionHeads * d.headDim;
    const kvDim = d.numKeyValueHeads * d.headDim;

    const inputLn = g.upload(await get(`${base}input_layernorm.weight`));
    const postAttnLn = g.upload(
      await get(`${base}post_attention_layernorm.weight`),
    );

    let qW: GPUBuffer;
    let kW: GPUBuffer;
    let vW: GPUBuffer;
    let qB: GPUBuffer | null = null;
    let kB: GPUBuffer | null = null;
    let vB: GPUBuffer | null = null;
    if (t.qkvFused) {
      const qkv = await get(`${base}self_attn.qkv_proj.weight`);
      qW = g.upload(qkv.subarray(0, qDim * d.hiddenSize));
      kW = g.upload(qkv.subarray(qDim * d.hiddenSize, (qDim + kvDim) * d.hiddenSize));
      vW = g.upload(
        qkv.subarray((qDim + kvDim) * d.hiddenSize, (qDim + 2 * kvDim) * d.hiddenSize),
      );
    } else {
      qW = g.upload(await get(`${base}self_attn.q_proj.weight`));
      kW = g.upload(await get(`${base}self_attn.k_proj.weight`));
      vW = g.upload(await get(`${base}self_attn.v_proj.weight`));
      if (t.qkvBias) {
        qB = g.upload(await get(`${base}self_attn.q_proj.bias`));
        kB = g.upload(await get(`${base}self_attn.k_proj.bias`));
        vB = g.upload(await get(`${base}self_attn.v_proj.bias`));
      }
    }
    const oW = g.upload(await get(`${base}self_attn.o_proj.weight`));

    let qNorm: GPUBuffer | null = null;
    let kNorm: GPUBuffer | null = null;
    if (t.qkNorm) {
      qNorm = g.upload(await get(`${base}self_attn.q_norm.weight`));
      kNorm = g.upload(await get(`${base}self_attn.k_norm.weight`));
    }

    let preFfLn: GPUBuffer | null = null;
    let postFfLn: GPUBuffer | null = null;
    if (t.extraMlpNorms) {
      preFfLn = g.upload(await get(`${base}pre_feedforward_layernorm.weight`));
      postFfLn = g.upload(await get(`${base}post_feedforward_layernorm.weight`));
    }

    let gateW: GPUBuffer;
    let upW: GPUBuffer;
    if (t.gateUpFused) {
      const gu = await get(`${base}mlp.gate_up_proj.weight`);
      const I = d.intermediateSize;
      gateW = g.upload(gu.subarray(0, I * d.hiddenSize));
      upW = g.upload(gu.subarray(I * d.hiddenSize, 2 * I * d.hiddenSize));
    } else {
      gateW = g.upload(await get(`${base}mlp.gate_proj.weight`));
      upW = g.upload(await get(`${base}mlp.up_proj.weight`));
    }
    const downW = g.upload(await get(`${base}mlp.down_proj.weight`));

    return {
      inputLn,
      qW,
      qB,
      kW,
      kB,
      vW,
      vB,
      oW,
      qNorm,
      kNorm,
      postAttnLn,
      preFfLn,
      postFfLn,
      gateW,
      upW,
      downW,
    };
  }

  /** Names of the tensors this shard will fetch (for progress accounting). */
  private tensorPlan(): string[] {
    const t = this.desc.traits;
    const names: string[] = [];
    if (this.range.isFirst) names.push("model.embed_tokens.weight");
    for (let i = this.range.layerStart; i < this.range.layerEnd; i++) {
      const b = `model.layers.${i}.`;
      names.push(`${b}input_layernorm.weight`, `${b}post_attention_layernorm.weight`);
      if (t.qkvFused) names.push(`${b}self_attn.qkv_proj.weight`);
      else {
        names.push(
          `${b}self_attn.q_proj.weight`,
          `${b}self_attn.k_proj.weight`,
          `${b}self_attn.v_proj.weight`,
        );
        if (t.qkvBias)
          names.push(
            `${b}self_attn.q_proj.bias`,
            `${b}self_attn.k_proj.bias`,
            `${b}self_attn.v_proj.bias`,
          );
      }
      names.push(`${b}self_attn.o_proj.weight`);
      if (t.qkNorm)
        names.push(`${b}self_attn.q_norm.weight`, `${b}self_attn.k_norm.weight`);
      if (t.extraMlpNorms)
        names.push(
          `${b}pre_feedforward_layernorm.weight`,
          `${b}post_feedforward_layernorm.weight`,
        );
      if (t.gateUpFused) names.push(`${b}mlp.gate_up_proj.weight`);
      else names.push(`${b}mlp.gate_proj.weight`, `${b}mlp.up_proj.weight`);
      names.push(`${b}mlp.down_proj.weight`);
    }
    if (this.range.isLast) {
      names.push("model.norm.weight");
      if (!this.desc.tieWordEmbeddings) names.push("lm_head.weight");
      else if (!this.range.isFirst) names.push("model.embed_tokens.weight");
    }
    return names;
  }

  private entry(name: string): TensorEntry {
    const e = this.index.tensors.get(name);
    if (!e) throw new Error(`tensor not found in safetensors: ${name}`);
    return e;
  }
}
