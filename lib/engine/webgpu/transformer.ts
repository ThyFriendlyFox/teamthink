/**
 * Generic decoder forward pass over a contiguous range of layers. The block
 * structure is driven by ArchTraits so one implementation covers Llama/Mistral/
 * Qwen2/Qwen3/Gemma(2)/Phi-3-style dense decoders. KV cache lives on the GPU,
 * one (k, v) pair per local layer; only hidden states cross shard boundaries.
 */

import type { ArchDescriptor } from "@/lib/engine/hf/config";
import { Gpu } from "@/lib/engine/webgpu/device";
import {
  appendKV,
  attention,
  glu,
  matmul,
  residual,
  rmsnorm,
  rope,
} from "@/lib/engine/webgpu/ops";

export interface LayerWeights {
  inputLn: GPUBuffer;
  qW: GPUBuffer;
  qB: GPUBuffer | null;
  kW: GPUBuffer;
  kB: GPUBuffer | null;
  vW: GPUBuffer;
  vB: GPUBuffer | null;
  oW: GPUBuffer;
  qNorm: GPUBuffer | null;
  kNorm: GPUBuffer | null;
  postAttnLn: GPUBuffer;
  preFfLn: GPUBuffer | null;
  postFfLn: GPUBuffer | null;
  gateW: GPUBuffer;
  upW: GPUBuffer;
  downW: GPUBuffer;
}

export interface LayerKV {
  k: GPUBuffer;
  v: GPUBuffer;
}

export interface ForwardCtx {
  desc: ArchDescriptor;
  cap: number;
  invFreq: GPUBuffer;
}

/**
 * Apply one decoder layer to `hidden` (updated in place). `posBuf` holds the
 * absolute position of each of the `seq` rows; `pastLen` is the KV length prior
 * to this step.
 */
export function forwardLayer(
  g: Gpu,
  c: ForwardCtx,
  w: LayerWeights,
  kv: LayerKV,
  hidden: GPUBuffer,
  posBuf: GPUBuffer,
  seq: number,
  pastLen: number,
): void {
  const d = c.desc;
  const t = d.traits;
  const H = d.hiddenSize;
  const hd = d.headDim;
  const nH = d.numAttentionHeads;
  const nKV = d.numKeyValueHeads;
  const I = d.intermediateSize;
  const qDim = nH * hd;
  const kvDim = nKV * hd;
  const eps = d.rmsNormEps;

  // --- attention sublayer ---
  {
    const ctx = g.beginPass();
    const n1 = g.storage(seq * H * 4);
    ctx.trash.push(n1);
    rmsnorm(g, ctx, hidden, w.inputLn, n1, seq, H, t.normOffset, eps);

    const q = g.storage(seq * qDim * 4);
    const k = g.storage(seq * kvDim * 4);
    const v = g.storage(seq * kvDim * 4);
    ctx.trash.push(q, k, v);
    matmul(g, ctx, n1, w.qW, w.qB, q, seq, qDim, H);
    matmul(g, ctx, n1, w.kW, w.kB, k, seq, kvDim, H);
    matmul(g, ctx, n1, w.vW, w.vB, v, seq, kvDim, H);

    if (t.qkNorm && w.qNorm && w.kNorm) {
      rmsnorm(g, ctx, q, w.qNorm, q, seq * nH, hd, false, eps);
      rmsnorm(g, ctx, k, w.kNorm, k, seq * nKV, hd, false, eps);
    }

    rope(g, ctx, q, c.invFreq, posBuf, seq, nH, hd);
    rope(g, ctx, k, c.invFreq, posBuf, seq, nKV, hd);
    appendKV(g, ctx, k, v, kv.k, kv.v, seq, nKV, hd, c.cap, pastLen);

    const attn = g.storage(seq * qDim * 4);
    ctx.trash.push(attn);
    attention(g, ctx, q, kv.k, kv.v, attn, {
      seq,
      nHeads: nH,
      nKV,
      headDim: hd,
      cap: c.cap,
      pastLen,
      scale: d.attnScale,
      softcap: t.attnLogitSoftcap ?? 0,
    });

    const oOut = g.storage(seq * H * 4);
    ctx.trash.push(oOut);
    matmul(g, ctx, attn, w.oW, null, oOut, seq, H, qDim);

    if (t.extraMlpNorms && w.postAttnLn) {
      const normed = g.storage(seq * H * 4);
      ctx.trash.push(normed);
      rmsnorm(g, ctx, oOut, w.postAttnLn, normed, seq, H, t.normOffset, eps);
      residual(g, ctx, hidden, normed, seq * H);
    } else {
      residual(g, ctx, hidden, oOut, seq * H);
    }
    g.submit(ctx);
  }

  // --- MLP sublayer ---
  {
    const ctx = g.beginPass();
    const n2 = g.storage(seq * H * 4);
    ctx.trash.push(n2);
    const ffNorm = t.extraMlpNorms && w.preFfLn ? w.preFfLn : w.postAttnLn;
    rmsnorm(g, ctx, hidden, ffNorm, n2, seq, H, t.normOffset, eps);

    const gate = g.storage(seq * I * 4);
    const up = g.storage(seq * I * 4);
    ctx.trash.push(gate, up);
    matmul(g, ctx, n2, w.gateW, null, gate, seq, I, H);
    matmul(g, ctx, n2, w.upW, null, up, seq, I, H);

    const gl = g.storage(seq * I * 4);
    ctx.trash.push(gl);
    glu(g, ctx, gate, up, gl, seq * I, t.mlpActivation === "silu" ? 0 : 1);

    const down = g.storage(seq * H * 4);
    ctx.trash.push(down);
    matmul(g, ctx, gl, w.downW, null, down, seq, H, I);

    if (t.extraMlpNorms && w.postFfLn) {
      const normed = g.storage(seq * H * 4);
      ctx.trash.push(normed);
      rmsnorm(g, ctx, down, w.postFfLn, normed, seq, H, t.normOffset, eps);
      residual(g, ctx, hidden, normed, seq * H);
    } else {
      residual(g, ctx, hidden, down, seq * H);
    }
    g.submit(ctx);
  }
}

/**
 * A large [rows, cols] matrix (embedding / lm_head) split across multiple GPU
 * buffers so each chunk stays under maxStorageBufferBindingSize.
 */
export class ChunkedMatrix {
  private chunks: GPUBuffer[] = [];
  private rowOffsets: number[] = [];
  private rowCounts: number[] = [];

  constructor(
    readonly rows: number,
    readonly cols: number,
  ) {}

  static fromF32(
    g: Gpu,
    data: Float32Array,
    rows: number,
    cols: number,
  ): ChunkedMatrix {
    const m = new ChunkedMatrix(rows, cols);
    const maxRows = Math.max(1, Math.floor(g.maxBindingBytes / (cols * 4)));
    for (let r = 0; r < rows; r += maxRows) {
      const cnt = Math.min(maxRows, rows - r);
      m.chunks.push(g.upload(data.subarray(r * cols, (r + cnt) * cols)));
      m.rowOffsets.push(r);
      m.rowCounts.push(cnt);
    }
    return m;
  }

  /** y[M, rows] = x[M, cols] @ this^T, accumulated chunk by chunk. */
  matmulInto(g: Gpu, x: GPUBuffer, y: GPUBuffer, M: number): void {
    const ctx = g.beginPass();
    for (let i = 0; i < this.chunks.length; i++) {
      matmul(g, ctx, x, this.chunks[i], null, y, M, this.rowCounts[i], this.cols, {
        nOffset: this.rowOffsets[i],
        nTotal: this.rows,
      });
    }
    g.submit(ctx);
  }

  destroy(): void {
    for (const c of this.chunks) c.destroy();
    this.chunks = [];
  }
}

/** CPU-side embedding lookup; avoids binding the full embedding matrix on GPU. */
export function embedGather(
  embed: Float32Array,
  tokens: number[],
  hidden: number,
  scale: number,
): Float32Array {
  const out = new Float32Array(tokens.length * hidden);
  for (let s = 0; s < tokens.length; s++) {
    const src = tokens[s] * hidden;
    for (let h = 0; h < hidden; h++) out[s * hidden + h] = embed[src + h] * scale;
  }
  return out;
}
