/**
 * Typed wrappers around the WGSL kernels. Each records one dispatch into an
 * open compute pass. Uniform/scratch buffers are pushed onto ctx.trash and
 * freed when the pass submits.
 */

import { Gpu, type EncCtx } from "@/lib/engine/webgpu/device";
import {
  APPEND_KV,
  ATTENTION,
  GLU,
  MATMUL,
  RESIDUAL,
  RMSNORM,
  ROPE,
  SOFTCAP,
} from "@/lib/engine/webgpu/kernels";

const WG = 64;
const ceil = (n: number, d: number) => Math.ceil(n / d);

export interface MatmulOpts {
  nOffset?: number;
  nTotal?: number;
}

/** Y[M, nTotal][nOffset:] = X[M,K] @ W[N,K]^T (+ bias). */
export function matmul(
  g: Gpu,
  ctx: EncCtx,
  x: GPUBuffer,
  w: GPUBuffer,
  bias: GPUBuffer | null,
  y: GPUBuffer,
  M: number,
  N: number,
  K: number,
  opts: MatmulOpts = {},
): void {
  const nTotal = opts.nTotal ?? N;
  const nOffset = opts.nOffset ?? 0;
  const dummy = bias ? null : g.storage(4);
  if (dummy) ctx.trash.push(dummy);
  const p = g.uniform(
    [M, N, K, bias ? 1 : 0, nOffset, nTotal],
    ["u", "u", "u", "u", "u", "u"],
  );
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("matmul", MATMUL), [x, w, bias ?? dummy!, y, p], [
    ceil(N, 8),
    ceil(M, 8),
    1,
  ]);
}

export function rmsnorm(
  g: Gpu,
  ctx: EncCtx,
  x: GPUBuffer,
  weight: GPUBuffer,
  y: GPUBuffer,
  M: number,
  K: number,
  offset: boolean,
  eps: number,
): void {
  const p = g.uniform([M, K, offset ? 1 : 0, eps], ["u", "u", "u", "f"]);
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("rmsnorm", RMSNORM), [x, weight, y, p], [
    ceil(M, WG),
    1,
    1,
  ]);
}

export function rope(
  g: Gpu,
  ctx: EncCtx,
  t: GPUBuffer,
  inv: GPUBuffer,
  pos: GPUBuffer,
  rows: number,
  heads: number,
  headDim: number,
): void {
  const half = Math.floor(headDim / 2);
  const p = g.uniform([rows, heads, headDim, half], ["u", "u", "u", "u"]);
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("rope", ROPE), [t, inv, pos, p], [
    ceil(half, WG),
    heads,
    rows,
  ]);
}

export function appendKV(
  g: Gpu,
  ctx: EncCtx,
  newK: GPUBuffer,
  newV: GPUBuffer,
  kCache: GPUBuffer,
  vCache: GPUBuffer,
  seq: number,
  nKV: number,
  headDim: number,
  cap: number,
  pastLen: number,
): void {
  const p = g.uniform([seq, nKV, headDim, cap, pastLen], [
    "u",
    "u",
    "u",
    "u",
    "u",
  ]);
  ctx.trash.push(p);
  g.encode(
    ctx,
    g.pipeline("appendkv", APPEND_KV),
    [newK, newV, kCache, vCache, p],
    [ceil(headDim, WG), nKV, seq],
  );
}

export interface AttnParams {
  seq: number;
  nHeads: number;
  nKV: number;
  headDim: number;
  cap: number;
  pastLen: number;
  scale: number;
  softcap: number;
}

export function attention(
  g: Gpu,
  ctx: EncCtx,
  q: GPUBuffer,
  kCache: GPUBuffer,
  vCache: GPUBuffer,
  out: GPUBuffer,
  a: AttnParams,
): void {
  const p = g.uniform(
    [a.seq, a.nHeads, a.nKV, a.headDim, a.cap, a.pastLen, a.scale, a.softcap],
    ["u", "u", "u", "u", "u", "u", "f", "f"],
  );
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("attention", ATTENTION), [q, kCache, vCache, out, p], [
    ceil(a.nHeads, 8),
    ceil(a.seq, 8),
    1,
  ]);
}

export function glu(
  g: Gpu,
  ctx: EncCtx,
  gate: GPUBuffer,
  up: GPUBuffer,
  out: GPUBuffer,
  n: number,
  act: 0 | 1,
): void {
  const p = g.uniform([n, act], ["u", "u"]);
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("glu", GLU), [gate, up, out, p], [ceil(n, WG), 1, 1]);
}

export function residual(
  g: Gpu,
  ctx: EncCtx,
  a: GPUBuffer,
  b: GPUBuffer,
  n: number,
): void {
  const p = g.uniform([n], ["u"]);
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("residual", RESIDUAL), [a, b, p], [ceil(n, WG), 1, 1]);
}

export function softcap(
  g: Gpu,
  ctx: EncCtx,
  x: GPUBuffer,
  n: number,
  cap: number,
): void {
  const p = g.uniform([n, cap], ["u", "f"]);
  ctx.trash.push(p);
  g.encode(ctx, g.pipeline("softcap", SOFTCAP), [x, p], [ceil(n, WG), 1, 1]);
}
