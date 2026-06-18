/**
 * WGSL compute kernels for the generic transformer executor. All tensors are
 * f32, row-major. Kernels are intentionally simple (correctness first); the
 * decode path runs with seq=1, so the dominant cost is the per-token matmuls.
 */

/** Y[M, nTotal] (slice [nOffset, nOffset+N)) = X[M,K] @ W[N,K]^T (+ bias[N]). */
export const MATMUL = /* wgsl */ `
struct P { M:u32, N:u32, K:u32, hasBias:u32, nOffset:u32, nTotal:u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (m >= p.M || n >= p.N) { return; }
  var acc = 0.0;
  let xoff = m * p.K;
  let woff = n * p.K;
  for (var k = 0u; k < p.K; k = k + 1u) {
    acc = acc + x[xoff + k] * w[woff + k];
  }
  if (p.hasBias == 1u) { acc = acc + bias[n]; }
  y[m * p.nTotal + p.nOffset + n] = acc;
}
`;

/** RMSNorm over the last dim K of X[M,K]; weight[K]; optional (1+w) offset. */
export const RMSNORM = /* wgsl */ `
struct P { M:u32, K:u32, offset:u32, eps:f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m = gid.x;
  if (m >= p.M) { return; }
  let off = m * p.K;
  var ss = 0.0;
  for (var k = 0u; k < p.K; k = k + 1u) { ss = ss + x[off + k] * x[off + k]; }
  let inv = 1.0 / sqrt(ss / f32(p.K) + p.eps);
  for (var k = 0u; k < p.K; k = k + 1u) {
    var wt = weight[k];
    if (p.offset == 1u) { wt = wt + 1.0; }
    y[off + k] = x[off + k] * inv * wt;
  }
}
`;

/** Rotary embedding in place over T[rows, heads, headDim]; inv[half]; pos[rows]. */
export const ROPE = /* wgsl */ `
struct P { rows:u32, heads:u32, headDim:u32, half:u32 };
@group(0) @binding(0) var<storage, read_write> t: array<f32>;
@group(0) @binding(1) var<storage, read> inv: array<f32>;
@group(0) @binding(2) var<storage, read> pos: array<i32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let head = gid.y;
  let row = gid.z;
  if (d >= p.half || head >= p.heads || row >= p.rows) { return; }
  let angle = f32(pos[row]) * inv[d];
  let c = cos(angle);
  let s = sin(angle);
  let base = (row * p.heads + head) * p.headDim;
  let a = t[base + d];
  let b = t[base + d + p.half];
  t[base + d] = a * c - b * s;
  t[base + d + p.half] = b * c + a * s;
}
`;

/** Scatter new K/V rows into the per-kv-head ring caches at position pastLen. */
export const APPEND_KV = /* wgsl */ `
struct P { seq:u32, nKV:u32, headDim:u32, cap:u32, pastLen:u32 };
@group(0) @binding(0) var<storage, read> newK: array<f32>;
@group(0) @binding(1) var<storage, read> newV: array<f32>;
@group(0) @binding(2) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let kv = gid.y;
  let s = gid.z;
  if (d >= p.headDim || kv >= p.nKV || s >= p.seq) { return; }
  let src = (s * p.nKV + kv) * p.headDim + d;
  let dst = (kv * p.cap + p.pastLen + s) * p.headDim + d;
  kCache[dst] = newK[src];
  vCache[dst] = newV[src];
}
`;

/** Causal multi-head attention with GQA, online softmax, optional soft-cap. */
export const ATTENTION = /* wgsl */ `
struct P { seq:u32, nHeads:u32, nKV:u32, headDim:u32, cap:u32, pastLen:u32, scale:f32, softcap:f32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kCache: array<f32>;
@group(0) @binding(2) var<storage, read> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  let i = gid.y;
  if (h >= p.nHeads || i >= p.seq) { return; }
  let group = p.nHeads / p.nKV;
  let kv = h / group;
  let qbase = (i * p.nHeads + h) * p.headDim;
  let qpos = p.pastLen + i;
  var m = -1e30;
  var l = 0.0;
  var acc: array<f32, 256>;
  for (var d = 0u; d < p.headDim; d = d + 1u) { acc[d] = 0.0; }
  for (var j = 0u; j <= qpos; j = j + 1u) {
    let kbase = (kv * p.cap + j) * p.headDim;
    var dot = 0.0;
    for (var d = 0u; d < p.headDim; d = d + 1u) {
      dot = dot + q[qbase + d] * kCache[kbase + d];
    }
    var score = dot * p.scale;
    if (p.softcap > 0.0) { score = p.softcap * tanh(score / p.softcap); }
    let newm = max(m, score);
    let corr = exp(m - newm);
    let pw = exp(score - newm);
    l = l * corr + pw;
    let vbase = (kv * p.cap + j) * p.headDim;
    for (var d = 0u; d < p.headDim; d = d + 1u) {
      acc[d] = acc[d] * corr + pw * vCache[vbase + d];
    }
    m = newm;
  }
  let obase = (i * p.nHeads + h) * p.headDim;
  let linv = 1.0 / l;
  for (var d = 0u; d < p.headDim; d = d + 1u) { out[obase + d] = acc[d] * linv; }
}
`;

/** Gated MLP activation: out = act(gate) * up. act 0=SiLU, 1=GELU(tanh). */
export const GLU = /* wgsl */ `
struct P { n:u32, act:u32 };
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let g = gate[i];
  var a = 0.0;
  if (p.act == 0u) {
    a = g / (1.0 + exp(-g));
  } else {
    a = 0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
  }
  out[i] = a * up[i];
}
`;

/** Elementwise a += b. */
export const RESIDUAL = /* wgsl */ `
struct P { n:u32 };
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  a[i] = a[i] + b[i];
}
`;

/** out[i] = cap * tanh(x[i]/cap) over a logits buffer. */
export const SOFTCAP = /* wgsl */ `
struct P { n:u32, cap:f32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  x[i] = p.cap * tanh(x[i] / p.cap);
}
`;
