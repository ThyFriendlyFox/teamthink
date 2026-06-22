/**
 * Reads a Hugging Face `config.json` and normalizes it into an architecture
 * descriptor plus a set of trait flags that the generic WebGPU executor uses to
 * branch the forward pass. This is the "universality lever": supporting a new
 * dense decoder family is mostly a matter of mapping its config here rather than
 * writing new kernels.
 */

/** Hugging Face CDN host for direct, browser->HF weight reads. */
export const HF_DIRECT_BASE = "https://huggingface.co";

/**
 * Build a resolve URL for a repo file. Points straight at the Hugging Face CDN
 * so weight range-reads happen browser->HF and never touch our origin — HF
 * serves permissive CORS and `Accept-Ranges: bytes` on `resolve/`, so partial
 * reads work directly from the browser. Gated repos that require a token aren't
 * supported in this fully-static deployment; use ungated repos.
 */
export function hfFileUrl(repo: string, file: string, rev = "main"): string {
  return `${HF_DIRECT_BASE}/${repo}/resolve/${rev}/${file}`;
}

export type ModelFamily =
  | "llama"
  | "mistral"
  | "qwen2"
  | "qwen3"
  | "gemma"
  | "gemma2"
  | "phi3";

export type MlpActivation = "silu" | "gelu_tanh";
export type RopeType = "default" | "llama3";

export interface RopeScaling {
  factor: number;
  lowFreqFactor: number;
  highFreqFactor: number;
  originalMaxPos: number;
}

/** Branch flags consumed by the generic decoder forward. */
export interface ArchTraits {
  family: ModelFamily;
  /** phi3: q/k/v projected by a single fused matrix. */
  qkvFused: boolean;
  /** phi3: gate+up projected by a single fused matrix. */
  gateUpFused: boolean;
  /** qwen2: additive bias on q/k/v projections. */
  qkvBias: boolean;
  /** qwen3: RMSNorm applied to q and k per-head before RoPE. */
  qkNorm: boolean;
  /** gemma family: RMSNorm uses (1 + weight). */
  normOffset: boolean;
  mlpActivation: MlpActivation;
  /** gemma: embeddings scaled by sqrt(hidden) after lookup. */
  embeddingScale: number | null;
  /** gemma2: soft-cap on attention logits. */
  attnLogitSoftcap: number | null;
  /** gemma2: soft-cap on final logits. */
  finalLogitSoftcap: number | null;
  /** gemma2: extra pre/post feed-forward RMSNorms around the MLP. */
  extraMlpNorms: boolean;
  ropeType: RopeType;
}

export interface ArchDescriptor {
  repo: string;
  modelType: string;
  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  rmsNormEps: number;
  ropeTheta: number;
  ropeScaling: RopeScaling | null;
  /** Attention score scale (1/sqrt(query_pre_attn_scalar) or 1/sqrt(headDim)). */
  attnScale: number;
  tieWordEmbeddings: boolean;
  eosTokenId: number | number[];
  maxPositionEmbeddings: number;
  traits: ArchTraits;
}

/** Raw shape of the fields we read from config.json. */
interface RawConfig {
  model_type?: string;
  architectures?: string[];
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  intermediate_size?: number;
  vocab_size?: number;
  rms_norm_eps?: number;
  rope_theta?: number;
  rope_scaling?: {
    rope_type?: string;
    type?: string;
    factor?: number;
    low_freq_factor?: number;
    high_freq_factor?: number;
    original_max_position_embeddings?: number;
  } | null;
  hidden_act?: string;
  hidden_activation?: string;
  attention_bias?: boolean;
  tie_word_embeddings?: boolean;
  eos_token_id?: number | number[];
  max_position_embeddings?: number;
  query_pre_attn_scalar?: number;
  attn_logit_softcapping?: number | null;
  final_logit_softcapping?: number | null;
  num_local_experts?: number;
  num_experts?: number;
}

const SUPPORTED_FAMILIES: Record<string, ModelFamily> = {
  llama: "llama",
  mistral: "mistral",
  qwen2: "qwen2",
  qwen2_vl: "qwen2",
  qwen3: "qwen3",
  gemma: "gemma",
  gemma2: "gemma2",
  gemma3_text: "gemma2",
  gemma3: "gemma2",
  phi3: "phi3",
  phi: "phi3",
  smollm: "llama",
  smollm2: "llama",
};

export async function fetchConfig(repo: string): Promise<RawConfig> {
  const res = await fetch(hfFileUrl(repo, "config.json"));
  if (!res.ok) {
    throw new Error(`config.json fetch failed for ${repo}: ${res.status}`);
  }
  return (await res.json()) as RawConfig;
}

export function describeArch(repo: string, raw: RawConfig): ArchDescriptor {
  const modelType = (raw.model_type ?? "").toLowerCase();
  const family = SUPPORTED_FAMILIES[modelType];
  if (!family) {
    throw new Error(
      `unsupported architecture "${modelType || "unknown"}" for ${repo}. ` +
        `Supported families: ${[...new Set(Object.values(SUPPORTED_FAMILIES))].join(", ")}.`,
    );
  }
  if ((raw.num_local_experts ?? raw.num_experts ?? 0) > 0) {
    throw new Error(`mixture-of-experts models are not yet supported (${repo}).`);
  }

  const hiddenSize = req(raw.hidden_size, "hidden_size");
  const numAttentionHeads = req(raw.num_attention_heads, "num_attention_heads");
  const numKeyValueHeads = raw.num_key_value_heads ?? numAttentionHeads;
  const headDim = raw.head_dim ?? Math.floor(hiddenSize / numAttentionHeads);
  if (headDim > 256) {
    throw new Error(`head_dim ${headDim} exceeds the executor limit of 256.`);
  }

  const isGemma = family === "gemma" || family === "gemma2";
  const isGemma2 = family === "gemma2";

  const act = (raw.hidden_activation ?? raw.hidden_act ?? "silu").toLowerCase();
  const mlpActivation: MlpActivation = act.includes("gelu")
    ? "gelu_tanh"
    : "silu";

  let ropeScaling: RopeScaling | null = null;
  let ropeType: RopeType = "default";
  if (raw.rope_scaling) {
    const rt = (raw.rope_scaling.rope_type ?? raw.rope_scaling.type ?? "").toLowerCase();
    if (rt === "llama3") {
      ropeType = "llama3";
      ropeScaling = {
        factor: raw.rope_scaling.factor ?? 8,
        lowFreqFactor: raw.rope_scaling.low_freq_factor ?? 1,
        highFreqFactor: raw.rope_scaling.high_freq_factor ?? 4,
        originalMaxPos:
          raw.rope_scaling.original_max_position_embeddings ?? 8192,
      };
    }
  }

  const queryPreAttn = isGemma2 ? raw.query_pre_attn_scalar ?? headDim : headDim;
  const attnScale = 1 / Math.sqrt(queryPreAttn);

  const traits: ArchTraits = {
    family,
    qkvFused: family === "phi3",
    gateUpFused: family === "phi3",
    qkvBias: family === "qwen2" || raw.attention_bias === true,
    qkNorm: family === "qwen3",
    normOffset: isGemma,
    mlpActivation,
    embeddingScale: isGemma ? Math.sqrt(hiddenSize) : null,
    attnLogitSoftcap: isGemma2 ? raw.attn_logit_softcapping ?? null : null,
    finalLogitSoftcap: isGemma2 ? raw.final_logit_softcapping ?? null : null,
    extraMlpNorms: isGemma2,
    ropeType,
  };

  return {
    repo,
    modelType,
    numLayers: req(raw.num_hidden_layers, "num_hidden_layers"),
    hiddenSize,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    intermediateSize: req(raw.intermediate_size, "intermediate_size"),
    vocabSize: req(raw.vocab_size, "vocab_size"),
    rmsNormEps: raw.rms_norm_eps ?? 1e-6,
    ropeTheta: raw.rope_theta ?? 10000,
    ropeScaling,
    attnScale,
    tieWordEmbeddings: raw.tie_word_embeddings ?? isGemma,
    eosTokenId: raw.eos_token_id ?? 0,
    maxPositionEmbeddings: raw.max_position_embeddings ?? 4096,
    traits,
  };
}

export async function fetchArchDescriptor(repo: string): Promise<ArchDescriptor> {
  return describeArch(repo, await fetchConfig(repo));
}

/** Precompute RoPE inverse frequencies (length headDim/2), applying scaling. */
export function computeInvFreq(desc: ArchDescriptor): Float32Array {
  const half = Math.floor(desc.headDim / 2);
  const inv = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    inv[k] = 1 / Math.pow(desc.ropeTheta, (2 * k) / desc.headDim);
  }
  if (desc.traits.ropeType === "llama3" && desc.ropeScaling) {
    const { factor, lowFreqFactor, highFreqFactor, originalMaxPos } =
      desc.ropeScaling;
    const lowWavelen = originalMaxPos / lowFreqFactor;
    const highWavelen = originalMaxPos / highFreqFactor;
    for (let k = 0; k < half; k++) {
      const wavelen = (2 * Math.PI) / inv[k];
      if (wavelen > lowWavelen) {
        inv[k] = inv[k] / factor;
      } else if (wavelen < highWavelen) {
        // unchanged
      } else {
        const smooth =
          (originalMaxPos / wavelen - lowFreqFactor) /
          (highFreqFactor - lowFreqFactor);
        inv[k] = (1 - smooth) * (inv[k] / factor) + smooth * inv[k];
      }
    }
  }
  return inv;
}

function req<T>(v: T | undefined, name: string): T {
  if (v === undefined || v === null) {
    throw new Error(`config.json missing required field: ${name}`);
  }
  return v;
}
