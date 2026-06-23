/**
 * Central configuration for the TeamThink grid: ICE servers, signaling TTLs,
 * heartbeat cadence, and the model registry shared across engines.
 */

export const ICE_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
  // Optional public TURN fallback for restrictive NATs. Replace with your own
  // credentials for production reliability.
  ...(process.env.NEXT_PUBLIC_TURN_URL
    ? [
        {
          urls: process.env.NEXT_PUBLIC_TURN_URL,
          username: process.env.NEXT_PUBLIC_TURN_USERNAME,
          credential: process.env.NEXT_PUBLIC_TURN_CREDENTIAL,
        } as RTCIceServer,
      ]
    : []),
];

/** Presence heartbeat cadence and the staleness window for re-claiming work. */
export const HEARTBEAT_INTERVAL_MS = 4000;
export const PEER_STALE_MS = 15000;
export const TASK_STALE_MS = 30000;

/**
 * Signaling transport tuning. The rendezvous is a KV-backed mailbox served by
 * `/api/signal` on our own deployment (no public WebRTC relays). It is
 * event-driven: a client holds one request open and the server returns as soon
 * as a message is waiting (or after `SIGNAL_POLL_HOLD_MS`), so there is no busy
 * polling. Mailboxes auto-expire so KV never accumulates state.
 *
 * Cost is bounded by the mesh design, not by the number of users: only one peer
 * per room (the lowest-id "greeter") holds the room mailbox open to greet
 * newcomers, and every other peer stops talking to the server entirely once it
 * has a mesh link — further connections are brokered peer-to-peer.
 */
export const SIGNAL_TTL_SECONDS = 120;
/** Max time the server holds a long-poll open before returning empty (ms). */
export const SIGNAL_POLL_HOLD_MS = 10000;
/** How often the server checks the mailbox while holding a long-poll (ms). */
export const SIGNAL_POLL_STEP_MS = 1000;
/** Client backoff after a failed/empty signaling request (ms). */
export const SIGNAL_RETRY_MS = 1500;
/** Base path of the signaling endpoint (override for a separate host). */
export const SIGNAL_ENDPOINT =
  process.env.NEXT_PUBLIC_SIGNAL_ENDPOINT ?? "/api/signal";

export type EngineKind = "webllm" | "transformers";

export type ModelModality = "text" | "vision" | "embedding";

export interface ModelSpec {
  /** Stable id used in task routing and CRDT state. */
  id: string;
  /** Human label shown in the UI. */
  label: string;
  engine: EngineKind;
  modality: ModelModality;
  /** Underlying model identifier passed to the engine. */
  modelId: string;
  /** Rough VRAM requirement (MB) used for capability scoring. */
  vramMb: number;
  /**
   * True for f16-quantized builds that need the WebGPU `shader-f16` feature.
   * Nodes whose adapter lacks it are filtered out of this model's candidates.
   */
  requiresShaderF16?: boolean;
  /**
   * Per-model WebLLM ChatOptions overrides merged at load time. Used for models
   * whose prebuilt MLC record needs adjustment — e.g. sliding-window models
   * (Gemma 2/3) require exactly one of `context_window_size` /
   * `sliding_window_size` to be positive, so we disable one here.
   */
  webllmOverrides?: Record<string, unknown>;
  /**
   * Hugging Face repo id (e.g. "HuggingFaceTB/SmolLM2-360M-Instruct"). When set,
   * the model runs via the distributed WebGPU pipeline: its layers are
   * partitioned across the pool and each peer range-fetches only its slice of
   * the `safetensors` weights. No offline preparation is required.
   */
  hfRepo?: string;
}

/**
 * The model registry. WebLLM ids must match MLC's prebuilt model list; the
 * Transformers.js entries use Hugging Face repo ids with ONNX weights.
 */
// All `modelId` values and VRAM figures are taken verbatim from the installed
// @mlc-ai/web-llm prebuilt model catalog. They must match exactly or the model
// library 404s at load time. Ordered small -> large for quick testing.
export const MODELS: ModelSpec[] = [
  {
    id: "smollm2-135m",
    label: "SmolLM2 135M",
    engine: "webllm",
    modality: "text",
    modelId: "SmolLM2-135M-Instruct-q0f16-MLC",
    vramMb: 360,
    requiresShaderF16: true,
  },
  {
    id: "smollm2-360m",
    label: "SmolLM2 360M",
    engine: "webllm",
    modality: "text",
    modelId: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    vramMb: 376,
    requiresShaderF16: true,
  },
  {
    id: "smollm2-360m-f32",
    label: "SmolLM2 360M (no-f16)",
    engine: "webllm",
    modality: "text",
    modelId: "SmolLM2-360M-Instruct-q4f32_1-MLC",
    vramMb: 580,
  },
  {
    id: "tinyllama-1.1b",
    label: "TinyLlama 1.1B Chat",
    engine: "webllm",
    modality: "text",
    modelId: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    vramMb: 697,
    requiresShaderF16: true,
  },
  {
    id: "gemma3-1b",
    label: "Gemma 3 1B Instruct",
    engine: "webllm",
    modality: "text",
    modelId: "gemma3-1b-it-q4f16_1-MLC",
    vramMb: 711,
    requiresShaderF16: true,
    // Use the full context window; disable sliding window (MLC allows only one).
    webllmOverrides: { sliding_window_size: -1 },
  },
  {
    id: "llama-3.2-1b",
    label: "Llama 3.2 1B Instruct",
    engine: "webllm",
    modality: "text",
    modelId: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    vramMb: 879,
    requiresShaderF16: true,
  },
  {
    id: "qwen2.5-0.5b",
    label: "Qwen2.5 0.5B Instruct",
    engine: "webllm",
    modality: "text",
    modelId: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    vramMb: 945,
    requiresShaderF16: true,
  },
  {
    id: "qwen3-0.6b",
    label: "Qwen3 0.6B",
    engine: "webllm",
    modality: "text",
    modelId: "Qwen3-0.6B-q4f16_1-MLC",
    vramMb: 1403,
    requiresShaderF16: true,
  },
  {
    id: "smollm2-1.7b",
    label: "SmolLM2 1.7B Instruct",
    engine: "webllm",
    modality: "text",
    modelId: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
    vramMb: 1774,
    requiresShaderF16: true,
  },
  {
    id: "gemma-2-2b",
    label: "Gemma 2 2B Instruct",
    engine: "webllm",
    modality: "text",
    modelId: "gemma-2-2b-it-q4f16_1-MLC",
    vramMb: 1895,
    requiresShaderF16: true,
    webllmOverrides: { sliding_window_size: -1 },
  },
  {
    id: "llama-3.2-3b-f32",
    label: "Llama 3.2 3B Instruct (no-f16)",
    engine: "webllm",
    modality: "text",
    modelId: "Llama-3.2-3B-Instruct-q4f32_1-MLC",
    vramMb: 2952,
  },
  {
    id: "phi-3.5-mini",
    label: "Phi 3.5 Mini Instruct",
    engine: "webllm",
    modality: "text",
    modelId: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    vramMb: 3672,
    requiresShaderF16: true,
  },
  {
    id: "smolvlm-256m",
    label: "SmolVLM 256M (vision)",
    engine: "transformers",
    modality: "vision",
    modelId: "HuggingFaceTB/SmolVLM-256M-Instruct",
    vramMb: 900,
  },
];

/**
 * Distributed (pipeline-parallel) models. Their layers are partitioned across
 * the pool at session time and each peer range-fetches only its slice of the
 * weights straight from Hugging Face, so the pool's total hostable model size
 * grows with the number of peers. Any ungated HF repo of a supported dense
 * decoder family works; these are curated known-good defaults. The `vramMb`
 * here is only a rough single-device hint and is not used for partitioning.
 */
export const SHARDED_MODELS: ModelSpec[] = [
  {
    id: "grid-smollm2-135m",
    label: "SmolLM2 135M",
    engine: "transformers",
    modality: "text",
    modelId: "HuggingFaceTB/SmolLM2-135M-Instruct",
    vramMb: 0,
    hfRepo: "HuggingFaceTB/SmolLM2-135M-Instruct",
  },
  {
    id: "grid-smollm2-360m",
    label: "SmolLM2 360M",
    engine: "transformers",
    modality: "text",
    modelId: "HuggingFaceTB/SmolLM2-360M-Instruct",
    vramMb: 0,
    hfRepo: "HuggingFaceTB/SmolLM2-360M-Instruct",
  },
  {
    id: "grid-tinyllama-1.1b",
    label: "TinyLlama 1.1B Chat",
    engine: "transformers",
    modality: "text",
    modelId: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    vramMb: 0,
    hfRepo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  },
  {
    id: "grid-qwen2.5-0.5b",
    label: "Qwen2.5 0.5B Instruct",
    engine: "transformers",
    modality: "text",
    modelId: "Qwen/Qwen2.5-0.5B-Instruct",
    vramMb: 0,
    hfRepo: "Qwen/Qwen2.5-0.5B-Instruct",
  },
  {
    id: "grid-qwen2.5-1.5b",
    label: "Qwen2.5 1.5B Instruct",
    engine: "transformers",
    modality: "text",
    modelId: "Qwen/Qwen2.5-1.5B-Instruct",
    vramMb: 0,
    hfRepo: "Qwen/Qwen2.5-1.5B-Instruct",
  },
];

export function getShardedModel(id: string): ModelSpec | undefined {
  return SHARDED_MODELS.find((m) => m.id === id);
}

export function getModel(id: string): ModelSpec | undefined {
  return (
    MODELS.find((m) => m.id === id) ?? SHARDED_MODELS.find((m) => m.id === id)
  );
}

export const DEFAULT_MODEL_ID = "smollm2-360m";
