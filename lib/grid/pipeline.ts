import type { ChatMessage } from "@/lib/engine/types";
import {
  fetchArchDescriptor,
  type ArchDescriptor,
} from "@/lib/engine/hf/config";
import type {
  PeerPresence,
  PipelinePlan,
  ShardAssignment,
} from "@/lib/grid/types";

/**
 * Planning and tokenization helpers for pipeline-parallel inference. The
 * orchestration (CRDT plan publish, readiness barrier, ring driving) lives in
 * GridNode; this module owns the pure pieces: building the model descriptor
 * from a Hugging Face repo, capacity-based shard placement, and prompt
 * (de)tokenization. There is no offline manifest — the model is partitioned
 * client-side across the pool once it forms.
 */

const KV_CAP = 2048;
const BUDGET_FRACTION = 0.5;
const MAX_PIPELINE_SHARDS = 8;

/**
 * Multi-peer pipelines stay gated until the WebGPU kernels are numerically
 * validated against a reference (see tools / the /dev/validate page). With the
 * gate off, a model still runs distributed-style on a single capable peer
 * (numShards = 1), which is the correctness path; set this to opt into
 * splitting a model across multiple peers.
 */
const MULTI_PEER_ENABLED =
  process.env.NEXT_PUBLIC_ENABLE_GRID_MULTI_PEER === "1";

const descCache = new Map<string, Promise<ArchDescriptor>>();

/** Fetch + normalize a repo's config.json into an architecture descriptor. */
export function buildModelDescriptor(repo: string): Promise<ArchDescriptor> {
  let p = descCache.get(repo);
  if (!p) {
    p = fetchArchDescriptor(repo);
    descCache.set(repo, p);
  }
  return p;
}

export interface PlanInput {
  modelId: string;
  repo: string;
  desc: ArchDescriptor;
  requester: string;
  peers: PeerPresence[];
  /** Measured RTT (ms) from the requester to each peer. */
  rtt: Map<string, number>;
  options: { temperature: number; topP: number; maxTokens: number };
  jobId: string;
  planId: string;
}

export type PlanResult =
  | { ok: true; plan: PipelinePlan }
  | { ok: false; error: string };

/** Bytes of f32 weights + KV cache for one decoder layer. */
function layerBytes(d: ArchDescriptor): number {
  const qDim = d.numAttentionHeads * d.headDim;
  const kvDim = d.numKeyValueHeads * d.headDim;
  const weights =
    d.hiddenSize * (2 * qDim + 2 * kvDim) + 3 * d.intermediateSize * d.hiddenSize;
  const kv = d.numKeyValueHeads * KV_CAP * d.headDim * 2;
  return (weights + kv) * 4;
}

function embedBytes(d: ArchDescriptor): number {
  return d.vocabSize * d.hiddenSize * 4;
}

/**
 * Partition the model's layers into contiguous ranges sized to each peer's
 * memory budget, ordered nearest-RTT first. The pool's total capacity grows
 * with the number of peers, so more peers can host larger models.
 */
export function buildPipelinePlan(input: PlanInput): PlanResult {
  const d = input.desc;
  const N = d.numLayers;
  const perLayer = layerBytes(d);
  const embed = embedBytes(d);
  const lmHead = embedBytes(d); // the tail also needs the vocab x hidden matrix

  const eligible = input.peers.filter((p) => p.caps.webgpu);
  if (eligible.length === 0) {
    return { ok: false, error: "no WebGPU-capable peers in the pool" };
  }

  const rttOf = (peerId: string) =>
    peerId === input.requester
      ? 0
      : (input.rtt.get(peerId) ?? Number.MAX_SAFE_INTEGER);
  const ordered = [...eligible].sort((a, b) => rttOf(a.peerId) - rttOf(b.peerId));

  const ranges: Array<{ peerId: string; start: number; end: number }> = [];
  let layer = 0;
  let placedLayers = 0;
  for (let pi = 0; pi < ordered.length && layer < N; pi++) {
    const peer = ordered[pi];
    const budget = peer.caps.memoryEstimateMb * 1024 * 1024 * BUDGET_FRACTION;
    const isFirst = layer === 0;
    const overhead = isFirst ? embed : 0;
    const capNotLast = Math.floor((budget - overhead) / perLayer);
    const capIfLast = Math.floor((budget - overhead - lmHead) / perLayer);
    const remaining = N - layer;

    if (remaining <= capIfLast) {
      ranges.push({ peerId: peer.peerId, start: layer, end: N });
      layer = N;
    } else if (capNotLast >= 1) {
      const take = Math.min(capNotLast, remaining);
      ranges.push({ peerId: peer.peerId, start: layer, end: layer + take });
      layer += take;
    }
    placedLayers = layer;
  }

  if (placedLayers < N) {
    const haveMb = Math.round(
      ordered.reduce((a, p) => a + p.caps.memoryEstimateMb, 0),
    );
    const needMb = Math.round(
      (N * perLayer + embed + lmHead) / BUDGET_FRACTION / 1024 / 1024,
    );
    return {
      ok: false,
      error: `not enough pool memory: placed ${placedLayers}/${N} layers across ${ordered.length} peer(s). Need ~${needMb}MB of GPU memory, have ~${haveMb}MB. Add more peers.`,
    };
  }
  if (ranges.length > MAX_PIPELINE_SHARDS) {
    return {
      ok: false,
      error: `model needs ${ranges.length} shards (max ${MAX_PIPELINE_SHARDS}); use larger peers`,
    };
  }
  if (!MULTI_PEER_ENABLED && ranges.length > 1) {
    return {
      ok: false,
      error: `this model needs ${ranges.length} peers, but multi-peer pipelines are disabled until the WebGPU kernels are validated. Run a model that fits one peer, or set NEXT_PUBLIC_ENABLE_GRID_MULTI_PEER=1.`,
    };
  }

  const shards: ShardAssignment[] = ranges.map((r, i) => ({
    peerId: r.peerId,
    shardIndex: i,
    layerStart: r.start,
    layerEnd: r.end,
    isFirst: i === 0,
    isLast: i === ranges.length - 1,
  }));

  return {
    ok: true,
    plan: {
      planId: input.planId,
      jobId: input.jobId,
      modelId: input.modelId,
      repo: input.repo,
      requester: input.requester,
      numShards: shards.length,
      shards,
      options: input.options,
    },
  };
}

export interface ShardRole {
  shardIndex: number | null;
  isFirst: boolean;
  isLast: boolean;
  nextPeerId: string | null;
  firstPeerId: string | null;
  isRequester: boolean;
}

export function roleFor(plan: PipelinePlan, peerId: string): ShardRole {
  const mine = plan.shards.find((s) => s.peerId === peerId);
  const shardIndex = mine ? mine.shardIndex : null;
  const next =
    shardIndex != null
      ? plan.shards.find((s) => s.shardIndex === shardIndex + 1)
      : undefined;
  const first = plan.shards.find((s) => s.shardIndex === 0);
  return {
    shardIndex,
    isFirst: shardIndex === 0,
    isLast: shardIndex === plan.numShards - 1,
    nextPeerId: next?.peerId ?? null,
    firstPeerId: first?.peerId ?? null,
    isRequester: plan.requester === peerId,
  };
}

// --- tokenization (requester side) ------------------------------------------

type LoadedTokenizer = {
  apply_chat_template: (messages: unknown, opts: Record<string, unknown>) => string;
  (text: string, opts?: Record<string, unknown>): Promise<{
    input_ids: { data: ArrayLike<bigint | number> };
  }>;
  decode: (ids: number[], opts?: Record<string, unknown>) => string;
};

const tokenizerCache = new Map<string, Promise<LoadedTokenizer>>();

export function loadTokenizer(source: string): Promise<LoadedTokenizer> {
  let p = tokenizerCache.get(source);
  if (!p) {
    p = import("@huggingface/transformers").then(async (tf) => {
      const auto = (tf as unknown as {
        AutoTokenizer: {
          from_pretrained: (s: string) => Promise<LoadedTokenizer>;
        };
      }).AutoTokenizer;
      return auto.from_pretrained(source);
    });
    tokenizerCache.set(source, p);
  }
  return p;
}

export async function encodePrompt(
  tok: LoadedTokenizer,
  messages: ChatMessage[],
): Promise<number[]> {
  const chat = messages.map((m) => ({ role: m.role, content: m.content }));
  const prompt = tok.apply_chat_template(chat, {
    add_generation_prompt: true,
    tokenize: false,
  });
  const enc = await tok(prompt, { add_special_tokens: false });
  return Array.from(enc.input_ids.data, (v) => Number(v));
}

export function decodeIds(tok: LoadedTokenizer, ids: number[]): string {
  return tok.decode(ids, { skip_special_tokens: true });
}
