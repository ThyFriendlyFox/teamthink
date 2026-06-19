import type { ChatMessage } from "@/lib/engine/types";
import type { DeviceCapabilities } from "@/lib/grid/capabilities";

export type TaskStatus =
  | "open"
  | "claimed"
  | "running"
  | "done"
  | "error";

/** A unit of inference work, replicated across peers via the CRDT. */
export interface TaskRecord {
  id: string;
  requester: string;
  modelId: string;
  messages: ChatMessage[];
  status: TaskStatus;
  claimedBy?: string;
  claimedAt?: number;
  result?: string;
  error?: string;
  createdAt: number;
  updatedAt: number;
}

/** Gossiped presence/capability heartbeat for a peer. */
export interface PeerPresence {
  peerId: string;
  caps: DeviceCapabilities;
  loadedModels: string[];
  activeJobs: number;
  /** Last heartbeat timestamp (ms). */
  ts: number;
  /** Whether this is the local node. */
  self?: boolean;
}

// --- pipeline-parallel (sharded) inference ----------------------------------

export interface ShardAssignment {
  peerId: string;
  shardIndex: number;
  /** Contiguous decoder layers this peer owns: [layerStart, layerEnd). */
  layerStart: number;
  layerEnd: number;
  /** Owns the embedding + chain head. */
  isFirst: boolean;
  /** Owns the final norm + lm_head + sampling. */
  isLast: boolean;
}

export interface PipelinePlan {
  planId: string;
  jobId: string;
  modelId: string;
  /** Hugging Face repo id; weights are range-fetched client-side per shard. */
  repo: string;
  requester: string;
  numShards: number;
  /** Ordered by shardIndex; shard 0 is the chain head. */
  shards: ShardAssignment[];
  options: { temperature: number; topP: number; maxTokens: number };
}

export type PipelineStatus =
  | "planning"
  | "warming"
  | "ready"
  | "queued"
  | "running"
  | "done"
  | "error";

/** A pipeline job, replicated across peers via the CRDT. */
export interface PipelineRecord {
  plan: PipelinePlan;
  status: PipelineStatus;
  /** peerId -> shard loaded and ready to run. */
  ready: Record<string, boolean>;
  result?: string;
  error?: string;
  createdAt: number;
  updatedAt: number;
}

export interface PipelineView {
  planId: string;
  modelId: string;
  status: PipelineStatus;
  numShards: number;
  readyCount: number;
  /** Per-shard layer ranges and the peer hosting each. */
  shards: { peerId: string; layerStart: number; layerEnd: number }[];
  /** Output text streamed so far (requester side). */
  text: string;
  tokensPerSec: number | null;
  error?: string;
}

/**
 * The model currently selected and warmed on the distribution network. Loading
 * begins on selection (not on first prompt), so this surfaces warm progress and
 * the layer partition across peers.
 */
export interface ProvisionedView {
  modelId: string;
  repo: string;
  status: PipelineStatus;
  numShards: number;
  readyCount: number;
  shards: { peerId: string; layerStart: number; layerEnd: number }[];
  error?: string;
  /** Per-shard download/warm progress on this device. */
  progress: { progress: number; text: string } | null;
}

export interface GridSnapshot {
  selfId: string;
  caps: DeviceCapabilities | null;
  peers: PeerPresence[];
  tasks: TaskRecord[];
  /** Streaming partial output keyed by task id. */
  streams: Record<string, string>;
  connected: boolean;
  activeModelId: string | null;
  modelLoad: { progress: number; text: string } | null;
  /** The model warmed on the grid (selected in the console), if any. */
  provisioned: ProvisionedView | null;
  /** One entry per submitted prompt (chat history) run on the provisioned model. */
  pipelines: PipelineView[];
}
