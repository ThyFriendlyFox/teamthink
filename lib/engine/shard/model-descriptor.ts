import type { ArchDescriptor } from "@/lib/engine/hf/config";

/**
 * A model is described to the executor by its ArchDescriptor (intrinsic, derived
 * from config.json) plus a per-peer ShardRange (plan-specific) telling a peer
 * which contiguous layer slice it owns and whether it is the head/tail.
 */
export type ModelDescriptor = ArchDescriptor;

export interface ShardRange {
  index: number;
  /** Inclusive first layer this shard owns. */
  layerStart: number;
  /** Exclusive last layer this shard owns. */
  layerEnd: number;
  /** Owns the embedding + is the chain head. */
  isFirst: boolean;
  /** Owns the final norm + lm_head + sampling. */
  isLast: boolean;
}

/** Tensor I/O across the worker boundary (matches the legacy ONNX runner). */
export interface TensorLike {
  dims: number[];
  /** float32 for hidden states / logits, BigInt64 for input ids. */
  data: Float32Array | BigInt64Array;
}
