/**
 * Shape of the `shards.json` manifest produced by tools/export_shards. Mirrors
 * the Python exporter's output exactly.
 */

export interface ShardMeta {
  index: number;
  file: string;
  layerStart: number;
  layerEnd: number;
  isFirst: boolean;
  isLast: boolean;
  /** "input_ids" for the first shard, otherwise "hidden_states". */
  inputName: string;
  /** "logits" for the last shard, otherwise "hidden_states_out". */
  outputName: string;
  numLocalLayers: number;
}

export interface ShardManifest {
  modelId: string;
  source: string;
  architecture: string;
  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
  ropeTheta: number;
  dtype: "fp32" | "fp16";
  eosTokenId: number | number[];
  shards: ShardMeta[];
}

export function isEos(
  tokenId: number,
  eos: number | number[] | undefined,
): boolean {
  if (eos == null) return false;
  return Array.isArray(eos) ? eos.includes(tokenId) : tokenId === eos;
}
