import type { ModelSpec } from "@/lib/config";
import type { ArchDescriptor } from "@/lib/engine/hf/config";
import type { ShardRange } from "@/lib/engine/shard/model-descriptor";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  /** Optional image as a data URL or http(s) URL, for vision models. */
  image?: string;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
}

export interface LoadProgress {
  progress: number; // 0..1
  text: string;
}

/** Common interface implemented by each in-browser engine. */
export interface InferenceEngine {
  readonly kind: ModelSpec["engine"];
  load(model: ModelSpec, onProgress: (p: LoadProgress) => void): Promise<void>;
  generate(
    model: ModelSpec,
    messages: ChatMessage[],
    opts: GenerateOptions,
    onToken: (token: string) => void,
  ): Promise<string>;
  unload(): Promise<void>;
}

// --- Worker message protocol -------------------------------------------------

/** Input to a single shard step: either token ids (first shard) or a hidden state. */
export type ShardInput =
  | { kind: "ids"; ids: number[] }
  | { kind: "hidden"; dims: number[]; data: ArrayBuffer };

/** Output of a shard step: a hidden state, or a sampled token id (last shard). */
export type ShardResult =
  | { kind: "hidden"; dims: number[]; data: ArrayBuffer }
  | { kind: "token"; tokenId: number };

export type WorkerRequest =
  | { type: "load"; reqId: string; modelId: string }
  | {
      type: "generate";
      reqId: string;
      modelId: string;
      messages: ChatMessage[];
      options: GenerateOptions;
    }
  | { type: "unload"; reqId: string }
  | {
      type: "shardLoad";
      reqId: string;
      descriptor: ArchDescriptor;
      range: ShardRange;
    }
  | {
      type: "shardRun";
      reqId: string;
      input: ShardInput;
      isLast: boolean;
      options: { temperature: number; topP: number };
    }
  | { type: "shardReset"; reqId: string };

export type WorkerResponse =
  | { type: "progress"; reqId: string; progress: number; text: string }
  | { type: "ready"; reqId: string }
  | { type: "token"; reqId: string; token: string }
  | { type: "done"; reqId: string; text: string }
  | { type: "error"; reqId: string; error: string }
  | { type: "shardLoaded"; reqId: string }
  | { type: "shardResult"; reqId: string; result: ShardResult };
