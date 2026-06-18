/// <reference lib="webworker" />
import { getModel } from "@/lib/config";
import { TransformersEngine } from "@/lib/engine/transformers";
import type {
  InferenceEngine,
  ShardResult,
  WorkerRequest,
  WorkerResponse,
} from "@/lib/engine/types";
import { WebLLMEngine } from "@/lib/engine/webllm";
import type { WebGPUShardRunner } from "@/lib/engine/shard/webgpu-shard-runner";
import type { ArchDescriptor } from "@/lib/engine/hf/config";
import type { ShardRange } from "@/lib/engine/shard/model-descriptor";

/**
 * Inference worker. Runs the active engine off the main thread and streams
 * progress/tokens back to the page. One engine per kind is cached so reloading
 * the same model is cheap. Also hosts a single pipeline shard runner when this
 * peer participates in a sharded job.
 */

const engines: Partial<Record<InferenceEngine["kind"], InferenceEngine>> = {};

function engineFor(kind: InferenceEngine["kind"]): InferenceEngine {
  let engine = engines[kind];
  if (!engine) {
    engine = kind === "webllm" ? new WebLLMEngine() : new TransformersEngine();
    engines[kind] = engine;
  }
  return engine;
}

let shardRunner: WebGPUShardRunner | null = null;

function post(msg: WorkerResponse, transfer?: Transferable[]) {
  (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

async function handleShardLoad(
  reqId: string,
  descriptor: ArchDescriptor,
  range: ShardRange,
): Promise<void> {
  const [{ WebGPUShardRunner }, { loadSafetensorsIndex }] = await Promise.all([
    import("@/lib/engine/shard/webgpu-shard-runner"),
    import("@/lib/engine/hf/safetensors"),
  ]);
  const index = await loadSafetensorsIndex(descriptor.repo);
  shardRunner?.dispose();
  shardRunner = new WebGPUShardRunner(descriptor, index, range);
  await shardRunner.load((progress, text) =>
    post({ type: "progress", reqId, progress, text }),
  );
  post({ type: "shardLoaded", reqId });
}

async function handleShardRun(
  reqId: string,
  req: Extract<WorkerRequest, { type: "shardRun" }>,
): Promise<void> {
  if (!shardRunner) throw new Error("shard not loaded");
  const { sampleToken, lastTokenLogits } = await import(
    "@/lib/engine/sampler"
  );

  const primary =
    req.input.kind === "ids"
      ? {
          dims: [1, req.input.ids.length],
          data: BigInt64Array.from(req.input.ids.map((n) => BigInt(n))),
        }
      : {
          dims: req.input.dims,
          data: new Float32Array(req.input.data),
        };

  const out = await shardRunner.run(primary);

  let result: ShardResult;
  const transfer: Transferable[] = [];
  if (req.isLast) {
    const logits = lastTokenLogits(out.data as Float32Array, out.dims);
    const tokenId = sampleToken(logits, {
      temperature: req.options.temperature,
      topP: req.options.topP,
    });
    result = { kind: "token", tokenId };
  } else {
    const f32 = (out.data as Float32Array).slice();
    const buf = f32.buffer as ArrayBuffer;
    transfer.push(buf);
    result = { kind: "hidden", dims: out.dims, data: buf };
  }
  post({ type: "shardResult", reqId, result }, transfer);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const req = e.data;
  try {
    if (req.type === "load") {
      const model = getModel(req.modelId);
      if (!model) throw new Error(`unknown model ${req.modelId}`);
      const engine = engineFor(model.engine);
      await engine.load(model, (p) =>
        post({
          type: "progress",
          reqId: req.reqId,
          progress: p.progress,
          text: p.text,
        }),
      );
      post({ type: "ready", reqId: req.reqId });
    } else if (req.type === "generate") {
      const model = getModel(req.modelId);
      if (!model) throw new Error(`unknown model ${req.modelId}`);
      const engine = engineFor(model.engine);
      const text = await engine.generate(
        model,
        req.messages,
        req.options,
        (token) => post({ type: "token", reqId: req.reqId, token }),
      );
      post({ type: "done", reqId: req.reqId, text });
    } else if (req.type === "unload") {
      await Promise.all(
        Object.values(engines).map((eng) => eng?.unload()),
      );
      shardRunner?.dispose();
      shardRunner = null;
      post({ type: "ready", reqId: req.reqId });
    } else if (req.type === "shardLoad") {
      await handleShardLoad(req.reqId, req.descriptor, req.range);
    } else if (req.type === "shardRun") {
      await handleShardRun(req.reqId, req);
    } else if (req.type === "shardReset") {
      shardRunner?.reset();
      post({ type: "ready", reqId: req.reqId });
    }
  } catch (err) {
    post({
      type: "error",
      reqId: req.reqId,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
