import type { ModelSpec } from "@/lib/config";
import type {
  ChatMessage,
  GenerateOptions,
  InferenceEngine,
  LoadProgress,
} from "@/lib/engine/types";

/**
 * Translate WebLLM/TVM device-init failures into actionable guidance. The most
 * common one on capable hardware is a WebGPU limit mismatch (the compiled model
 * kernel requests more storage buffers per shader stage than the adapter grants
 * at device-creation time), which is resolved by a newer browser build rather
 * than a code change.
 */
function describeLoadError(label: string, err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  if (/maxStorageBuffersPerShaderStage|exceeds limit/i.test(raw)) {
    return (
      `${label} needs more WebGPU storage buffers per shader stage than this ` +
      `device grants. Update to the latest Chrome/Edge (which raises the limit) ` +
      `or pick a different model. (${raw})`
    );
  }
  if (/requestDevice|requestAdapter|no available adapter/i.test(raw)) {
    return `${label} could not initialize WebGPU on this device. (${raw})`;
  }
  return `${label} failed to load: ${raw}`;
}

/**
 * WebLLM (MLC) engine — prebuilt quantized chat LLMs on WebGPU. Closest to an
 * Ollama-style local chat runtime.
 */
export class WebLLMEngine implements InferenceEngine {
  readonly kind = "webllm" as const;
  private engine: unknown = null;
  private loadedModelId: string | null = null;

  async load(
    model: ModelSpec,
    onProgress: (p: LoadProgress) => void,
  ): Promise<void> {
    if (this.loadedModelId === model.modelId && this.engine) return;
    const webllm = await import("@mlc-ai/web-llm");
    await this.unload();
    const createEngine = webllm.CreateMLCEngine as (
      modelId: string,
      engineConfig?: unknown,
      chatOpts?: unknown,
    ) => Promise<unknown>;
    try {
      this.engine = await createEngine(
        model.modelId,
        {
          initProgressCallback: (r: { progress: number; text: string }) =>
            onProgress({ progress: r.progress ?? 0, text: r.text ?? "" }),
        },
        model.webllmOverrides,
      );
    } catch (err) {
      throw new Error(describeLoadError(model.label, err));
    }
    this.loadedModelId = model.modelId;
  }

  async generate(
    model: ModelSpec,
    messages: ChatMessage[],
    opts: GenerateOptions,
    onToken: (token: string) => void,
  ): Promise<string> {
    if (!this.engine || this.loadedModelId !== model.modelId) {
      await this.load(model, () => {});
    }
    const engine = this.engine as {
      chat: {
        completions: {
          create: (args: unknown) => Promise<
            AsyncIterable<{
              choices: { delta: { content?: string } }[];
            }>
          >;
        };
      };
    };

    const chunks = await engine.chat.completions.create({
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      stream: true,
      temperature: opts.temperature ?? 0.7,
      max_tokens: opts.maxTokens ?? 512,
    });

    let full = "";
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content ?? "";
      if (delta) {
        full += delta;
        onToken(delta);
      }
    }
    return full;
  }

  async unload(): Promise<void> {
    const engine = this.engine as { unload?: () => Promise<void> } | null;
    if (engine?.unload) {
      try {
        await engine.unload();
      } catch {
        // ignore
      }
    }
    this.engine = null;
    this.loadedModelId = null;
  }
}
