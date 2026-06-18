"use client";

import { useState } from "react";
import { buildModelDescriptor } from "@/lib/grid/pipeline";
import { loadSafetensorsIndex } from "@/lib/engine/hf/safetensors";
import { WebGPUShardRunner } from "@/lib/engine/shard/webgpu-shard-runner";
import { lastTokenLogits, sampleToken } from "@/lib/engine/sampler";
import { loadTokenizer } from "@/lib/grid/pipeline";

/**
 * Developer validation harness. Runs a small model end-to-end through the
 * WebGPU executor as a single shard (numShards = 1) and greedily decodes a few
 * tokens, alongside a transformers.js reference, so the kernels can be checked
 * numerically before multi-peer pipelines are enabled. Browser + WebGPU only.
 */

const DEFAULT_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct";

export default function ValidatePage() {
  const [repo, setRepo] = useState(DEFAULT_REPO);
  const [promptText, setPromptText] = useState("The capital of France is");
  const [maxTokens, setMaxTokens] = useState(16);
  const [log, setLog] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const push = (s: string) => setLog((l) => [...l, s]);

  async function run() {
    setBusy(true);
    setLog([]);
    try {
      push(`building descriptor for ${repo}…`);
      const desc = await buildModelDescriptor(repo);
      push(
        `arch=${desc.traits.family} layers=${desc.numLayers} hidden=${desc.hiddenSize} heads=${desc.numAttentionHeads}/${desc.numKeyValueHeads} vocab=${desc.vocabSize}`,
      );

      const index = await loadSafetensorsIndex(repo);
      push(`safetensors: ${index.tensors.size} tensors indexed`);

      const tok = await loadTokenizer(repo);
      const enc = await tok(promptText, { add_special_tokens: true });
      const ids = Array.from(enc.input_ids.data, (v) => Number(v));
      push(`prompt tokens (${ids.length}): [${ids.join(", ")}]`);

      const runner = new WebGPUShardRunner(desc, index, {
        index: 0,
        layerStart: 0,
        layerEnd: desc.numLayers,
        isFirst: true,
        isLast: true,
      });
      push("loading shard (range-fetching weights)…");
      const t0 = performance.now();
      await runner.load((f, text) => {
        if (Math.round(f * 100) % 25 === 0) push(`  ${text}`);
      });
      push(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

      // Prefill, then greedy-decode.
      const out: number[] = [];
      let res = await runner.run({
        dims: [1, ids.length],
        data: BigInt64Array.from(ids.map((n) => BigInt(n))),
      });
      for (let step = 0; step < maxTokens; step++) {
        const logits = lastTokenLogits(res.data as Float32Array, res.dims);
        const next = sampleToken(logits, { temperature: 0 });
        out.push(next);
        res = await runner.run({
          dims: [1, 1],
          data: BigInt64Array.from([BigInt(next)]),
        });
      }
      runner.dispose();

      const webgpuText = tok.decode(out, { skip_special_tokens: true });
      push(`WebGPU greedy ids: [${out.join(", ")}]`);
      push(`WebGPU greedy text: ${JSON.stringify(webgpuText)}`);

      push("running transformers.js reference…");
      const ref = await referenceGenerate(repo, promptText, maxTokens);
      push(`reference text: ${JSON.stringify(ref)}`);
      push(
        webgpuText.trim() && ref.includes(webgpuText.trim().slice(0, 8))
          ? "MATCH (prefix) ✓"
          : "DIVERGED — inspect kernels",
      );
    } catch (err) {
      push(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ padding: 24, fontFamily: "system-ui", maxWidth: 900 }}>
      <h1>WebGPU executor validation</h1>
      <p style={{ color: "#888" }}>
        Single-shard greedy decode vs. a transformers.js reference. Requires
        WebGPU.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", margin: "12px 0" }}>
        <input
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          style={{ flex: 1, minWidth: 320, padding: 8 }}
        />
        <input
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          style={{ flex: 1, minWidth: 240, padding: 8 }}
        />
        <input
          type="number"
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          style={{ width: 80, padding: 8 }}
        />
        <button onClick={run} disabled={busy} style={{ padding: "8px 16px" }}>
          {busy ? "running…" : "Run"}
        </button>
      </div>
      <pre
        style={{
          background: "#111",
          color: "#0f0",
          padding: 16,
          borderRadius: 8,
          whiteSpace: "pre-wrap",
          minHeight: 200,
        }}
      >
        {log.join("\n")}
      </pre>
    </main>
  );
}

async function referenceGenerate(
  repo: string,
  prompt: string,
  maxTokens: number,
): Promise<string> {
  const tf = (await import("@huggingface/transformers")) as unknown as {
    pipeline: (
      task: string,
      model: string,
      opts?: Record<string, unknown>,
    ) => Promise<
      (
        text: string,
        opts: Record<string, unknown>,
      ) => Promise<Array<{ generated_text: string }>>
    >;
  };
  const gen = await tf.pipeline("text-generation", repo, { dtype: "fp32" });
  const res = await gen(prompt, {
    max_new_tokens: maxTokens,
    do_sample: false,
    return_full_text: false,
  });
  return res?.[0]?.generated_text ?? "";
}
