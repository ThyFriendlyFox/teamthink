"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { getModel, SHARDED_MODELS } from "@/lib/config";
import type { ChatMessage } from "@/lib/engine/types";
import type { GridNode } from "@/lib/grid/scheduler";
import type {
  GridSnapshot,
  PipelineStatus,
  PipelineView,
  ProvisionedView,
} from "@/lib/grid/types";

const CUSTOM = "__custom__";

export function InferenceConsole({
  node,
  snapshot,
}: {
  node: GridNode;
  snapshot: GridSnapshot;
}) {
  const [modelId, setModelId] = useState(SHARDED_MODELS[0]?.id ?? CUSTOM);
  const [customRepo, setCustomRepo] = useState("");
  const [prompt, setPrompt] = useState("");

  const isCustom = modelId === CUSTOM;
  const model = isCustom ? null : getModel(modelId);
  const provisioned = snapshot.provisioned;

  const webgpuPeers = useMemo(
    () => snapshot.peers.filter((p) => p.caps.webgpu).length,
    [snapshot.peers],
  );

  // Selecting a registry model warms it on the grid immediately (loading is
  // decoupled from prompting). Only auto-warm on a compute-capable device so
  // request-only peers don't pin a remote peer's GPU just by opening the page.
  const canHost = !!snapshot.caps?.webgpu;
  const provisionedRef = useRef<string>("");
  useEffect(() => {
    if (isCustom || !model?.hfRepo || !canHost) return;
    const key = `${model.id}:${model.hfRepo}`;
    if (provisionedRef.current === key) return;
    provisionedRef.current = key;
    void node.provision(model.id, model.hfRepo);
  }, [isCustom, model, node, canHost]);

  function loadCustom() {
    const repo = customRepo.trim();
    if (!repo) return;
    provisionedRef.current = `custom:${repo}`;
    void node.provisionRepo(repo);
  }

  function submit() {
    const text = prompt.trim();
    if (!text) return;
    const messages: ChatMessage[] = [{ role: "user", content: text }];
    if (isCustom) {
      const repo = customRepo.trim();
      if (!repo) return;
      if (provisionedRef.current !== `custom:${repo}`) loadCustom();
    } else if (model?.hfRepo) {
      void node.provision(model.id, model.hfRepo);
    }
    if (node.runPrompt(messages)) setPrompt("");
  }

  const canSend = !!prompt.trim() && (!isCustom || !!customRepo.trim());

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Inference console</CardTitle>
        <Badge tone={webgpuPeers >= 1 ? "positive" : "warning"} dot>
          {webgpuPeers >= 1
            ? `${webgpuPeers} compute ${webgpuPeers === 1 ? "node" : "nodes"}`
            : "needs a compute node"}
        </Badge>
      </CardHeader>

      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="h-10 rounded-xl border border-border bg-canvas px-3 text-sm text-ink outline-none focus:border-accent"
          >
            {SHARDED_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
            <option value={CUSTOM}>Custom HF repo…</option>
          </select>
          {isCustom && (
            <>
              <input
                type="text"
                value={customRepo}
                onChange={(e) => setCustomRepo(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && loadCustom()}
                placeholder="org/model-id (e.g. Qwen/Qwen2.5-1.5B-Instruct)"
                className="h-10 min-w-[16rem] flex-1 rounded-xl border border-border bg-canvas px-3 text-sm text-ink outline-none focus:border-accent"
              />
              <Button
                size="sm"
                variant="secondary"
                onClick={loadCustom}
                disabled={!customRepo.trim()}
              >
                Load
              </Button>
            </>
          )}
        </div>

        {provisioned && <ProvisionedStatus provisioned={provisioned} />}

        <div className="flex gap-2">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            rows={2}
            placeholder="Ask the grid something… (⌘/Ctrl + Enter)"
            className="flex-1 resize-none rounded-xl border border-border bg-canvas px-4 py-3 text-sm text-ink outline-none placeholder:text-ink-subtle focus:border-accent focus:ring-2 focus:ring-accent/30"
          />
          <Button onClick={submit} disabled={!canSend}>
            Send
          </Button>
        </div>
      </div>

      <div className="scroll-thin mt-5 flex-1 space-y-4 overflow-y-auto pr-1">
        {snapshot.pipelines
          .slice()
          .reverse()
          .map((p) => (
            <PipelineBubble key={p.planId} pipeline={p} />
          ))}
        {snapshot.pipelines.length === 0 && (
          <div className="rounded-xl border border-dashed border-border px-4 py-10 text-center text-sm text-ink-subtle">
            {provisioned
              ? "Model ready. Send a prompt to run it on the grid."
              : "Pick a model to load it on the grid, then send a prompt."}
          </div>
        )}
      </div>
    </Card>
  );
}

function ProvisionedStatus({ provisioned }: { provisioned: ProvisionedView }) {
  const model = getModel(provisioned.modelId);
  const label = model?.label ?? provisioned.repo ?? provisioned.modelId;
  const pct = provisioned.progress
    ? Math.round(provisioned.progress.progress * 100)
    : null;
  return (
    <div className="rounded-xl border border-border bg-surface-sunken p-3">
      <div className="flex items-center justify-between gap-2 text-xs text-ink-muted">
        <div className="flex items-center gap-2">
          <Badge tone="accent">{label}</Badge>
          <span>
            {provisioned.readyCount}/{provisioned.numShards || 1} shards
          </span>
        </div>
        <Badge tone={pipelineTone[provisioned.status]} dot>
          {provisioned.status}
        </Badge>
      </div>
      {provisioned.status === "error" && provisioned.error && (
        <p className="mt-2 text-xs text-danger">{provisioned.error}</p>
      )}
      {provisioned.progress && pct != null && provisioned.status !== "error" && (
        <div className="mt-2">
          <div className="mb-1 flex justify-between text-[11px] text-ink-subtle">
            <span className="truncate">
              {provisioned.progress.text || "warming"}
            </span>
            <span className="tabular-nums">{pct}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-surface">
            <div
              className="h-full rounded-full bg-accent transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}
      {provisioned.shards.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 text-[11px] text-ink-subtle">
          {provisioned.shards.map((s, i) => (
            <span
              key={i}
              className="rounded-md border border-border bg-surface px-1.5 py-0.5 tabular-nums"
            >
              {s.peerId.slice(0, 6)} · L{s.layerStart}–{s.layerEnd - 1}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

const pipelineTone: Record<
  PipelineStatus,
  "neutral" | "accent" | "positive" | "warning" | "danger"
> = {
  planning: "warning",
  warming: "warning",
  ready: "positive",
  queued: "neutral",
  running: "accent",
  done: "positive",
  error: "danger",
};

function PipelineBubble({ pipeline }: { pipeline: PipelineView }) {
  const model = getModel(pipeline.modelId);
  const running = pipeline.status === "running";
  const output =
    pipeline.status === "error" ? (pipeline.error ?? "failed") : pipeline.text;

  return (
    <div className="animate-fade-in rounded-xl border border-accent/40 bg-surface-sunken p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs text-ink-muted">
          <Badge tone="accent">{model?.label ?? pipeline.modelId}</Badge>
          {pipeline.tokensPerSec != null && (
            <span className="tabular-nums">
              {pipeline.tokensPerSec.toFixed(1)} tok/s
            </span>
          )}
        </div>
        <Badge tone={pipelineTone[pipeline.status]} dot>
          {pipeline.status}
        </Badge>
      </div>

      {(output || running || pipeline.status === "queued") && (
        <div className="mt-3 whitespace-pre-wrap rounded-lg border border-border bg-surface p-3 font-mono text-sm text-ink">
          {output || (
            <span className="animate-pulse-soft text-ink-subtle">
              {pipeline.status === "queued" ? "queued…" : "waiting for tokens…"}
            </span>
          )}
          {running && output && <span className="animate-pulse-soft">▍</span>}
        </div>
      )}
    </div>
  );
}
