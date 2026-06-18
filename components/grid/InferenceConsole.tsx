"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { DEFAULT_MODEL_ID, getModel, MODELS, SHARDED_MODELS } from "@/lib/config";
import type { ChatMessage } from "@/lib/engine/types";
import type { GridNode } from "@/lib/grid/scheduler";
import type {
  GridSnapshot,
  PipelineStatus,
  PipelineView,
  TaskRecord,
  TaskStatus,
} from "@/lib/grid/types";

const statusTone: Record<
  TaskStatus,
  "neutral" | "accent" | "positive" | "warning" | "danger"
> = {
  open: "warning",
  claimed: "accent",
  running: "accent",
  done: "positive",
  error: "danger",
};

export function InferenceConsole({
  node,
  snapshot,
}: {
  node: GridNode;
  snapshot: GridSnapshot;
}) {
  const CUSTOM = "__custom__";
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);
  const [customRepo, setCustomRepo] = useState("");
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState<string | null>(null);

  const isCustom = modelId === CUSTOM;
  const model = getModel(modelId);
  const isVision = model?.modality === "vision";
  const isSharded = isCustom || !!model?.hfRepo;
  const allModels = useMemo(() => [...MODELS, ...SHARDED_MODELS], []);

  const webgpuPeers = useMemo(
    () => snapshot.peers.filter((p) => p.caps.webgpu).length,
    [snapshot.peers],
  );

  const capableForModel = useMemo(() => {
    if (isSharded) {
      // Distributed models partition across the pool; one capable peer can host
      // a small model whole, more peers unlock larger ones.
      return webgpuPeers >= 1;
    }
    return snapshot.peers.some(
      (p) => p.caps.webgpu && p.caps.compatibleModelIds.includes(modelId),
    );
  }, [snapshot.peers, modelId, isSharded, webgpuPeers]);

  function submit() {
    if (!prompt.trim()) return;
    const messages: ChatMessage[] = [
      {
        role: "user",
        content: prompt.trim(),
        ...(isVision && image ? { image } : {}),
      },
    ];
    if (isCustom) {
      const repo = customRepo.trim();
      if (!repo) return;
      node.submitRepo(repo, messages);
    } else {
      node.submit(modelId, messages);
    }
    setPrompt("");
    setImage(null);
  }

  async function onFile(file: File | null) {
    if (!file) return setImage(null);
    const reader = new FileReader();
    reader.onload = () => setImage(reader.result as string);
    reader.readAsDataURL(file);
  }

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Inference console</CardTitle>
        <Badge tone={capableForModel ? "positive" : "warning"} dot>
          {isSharded
            ? capableForModel
              ? "enough peers"
              : "needs more peers"
            : capableForModel
              ? "node available"
              : "no capable node"}
        </Badge>
      </CardHeader>

      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="h-10 rounded-xl border border-border bg-canvas px-3 text-sm text-ink outline-none focus:border-accent"
          >
            {allModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
                {m.hfRepo ? " · grid" : ` · ${m.engine}`}
              </option>
            ))}
            <option value={CUSTOM}>Custom HF repo… · grid</option>
          </select>
          {isCustom && (
            <input
              type="text"
              value={customRepo}
              onChange={(e) => setCustomRepo(e.target.value)}
              placeholder="org/model-id (e.g. Qwen/Qwen2.5-1.5B-Instruct)"
              className="h-10 min-w-[18rem] flex-1 rounded-xl border border-border bg-canvas px-3 text-sm text-ink outline-none focus:border-accent"
            />
          )}
          {isVision && (
            <input
              type="file"
              accept="image/*"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
              className="text-xs text-ink-muted file:mr-2 file:rounded-lg file:border file:border-border file:bg-surface file:px-3 file:py-1.5 file:text-ink"
            />
          )}
        </div>

        <div className="flex gap-2">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            rows={2}
            placeholder={
              isVision
                ? "Describe what to do with the image…"
                : "Ask the grid something… (⌘/Ctrl + Enter)"
            }
            className="flex-1 resize-none rounded-xl border border-border bg-canvas px-4 py-3 text-sm text-ink outline-none placeholder:text-ink-subtle focus:border-accent focus:ring-2 focus:ring-accent/30"
          />
          <Button
            onClick={submit}
            disabled={!prompt.trim() || (isCustom && !customRepo.trim())}
          >
            Send
          </Button>
        </div>
        {isVision && image && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={image}
            alt="attachment preview"
            className="h-16 w-16 rounded-lg border border-border object-cover"
          />
        )}
      </div>

      <div className="scroll-thin mt-5 flex-1 space-y-4 overflow-y-auto pr-1">
        {snapshot.pipelines.map((p) => (
          <PipelineBubble key={p.planId} pipeline={p} />
        ))}
        {snapshot.tasks.map((task) => (
          <TaskBubble
            key={task.id}
            task={task}
            stream={snapshot.streams[task.id]}
            selfId={snapshot.selfId}
          />
        ))}
        {snapshot.tasks.length === 0 && snapshot.pipelines.length === 0 && (
          <div className="rounded-xl border border-dashed border-border px-4 py-10 text-center text-sm text-ink-subtle">
            No requests yet. Send one to the grid.
          </div>
        )}
      </div>
    </Card>
  );
}

const pipelineTone: Record<
  PipelineStatus,
  "neutral" | "accent" | "positive" | "warning" | "danger"
> = {
  planning: "warning",
  warming: "warning",
  running: "accent",
  done: "positive",
  error: "danger",
};

function PipelineBubble({ pipeline }: { pipeline: PipelineView }) {
  const model = getModel(pipeline.modelId);
  const running = pipeline.status === "running" || pipeline.status === "warming";
  const output =
    pipeline.status === "error"
      ? (pipeline.error ?? "failed")
      : pipeline.text;

  return (
    <div className="animate-fade-in rounded-xl border border-accent/40 bg-surface-sunken p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs text-ink-muted">
          <Badge tone="accent">{model?.label ?? pipeline.modelId}</Badge>
          <span>
            pipeline · {pipeline.readyCount}/{pipeline.numShards} shards
          </span>
          {pipeline.tokensPerSec != null && (
            <span className="tabular-nums">
              · {pipeline.tokensPerSec.toFixed(1)} tok/s
            </span>
          )}
        </div>
        <Badge tone={pipelineTone[pipeline.status]} dot>
          {pipeline.status}
        </Badge>
      </div>

      {pipeline.shards.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 text-[11px] text-ink-subtle">
          {pipeline.shards.map((s, i) => (
            <span
              key={i}
              className="rounded-md border border-border bg-surface px-1.5 py-0.5 tabular-nums"
            >
              {s.peerId.slice(0, 6)} · L{s.layerStart}–{s.layerEnd - 1}
            </span>
          ))}
        </div>
      )}

      {(output || running) && (
        <div className="mt-3 whitespace-pre-wrap rounded-lg border border-border bg-surface p-3 font-mono text-sm text-ink">
          {output || (
            <span className="animate-pulse-soft text-ink-subtle">
              {pipeline.status === "warming"
                ? "warming shards…"
                : "waiting for tokens…"}
            </span>
          )}
          {pipeline.status === "running" && output && (
            <span className="animate-pulse-soft">▍</span>
          )}
        </div>
      )}
    </div>
  );
}

function TaskBubble({
  task,
  stream,
  selfId,
}: {
  task: TaskRecord;
  stream?: string;
  selfId: string;
}) {
  const model = getModel(task.modelId);
  const output =
    task.status === "done"
      ? task.result ?? ""
      : task.status === "error"
        ? task.error ?? "failed"
        : (stream ?? "");
  const mine = task.requester === selfId;

  return (
    <div className="animate-fade-in rounded-xl border border-border bg-surface-sunken p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs text-ink-muted">
          <Badge tone="neutral">{model?.label ?? task.modelId}</Badge>
          {mine ? (
            <span>requested here</span>
          ) : (
            <span>from {task.requester.slice(0, 8)}</span>
          )}
          {task.claimedBy && (
            <span>· on {task.claimedBy.slice(0, 8)}</span>
          )}
        </div>
        <Badge tone={statusTone[task.status]} dot>
          {task.status}
        </Badge>
      </div>

      <div className="mt-2 text-sm text-ink">
        <span className="text-ink-muted">{task.messages.at(-1)?.content}</span>
      </div>

      {(output || task.status === "running" || task.status === "claimed") && (
        <div className="mt-3 whitespace-pre-wrap rounded-lg border border-border bg-surface p-3 font-mono text-sm text-ink">
          {output || (
            <span className="animate-pulse-soft text-ink-subtle">
              waiting for tokens…
            </span>
          )}
          {(task.status === "running" || task.status === "claimed") &&
            output && <span className="animate-pulse-soft">▍</span>}
        </div>
      )}
    </div>
  );
}
