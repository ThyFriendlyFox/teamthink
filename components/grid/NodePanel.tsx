"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { MODELS } from "@/lib/config";
import type { GridNode } from "@/lib/grid/scheduler";
import type { GridSnapshot } from "@/lib/grid/types";

export function NodePanel({
  node,
  snapshot,
}: {
  node: GridNode;
  snapshot: GridSnapshot;
}) {
  const { caps, activeModelId, modelLoad } = snapshot;
  const activePipelines = snapshot.pipelines.filter(
    (p) => p.status === "warming" || p.status === "running",
  );
  const [hosting, setHosting] = useState(false);
  const compatible = caps
    ? MODELS.filter((m) => caps.compatibleModelIds.includes(m.id))
    : [];

  async function host(modelId: string) {
    setHosting(true);
    try {
      await node.setActiveModel(modelId);
    } finally {
      setHosting(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>This device</CardTitle>
      </CardHeader>

      {!caps && <p className="text-sm text-ink-muted">Detecting hardware…</p>}

      {caps && !caps.webgpu && (
        <div className="rounded-xl border border-border bg-surface-sunken p-4 text-sm text-ink-muted">
          WebGPU isn&apos;t available here, so this device joins as a
          request-only node. It can submit work to the grid but won&apos;t run
          inference.
        </div>
      )}

      {caps?.webgpu && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="positive" dot>
              WebGPU ready
            </Badge>
            {caps.gpuVendor && <Badge>{caps.gpuVendor}</Badge>}
            <Badge>~{(caps.memoryEstimateMb / 1024).toFixed(1)} GB usable</Badge>
            {caps.shaderF16 && <Badge>f16</Badge>}
            {activePipelines.map((p) => (
              <Badge key={p.planId} tone="accent" dot>
                pipeline {p.readyCount}/{p.numShards}
              </Badge>
            ))}
          </div>

          <div>
            <div className="mb-2 text-xs uppercase tracking-wide text-ink-subtle">
              Host a model
            </div>
            <div className="flex flex-wrap gap-2">
              {compatible.map((m) => {
                const active = activeModelId === m.id;
                return (
                  <Button
                    key={m.id}
                    size="sm"
                    variant={active ? "primary" : "secondary"}
                    disabled={hosting}
                    onClick={() => host(m.id)}
                  >
                    {m.label}
                  </Button>
                );
              })}
              {compatible.length === 0 && (
                <span className="text-sm text-ink-subtle">
                  No models fit this device&apos;s estimated memory.
                </span>
              )}
            </div>
          </div>

          {modelLoad && (
            <div>
              <div className="mb-1 flex justify-between text-xs text-ink-muted">
                <span className="truncate">{modelLoad.text || "loading"}</span>
                <span className="tabular-nums">
                  {Math.round(modelLoad.progress * 100)}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-sunken">
                <div
                  className="h-full rounded-full bg-accent transition-all"
                  style={{ width: `${Math.round(modelLoad.progress * 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
