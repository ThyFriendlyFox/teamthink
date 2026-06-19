"use client";

import { Badge } from "@/components/ui/Badge";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import type { GridSnapshot } from "@/lib/grid/types";

export function NodePanel({ snapshot }: { snapshot: GridSnapshot }) {
  const { caps, provisioned } = snapshot;
  const hostsShard =
    !!provisioned?.shards.some((s) => s.peerId === snapshot.selfId);

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
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="positive" dot>
            WebGPU ready
          </Badge>
          {caps.gpuVendor && <Badge>{caps.gpuVendor}</Badge>}
          <Badge>~{(caps.memoryEstimateMb / 1024).toFixed(1)} GB usable</Badge>
          {caps.shaderF16 && <Badge>f16</Badge>}
          {hostsShard && provisioned && (
            <Badge tone="accent" dot>
              hosting shard
            </Badge>
          )}
        </div>
      )}
    </Card>
  );
}
