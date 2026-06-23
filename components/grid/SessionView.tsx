"use client";

import { InferenceConsole } from "@/components/grid/InferenceConsole";
import { InviteBar } from "@/components/grid/InviteBar";
import { NodePanel } from "@/components/grid/NodePanel";
import { PeerList } from "@/components/grid/PeerList";
import { Stat } from "@/components/ui/Stat";
import { useGridNode } from "@/lib/grid/useGridNode";

export function SessionView({ roomId }: { roomId: string }) {
  const { node, snapshot } = useGridNode(roomId);

  const computeNodes = snapshot.peers.filter((p) => p.caps.webgpu).length;
  const activeJobs = snapshot.peers.reduce((n, p) => n + p.activeJobs, 0);
  const openTasks = snapshot.tasks.filter((t) => t.status === "open").length;

  return (
    <main className="flex-1">
      <InviteBar
        roomId={roomId}
        connected={snapshot.connected}
        createInvite={() => node.createInvite()}
      />

      <div className="mx-auto grid max-w-6xl gap-5 px-6 py-6 lg:grid-cols-[360px_1fr]">
        <div className="space-y-5">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Nodes" value={snapshot.peers.length} />
            <Stat label="Compute" value={computeNodes} />
            <Stat
              label="Active"
              value={activeJobs}
              hint={openTasks ? `${openTasks} queued` : undefined}
            />
          </div>
          <NodePanel snapshot={snapshot} />
          <PeerList peers={snapshot.peers} />
        </div>

        <div className="min-h-[70vh]">
          <InferenceConsole node={node} snapshot={snapshot} />
        </div>
      </div>
    </main>
  );
}
