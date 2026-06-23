"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { GridNode } from "@/lib/grid/scheduler";
import type { GridSnapshot } from "@/lib/grid/types";

/**
 * Creates and owns a GridNode for the lifetime of a session component, exposing
 * its snapshot via useSyncExternalStore for tear-free React updates.
 */
export function useGridNode(roomId: string) {
  // Lazy state init creates the node exactly once; safe to read during render.
  const [node] = useState(() => new GridNode(roomId));
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // An offer-in-link arrives in the URL fragment (`#i=...`), which never hits
    // the server. Consume it, then strip it so a single-use offer isn't reused
    // on refresh or reshare.
    let invite: string | null = null;
    if (typeof window !== "undefined") {
      const hash = window.location.hash.slice(1);
      if (hash.includes("i=")) {
        invite = hash;
        window.history.replaceState(
          null,
          "",
          window.location.pathname + window.location.search,
        );
      }
    }
    void node.start(invite).then(() => {
      if (!cancelled) setReady(true);
    });
    return () => {
      cancelled = true;
      node.stop();
    };
  }, [node]);

  const snapshot = useSyncExternalStore<GridSnapshot>(
    (cb) => node.subscribe(cb),
    () => node.getSnapshot(),
    () => node.getSnapshot(),
  );

  return { node, snapshot, ready };
}
