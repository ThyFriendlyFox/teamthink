"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export function InviteBar({
  roomId,
  connected,
}: {
  roomId: string;
  connected: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const path = `/s?r=${roomId}`;

  async function copy() {
    const url =
      typeof window !== "undefined" ? `${window.location.origin}${path}` : path;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore clipboard failures
    }
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border bg-surface/70 px-6 py-4 backdrop-blur">
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="font-display text-lg text-ink hover:text-accent-strong"
        >
          TeamThink
        </Link>
        <span className="text-ink-subtle">/</span>
        <span className="font-mono text-sm text-ink-muted">{roomId}</span>
        <Badge tone={connected ? "positive" : "neutral"} dot>
          {connected ? "connected" : "waiting for peers"}
        </Badge>
      </div>
      <div className="flex items-center gap-2">
        <code className="hidden max-w-[280px] truncate rounded-lg border border-border bg-canvas px-3 py-1.5 text-xs text-ink-muted sm:block">
          {path}
        </code>
        <Button size="sm" variant="secondary" onClick={copy}>
          {copied ? "Copied" : "Copy invite"}
        </Button>
      </div>
    </div>
  );
}
