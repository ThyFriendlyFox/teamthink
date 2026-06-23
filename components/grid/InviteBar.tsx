"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export function InviteBar({
  roomId,
  connected,
  createInvite,
}: {
  roomId: string;
  connected: boolean;
  /** Mints an offer-in-link blob (the cheap, instant-connect invite). */
  createInvite?: () => Promise<string>;
}) {
  const [copied, setCopied] = useState<"room" | "live" | null>(null);
  const [minting, setMinting] = useState(false);
  const path = `/s?r=${roomId}`;

  const origin = typeof window !== "undefined" ? window.location.origin : "";

  async function copyRoom() {
    try {
      await navigator.clipboard.writeText(`${origin}${path}`);
      setCopied("room");
      setTimeout(() => setCopied(null), 1500);
    } catch {
      // ignore clipboard failures
    }
  }

  async function copyLive() {
    if (!createInvite) return;
    setMinting(true);
    try {
      const blob = await createInvite();
      await navigator.clipboard.writeText(`${origin}${path}#${blob}`);
      setCopied("live");
      setTimeout(() => setCopied(null), 1500);
    } catch {
      // ignore clipboard / mint failures
    } finally {
      setMinting(false);
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
        <Button size="sm" variant="secondary" onClick={copyRoom}>
          {copied === "room" ? "Copied" : "Copy invite"}
        </Button>
        {createInvite ? (
          <Button size="sm" onClick={copyLive} disabled={minting}>
            {copied === "live"
              ? "Copied"
              : minting
                ? "Minting…"
                : "Copy quick-join link"}
          </Button>
        ) : null}
      </div>
    </div>
  );
}
