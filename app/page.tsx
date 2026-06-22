"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { generateRoomId } from "@/lib/id";

export default function Home() {
  const router = useRouter();
  const [creating, setCreating] = useState(false);
  const [joinValue, setJoinValue] = useState("");

  function createSession() {
    setCreating(true);
    // Room ids are minted client-side; the id in the link is all a peer needs
    // to join the pool over public signaling. No server round-trip.
    router.push(`/s?r=${generateRoomId()}`);
  }

  function joinSession() {
    const code = extractRoomId(joinValue.trim());
    if (code) router.push(`/s?r=${code}`);
  }

  return (
    <main className="flex-1">
      <div className="mx-auto flex min-h-[calc(100vh-0px)] max-w-3xl flex-col px-6 py-16 sm:py-24">
        <header className="animate-fade-in">
          <Badge tone="accent" dot>
            WebGPU inference grid
          </Badge>
          <h1 className="mt-6 font-display text-5xl leading-tight text-ink sm:text-6xl">
            TeamThink
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-muted">
            Start a session, invite devices with a link, and run model inference
            across a peer-to-peer mesh. Each device that joins becomes a compute
            node; requests are routed to whoever has the GPU to spare.
          </p>
        </header>

        <section
          className="mt-12 grid gap-4 sm:grid-cols-2"
          style={{ animationDelay: "60ms" }}
        >
          <div className="animate-fade-in rounded-2xl border border-border bg-surface p-6">
            <h2 className="font-display text-xl text-ink">Host a session</h2>
            <p className="mt-2 text-sm text-ink-muted">
              Create a new grid and get an invite link to share.
            </p>
            <Button
              size="lg"
              className="mt-5 w-full"
              onClick={createSession}
              disabled={creating}
            >
              {creating ? "Creating…" : "Create session"}
            </Button>
          </div>

          <div className="animate-fade-in rounded-2xl border border-border bg-surface p-6">
            <h2 className="font-display text-xl text-ink">Join a session</h2>
            <p className="mt-2 text-sm text-ink-muted">
              Paste an invite link or enter a session code.
            </p>
            <div className="mt-5 flex gap-2">
              <input
                value={joinValue}
                onChange={(e) => setJoinValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && joinSession()}
                placeholder="link or code"
                className="h-12 flex-1 rounded-2xl border border-border bg-canvas px-4 text-sm text-ink outline-none placeholder:text-ink-subtle focus:border-accent focus:ring-2 focus:ring-accent/30"
              />
              <Button
                size="lg"
                variant="secondary"
                onClick={joinSession}
                disabled={!joinValue.trim()}
              >
                Join
              </Button>
            </div>
          </div>
        </section>

        <section className="mt-16 animate-fade-in">
          <h3 className="font-display text-lg text-ink">How it works</h3>
          <ol className="mt-4 space-y-3 text-sm text-ink-muted">
            <li className="flex gap-3">
              <span className="font-mono text-accent-strong">01</span>
              Devices connect directly over WebRTC; a tiny rendezvous only
              brokers the initial handshake.
            </li>
            <li className="flex gap-3">
              <span className="font-mono text-accent-strong">02</span>
              Each node gossips its capabilities and load; shared session state
              is replicated across peers as a CRDT.
            </li>
            <li className="flex gap-3">
              <span className="font-mono text-accent-strong">03</span>
              Capable nodes claim open requests, run them on WebGPU, and stream
              tokens back to the requester.
            </li>
          </ol>
        </section>
      </div>
    </main>
  );
}

function extractRoomId(value: string): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    // Current form: /s?r=<id>. Legacy form: /s/<id>.
    const q = url.searchParams.get("r");
    if (q && /^[a-z0-9]+$/i.test(q)) return q;
    const match = url.pathname.match(/\/s\/([a-z0-9]+)/i);
    if (match) return match[1];
  } catch {
    // not a URL; treat as a raw code
  }
  return /^[a-z0-9]+$/i.test(value) ? value : null;
}
