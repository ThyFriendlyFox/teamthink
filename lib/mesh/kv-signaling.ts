import { SIGNAL_ENDPOINT, SIGNAL_RETRY_MS } from "@/lib/config";

/**
 * Client transport for the KV signaling mailbox (`/api/signal`). It speaks to
 * our own deployment — no public WebRTC relays — and is event-driven: each
 * "box" we care about is drained by a single held-open long-poll that returns
 * the moment a message lands, then immediately re-issues. There is no busy
 * polling and no persistent socket.
 *
 * A peer subscribes to a small, changing set of boxes:
 *   - its own peer id (private inbox) while it is still establishing links,
 *   - the `room` mailbox, but only while it is the elected greeter,
 *   - an invite key, while it has an outstanding offer-in-link invite.
 *
 * `MeshClient` adds/removes boxes as the mesh state changes, so in a stable
 * mesh only the single greeter is talking to the server at all.
 */

export interface KvSignalingEvents {
  /** A message drained from `box` (one of our subscribed boxes). */
  onMessage: (box: string, msg: Record<string, unknown>) => void;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export class KvSignaling {
  private loops = new Map<string, AbortController>();
  private stopped = false;

  constructor(
    private readonly room: string,
    private readonly events: KvSignalingEvents,
  ) {}

  /** Begin draining `box`; no-op if already subscribed. */
  listen(box: string): void {
    if (this.stopped || this.loops.has(box)) return;
    const ctrl = new AbortController();
    this.loops.set(box, ctrl);
    void this.drainLoop(box, ctrl);
  }

  /** Stop draining `box`. */
  unlisten(box: string): void {
    const ctrl = this.loops.get(box);
    if (!ctrl) return;
    this.loops.delete(box);
    ctrl.abort();
  }

  /** Drop a message into `box` for whoever is draining it. */
  async publish(box: string, msg: Record<string, unknown>): Promise<void> {
    if (this.stopped) return;
    try {
      await fetch(SIGNAL_ENDPOINT, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ room: this.room, box, msg }),
        keepalive: true,
      });
    } catch {
      // best-effort; the sender retries the handshake on its own cadence
    }
  }

  stop(): void {
    this.stopped = true;
    for (const ctrl of this.loops.values()) ctrl.abort();
    this.loops.clear();
  }

  private async drainLoop(box: string, ctrl: AbortController): Promise<void> {
    const url = `${SIGNAL_ENDPOINT}?room=${encodeURIComponent(
      this.room,
    )}&box=${encodeURIComponent(box)}`;
    while (!this.stopped && this.loops.get(box) === ctrl) {
      try {
        const res = await fetch(url, { signal: ctrl.signal });
        if (!res.ok) {
          await sleep(SIGNAL_RETRY_MS);
          continue;
        }
        const data = (await res.json()) as {
          msgs?: Record<string, unknown>[];
        };
        for (const msg of data.msgs ?? []) {
          if (msg && typeof msg === "object") this.events.onMessage(box, msg);
        }
        // Loop straight back into the next long-poll (push-like).
      } catch (err) {
        if (ctrl.signal.aborted) return; // unsubscribed/stopped
        void err;
        await sleep(SIGNAL_RETRY_MS);
      }
    }
  }
}
