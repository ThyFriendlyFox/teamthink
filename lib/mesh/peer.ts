import {
  ICE_SERVERS,
  SIGNAL_POLL_BACKOFF,
  SIGNAL_POLL_MAX_MS,
  SIGNAL_POLL_MIN_MS,
} from "@/lib/config";
import type { SignalMessage } from "@/lib/server/signaling-store";

/**
 * Client-side WebRTC full-mesh manager. Uses the KV-backed signaling mailbox
 * (/api/signal) only to bootstrap connections; once a data channel opens, all
 * traffic flows peer-to-peer.
 *
 * Framing: every data-channel frame is a Uint8Array whose first byte is a
 * channel tag, letting multiple logical streams (CRDT sync, app messages)
 * share one channel.
 */

export const CHANNEL_CRDT = 0;
export const CHANNEL_APP = 1;
/** Pipeline-parallel inference traffic (chunked tensors, tokens, control). */
export const CHANNEL_PIPE = 2;

type FrameHandler = (peerId: string, payload: Uint8Array) => void;

interface MeshEvents {
  onPeerOpen?: (peerId: string) => void;
  onPeerClose?: (peerId: string) => void;
  onPeersChange?: (peerIds: string[]) => void;
}

interface Connection {
  pc: RTCPeerConnection;
  channel?: RTCDataChannel;
  pendingCandidates: RTCIceCandidateInit[];
  remoteSet: boolean;
  open: boolean;
}

export class MeshClient {
  private connections = new Map<string, Connection>();
  private handlers = new Map<number, Set<FrameHandler>>();
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private pollDelay = SIGNAL_POLL_MIN_MS;
  private polling = false;
  private peerSig = "";
  private stopped = false;

  constructor(
    readonly roomId: string,
    readonly peerId: string,
    private events: MeshEvents = {},
  ) {}

  async start(): Promise<void> {
    await this.signal({ action: "join", roomId: this.roomId, peerId: this.peerId });
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", this.onVisibility);
    }
    void this.tick();
  }

  stop(): void {
    this.stopped = true;
    if (typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.onVisibility);
    }
    if (this.pollTimer) clearTimeout(this.pollTimer);
    for (const [, conn] of this.connections) {
      conn.channel?.close();
      conn.pc.close();
    }
    this.connections.clear();
    // Best-effort presence removal.
    void this.signal({
      action: "leave",
      roomId: this.roomId,
      peerId: this.peerId,
    }).catch(() => {});
  }

  /** Subscribe to frames on a logical channel. Returns an unsubscribe fn. */
  on(channel: number, handler: FrameHandler): () => void {
    let set = this.handlers.get(channel);
    if (!set) {
      set = new Set();
      this.handlers.set(channel, set);
    }
    set.add(handler);
    return () => set!.delete(handler);
  }

  broadcast(channel: number, payload: Uint8Array): void {
    for (const [peerId] of this.connections) this.sendTo(peerId, channel, payload);
  }

  sendTo(peerId: string, channel: number, payload: Uint8Array): boolean {
    const conn = this.connections.get(peerId);
    if (!conn?.channel || conn.channel.readyState !== "open") return false;
    const frame = new Uint8Array(payload.length + 1);
    frame[0] = channel;
    frame.set(payload, 1);
    conn.channel.send(frame);
    return true;
  }

  get connectedPeers(): string[] {
    return [...this.connections.entries()]
      .filter(([, c]) => c.open)
      .map(([id]) => id);
  }

  // --- signaling loop -------------------------------------------------------

  /** Schedule the next poll, replacing any pending timer. */
  private scheduleNext(delay: number): void {
    if (this.stopped) return;
    if (this.pollTimer) clearTimeout(this.pollTimer);
    this.pollTimer = setTimeout(() => void this.tick(), delay);
  }

  private async tick(): Promise<void> {
    if (this.stopped || this.polling) return;
    this.polling = true;
    let active = false;
    try {
      active = await this.poll();
    } finally {
      this.polling = false;
    }
    if (this.stopped) return;
    this.pollDelay = this.nextPollDelay(active);
    this.scheduleNext(this.pollDelay);
  }

  /**
   * Fast cadence while connecting or when state is changing; otherwise back off
   * geometrically toward the max so a stable, fully-connected mesh barely
   * touches the server. Hidden tabs go straight to the max.
   */
  private nextPollDelay(active: boolean): number {
    const hidden = typeof document !== "undefined" && document.hidden;
    if (active && !hidden) return SIGNAL_POLL_MIN_MS;
    const backed = Math.min(
      SIGNAL_POLL_MAX_MS,
      Math.round(this.pollDelay * SIGNAL_POLL_BACKOFF),
    );
    return hidden ? SIGNAL_POLL_MAX_MS : backed;
  }

  /** A tab returning to the foreground should reconnect/discover promptly. */
  private onVisibility = (): void => {
    if (typeof document === "undefined" || document.hidden || this.stopped) {
      return;
    }
    this.pollDelay = SIGNAL_POLL_MIN_MS;
    if (!this.polling) this.scheduleNext(0);
  };

  /** Returns whether this poll observed activity (keeps the cadence fast). */
  private async poll(): Promise<boolean> {
    const res = await this.signal({
      action: "poll",
      roomId: this.roomId,
      peerId: this.peerId,
    });
    if (!res) return false;
    const { messages = [], peers = [] } = res as {
      messages?: SignalMessage[];
      peers?: string[];
    };

    for (const peerId of peers) {
      if (!this.connections.has(peerId) && this.shouldInitiate(peerId)) {
        await this.initiate(peerId);
      }
    }
    // Drop connections to peers that left.
    for (const [peerId, conn] of this.connections) {
      if (!peers.includes(peerId) && !conn.open) {
        // keep open connections even if presence momentarily lapses
        this.teardown(peerId);
      }
    }

    for (const msg of messages) await this.handleSignal(msg);
    this.events.onPeersChange?.(this.connectedPeers);

    // Stay fast while handshakes are in flight or the peer set is shifting;
    // a stable, fully-connected mesh produces none of these and backs off.
    const peerSig = [...peers].sort().join(",");
    const peersChanged = peerSig !== this.peerSig;
    this.peerSig = peerSig;
    const establishing = [...this.connections.values()].some((c) => !c.open);
    return messages.length > 0 || peersChanged || establishing;
  }

  /** Deterministic tie-break so exactly one side creates the offer. */
  private shouldInitiate(peerId: string): boolean {
    return this.peerId < peerId;
  }

  // --- connection setup -----------------------------------------------------

  private createConnection(peerId: string): Connection {
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    const conn: Connection = {
      pc,
      pendingCandidates: [],
      remoteSet: false,
      open: false,
    };
    this.connections.set(peerId, conn);

    pc.onicecandidate = (e) => {
      if (e.candidate) {
        void this.send(peerId, "candidate", e.candidate.toJSON());
      }
    };
    pc.onconnectionstatechange = () => {
      if (
        pc.connectionState === "failed" ||
        pc.connectionState === "closed" ||
        pc.connectionState === "disconnected"
      ) {
        if (conn.open) {
          conn.open = false;
          this.events.onPeerClose?.(peerId);
        }
      }
    };
    pc.ondatachannel = (e) => this.bindChannel(peerId, conn, e.channel);
    return conn;
  }

  private bindChannel(
    peerId: string,
    conn: Connection,
    channel: RTCDataChannel,
  ): void {
    channel.binaryType = "arraybuffer";
    conn.channel = channel;
    channel.onopen = () => {
      conn.open = true;
      this.events.onPeerOpen?.(peerId);
      this.events.onPeersChange?.(this.connectedPeers);
    };
    channel.onclose = () => {
      if (conn.open) {
        conn.open = false;
        this.events.onPeerClose?.(peerId);
      }
    };
    channel.onmessage = (e) => {
      const buf = new Uint8Array(e.data as ArrayBuffer);
      const tag = buf[0];
      const payload = buf.subarray(1);
      const set = this.handlers.get(tag);
      if (set) for (const h of set) h(peerId, payload);
    };
  }

  private async initiate(peerId: string): Promise<void> {
    const conn = this.createConnection(peerId);
    const channel = conn.pc.createDataChannel("tt", { ordered: true });
    this.bindChannel(peerId, conn, channel);
    const offer = await conn.pc.createOffer();
    await conn.pc.setLocalDescription(offer);
    await this.send(peerId, "offer", offer);
  }

  private async handleSignal(msg: SignalMessage): Promise<void> {
    let conn = this.connections.get(msg.from);

    if (msg.kind === "offer") {
      if (!conn) conn = this.createConnection(msg.from);
      await conn.pc.setRemoteDescription(
        msg.data as RTCSessionDescriptionInit,
      );
      conn.remoteSet = true;
      await this.flushCandidates(conn);
      const answer = await conn.pc.createAnswer();
      await conn.pc.setLocalDescription(answer);
      await this.send(msg.from, "answer", answer);
    } else if (msg.kind === "answer") {
      if (!conn) return;
      await conn.pc.setRemoteDescription(
        msg.data as RTCSessionDescriptionInit,
      );
      conn.remoteSet = true;
      await this.flushCandidates(conn);
    } else if (msg.kind === "candidate") {
      if (!conn) return;
      const cand = msg.data as RTCIceCandidateInit;
      if (conn.remoteSet) {
        await conn.pc.addIceCandidate(cand).catch(() => {});
      } else {
        conn.pendingCandidates.push(cand);
      }
    }
  }

  private async flushCandidates(conn: Connection): Promise<void> {
    for (const cand of conn.pendingCandidates) {
      await conn.pc.addIceCandidate(cand).catch(() => {});
    }
    conn.pendingCandidates = [];
  }

  private teardown(peerId: string): void {
    const conn = this.connections.get(peerId);
    if (!conn) return;
    conn.channel?.close();
    conn.pc.close();
    this.connections.delete(peerId);
  }

  // --- transport helpers ----------------------------------------------------

  private async send(
    to: string,
    kind: SignalMessage["kind"],
    data: unknown,
  ): Promise<void> {
    await this.signal({
      action: "send",
      roomId: this.roomId,
      message: { from: this.peerId, to, kind, data, ts: Date.now() },
    });
  }

  private async signal(body: unknown): Promise<unknown> {
    try {
      const res = await fetch("/api/signal", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }
}
