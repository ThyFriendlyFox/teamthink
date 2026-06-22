import { ICE_SERVERS, SIGNALING_SERVERS } from "@/lib/config";
import { SignalingClient } from "@/lib/mesh/signaling";

/**
 * Client-side WebRTC full-mesh manager. Peers find each other through a public
 * pub/sub signaling relay (see `signaling.ts`) keyed by the room id, exchange
 * the SDP/ICE handshake there, and then talk directly over WebRTC data
 * channels. Nothing — not signaling, not data — touches our own origin; the
 * deployment only serves the static page.
 *
 * Discovery is push-based, not polled: a peer announces itself once on join
 * (with a small retry burst to ride out relay races) and again on reconnect;
 * existing peers reply so the newcomer learns them, then a deterministic
 * tie-break decides who sends the offer. A stable mesh produces no signaling
 * traffic at all.
 *
 * Framing: every data-channel frame is a Uint8Array whose first byte is a
 * channel tag, letting multiple logical streams (CRDT sync, app messages,
 * pipeline tensors) share one channel.
 */

export const CHANNEL_CRDT = 0;
export const CHANNEL_APP = 1;
/** Pipeline-parallel inference traffic (chunked tensors, tokens, control). */
export const CHANNEL_PIPE = 2;

type FrameHandler = (peerId: string, payload: Uint8Array) => void;
type WebrtcKind = "offer" | "answer" | "candidate";

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

/** Re-announce schedule (ms after join/reconnect) to survive relay races. */
const ANNOUNCE_BURST_MS = [0, 1500, 4000];

export class MeshClient {
  private connections = new Map<string, Connection>();
  private handlers = new Map<number, Set<FrameHandler>>();
  private signaling: SignalingClient;
  private announceTimers: ReturnType<typeof setTimeout>[] = [];
  private stopped = false;

  constructor(
    readonly roomId: string,
    readonly peerId: string,
    private events: MeshEvents = {},
  ) {
    this.signaling = new SignalingClient(
      `teamthink/${roomId}`,
      SIGNALING_SERVERS,
      {
        onMessage: (msg) => this.onSignal(msg),
        onOpen: () => this.announceBurst(),
      },
    );
  }

  async start(): Promise<void> {
    this.signaling.start();
  }

  stop(): void {
    this.stopped = true;
    for (const t of this.announceTimers) clearTimeout(t);
    this.announceTimers = [];
    // Best-effort departure notice so peers tear down promptly.
    this.signaling.publish({ kind: "bye", from: this.peerId });
    this.signaling.stop();
    for (const [, conn] of this.connections) {
      conn.channel?.close();
      conn.pc.close();
    }
    this.connections.clear();
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

  // --- discovery / signaling ------------------------------------------------

  /** Announce presence a few times to ride out relay/connect races. */
  private announceBurst(): void {
    for (const t of this.announceTimers) clearTimeout(t);
    this.announceTimers = ANNOUNCE_BURST_MS.map((d) =>
      setTimeout(() => {
        if (this.stopped) return;
        this.signaling.publish({ kind: "announce", from: this.peerId });
      }, d),
    );
  }

  private onSignal(msg: Record<string, unknown>): void {
    const from = msg.from as string | undefined;
    if (!from || from === this.peerId) return; // ignore our own fan-out
    const kind = msg.kind as string | undefined;

    if (kind === "announce") {
      const to = msg.to as string | undefined;
      if (to && to !== this.peerId) return; // a directed reply for someone else
      this.onAnnounce(from, Boolean(to));
    } else if (kind === "bye") {
      this.teardown(from, true);
    } else if (kind === "webrtc" && msg.to === this.peerId) {
      const sig = msg.signal as { kind: WebrtcKind; data: unknown } | undefined;
      if (sig) void this.handleSignal(from, sig.kind, sig.data);
    }
  }

  private onAnnounce(from: string, directed: boolean): void {
    if (this.connections.has(from)) return;
    // Reply to a broadcast announce so the newcomer learns about us; a directed
    // reply needs no further reply (avoids a loop).
    if (!directed) {
      this.signaling.publish({
        kind: "announce",
        from: this.peerId,
        to: from,
      });
    }
    if (this.shouldInitiate(from)) void this.initiate(from);
  }

  /** Deterministic tie-break so exactly one side creates the offer. */
  private shouldInitiate(peerId: string): boolean {
    return this.peerId < peerId;
  }

  private send(to: string, kind: WebrtcKind, data: unknown): void {
    this.signaling.publish({
      kind: "webrtc",
      from: this.peerId,
      to,
      signal: { kind, data },
    });
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
      if (e.candidate) this.send(peerId, "candidate", e.candidate.toJSON());
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
    if (this.connections.has(peerId)) return;
    const conn = this.createConnection(peerId);
    const channel = conn.pc.createDataChannel("tt", { ordered: true });
    this.bindChannel(peerId, conn, channel);
    const offer = await conn.pc.createOffer();
    await conn.pc.setLocalDescription(offer);
    this.send(peerId, "offer", offer);
  }

  private async handleSignal(
    from: string,
    kind: WebrtcKind,
    data: unknown,
  ): Promise<void> {
    let conn = this.connections.get(from);

    if (kind === "offer") {
      if (!conn) conn = this.createConnection(from);
      await conn.pc.setRemoteDescription(data as RTCSessionDescriptionInit);
      conn.remoteSet = true;
      await this.flushCandidates(conn);
      const answer = await conn.pc.createAnswer();
      await conn.pc.setLocalDescription(answer);
      this.send(from, "answer", answer);
    } else if (kind === "answer") {
      if (!conn) return;
      await conn.pc.setRemoteDescription(data as RTCSessionDescriptionInit);
      conn.remoteSet = true;
      await this.flushCandidates(conn);
    } else if (kind === "candidate") {
      if (!conn) return;
      const cand = data as RTCIceCandidateInit;
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

  private teardown(peerId: string, emitClose = false): void {
    const conn = this.connections.get(peerId);
    if (!conn) return;
    const wasOpen = conn.open;
    conn.channel?.close();
    conn.pc.close();
    this.connections.delete(peerId);
    if (emitClose && wasOpen) {
      this.events.onPeerClose?.(peerId);
      this.events.onPeersChange?.(this.connectedPeers);
    }
  }
}
