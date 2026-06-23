import { ICE_SERVERS } from "@/lib/config";
import { KvSignaling } from "@/lib/mesh/kv-signaling";
import { decodeInvite, encodeInvite, type InvitePayload } from "@/lib/mesh/invite";

/**
 * Client-side WebRTC full-mesh manager.
 *
 * Rendezvous runs over our own KV-backed signaling endpoint (`/api/signal`) —
 * no public WebRTC relays — and only ever brokers the *first* link for a new
 * peer. The cost is bounded by design:
 *
 *   - Only one peer per room (the lowest-id "greeter") holds the room mailbox
 *     open to greet newcomers. Everyone else stops talking to the server the
 *     moment they have a mesh link.
 *   - A newcomer connects to a single seed (the greeter, or — via an
 *     offer-in-link invite — whoever minted the link). From there the mesh is
 *     self-sustaining: peers gossip membership (PEX) over their data channels
 *     and broker each *new* pair's handshake through a common neighbour, so
 *     filling the mesh costs the server nothing.
 *   - Handshakes are non-trickle: ICE is gathered up front and the whole
 *     offer/answer travels as a single message, so a join is ~one or two tiny
 *     KV writes, never a stream.
 *
 * A stable mesh sends zero signaling traffic. Inference/sync/tokens are always
 * peer-to-peer and never touch the server.
 *
 * Framing: every data-channel frame is a Uint8Array whose first byte is a
 * channel tag, multiplexing CRDT sync, app messages, pipeline tensors, and
 * internal mesh control over one channel.
 */

export const CHANNEL_CRDT = 0;
export const CHANNEL_APP = 1;
/** Pipeline-parallel inference traffic (chunked tensors, tokens, control). */
export const CHANNEL_PIPE = 2;
/** Internal: peer-exchange + brokered handshake. Not for application use. */
const CHANNEL_MESH = 3;

/** Mailbox name the elected greeter watches for newcomers. */
const ROOM_BOX = "room";
/** Re-announce schedule (ms after join) to ride out races / mailbox expiry. */
const ANNOUNCE_BURST_MS = [0, 1500, 4000];
/** Cap on waiting for ICE gathering before sending the SDP anyway. */
const ICE_GATHER_TIMEOUT_MS = 3000;
/** If a mesh-brokered handshake hasn't opened in this long, give up on it. */
const BROKER_FALLBACK_MS = 9000;

type FrameHandler = (peerId: string, payload: Uint8Array) => void;

interface MeshEvents {
  onPeerOpen?: (peerId: string) => void;
  onPeerClose?: (peerId: string) => void;
  onPeersChange?: (peerIds: string[]) => void;
}

interface Connection {
  pc: RTCPeerConnection;
  /** Mutable: an invite connection starts keyed by its invite key, then is
   * re-keyed to the answerer's peer id once the answer arrives. */
  peerId: string;
  channel?: RTCDataChannel;
  open: boolean;
  /** True if this link was bootstrapped over KV (vs. brokered via the mesh). */
  viaKv: boolean;
  brokerTimer?: ReturnType<typeof setTimeout>;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

function randId(): string {
  const a = new Uint8Array(12);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

function waitForIce(pc: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    };
    const check = () => {
      if (pc.iceGatheringState === "complete") finish();
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(finish, timeoutMs);
  });
}

export class MeshClient {
  private connections = new Map<string, Connection>();
  /** Outstanding offer-in-link invites, keyed by invite key (pre-answer). */
  private pendingInvites = new Map<string, Connection>();
  private handlers = new Map<number, Set<FrameHandler>>();
  private kv: KvSignaling;
  /** Boxes we currently hold open on the signaling endpoint. */
  private subscribed = new Set<string>();
  /** For a peer we can't reach directly, the neighbour that brokers to it. */
  private brokerFor = new Map<string, string>();
  private announceTimers: ReturnType<typeof setTimeout>[] = [];
  private stopped = false;

  constructor(
    readonly roomId: string,
    readonly peerId: string,
    private events: MeshEvents = {},
  ) {
    this.kv = new KvSignaling(roomId, {
      onMessage: (box, msg) => this.onKv(box, msg),
    });
  }

  /** `invite` is the decoded payload or raw `#i=...` blob if this tab opened an
   * offer-in-link, in which case we connect to the inviter directly. */
  async start(invite?: InvitePayload | string | null): Promise<void> {
    const payload =
      typeof invite === "string" ? decodeInvite(invite) : (invite ?? null);
    if (payload) {
      // Fast path: connect to the inviter directly, no announce / no greeter.
      void this.acceptInvite(payload);
    }
    this.refreshKv();
    this.announceBurst();
  }

  stop(): void {
    this.stopped = true;
    for (const t of this.announceTimers) clearTimeout(t);
    this.announceTimers = [];
    // Best-effort departure notice so peers tear down promptly.
    this.broadcastMesh({ t: "bye", from: this.peerId });
    this.kv.stop();
    this.subscribed.clear();
    for (const conn of this.connections.values()) this.closeConn(conn);
    for (const conn of this.pendingInvites.values()) this.closeConn(conn);
    this.connections.clear();
    this.pendingInvites.clear();
    this.brokerFor.clear();
  }

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
    return [...this.connections.values()]
      .filter((c) => c.open)
      .map((c) => c.peerId);
  }

  /**
   * Mint an offer-in-link invite. Creates a ready-to-go offer (ICE gathered)
   * and returns the encoded blob to embed in a `#i=...` link. We hold the
   * invite's one-time mailbox open only until the answer arrives.
   */
  async createInvite(): Promise<string> {
    const key = randId();
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    const conn: Connection = { pc, peerId: key, open: false, viaKv: true };
    this.pendingInvites.set(key, conn);
    this.setupPc(conn);
    const channel = pc.createDataChannel("tt", { ordered: true });
    this.bindChannel(conn, channel);
    this.refreshKv();
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIce(pc, ICE_GATHER_TIMEOUT_MS);
    return encodeInvite({
      v: 1,
      from: this.peerId,
      key,
      sdp: (pc.localDescription ?? offer).sdp ?? "",
    });
  }

  // --- signaling ------------------------------------------------------------

  /** Reconcile which signaling boxes we hold open with what we actually need. */
  private refreshKv(): void {
    if (this.stopped) return;
    const desired = new Set<string>();
    const alone = this.connectedPeers.length === 0;
    const greeter = this.isGreeter();
    if (alone || greeter || this.hasPendingKv()) desired.add(this.peerId);
    if (greeter) desired.add(ROOM_BOX);
    for (const key of this.pendingInvites.keys()) desired.add(key);

    for (const box of desired) {
      if (!this.subscribed.has(box)) {
        this.subscribed.add(box);
        this.kv.listen(box);
      }
    }
    for (const box of [...this.subscribed]) {
      if (!desired.has(box)) {
        this.subscribed.delete(box);
        this.kv.unlisten(box);
      }
    }
  }

  /** Lowest-id peer (among us + our direct links) greets newcomers. */
  private isGreeter(): boolean {
    let min = this.peerId;
    for (const id of this.connectedPeers) if (id < min) min = id;
    return min === this.peerId;
  }

  private hasPendingKv(): boolean {
    for (const c of this.connections.values()) if (c.viaKv && !c.open) return true;
    return false;
  }

  private announceBurst(): void {
    for (const t of this.announceTimers) clearTimeout(t);
    this.announceTimers = ANNOUNCE_BURST_MS.map((d) =>
      setTimeout(() => {
        if (this.stopped || this.connectedPeers.length > 0) return;
        void this.kv.publish(ROOM_BOX, { kind: "announce", from: this.peerId });
      }, d),
    );
  }

  private onKv(box: string, msg: Record<string, unknown>): void {
    const from = msg.from as string | undefined;
    const kind = msg.kind as string | undefined;

    if (box === ROOM_BOX) {
      if (kind === "announce" && from) this.onAnnounce(from);
      return;
    }
    // Otherwise box is our own inbox or an outstanding invite key.
    if (kind !== "sdp" || !from) return;
    const desc = msg.desc as RTCSessionDescriptionInit | undefined;
    if (!desc) return;
    if (this.pendingInvites.has(box)) {
      void this.resolveInvite(box, from, desc);
    } else {
      void this.handleSdp(from, desc, undefined);
    }
  }

  /** A newcomer announced on the room mailbox; greet them with an offer. */
  private onAnnounce(from: string): void {
    if (from === this.peerId || this.connections.has(from)) return;
    const established = this.connectedPeers.length > 0;
    // A settled mesh member always initiates to a newcomer; two unestablished
    // peers fall back to a deterministic tie-break to avoid double-offering.
    if (established || this.shouldInitiate(from)) void this.initiate(from, undefined);
  }

  private shouldInitiate(peerId: string): boolean {
    return this.peerId < peerId;
  }

  /** Route an SDP to `to`: through a connected broker when we have one, else
   * over the KV mailbox (the seed/bootstrap path). */
  private async routeSignal(
    to: string,
    desc: RTCSessionDescriptionInit,
  ): Promise<void> {
    const broker = this.brokerFor.get(to);
    const brokerConn = broker ? this.connections.get(broker) : undefined;
    if (broker && broker !== to && brokerConn?.open) {
      this.sendMesh(broker, { t: "sig", to, from: this.peerId, desc });
      return;
    }
    await this.kv.publish(to, { kind: "sdp", from: this.peerId, desc });
  }

  // --- peer exchange (PEX) over the mesh ------------------------------------

  private sendMesh(peerId: string, obj: Record<string, unknown>): boolean {
    return this.sendTo(peerId, CHANNEL_MESH, enc.encode(JSON.stringify(obj)));
  }

  private broadcastMesh(obj: Record<string, unknown>): void {
    const frame = enc.encode(JSON.stringify(obj));
    for (const [peerId] of this.connections) {
      this.sendTo(peerId, CHANNEL_MESH, frame);
    }
  }

  private gossipPeers(): void {
    const peers = this.connectedPeers;
    for (const peerId of peers) {
      this.sendMesh(peerId, { t: "pex", peers: peers.filter((p) => p !== peerId) });
    }
  }

  private onMeshControl(deliveredBy: string, payload: Uint8Array): void {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(dec.decode(payload)) as Record<string, unknown>;
    } catch {
      return;
    }
    const t = msg.t as string | undefined;

    if (t === "pex") {
      const peers = (msg.peers as string[] | undefined) ?? [];
      for (const p of peers) {
        if (p === this.peerId || this.connections.has(p)) continue;
        // Brokered through the neighbour that told us — it knows both of us.
        this.maybeInitiate(p, deliveredBy);
      }
    } else if (t === "sig") {
      const to = msg.to as string | undefined;
      const from = msg.from as string | undefined;
      const desc = msg.desc as RTCSessionDescriptionInit | undefined;
      if (!to || !from) return;
      if (to === this.peerId) {
        if (desc) void this.handleSdp(from, desc, deliveredBy);
      } else {
        this.sendMesh(to, msg); // one-hop forward
      }
    } else if (t === "bye") {
      const byeFrom = msg.from as string | undefined;
      if (byeFrom) this.teardown(byeFrom, true);
    }
  }

  // --- connection setup -----------------------------------------------------

  private setupPc(conn: Connection): void {
    const pc = conn.pc;
    pc.onconnectionstatechange = () => {
      const s = pc.connectionState;
      if (s === "failed" || s === "closed") {
        this.teardown(conn.peerId, true);
      } else if (s === "disconnected" && conn.open) {
        conn.open = false;
        this.events.onPeerClose?.(conn.peerId);
        this.events.onPeersChange?.(this.connectedPeers);
        this.refreshKv();
      }
    };
    pc.ondatachannel = (e) => this.bindChannel(conn, e.channel);
  }

  private createConnection(peerId: string, viaKv: boolean): Connection {
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    const conn: Connection = { pc, peerId, open: false, viaKv };
    this.connections.set(peerId, conn);
    this.setupPc(conn);
    return conn;
  }

  private bindChannel(conn: Connection, channel: RTCDataChannel): void {
    channel.binaryType = "arraybuffer";
    conn.channel = channel;
    channel.onopen = () => {
      conn.open = true;
      if (conn.brokerTimer) {
        clearTimeout(conn.brokerTimer);
        conn.brokerTimer = undefined;
      }
      this.brokerFor.delete(conn.peerId);
      this.events.onPeerOpen?.(conn.peerId);
      this.events.onPeersChange?.(this.connectedPeers);
      this.gossipPeers();
      this.refreshKv();
    };
    channel.onclose = () => {
      if (conn.open) {
        conn.open = false;
        this.events.onPeerClose?.(conn.peerId);
        this.events.onPeersChange?.(this.connectedPeers);
        this.refreshKv();
      }
    };
    channel.onmessage = (e) => {
      const buf = new Uint8Array(e.data as ArrayBuffer);
      const tag = buf[0];
      const body = buf.subarray(1);
      if (tag === CHANNEL_MESH) {
        this.onMeshControl(conn.peerId, body);
        return;
      }
      const set = this.handlers.get(tag);
      if (set) for (const h of set) h(conn.peerId, body);
    };
  }

  private maybeInitiate(peerId: string, broker: string | undefined): void {
    if (this.connections.has(peerId)) return;
    if (this.shouldInitiate(peerId)) void this.initiate(peerId, broker);
  }

  private async initiate(
    peerId: string,
    broker: string | undefined,
  ): Promise<void> {
    if (this.connections.has(peerId)) return;
    if (broker) this.brokerFor.set(peerId, broker);
    const conn = this.createConnection(peerId, broker === undefined);
    const channel = conn.pc.createDataChannel("tt", { ordered: true });
    this.bindChannel(conn, channel);
    this.refreshKv(); // ensure our inbox is open to receive the answer
    if (broker) {
      conn.brokerTimer = setTimeout(() => {
        if (this.stopped || conn.open) return;
        // Brokered handshake stalled; drop it so PEX can retry via another
        // neighbour as the mesh changes.
        this.teardown(peerId);
      }, BROKER_FALLBACK_MS);
    }
    const offer = await conn.pc.createOffer();
    await conn.pc.setLocalDescription(offer);
    await waitForIce(conn.pc, ICE_GATHER_TIMEOUT_MS);
    await this.routeSignal(peerId, conn.pc.localDescription ?? offer);
  }

  private async handleSdp(
    from: string,
    desc: RTCSessionDescriptionInit,
    broker: string | undefined,
  ): Promise<void> {
    let conn = this.connections.get(from);

    if (desc.type === "offer") {
      if (!conn) conn = this.createConnection(from, broker === undefined);
      if (broker) this.brokerFor.set(from, broker);
      await conn.pc.setRemoteDescription(desc);
      const answer = await conn.pc.createAnswer();
      await conn.pc.setLocalDescription(answer);
      await waitForIce(conn.pc, ICE_GATHER_TIMEOUT_MS);
      await this.routeSignal(from, conn.pc.localDescription ?? answer);
    } else if (desc.type === "answer") {
      if (!conn) return;
      await conn.pc.setRemoteDescription(desc);
    }
    this.refreshKv();
  }

  // --- offer-in-link invites ------------------------------------------------

  private async acceptInvite(p: InvitePayload): Promise<void> {
    if (p.from === this.peerId || this.connections.has(p.from)) return;
    const conn = this.createConnection(p.from, true);
    await conn.pc.setRemoteDescription({ type: "offer", sdp: p.sdp });
    const answer = await conn.pc.createAnswer();
    await conn.pc.setLocalDescription(answer);
    await waitForIce(conn.pc, ICE_GATHER_TIMEOUT_MS);
    // Reply straight to the invite's one-time mailbox; the inviter is waiting.
    await this.kv.publish(p.key, {
      kind: "sdp",
      from: this.peerId,
      desc: conn.pc.localDescription ?? answer,
    });
  }

  private async resolveInvite(
    key: string,
    from: string,
    desc: RTCSessionDescriptionInit,
  ): Promise<void> {
    const conn = this.pendingInvites.get(key);
    if (!conn) return;
    this.pendingInvites.delete(key);
    this.subscribed.delete(key);
    this.kv.unlisten(key);
    if (this.connections.has(from)) {
      this.closeConn(conn); // already linked another way
      this.refreshKv();
      return;
    }
    conn.peerId = from;
    this.connections.set(from, conn);
    await conn.pc.setRemoteDescription(desc);
    this.refreshKv();
  }

  // --- teardown -------------------------------------------------------------

  private closeConn(conn: Connection): void {
    if (conn.brokerTimer) clearTimeout(conn.brokerTimer);
    try {
      conn.channel?.close();
      conn.pc.close();
    } catch {
      // ignore
    }
  }

  private teardown(id: string, emitClose = false): void {
    const conn = this.connections.get(id);
    if (conn) {
      const wasOpen = conn.open;
      this.closeConn(conn);
      this.connections.delete(id);
      this.brokerFor.delete(id);
      if (emitClose && wasOpen) {
        this.events.onPeerClose?.(conn.peerId);
        this.events.onPeersChange?.(this.connectedPeers);
      }
      this.maybeReseed();
      this.refreshKv();
      return;
    }
    const pending = this.pendingInvites.get(id);
    if (pending) {
      this.closeConn(pending);
      this.pendingInvites.delete(id);
      this.subscribed.delete(id);
      this.kv.unlisten(id);
      this.refreshKv();
    }
  }

  /** If we've fallen out of the mesh entirely, re-announce to find a seed. */
  private maybeReseed(): void {
    if (this.stopped || this.connections.size > 0) return;
    this.announceBurst();
  }
}
