/**
 * Public WebRTC signaling transport. Connects to a free, public pub/sub
 * signaling relay (the y-webrtc signaling protocol) over a single WebSocket and
 * relays the tiny SDP/ICE handshake between peers in the same room topic.
 *
 * This deliberately uses third-party public infrastructure (or a relay you run
 * yourself) rather than our own backend: the deployment only serves the static
 * page, and no signaling — let alone inference data — ever touches our origin.
 * Once peers complete the handshake, all traffic is direct peer-to-peer; this
 * socket stays open, idle, only to learn about peers who join later.
 *
 * Protocol (matches y-webrtc's signaling servers):
 *   client -> { type: "subscribe", topics: [topic] }
 *   client -> { type: "publish", topic, ...payload }   (fan-out to subscribers)
 *   client -> { type: "ping" }  /  server -> { type: "pong" }
 *   server -> { type: "publish", topic, ...payload }
 */

export interface SignalingEvents {
  /** A message published to our topic by some peer (including, possibly, us). */
  onMessage: (msg: Record<string, unknown>) => void;
  /** Fired whenever the socket (re)connects and the topic is (re)subscribed. */
  onOpen?: () => void;
}

const PING_INTERVAL_MS = 25000;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 15000;

export class SignalingClient {
  private ws: WebSocket | null = null;
  private serverIndex = 0;
  private attempt = 0;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private sendQueue: Record<string, unknown>[] = [];
  private stopped = false;

  constructor(
    private readonly topic: string,
    private readonly servers: string[],
    private readonly events: SignalingEvents,
  ) {}

  start(): void {
    if (this.servers.length === 0) {
      console.warn("[signaling] no signaling servers configured");
      return;
    }
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    try {
      this.ws?.close();
    } catch {
      // ignore
    }
    this.ws = null;
  }

  /** Fan a message out to all peers subscribed to the room topic. */
  publish(payload: Record<string, unknown>): void {
    this.send({ type: "publish", topic: this.topic, ...payload });
  }

  private connect(): void {
    if (this.stopped) return;
    const url = this.servers[this.serverIndex % this.servers.length];
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      if (this.stopped) return;
      this.attempt = 0;
      this.rawSend({ type: "subscribe", topics: [this.topic] });
      this.flushQueue();
      this.startPing();
      this.events.onOpen?.();
    };

    ws.onmessage = (e) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(typeof e.data === "string" ? e.data : "") as Record<
          string,
          unknown
        >;
      } catch {
        return;
      }
      if (msg.type === "publish" && msg.topic === this.topic) {
        this.events.onMessage(msg);
      }
    };

    ws.onerror = () => {
      try {
        ws.close();
      } catch {
        // ignore; onclose will handle reconnect
      }
    };

    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.stopPing();
      // Rotate to the next server on each failed/closed connection.
      this.serverIndex += 1;
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer) return;
    const delay = Math.min(
      RECONNECT_MAX_MS,
      RECONNECT_BASE_MS * 2 ** Math.min(this.attempt, 4),
    );
    this.attempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(
      () => this.rawSend({ type: "ping" }),
      PING_INTERVAL_MS,
    );
  }

  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private clearTimers(): void {
    this.stopPing();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private send(obj: Record<string, unknown>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.rawSend(obj);
    } else {
      // Buffer until the socket (re)connects so a publish is never lost.
      this.sendQueue.push(obj);
      if (this.sendQueue.length > 256) this.sendQueue.shift();
    }
  }

  private flushQueue(): void {
    const queued = this.sendQueue;
    this.sendQueue = [];
    for (const obj of queued) this.rawSend(obj);
  }

  private rawSend(obj: Record<string, unknown>): void {
    try {
      this.ws?.send(JSON.stringify(obj));
    } catch {
      // ignore transient send failures; reconnect will re-subscribe
    }
  }
}
