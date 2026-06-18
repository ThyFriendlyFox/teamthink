import { Redis } from "@upstash/redis";
import { ROOM_TTL_SECONDS, SIGNAL_TTL_SECONDS } from "@/lib/config";

/**
 * Signaling/rendezvous store. Backed by Upstash Redis (Vercel KV) when the
 * REST credentials are present; otherwise falls back to an in-memory store so
 * local development works without provisioning anything.
 *
 * Only carries bootstrap data: the room peer registry and short-lived SDP/ICE
 * mailboxes. No inference data ever passes through here.
 */

export interface SignalMessage {
  from: string;
  to: string;
  /** WebRTC signaling payload: offer | answer | candidate. */
  kind: "offer" | "answer" | "candidate";
  data: unknown;
  ts: number;
}

interface SignalingStore {
  registerPeer(roomId: string, peerId: string): Promise<void>;
  listPeers(roomId: string): Promise<string[]>;
  removePeer(roomId: string, peerId: string): Promise<void>;
  push(roomId: string, message: SignalMessage): Promise<void>;
  drain(roomId: string, peerId: string): Promise<SignalMessage[]>;
  saveSnapshot(roomId: string, snapshot: string): Promise<void>;
  loadSnapshot(roomId: string): Promise<string | null>;
}

const peersKey = (roomId: string) => `tt:room:${roomId}:peers`;
const inboxKey = (roomId: string, peerId: string) =>
  `tt:room:${roomId}:inbox:${peerId}`;
const snapshotKey = (roomId: string) => `tt:room:${roomId}:snapshot`;

/** Peers are considered present if seen within this window. */
const PEER_PRESENCE_MS = 20000;

function getRedis(): Redis | null {
  // Accept the standard Upstash/Vercel KV names as well as the names produced
  // when the Vercel integration is configured with an "UPSTASH_REDIS_REST"
  // storage prefix (which yields UPSTASH_REDIS_REST_KV_REST_API_*).
  const url =
    process.env.UPSTASH_REDIS_REST_URL ??
    process.env.KV_REST_API_URL ??
    process.env.UPSTASH_REDIS_REST_KV_REST_API_URL;
  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ??
    process.env.KV_REST_API_TOKEN ??
    process.env.UPSTASH_REDIS_REST_KV_REST_API_TOKEN;
  if (!url || !token) return null;
  return new Redis({ url, token });
}

class RedisStore implements SignalingStore {
  constructor(private redis: Redis) {}

  async registerPeer(roomId: string, peerId: string): Promise<void> {
    const key = peersKey(roomId);
    await this.redis.zadd(key, { score: Date.now(), member: peerId });
    await this.redis.expire(key, ROOM_TTL_SECONDS);
  }

  async listPeers(roomId: string): Promise<string[]> {
    const key = peersKey(roomId);
    const cutoff = Date.now() - PEER_PRESENCE_MS;
    await this.redis.zremrangebyscore(key, 0, cutoff);
    return (await this.redis.zrange<string[]>(key, 0, -1)) ?? [];
  }

  async removePeer(roomId: string, peerId: string): Promise<void> {
    await this.redis.zrem(peersKey(roomId), peerId);
  }

  async push(roomId: string, message: SignalMessage): Promise<void> {
    const key = inboxKey(roomId, message.to);
    await this.redis.rpush(key, JSON.stringify(message));
    await this.redis.expire(key, SIGNAL_TTL_SECONDS);
  }

  async drain(roomId: string, peerId: string): Promise<SignalMessage[]> {
    const key = inboxKey(roomId, peerId);
    const items = (await this.redis.lrange<string>(key, 0, -1)) ?? [];
    if (items.length > 0) await this.redis.del(key);
    return items.map((i) =>
      typeof i === "string" ? (JSON.parse(i) as SignalMessage) : (i as SignalMessage),
    );
  }

  async saveSnapshot(roomId: string, snapshot: string): Promise<void> {
    await this.redis.set(snapshotKey(roomId), snapshot, {
      ex: ROOM_TTL_SECONDS,
    });
  }

  async loadSnapshot(roomId: string): Promise<string | null> {
    return (await this.redis.get<string>(snapshotKey(roomId))) ?? null;
  }
}

interface MemPeer {
  ts: number;
}

class MemoryStore implements SignalingStore {
  private peers = new Map<string, Map<string, MemPeer>>();
  private inbox = new Map<string, SignalMessage[]>();
  private snapshots = new Map<string, string>();

  async registerPeer(roomId: string, peerId: string): Promise<void> {
    let room = this.peers.get(roomId);
    if (!room) {
      room = new Map();
      this.peers.set(roomId, room);
    }
    room.set(peerId, { ts: Date.now() });
  }

  async listPeers(roomId: string): Promise<string[]> {
    const room = this.peers.get(roomId);
    if (!room) return [];
    const cutoff = Date.now() - PEER_PRESENCE_MS;
    const live: string[] = [];
    for (const [id, p] of room) {
      if (p.ts < cutoff) room.delete(id);
      else live.push(id);
    }
    return live;
  }

  async removePeer(roomId: string, peerId: string): Promise<void> {
    this.peers.get(roomId)?.delete(peerId);
  }

  async push(roomId: string, message: SignalMessage): Promise<void> {
    const key = inboxKey(roomId, message.to);
    const list = this.inbox.get(key) ?? [];
    list.push(message);
    this.inbox.set(key, list);
  }

  async drain(roomId: string, peerId: string): Promise<SignalMessage[]> {
    const key = inboxKey(roomId, peerId);
    const list = this.inbox.get(key) ?? [];
    this.inbox.set(key, []);
    return list;
  }

  async saveSnapshot(roomId: string, snapshot: string): Promise<void> {
    this.snapshots.set(roomId, snapshot);
  }

  async loadSnapshot(roomId: string): Promise<string | null> {
    return this.snapshots.get(roomId) ?? null;
  }
}

let store: SignalingStore | null = null;
let warned = false;

export function getSignalingStore(): SignalingStore {
  if (store) return store;
  const redis = getRedis();
  if (redis) {
    store = new RedisStore(redis);
  } else {
    if (!warned) {
      console.warn(
        "[signaling] No Upstash/Vercel KV credentials found; using in-memory store. " +
          "Cross-instance signaling will not work in production without KV.",
      );
      warned = true;
    }
    store = new MemoryStore();
  }
  return store;
}
