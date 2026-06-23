import type { NextRequest } from "next/server";
import {
  SIGNAL_POLL_HOLD_MS,
  SIGNAL_POLL_STEP_MS,
  SIGNAL_TTL_SECONDS,
} from "@/lib/config";
import { getKv } from "@/lib/server/kv";

/**
 * Event-driven signaling mailbox — the *only* server endpoint in the app.
 *
 * It brokers nothing but the brief WebRTC handshake: peers drop tiny SDP
 * messages into per-recipient "boxes" (`POST`) and drain their own box with a
 * held-open long-poll (`GET`) that returns the instant a message arrives, or
 * empty after `SIGNAL_POLL_HOLD_MS`. There is no busy polling, and boxes
 * auto-expire, so KV holds no durable state. Once peers are in the mesh they
 * stop calling this endpoint entirely — further connections are brokered
 * peer-to-peer (see `lib/mesh/peer.ts`).
 *
 * A "box" is either a peer id (its private inbox), an invite key (for the
 * offer-in-link fast path), or the literal `room` mailbox that the elected
 * greeter watches for newcomers.
 */

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ROOM_RE = /^[a-zA-Z0-9_-]{1,64}$/;
const BOX_RE = /^[a-zA-Z0-9_-]{1,128}$/;
const MAX_DRAIN = 64;

function key(room: string, box: string): string {
  return `tt:sig:${room}:${box}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

export async function POST(req: NextRequest): Promise<Response> {
  const kv = getKv();
  if (!kv) return json({ error: "signaling-unconfigured" }, 503);

  let body: { room?: string; box?: string; msg?: unknown };
  try {
    body = (await req.json()) as typeof body;
  } catch {
    return json({ error: "bad-json" }, 400);
  }
  const { room, box, msg } = body;
  if (!room || !box || msg == null || !ROOM_RE.test(room) || !BOX_RE.test(box)) {
    return json({ error: "bad-request" }, 400);
  }

  const k = key(room, box);
  // Store the message as a JSON string so KV round-trips it verbatim.
  await kv.rpush(k, JSON.stringify(msg));
  await kv.expire(k, SIGNAL_TTL_SECONDS);
  return json({ ok: true });
}

export async function GET(req: NextRequest): Promise<Response> {
  const kv = getKv();
  if (!kv) return json({ msgs: [] });

  const { searchParams } = new URL(req.url);
  const room = searchParams.get("room") ?? "";
  const box = searchParams.get("box") ?? "";
  if (!ROOM_RE.test(room) || !BOX_RE.test(box)) {
    return json({ error: "bad-request" }, 400);
  }

  const k = key(room, box);
  const deadline = Date.now() + SIGNAL_POLL_HOLD_MS;
  // Hold the request open, checking the mailbox periodically, until something
  // is waiting or we hit the budget. The client immediately re-requests, so
  // this behaves like a push without a persistent socket.
  for (;;) {
    const raw = (await kv.lpop(k, MAX_DRAIN)) as unknown;
    const items = Array.isArray(raw) ? raw : raw != null ? [raw] : [];
    if (items.length > 0) {
      return json({ msgs: items.map(parseItem).filter((m) => m != null) });
    }
    if (Date.now() >= deadline || req.signal.aborted) break;
    await sleep(SIGNAL_POLL_STEP_MS);
  }
  return json({ msgs: [] });
}

function parseItem(item: unknown): unknown {
  if (typeof item !== "string") return item; // KV may auto-deserialize
  try {
    return JSON.parse(item);
  } catch {
    return null;
  }
}
