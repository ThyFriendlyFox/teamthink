import { NextRequest, NextResponse } from "next/server";
import { generateRoomId } from "@/lib/id";
import { getSignalingStore } from "@/lib/server/signaling-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Absolute origin of this deployment, honoring proxy headers. */
function originOf(req: NextRequest): string {
  const env = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "");
  if (env) return env;
  const proto = req.headers.get("x-forwarded-proto") ?? "https";
  const host = req.headers.get("x-forwarded-host") ?? req.headers.get("host");
  return host ? `${proto}://${host}` : req.nextUrl.origin;
}

function joinDetails(req: NextRequest, roomId: string) {
  const origin = originOf(req);
  return {
    roomId,
    joinUrl: `${origin}/s/${roomId}`,
    signalingUrl: `${origin}/api/signal`,
  };
}

/**
 * Create a session room. Headless/zero-install friendly: a single
 * `curl -X POST <site>/api/room` returns the join URL to open or share. Add
 * `?format=txt` to get just the URL (handy over ssh/pipes).
 */
export async function POST(req: NextRequest) {
  const roomId = generateRoomId();
  const details = joinDetails(req, roomId);
  if (req.nextUrl.searchParams.get("format") === "txt") {
    return new NextResponse(`${details.joinUrl}\n`, {
      headers: { "content-type": "text/plain; charset=utf-8" },
    });
  }
  return NextResponse.json(details);
}

/** Report whether a room currently has any present peers, plus join details. */
export async function GET(req: NextRequest) {
  const roomId = req.nextUrl.searchParams.get("roomId");
  if (!roomId) {
    return NextResponse.json({ error: "roomId required" }, { status: 400 });
  }
  const store = getSignalingStore();
  const peers = await store.listPeers(roomId);
  const details = joinDetails(req, roomId);
  if (req.nextUrl.searchParams.get("format") === "txt") {
    return new NextResponse(`${details.joinUrl}\n`, {
      headers: { "content-type": "text/plain; charset=utf-8" },
    });
  }
  return NextResponse.json({ ...details, peerCount: peers.length });
}
