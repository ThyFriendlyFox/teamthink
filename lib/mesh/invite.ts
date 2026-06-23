/**
 * Offer-in-link invites. An online peer can mint a link that already carries a
 * complete WebRTC offer (SDP with ICE candidates gathered up front) plus a
 * one-time inbox key. The person who opens the link produces an answer and
 * drops it in that inbox — a single KV write — and the two connect directly.
 *
 * This is the cheapest possible join: no announce, no greeter listening for an
 * arbitrary newcomer, just one message back to the inviter who is already
 * waiting on exactly that key. After first contact the mesh takes over.
 *
 * The payload rides in the URL *fragment* (`#i=...`), which browsers never send
 * to the server, so even the offer never touches our origin.
 */

export interface InvitePayload {
  v: 1;
  /** Peer id of the inviter (so the joiner can label the connection). */
  from: string;
  /** One-time mailbox key the answer is posted to. */
  key: string;
  /** The inviter's offer SDP (ICE already gathered, non-trickle). */
  sdp: string;
}

const PREFIX = "i=";

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(s: string): Uint8Array {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function encodeInvite(p: InvitePayload): string {
  const json = JSON.stringify(p);
  return PREFIX + toBase64Url(new TextEncoder().encode(json));
}

/** Parse an invite from a full URL, a `#...` hash, or a bare `i=...` blob. */
export function decodeInvite(input: string): InvitePayload | null {
  if (!input) return null;
  let frag = input;
  const hashAt = frag.indexOf("#");
  if (hashAt >= 0) frag = frag.slice(hashAt + 1);
  // Allow a leading "i=" with or without other fragment params.
  const part = frag
    .split("&")
    .map((s) => s.trim())
    .find((s) => s.startsWith(PREFIX));
  const blob = part ? part.slice(PREFIX.length) : null;
  if (!blob) return null;
  try {
    const json = new TextDecoder().decode(fromBase64Url(blob));
    const obj = JSON.parse(json) as InvitePayload;
    if (obj && obj.v === 1 && obj.from && obj.key && obj.sdp) return obj;
  } catch {
    // fall through
  }
  return null;
}
