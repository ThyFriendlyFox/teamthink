/**
 * Wire protocol for pipeline-parallel inference over the CHANNEL_PIPE data
 * channel. Logical messages (token starts, activation tensors, sampled tokens,
 * control) are encoded to a single buffer and split into ordered chunks that
 * stay under the SCTP per-message size limit. The data channel is `ordered`, so
 * reassembly is a simple in-sequence accumulation keyed by message id.
 */

export type GenOptions = {
  temperature: number;
  topP: number;
  maxTokens: number;
};

export type PipeMessage =
  | {
      kind: "start";
      jobId: string;
      planId: string;
      tokenIds: number[];
      options: GenOptions;
    }
  | {
      kind: "activation";
      jobId: string;
      step: number;
      dtype: "f32" | "f16";
      dims: number[];
      data: ArrayBuffer;
    }
  | {
      kind: "token";
      jobId: string;
      step: number;
      tokenId: number;
      done: boolean;
      /** "head" = feed back to shard 0 for the next step; "sink" = stream to requester. */
      to: "head" | "sink";
    }
  | { kind: "abort"; jobId: string; reason: string }
  | { kind: "ping"; nonce: number }
  | { kind: "pong"; nonce: number }
  | { kind: "ready"; jobId: string; shardIndex: number };

const MAX_CHUNK = 16000; // bytes of payload per data-channel frame
const HEADER_BYTES = 8; // msgId(u32) + partIdx(u16) + partCount(u16)

let msgSeq = 1;

// --- logical message encode/decode ------------------------------------------

function encodeLogical(msg: PipeMessage): Uint8Array {
  let tail: ArrayBuffer | null = null;
  let header: Record<string, unknown>;

  if (msg.kind === "activation") {
    header = {
      kind: msg.kind,
      jobId: msg.jobId,
      step: msg.step,
      dtype: msg.dtype,
      dims: msg.dims,
    };
    tail = msg.data;
  } else {
    header = msg as unknown as Record<string, unknown>;
  }

  const json = new TextEncoder().encode(JSON.stringify(header));
  const tailLen = tail ? tail.byteLength : 0;
  const out = new Uint8Array(4 + json.length + tailLen);
  new DataView(out.buffer).setUint32(0, json.length);
  out.set(json, 4);
  if (tail) out.set(new Uint8Array(tail), 4 + json.length);
  return out;
}

function decodeLogical(buf: Uint8Array): PipeMessage {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const jsonLen = view.getUint32(0);
  const json = new TextDecoder().decode(buf.subarray(4, 4 + jsonLen));
  const header = JSON.parse(json) as Record<string, unknown>;
  if (header.kind === "activation") {
    const tail = buf.subarray(4 + jsonLen);
    // Copy out so the backing buffer can be reused/GC'd.
    const data = tail.slice().buffer as ArrayBuffer;
    return {
      kind: "activation",
      jobId: header.jobId as string,
      step: header.step as number,
      dtype: header.dtype as "f32" | "f16",
      dims: header.dims as number[],
      data,
    };
  }
  return header as unknown as PipeMessage;
}

// --- chunking ----------------------------------------------------------------

/** Encode a logical message into ordered data-channel frames. */
export function encodePipe(msg: PipeMessage): Uint8Array[] {
  const logical = encodeLogical(msg);
  const partCount = Math.max(1, Math.ceil(logical.length / MAX_CHUNK));
  const msgId = msgSeq++ >>> 0;
  const frames: Uint8Array[] = [];
  for (let part = 0; part < partCount; part++) {
    const start = part * MAX_CHUNK;
    const slice = logical.subarray(start, start + MAX_CHUNK);
    const frame = new Uint8Array(HEADER_BYTES + slice.length);
    const dv = new DataView(frame.buffer);
    dv.setUint32(0, msgId);
    dv.setUint16(4, part);
    dv.setUint16(6, partCount);
    frame.set(slice, HEADER_BYTES);
    frames.push(frame);
  }
  return frames;
}

interface Partial {
  parts: (Uint8Array | undefined)[];
  received: number;
  partCount: number;
}

/** Reassembles chunked frames back into logical messages. */
export class PipeReassembler {
  private pending = new Map<number, Partial>();

  push(frame: Uint8Array): PipeMessage | null {
    const dv = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
    const msgId = dv.getUint32(0);
    const partIdx = dv.getUint16(4);
    const partCount = dv.getUint16(6);
    const payload = frame.subarray(HEADER_BYTES);

    if (partCount === 1) {
      return decodeLogical(payload.slice());
    }

    let entry = this.pending.get(msgId);
    if (!entry) {
      entry = { parts: new Array(partCount), received: 0, partCount };
      this.pending.set(msgId, entry);
    }
    if (!entry.parts[partIdx]) {
      entry.parts[partIdx] = payload.slice();
      entry.received++;
    }
    if (entry.received < entry.partCount) return null;

    this.pending.delete(msgId);
    let total = 0;
    for (const p of entry.parts) total += p!.length;
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const p of entry.parts) {
      merged.set(p!, offset);
      offset += p!.length;
    }
    return decodeLogical(merged);
  }
}

// --- float16 helpers (optional activation compression) ----------------------

export function f32ToF16Bytes(src: Float32Array): ArrayBuffer {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = floatToHalf(src[i]);
  return out.buffer as ArrayBuffer;
}

export function f16BytesToF32(buf: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

function floatToHalf(val: number): number {
  const f = new Float32Array(1);
  const i = new Int32Array(f.buffer);
  f[0] = val;
  const x = i[0];
  const sign = (x >> 16) & 0x8000;
  const mantissa = x & 0x007fffff;
  const exp = ((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | (mantissa >> 13);
}

function halfToFloat(h: number): number {
  const sign = (h & 0x8000) << 16;
  let exp = (h >> 10) & 0x1f;
  let mantissa = h & 0x03ff;
  if (exp === 0) {
    if (mantissa === 0) {
      const f = new Int32Array([sign]);
      return new Float32Array(f.buffer)[0];
    }
    // subnormal
    exp = 1;
    while ((mantissa & 0x0400) === 0) {
      mantissa <<= 1;
      exp--;
    }
    mantissa &= 0x03ff;
  } else if (exp === 31) {
    const f = new Int32Array([sign | 0x7f800000 | (mantissa << 13)]);
    return new Float32Array(f.buffer)[0];
  }
  const bits = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
  const f = new Int32Array([bits]);
  return new Float32Array(f.buffer)[0];
}
