/**
 * Minimal `safetensors` reader that fetches individual tensors by HTTP Range
 * request (through the HF proxy). The safetensors layout is:
 *
 *   [ 8 bytes: header length (u64 LE) ][ header JSON ][ tensor data ]
 *
 * The header maps each tensor name to its dtype, shape, and byte offsets within
 * the data section. We read the header once, then range-fetch only the tensors
 * a given shard owns and normalize them to f32 for the WebGPU executor.
 */

import { hfFileUrl } from "@/lib/engine/hf/config";

export type SafetensorDType =
  | "F32"
  | "F16"
  | "BF16"
  | "F64"
  | "I64"
  | "I32"
  | "I16"
  | "I8"
  | "U8";

export interface TensorEntry {
  name: string;
  file: string;
  dtype: SafetensorDType;
  shape: number[];
  /** Absolute byte range within `file` (inclusive start, exclusive end). */
  byteStart: number;
  byteEnd: number;
}

export interface SafetensorsIndex {
  repo: string;
  tensors: Map<string, TensorEntry>;
}

interface RawHeaderEntry {
  dtype: SafetensorDType;
  shape: number[];
  data_offsets: [number, number];
}

interface IndexJson {
  weight_map: Record<string, string>;
}

const headerCache = new Map<string, Promise<SafetensorsIndex>>();

/** Read the safetensors header(s) for a repo and build a tensor offset table. */
export function loadSafetensorsIndex(repo: string): Promise<SafetensorsIndex> {
  let p = headerCache.get(repo);
  if (!p) {
    p = buildIndex(repo);
    headerCache.set(repo, p);
  }
  return p;
}

async function buildIndex(repo: string): Promise<SafetensorsIndex> {
  const files = await resolveFiles(repo);
  const tensors = new Map<string, TensorEntry>();
  for (const file of files) {
    const { header, dataStart } = await readHeader(repo, file);
    for (const [name, meta] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const entry = meta as unknown as RawHeaderEntry;
      tensors.set(name, {
        name,
        file,
        dtype: entry.dtype,
        shape: entry.shape,
        byteStart: dataStart + entry.data_offsets[0],
        byteEnd: dataStart + entry.data_offsets[1],
      });
    }
  }
  return { repo, tensors };
}

/** Determine which `.safetensors` file(s) hold the weights. */
async function resolveFiles(repo: string): Promise<string[]> {
  const idxRes = await fetch(hfFileUrl(repo, "model.safetensors.index.json"));
  if (idxRes.ok) {
    const idx = (await idxRes.json()) as IndexJson;
    return [...new Set(Object.values(idx.weight_map))];
  }
  // Single-file model.
  return ["model.safetensors"];
}

interface HeaderResult {
  header: Record<string, unknown>;
  dataStart: number;
}

async function readHeader(repo: string, file: string): Promise<HeaderResult> {
  const url = hfFileUrl(repo, file);
  const lenBuf = await rangeBytes(url, 0, 7);
  const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
  const headerBuf = await rangeBytes(url, 8, 8 + headerLen - 1);
  const json = new TextDecoder().decode(new Uint8Array(headerBuf));
  return {
    header: JSON.parse(json) as Record<string, unknown>,
    dataStart: 8 + headerLen,
  };
}

async function rangeBytes(
  url: string,
  start: number,
  end: number,
): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (!res.ok && res.status !== 206) {
    throw new Error(`range fetch failed (${res.status}) for ${url}`);
  }
  return res.arrayBuffer();
}

export function tensorElements(entry: TensorEntry): number {
  return entry.shape.reduce((a, b) => a * b, 1);
}

/** Range-fetch a tensor and convert its bytes to a Float32Array. */
export async function fetchTensorF32(
  repo: string,
  entry: TensorEntry,
  onBytes?: (received: number, total: number) => void,
): Promise<Float32Array> {
  const url = hfFileUrl(repo, entry.file);
  const total = entry.byteEnd - entry.byteStart;
  const res = await fetch(url, {
    headers: { Range: `bytes=${entry.byteStart}-${entry.byteEnd - 1}` },
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`tensor fetch failed (${res.status}) for ${entry.name}`);
  }
  const buf = await res.arrayBuffer();
  onBytes?.(total, total);
  return convertToF32(buf, entry.dtype, tensorElements(entry));
}

function convertToF32(
  buf: ArrayBuffer,
  dtype: SafetensorDType,
  count: number,
): Float32Array {
  switch (dtype) {
    case "F32":
      return new Float32Array(buf, 0, count);
    case "F16": {
      const u16 = new Uint16Array(buf, 0, count);
      const out = new Float32Array(count);
      for (let i = 0; i < count; i++) out[i] = halfToFloat(u16[i]);
      return out;
    }
    case "BF16": {
      const u16 = new Uint16Array(buf, 0, count);
      const out = new Float32Array(count);
      const i32 = new Int32Array(out.buffer);
      for (let i = 0; i < count; i++) i32[i] = u16[i] << 16;
      return out;
    }
    case "F64": {
      const f64 = new Float64Array(buf, 0, count);
      return Float32Array.from(f64);
    }
    default:
      throw new Error(`unsupported tensor dtype ${dtype}`);
  }
}

function halfToFloat(h: number): number {
  const sign = (h & 0x8000) << 16;
  let exp = (h >> 10) & 0x1f;
  let mantissa = h & 0x03ff;
  if (exp === 0) {
    if (mantissa === 0) {
      return new Float32Array(new Int32Array([sign]).buffer)[0];
    }
    exp = 1;
    while ((mantissa & 0x0400) === 0) {
      mantissa <<= 1;
      exp--;
    }
    mantissa &= 0x03ff;
  } else if (exp === 31) {
    return new Float32Array(
      new Int32Array([sign | 0x7f800000 | (mantissa << 13)]).buffer,
    )[0];
  }
  const bits = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
  return new Float32Array(new Int32Array([bits]).buffer)[0];
}
