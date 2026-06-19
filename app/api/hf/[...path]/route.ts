import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Range/CORS proxy for Hugging Face model files. Browsers range-fetch only the
 * byte slices of `safetensors` they need (per-shard weights), but the public HF
 * CDN does not reliably expose CORS for cross-origin range requests. This route
 * forwards GET requests to huggingface.co, passing the `Range` header through
 * and returning `206`/`Content-Range` so the client can do partial reads.
 *
 * Inference still runs entirely in the browser; only the weight bytes pass
 * through this origin.
 */

const HF_BASE = "https://huggingface.co";

/** Server-side HF access token (set in the deploy env, never exposed). */
function hfToken(): string | undefined {
  return (
    process.env.HF_TOKEN ||
    process.env.HUGGING_FACE_HUB_TOKEN ||
    process.env.HUGGINGFACE_TOKEN ||
    process.env.HF_API_KEY ||
    undefined
  );
}

function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,HEAD,OPTIONS",
    "Access-Control-Allow-Headers": "Range, Content-Type",
    "Access-Control-Expose-Headers":
      "Content-Range, Content-Length, Accept-Ranges, Content-Type",
  };
}

export function OPTIONS(): Response {
  return new Response(null, { status: 204, headers: corsHeaders() });
}

async function proxy(req: NextRequest, path: string[]): Promise<Response> {
  if (!path || path.length === 0) {
    return new Response("missing path", { status: 400, headers: corsHeaders() });
  }
  // Only allow the resolve/ paths that serve model files; reject anything else.
  const target = `${HF_BASE}/${path.map(encodeURIComponent).join("/")}`;

  const fwd: Record<string, string> = {};
  const range = req.headers.get("range");
  if (range) fwd.Range = range;
  // A host-provided token lifts the strict anonymous rate limits and unlocks
  // gated repos. It never reaches the browser — only this server attaches it.
  const token = hfToken();
  if (token) fwd.Authorization = `Bearer ${token}`;

  const upstream = await fetch(target, {
    method: req.method === "HEAD" ? "HEAD" : "GET",
    headers: fwd,
    redirect: "follow",
  });

  const headers = new Headers(corsHeaders());
  for (const h of [
    "content-type",
    "content-length",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
  ]) {
    const v = upstream.headers.get(h);
    if (v) headers.set(h, v);
  }
  if (!headers.has("accept-ranges")) headers.set("accept-ranges", "bytes");

  return new Response(req.method === "HEAD" ? null : upstream.body, {
    status: upstream.status,
    headers,
  });
}

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await ctx.params;
  return proxy(req, path);
}

export async function HEAD(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  const { path } = await ctx.params;
  return proxy(req, path);
}
