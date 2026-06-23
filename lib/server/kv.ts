import { Redis } from "@upstash/redis";

/**
 * KV (Upstash Redis) handle for the signaling mailbox. Reads the standard
 * credentials Vercel injects for its KV / Upstash integration. If none are
 * configured the signaling endpoint degrades gracefully (returns "unavailable")
 * rather than throwing, so the static page still deploys without KV.
 */

let cached: Redis | null | undefined;

export function getKv(): Redis | null {
  if (cached !== undefined) return cached;
  const url =
    process.env.KV_REST_API_URL ?? process.env.UPSTASH_REDIS_REST_URL ?? "";
  const token =
    process.env.KV_REST_API_TOKEN ?? process.env.UPSTASH_REDIS_REST_TOKEN ?? "";
  cached = url && token ? new Redis({ url, token }) : null;
  return cached;
}
