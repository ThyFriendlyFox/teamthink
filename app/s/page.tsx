"use client";

import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { SessionView } from "@/components/grid/SessionView";

/**
 * Session route. The room id rides in the query string (`/s?r=<id>`) rather
 * than a dynamic path segment so the whole app ships as a static export — the
 * deployment serves one `/s` page and the room is read client-side from the
 * link. No server route is involved in joining a room.
 */
function Session() {
  const params = useSearchParams();
  const roomId = params.get("r")?.trim() ?? "";

  if (!roomId) {
    return (
      <main className="flex flex-1 items-center justify-center px-6 py-24">
        <div className="text-center">
          <p className="text-ink-muted">No session in this link.</p>
          <Link
            href="/"
            className="mt-3 inline-block font-display text-lg text-accent-strong"
          >
            Start or join a session
          </Link>
        </div>
      </main>
    );
  }

  return <SessionView roomId={roomId} />;
}

export default function SessionPage() {
  return (
    <Suspense fallback={null}>
      <Session />
    </Suspense>
  );
}
