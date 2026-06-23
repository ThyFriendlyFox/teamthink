# TeamThink — Serverless WebGPU Inference Grid

Spin up a session, share an invite link, and run model inference across a
peer-to-peer mesh of browsers. Each device that joins becomes a WebGPU compute
node; inference requests are routed to whichever peer has capacity.

The deployment is the **static page plus one tiny, event-driven signaling
endpoint** (`/api/signal`, backed by KV). That endpoint only brokers the brief
WebRTC handshake that lets a newcomer find a first peer — it is long-poll/push,
never a busy poll — and once a peer is in the mesh, every further connection is
brokered peer-to-peer with no server involvement. Model weights are fetched
directly from the Hugging Face CDN. Nothing of ours sits in the data path.

## Architecture

- **Signaling (KV mailbox, cost-bounded):** the rendezvous is a KV-backed
  mailbox served by `/api/signal` — no public WebRTC relays. It is event-driven
  (a held-open long-poll that returns the instant a message lands), and mailboxes
  auto-expire so no durable state accumulates. Cost is bounded by the mesh
  design, not by the number of users:
  - Only **one peer per room** (the lowest-id "greeter") holds the room mailbox
    open to greet newcomers. Everyone else stops talking to the server the moment
    they have a mesh link.
  - A newcomer connects to a single **seed** (the greeter), then the mesh fills
    itself in: peers gossip membership (PEX) over their data channels and broker
    each new pair's handshake through a common neighbour. A stable mesh sends
    **zero** signaling traffic.
- **Offer-in-link (the cheapest join):** an online peer can mint a "quick-join"
  link that already carries a complete WebRTC offer (ICE gathered up front) in
  the URL **fragment** (`#i=...`, which never hits the server). Opening it
  produces an answer that's dropped in a one-time mailbox the inviter is already
  waiting on — a single KV write, no announce, no greeter. The mesh takes over
  after first contact.
- **Data plane (peer-to-peer):** a [Yjs](https://github.com/yjs/yjs) document is
  replicated across peers over WebRTC data channels. Presence/capability
  heartbeats are gossiped; a shared task map drives scheduling. Tokens stream
  directly from the runner to the requester. Late joiners sync the document from
  any connected peer (no server-persisted snapshot).
- **Weights:** range-fetched straight from the Hugging Face CDN by each browser.
- **Inference:** runs in a Web Worker behind a pluggable engine interface —
  [WebLLM](https://github.com/mlc-ai/web-llm) (chat LLMs) and
  [Transformers.js](https://github.com/huggingface/transformers.js) (vision and
  more). VRAM-pooling / model sharding implements the same interface.

```
Browser A ─┐   /api/signal (KV)    ┌─ Browser B
           ├── seed handshake only ┤    (WebGPU compute)
Browser C ─┘  then peer-brokered   └─ Browser D
     │                                       │
     └────── WebRTC data channels: Yjs CRDT + token streams ──────┘
```

## Requirements

- Node 20+, [pnpm](https://pnpm.io) 9+.
- A WebGPU-capable browser (recent Chrome/Edge) to act as a compute node.
  Browsers without WebGPU join as request-only nodes.

## Develop

```bash
pnpm install
pnpm dev
```

Open the app, click **Create session**, then open the invite link in a second
browser/tab/device to join the mesh.

## Configure

Signaling needs a KV store (Upstash Redis / Vercel KV). The endpoint reads the
standard credentials Vercel injects for its KV integration; if none are present
it degrades gracefully (returns "unavailable") and the page still deploys.

- `KV_REST_API_URL` / `KV_REST_API_TOKEN` (or `UPSTASH_REDIS_REST_URL` /
  `UPSTASH_REDIS_REST_TOKEN`) — the KV mailbox used by `/api/signal`. On Vercel,
  adding the KV / Upstash integration injects these automatically.
- `NEXT_PUBLIC_SIGNAL_ENDPOINT` — override the signaling endpoint (default
  `/api/signal`) if you host it separately.
- `NEXT_PUBLIC_TURN_URL` / `NEXT_PUBLIC_TURN_USERNAME` /
  `NEXT_PUBLIC_TURN_CREDENTIAL` — optional TURN relay for restrictive NATs.

KV usage is intentionally tiny: signaling messages are a couple of small writes
per join, and only the per-room greeter holds a long-poll open. Inference data
never touches KV or the server.

## Deploy

1. Import the repo into Vercel (framework auto-detected as Next.js) and add a KV
   / Upstash integration so the signaling credentials above are present.
2. Deploy. `pnpm build` runs `next build --webpack` (pinned for the inference
   worker + ML package bundling). The output is the static page plus the single
   `/api/signal` function — no other serverless functions, no proxied bandwidth.

## Project layout

```
app/
  page.tsx              landing: create / join a session
  s/page.tsx            session route (room id from ?r=, invite from #i=)
  api/signal/route.ts   event-driven KV signaling mailbox (the only function)
components/
  ui/                   Claude-style primitives (Button, Card, Badge, Stat)
  grid/                 session UI (peers, node panel, inference console)
lib/
  config.ts             ICE servers, signaling tuning, model registry
  server/kv.ts          KV (Upstash Redis) handle for the mailbox
  mesh/peer.ts          WebRTC full-mesh: KV seed + peer-brokered growth
  mesh/kv-signaling.ts  HTTP long-poll signaling transport
  mesh/invite.ts        offer-in-link encode/decode
  mesh/yjs-provider.ts  minimal Yjs sync over data channels
  grid/capabilities.ts  WebGPU/VRAM detection
  grid/scheduler.ts     decentralized task claiming + streaming
  engine/               InferenceEngine interface + WebLLM/Transformers engines
workers/
  inference.worker.ts   runs the active engine off the main thread
```

## Notes & limits

- Full mesh suits tens of peers; larger grids need a partial-mesh/gossip
  topology.
- Routing distributes whole requests across devices. Pooling VRAM to run a model
  larger than any single device (pipeline/tensor sharding) is experimental and
  latency-bound; it is intentionally deferred behind the engine interface.
