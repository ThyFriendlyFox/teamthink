# TeamThink — Serverless WebGPU Inference Grid

Spin up a session, share an invite link, and run model inference across a
peer-to-peer mesh of browsers. Each device that joins becomes a WebGPU compute
node; inference requests are routed to whichever peer has capacity.

The app ships as a **fully static site** — the deployment only serves the page.
There is no backend, no database, and no API routes: signaling runs over public
WebRTC relays and model weights are fetched directly from the Hugging Face CDN.
Nothing of ours sits in the data path.

## Architecture

- **Signaling (no backend):** peers find each other through a public pub/sub
  WebRTC signaling relay (the [y-webrtc](https://github.com/yjs/y-webrtc)
  protocol) keyed by the room id, and exchange only the SDP/ICE handshake there.
  A stable mesh produces no signaling traffic. Override the relays with
  `NEXT_PUBLIC_SIGNALING_URLS` (e.g. to self-host one).
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
Browser A ─┐     public WebRTC      ┌─ Browser B
           ├──  signaling relay  ───┤    (WebGPU compute)
Browser C ─┘    SDP/ICE handshake   └─ Browser D
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

No configuration is required to run. Optional environment variables:

- `NEXT_PUBLIC_SIGNALING_URLS` — comma-separated WebRTC signaling relay URLs to
  use instead of the public defaults. The defaults are shared community
  y-webrtc relays; for a reliable deployment run your own (it's a tiny
  WebSocket process) and point this at it:

  ```bash
  PORT=4444 npx y-webrtc-signaling   # or: node node_modules/y-webrtc/bin/server.js
  # then set NEXT_PUBLIC_SIGNALING_URLS=wss://your-host:4444
  ```

  The relay only brokers the SDP/ICE handshake — no inference data flows
  through it.
- `NEXT_PUBLIC_TURN_URL` / `NEXT_PUBLIC_TURN_USERNAME` /
  `NEXT_PUBLIC_TURN_CREDENTIAL` — optional TURN relay for restrictive NATs.

## Deploy

The build emits a static site to `out/` (`output: "export"`), so it can be
hosted on any static host or CDN.

1. Import the repo into Vercel (framework auto-detected as Next.js). It serves
   the static export — no serverless functions are created.
2. Deploy. `pnpm build` runs `next build --webpack` (pinned for the inference
   worker + ML package bundling) and writes `out/`.

## Project layout

```
app/
  page.tsx              landing: create / join a session
  s/[roomId]/page.tsx   session route
  api/room/route.ts     mint room ids / presence count
  api/signal/route.ts   signaling mailbox + snapshot persistence
components/
  ui/                   Claude-style primitives (Button, Card, Badge, Stat)
  grid/                 session UI (peers, node panel, inference console)
lib/
  config.ts             ICE servers, TTLs, model registry
  mesh/peer.ts          WebRTC full-mesh over KV signaling
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
