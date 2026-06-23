import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The deployment is the static page plus one event-driven signaling endpoint
  // (`/api/signal`) backed by KV. That endpoint only brokers the brief WebRTC
  // handshake — it is long-poll/push, never a busy poll — and after a peer is
  // in the mesh, new connections are brokered peer-to-peer (no server). Weights
  // are still fetched directly from the Hugging Face CDN.
  reactCompiler: true,
  webpack: (config) => {
    // Transformers.js / onnxruntime-web reference Node built-ins that don't
    // exist in the browser or inference worker; stub them out.
    config.resolve = config.resolve ?? {};
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
};

export default nextConfig;
