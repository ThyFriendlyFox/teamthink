import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Ship as a fully static site: the deployment only hosts the page. There are
  // no API routes, no server runtime — signaling runs over public WebRTC relays
  // and weights are fetched directly from the Hugging Face CDN.
  output: "export",
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
