import type { Metadata } from "next";
import { bodySans, displaySerif, mono } from "./fonts";
import "./globals.css";

export const metadata: Metadata = {
  title: "TeamThink — Shared WebGPU Inference Grid",
  description:
    "Spin up a session, invite devices, and run model inference across a peer-to-peer WebGPU grid.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${bodySans.variable} ${displaySerif.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
