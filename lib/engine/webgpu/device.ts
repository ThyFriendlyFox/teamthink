/// <reference types="@webgpu/types" />

/**
 * Thin WebGPU helper: device acquisition, buffer creation/upload/readback, a
 * compute-pipeline cache, and a single-pass dispatch recorder. All compute in
 * the executor is f32. Storage-buffer usage per kernel is kept to <= 4 to stay
 * well under the `maxStorageBuffersPerShaderStage` floor of 8.
 */

export interface EncCtx {
  encoder: GPUCommandEncoder;
  pass: GPUComputePassEncoder;
  /** Transient buffers (uniforms, scratch) freed after the pass submits. */
  trash: GPUBuffer[];
}

export class Gpu {
  readonly device: GPUDevice;
  readonly maxBindingBytes: number;
  private pipelines = new Map<string, GPUComputePipeline>();

  private constructor(device: GPUDevice, maxBindingBytes: number) {
    this.device = device;
    this.maxBindingBytes = maxBindingBytes;
  }

  static async create(): Promise<Gpu> {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu) throw new Error("WebGPU is not available in this browser");
    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("no WebGPU adapter");

    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
        maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
      },
    });
    device.lost.then((info) => {
      console.error("WebGPU device lost:", info.message);
    });
    return new Gpu(device, lim.maxStorageBufferBindingSize);
  }

  storage(bytes: number): GPUBuffer {
    return this.device.createBuffer({
      size: align4(bytes),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
  }

  upload(data: Float32Array | Int32Array | Uint32Array): GPUBuffer {
    const buf = this.storage(data.byteLength);
    this.device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
    return buf;
  }

  /** Build a uniform buffer from a packed ArrayBuffer of i32/f32 fields. */
  uniform(words: Array<number>, kinds: Array<"i" | "u" | "f">): GPUBuffer {
    const ab = new ArrayBuffer(align16(words.length * 4));
    const dv = new DataView(ab);
    for (let i = 0; i < words.length; i++) {
      if (kinds[i] === "f") dv.setFloat32(i * 4, words[i], true);
      else if (kinds[i] === "u") dv.setUint32(i * 4, words[i] >>> 0, true);
      else dv.setInt32(i * 4, words[i] | 0, true);
    }
    const buf = this.device.createBuffer({
      size: ab.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, ab);
    return buf;
  }

  pipeline(key: string, code: string): GPUComputePipeline {
    let p = this.pipelines.get(key);
    if (!p) {
      const shaderModule = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** Record one dispatch into an open compute pass. */
  encode(
    ctx: EncCtx,
    pipeline: GPUComputePipeline,
    buffers: GPUBuffer[],
    workgroups: [number, number, number],
  ): void {
    const entries = buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    }));
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    ctx.pass.setPipeline(pipeline);
    ctx.pass.setBindGroup(0, bindGroup);
    ctx.pass.dispatchWorkgroups(
      Math.max(1, workgroups[0]),
      Math.max(1, workgroups[1]),
      Math.max(1, workgroups[2]),
    );
  }

  beginPass(): EncCtx {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    return { encoder, pass, trash: [] };
  }

  submit(ctx: EncCtx): void {
    ctx.pass.end();
    this.device.queue.submit([ctx.encoder.finish()]);
    for (const b of ctx.trash) b.destroy();
    ctx.trash.length = 0;
  }

  /** Copy bytes between two buffers (own encoder + submit). */
  copy(
    src: GPUBuffer,
    srcOffset: number,
    dst: GPUBuffer,
    dstOffset: number,
    bytes: number,
  ): void {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, align4(bytes));
    this.device.queue.submit([encoder.finish()]);
  }

  async read(src: GPUBuffer, bytes: number): Promise<Float32Array> {
    const size = align4(bytes);
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out.subarray(0, Math.floor(bytes / 4));
  }
}

let cached: Promise<Gpu> | null = null;
/** Process-wide WebGPU device (one per worker). */
export function getGpu(): Promise<Gpu> {
  if (!cached) cached = Gpu.create();
  return cached;
}

function align4(n: number): number {
  return Math.ceil(n / 4) * 4;
}
function align16(n: number): number {
  return Math.ceil(n / 16) * 16;
}
