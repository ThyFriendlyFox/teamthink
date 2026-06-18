import * as Y from "yjs";
import {
  DEFAULT_MODEL_ID,
  getModel,
  HEARTBEAT_INTERVAL_MS,
  PEER_STALE_MS,
  TASK_STALE_MS,
} from "@/lib/config";
import type { ChatMessage, GenerateOptions } from "@/lib/engine/types";
import { InferenceClient } from "@/lib/engine/worker-client";
import {
  detectCapabilities,
  modelFits,
  type DeviceCapabilities,
} from "@/lib/grid/capabilities";
import type {
  GridSnapshot,
  PeerPresence,
  PipelineRecord,
  PipelineView,
  TaskRecord,
} from "@/lib/grid/types";
import {
  buildModelDescriptor,
  buildPipelinePlan,
  decodeIds,
  encodePrompt,
  loadTokenizer,
  roleFor,
} from "@/lib/grid/pipeline";
import { CHANNEL_APP, CHANNEL_PIPE, MeshClient } from "@/lib/mesh/peer";
import {
  encodePipe,
  PipeReassembler,
  type PipeMessage,
} from "@/lib/mesh/tensor-frame";
import { isEos } from "@/lib/engine/shard/manifest";
import type { ShardRange } from "@/lib/engine/shard/model-descriptor";
import { MeshYjsProvider } from "@/lib/mesh/yjs-provider";
import { generatePeerId } from "@/lib/id";

const enc = new TextEncoder();
const dec = new TextDecoder();

type AppMessage =
  | { t: "presence"; p: PeerPresence }
  | { t: "token"; jobId: string; token: string }
  | { t: "stage"; jobId: string; stage: string; progress: number };

const MAX_CONCURRENT = 1;
/** Window to let CRDT claims converge before committing to run. */
const CLAIM_SETTLE_MS = 350;
/** If a pipeline step makes no progress within this window, abort. */
const PIPE_STEP_TIMEOUT_MS = 30000;

/** Local runtime state for a pipeline job this peer participates in. */
interface PipeJobState {
  planId: string;
  jobId: string;
  manifestSource: string;
  options: { temperature: number; topP: number; maxTokens: number };
  isFirst: boolean;
  isLast: boolean;
  nextPeerId: string | null;
  firstPeerId: string | null;
  requester: string;
  isRequester: boolean;
  /** Requester-side accumulated output token ids. */
  outIds: number[];
  /** Last-shard generated-token counter (for maxTokens stop). */
  generated: number;
  startedAt: number;
  stopped: boolean;
}

/**
 * GridNode ties the mesh, CRDT, presence gossip, and inference worker together.
 * It implements decentralized task scheduling: capable peers locally project
 * who should claim each open task (no central scheduler), claim via the CRDT,
 * run inference, and stream tokens directly to the requester.
 */
export class GridNode {
  readonly peerId: string;
  private mesh: MeshClient;
  private doc = new Y.Doc();
  private provider: MeshYjsProvider;
  private tasks: Y.Map<TaskRecord>;
  private inference = new InferenceClient();

  private caps: DeviceCapabilities | null = null;
  private presence = new Map<string, PeerPresence>();
  private streams = new Map<string, string>();
  private loadedModels = new Set<string>();
  private activeJobs = 0;
  private activeModelId: string | null = null;
  private modelLoad: { progress: number; text: string } | null = null;
  private runningTasks = new Set<string>();

  // --- pipeline-parallel state ----------------------------------------------
  private pipelines: Y.Map<PipelineRecord>;
  private reassemblers = new Map<string, PipeReassembler>();
  private rtt = new Map<string, number>();
  private pingSentAt = new Map<number, number>();
  private pipeJobs = new Map<string, PipeJobState>();
  private warmedPlan: string | null = null;
  private pipeText = new Map<string, string>();
  private pipeTps = new Map<string, number | null>();
  private pipeStepTimers = new Map<string, ReturnType<typeof setTimeout>>();

  private listeners = new Set<() => void>();
  private snapshot: GridSnapshot;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  private started = false;

  constructor(readonly roomId: string) {
    this.peerId = generatePeerId();
    this.mesh = new MeshClient(roomId, this.peerId, {
      onPeerOpen: (peerId) => this.onPeerOpen(peerId),
      onPeerClose: () => this.recompute(),
      onPeersChange: () => this.recompute(),
    });
    this.provider = new MeshYjsProvider(this.mesh, this.doc);
    this.tasks = this.doc.getMap<TaskRecord>("tasks");
    this.pipelines = this.doc.getMap<PipelineRecord>("pipelines");
    this.snapshot = this.emptySnapshot();
    this.tasks.observeDeep(() => this.onTasksChanged());
    this.pipelines.observeDeep(() => this.onPipelinesChanged());
  }

  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;

    this.caps = await detectCapabilities();

    // Load any persisted CRDT snapshot for late joiners / cold start.
    await this.loadSnapshot();

    this.mesh.on(CHANNEL_APP, (peerId, payload) =>
      this.onAppMessage(peerId, payload),
    );
    this.mesh.on(CHANNEL_PIPE, (peerId, payload) =>
      this.onPipeFrame(peerId, payload),
    );

    await this.mesh.start();

    this.updateSelfPresence();
    this.heartbeatTimer = setInterval(
      () => this.heartbeat(),
      HEARTBEAT_INTERVAL_MS,
    );
    this.watchdogTimer = setInterval(() => this.watchdog(), 5000);
    this.recompute();
  }

  stop(): void {
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    if (this.watchdogTimer) clearInterval(this.watchdogTimer);
    for (const t of this.pipeStepTimers.values()) clearTimeout(t);
    this.pipeStepTimers.clear();
    this.provider.destroy();
    this.mesh.stop();
    this.inference.terminate();
  }

  // --- public API for the UI ------------------------------------------------

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): GridSnapshot {
    return this.snapshot;
  }

  /** Submit an inference request to the grid. Returns the task/job id. */
  submit(modelId: string, messages: ChatMessage[]): string {
    const model = getModel(modelId);
    if (model?.hfRepo) {
      return this.submitPipeline(model.id, model.hfRepo, messages);
    }
    const id = `t_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const now = Date.now();
    const task: TaskRecord = {
      id,
      requester: this.peerId,
      modelId,
      messages,
      status: "open",
      createdAt: now,
      updatedAt: now,
    };
    this.tasks.set(id, task);
    this.streams.set(id, "");
    this.recompute();
    return id;
  }

  /**
   * Submit a distributed inference request against an arbitrary Hugging Face
   * repo (the model is partitioned across the pool client-side). Returns the
   * job id.
   */
  submitRepo(repo: string, messages: ChatMessage[]): string {
    return this.submitPipeline(repo, repo, messages);
  }

  /** Preload a model so this device advertises itself as a provider for it. */
  async setActiveModel(modelId: string): Promise<void> {
    if (!this.caps?.webgpu) return;
    const model = getModel(modelId);
    if (!model || !modelFits(model, this.caps.memoryEstimateMb)) return;
    this.activeModelId = modelId;
    this.modelLoad = { progress: 0, text: "starting" };
    this.recompute();
    try {
      await this.inference.load(modelId, (progress, text) => {
        this.modelLoad = { progress, text };
        this.recompute();
      });
      this.loadedModels.add(modelId);
      this.modelLoad = null;
      this.updateSelfPresence();
    } catch (err) {
      this.modelLoad = {
        progress: 0,
        text: err instanceof Error ? err.message : "load failed",
      };
    }
    this.recompute();
  }

  // --- presence / heartbeat -------------------------------------------------

  private updateSelfPresence(): void {
    if (!this.caps) return;
    const self: PeerPresence = {
      peerId: this.peerId,
      caps: this.caps,
      loadedModels: [...this.loadedModels],
      activeJobs: this.activeJobs,
      ts: Date.now(),
      self: true,
    };
    this.presence.set(this.peerId, self);
    this.recompute();
  }

  private heartbeat(): void {
    this.updateSelfPresence();
    const self = this.presence.get(this.peerId);
    if (self) this.broadcastApp({ t: "presence", p: { ...self, self: false } });
    this.prunePresence();
    this.pingPeers();
  }

  /** Measure RTT to connected peers for pipeline chain ordering. */
  private pingPeers(): void {
    for (const peerId of this.mesh.connectedPeers) {
      const nonce = Math.floor(Math.random() * 0xffffffff);
      this.pingSentAt.set(nonce, performance.now());
      this.sendPipe(peerId, { kind: "ping", nonce });
    }
  }

  private prunePresence(): void {
    const cutoff = Date.now() - PEER_STALE_MS;
    let changed = false;
    for (const [id, p] of this.presence) {
      if (!p.self && p.ts < cutoff) {
        this.presence.delete(id);
        changed = true;
      }
    }
    if (changed) this.recompute();
  }

  private onPeerOpen(peerId: string): void {
    // Sync CRDT state and announce ourselves to the new peer.
    this.provider.syncWithPeer(peerId);
    const self = this.presence.get(this.peerId);
    if (self)
      this.mesh.sendTo(
        peerId,
        CHANNEL_APP,
        encodeApp({ t: "presence", p: { ...self, self: false } }),
      );
    this.recompute();
  }

  private onAppMessage(peerId: string, payload: Uint8Array): void {
    let msg: AppMessage;
    try {
      msg = JSON.parse(dec.decode(payload)) as AppMessage;
    } catch {
      return;
    }
    if (msg.t === "presence") {
      this.presence.set(msg.p.peerId, { ...msg.p, self: false });
      this.recompute();
    } else if (msg.t === "token") {
      const prev = this.streams.get(msg.jobId) ?? "";
      this.streams.set(msg.jobId, prev + msg.token);
      this.recompute();
    } else if (msg.t === "stage") {
      this.recompute();
    }
  }

  // --- scheduling -----------------------------------------------------------

  private onTasksChanged(): void {
    this.recompute();
    void this.evaluateOpenTasks();
    this.maybePersistSnapshot();
  }

  /**
   * Local projection of who should claim each open task. Each node computes the
   * best candidate from its presence view and only claims when it wins.
   */
  private async evaluateOpenTasks(): Promise<void> {
    if (!this.caps?.webgpu) return;
    if (this.activeJobs >= MAX_CONCURRENT) return;

    for (const [id, task] of this.tasks.entries()) {
      if (task.status !== "open") continue;
      if (this.runningTasks.has(id)) continue;
      const winner = this.bestCandidate(task);
      if (winner === this.peerId) {
        await this.claimAndRun(id);
        if (this.activeJobs >= MAX_CONCURRENT) return;
      }
    }
  }

  /** Pick the best capable peer for a task, or null if none are capable. */
  private bestCandidate(task: TaskRecord): string | null {
    const model = getModel(task.modelId);
    if (!model) return null;
    let best: { id: string; score: number } | null = null;
    for (const p of this.presence.values()) {
      if (!p.caps.webgpu) continue;
      if (!modelFits(model, p.caps.memoryEstimateMb)) continue;
      if (p.activeJobs >= MAX_CONCURRENT) continue;
      const score = this.scoreCandidate(p, task);
      if (
        !best ||
        score > best.score ||
        (score === best.score && p.peerId < best.id)
      ) {
        best = { id: p.peerId, score };
      }
    }
    return best?.id ?? null;
  }

  private scoreCandidate(p: PeerPresence, task: TaskRecord): number {
    let score = 0;
    if (p.loadedModels.includes(task.modelId)) score += 1000;
    score += p.caps.memoryEstimateMb / 100;
    score -= p.activeJobs * 500;
    return score;
  }

  private async claimAndRun(id: string): Promise<void> {
    const task = this.tasks.get(id);
    if (!task || task.status !== "open") return;

    this.runningTasks.add(id);
    this.activeJobs += 1;
    this.updateSelfPresence();

    this.patchTask(id, {
      status: "claimed",
      claimedBy: this.peerId,
      claimedAt: Date.now(),
    });

    // Let concurrent claims converge, then verify we still hold the claim.
    await sleep(CLAIM_SETTLE_MS);
    const after = this.tasks.get(id);
    if (!after || after.claimedBy !== this.peerId) {
      this.runningTasks.delete(id);
      this.activeJobs = Math.max(0, this.activeJobs - 1);
      this.updateSelfPresence();
      return;
    }

    await this.runTask(after);
  }

  private async runTask(task: TaskRecord): Promise<void> {
    const id = task.id;
    try {
      this.patchTask(id, { status: "running" });

      this.activeModelId = task.modelId;
      if (!this.loadedModels.has(task.modelId)) {
        this.modelLoad = { progress: 0, text: "loading model" };
        this.recompute();
        await this.inference.load(task.modelId, (progress, text) => {
          this.modelLoad = { progress, text };
          this.sendToRequester(task, {
            t: "stage",
            jobId: id,
            stage: text,
            progress,
          });
          this.recompute();
        });
        this.loadedModels.add(task.modelId);
        this.modelLoad = null;
        this.updateSelfPresence();
      }

      const options: GenerateOptions = { maxTokens: 512, temperature: 0.7 };
      const text = await this.inference.generate(
        task.modelId,
        task.messages,
        options,
        (token) => {
          // Stream to the requester (or locally if we are the requester).
          if (task.requester === this.peerId) {
            const prev = this.streams.get(id) ?? "";
            this.streams.set(id, prev + token);
            this.recompute();
          } else {
            this.sendToRequester(task, { t: "token", jobId: id, token });
          }
        },
      );

      this.patchTask(id, { status: "done", result: text });
    } catch (err) {
      this.patchTask(id, {
        status: "error",
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      this.runningTasks.delete(id);
      this.activeJobs = Math.max(0, this.activeJobs - 1);
      this.updateSelfPresence();
      void this.evaluateOpenTasks();
    }
  }

  private sendToRequester(task: TaskRecord, msg: AppMessage): void {
    this.mesh.sendTo(task.requester, CHANNEL_APP, encodeApp(msg));
  }

  /** Revert tasks claimed by peers that have gone stale. */
  private watchdog(): void {
    const now = Date.now();
    for (const [id, task] of this.tasks.entries()) {
      if (task.status !== "claimed" && task.status !== "running") continue;
      if (task.claimedBy === this.peerId) continue;
      const claimer = task.claimedBy
        ? this.presence.get(task.claimedBy)
        : undefined;
      const claimerAlive = claimer && now - claimer.ts < PEER_STALE_MS;
      const stale = now - (task.updatedAt ?? 0) > TASK_STALE_MS;
      if (!claimerAlive && stale) {
        this.patchTask(id, {
          status: "open",
          claimedBy: undefined,
          claimedAt: undefined,
        });
      }
    }
    this.pipelineWatchdog();
  }

  private patchTask(id: string, patch: Partial<TaskRecord>): void {
    const current = this.tasks.get(id);
    if (!current) return;
    this.tasks.set(id, { ...current, ...patch, updatedAt: Date.now() });
  }

  // --- pipeline-parallel (sharded) inference --------------------------------

  /** Build a plan, publish it, and (once shards warm up) drive generation. */
  private submitPipeline(
    modelId: string,
    repo: string,
    messages: ChatMessage[],
  ): string {
    const jobId = `j_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const planId = jobId;
    void this.planAndPublish(modelId, repo, messages, jobId, planId);
    return jobId;
  }

  private async planAndPublish(
    modelId: string,
    repo: string,
    messages: ChatMessage[],
    jobId: string,
    planId: string,
  ): Promise<void> {
    try {
      const desc = await buildModelDescriptor(repo);
      const options = { temperature: 0.7, topP: 0.95, maxTokens: 256 };
      const result = buildPipelinePlan({
        modelId,
        repo,
        desc,
        requester: this.peerId,
        peers: [...this.presence.values()],
        rtt: this.rtt,
        options,
        jobId,
        planId,
      });
      if (!result.ok) {
        this.publishPipelineError(planId, modelId, repo, result.error);
        return;
      }
      // Stash the prompt so we can tokenize once shards are ready.
      this.pendingPrompts.set(jobId, { source: repo, messages });
      const record: PipelineRecord = {
        plan: result.plan,
        status: "warming",
        ready: {},
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      this.pipelines.set(planId, record);
      this.recompute();
    } catch (err) {
      this.publishPipelineError(
        planId,
        modelId,
        repo,
        err instanceof Error ? err.message : String(err),
      );
    }
  }

  private pendingPrompts = new Map<
    string,
    { source: string; messages: ChatMessage[] }
  >();

  private publishPipelineError(
    planId: string,
    modelId: string,
    repo: string,
    error: string,
  ): void {
    const existing = this.pipelines.get(planId);
    const plan = existing?.plan ?? {
      planId,
      jobId: planId,
      modelId,
      repo,
      requester: this.peerId,
      numShards: 0,
      shards: [],
      options: { temperature: 0.7, topP: 0.95, maxTokens: 256 },
    };
    this.pipelines.set(planId, {
      plan,
      status: "error",
      ready: existing?.ready ?? {},
      error,
      createdAt: existing?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
    });
    this.recompute();
  }

  /** Observe plan changes: warm our assigned shard, and (requester) start. */
  private onPipelinesChanged(): void {
    this.recompute();
    for (const [planId, record] of this.pipelines.entries()) {
      if (record.status === "error" || record.status === "done") continue;
      const role = roleFor(record.plan, this.peerId);

      // Warm our shard if assigned and not already warming/warmed.
      if (
        role.shardIndex != null &&
        this.warmedPlan !== planId &&
        !record.ready[this.peerId]
      ) {
        void this.warmShard(planId, record);
      }

      // Requester: once every shard is ready, kick off generation.
      if (
        role.isRequester &&
        record.status === "warming" &&
        this.allShardsReady(record)
      ) {
        void this.startPipeline(planId, record);
      }
    }
  }

  private allShardsReady(record: PipelineRecord): boolean {
    return record.plan.shards.every((s) => record.ready[s.peerId]);
  }

  private async warmShard(
    planId: string,
    record: PipelineRecord,
  ): Promise<void> {
    if (this.warmedPlan && this.warmedPlan !== planId) return; // one plan at a time
    this.warmedPlan = planId;
    const role = roleFor(record.plan, this.peerId);
    if (role.shardIndex == null) return;
    const assignment = record.plan.shards.find((s) => s.peerId === this.peerId);
    if (!assignment) return;
    const range: ShardRange = {
      index: assignment.shardIndex,
      layerStart: assignment.layerStart,
      layerEnd: assignment.layerEnd,
      isFirst: assignment.isFirst,
      isLast: assignment.isLast,
    };
    try {
      const desc = await buildModelDescriptor(record.plan.repo);
      await this.inference.shardLoad(desc, range, (progress, text) => {
        this.modelLoad = { progress, text };
        this.recompute();
      });
      this.modelLoad = null;
      const cur = this.pipelines.get(planId);
      if (!cur) return;
      this.pipelines.set(planId, {
        ...cur,
        ready: { ...cur.ready, [this.peerId]: true },
        updatedAt: Date.now(),
      });
      this.recompute();
    } catch (err) {
      this.warmedPlan = null;
      this.publishPipelineError(
        planId,
        record.plan.modelId,
        record.plan.repo,
        err instanceof Error ? err.message : String(err),
      );
    }
  }

  private async startPipeline(
    planId: string,
    record: PipelineRecord,
  ): Promise<void> {
    const pending = this.pendingPrompts.get(record.plan.jobId);
    if (!pending) return;
    this.pendingPrompts.delete(record.plan.jobId);

    const cur = this.pipelines.get(planId);
    if (cur) {
      this.pipelines.set(planId, {
        ...cur,
        status: "running",
        updatedAt: Date.now(),
      });
    }

    const tok = await loadTokenizer(pending.source);
    const ids = await encodePrompt(tok, pending.messages);
    const head = record.plan.shards.find((s) => s.shardIndex === 0)!.peerId;
    this.sendPipe(head, {
      kind: "start",
      jobId: record.plan.jobId,
      planId,
      tokenIds: ids,
      options: record.plan.options,
    });
  }

  // --- pipe transport / message handling ------------------------------------

  private onPipeFrame(peerId: string, payload: Uint8Array): void {
    let re = this.reassemblers.get(peerId);
    if (!re) {
      re = new PipeReassembler();
      this.reassemblers.set(peerId, re);
    }
    const msg = re.push(payload);
    if (msg) void this.onPipeMessage(peerId, msg);
  }

  private sendPipe(peerId: string, msg: PipeMessage): void {
    if (peerId === this.peerId) {
      // Loopback for a peer that hosts a shard for its own request.
      void this.onPipeMessage(this.peerId, msg);
      return;
    }
    for (const frame of encodePipe(msg)) {
      this.mesh.sendTo(peerId, CHANNEL_PIPE, frame);
    }
  }

  private ensurePipeJob(planId: string, jobId: string): PipeJobState | null {
    const existing = this.pipeJobs.get(jobId);
    if (existing) return existing;
    const record = this.pipelines.get(planId);
    if (!record) return null;
    const role = roleFor(record.plan, this.peerId);
    const state: PipeJobState = {
      planId,
      jobId,
      manifestSource: "",
      options: record.plan.options,
      isFirst: role.isFirst,
      isLast: role.isLast,
      nextPeerId: role.nextPeerId,
      firstPeerId: role.firstPeerId,
      requester: record.plan.requester,
      isRequester: role.isRequester,
      outIds: [],
      generated: 0,
      startedAt: performance.now(),
      stopped: false,
    };
    this.pipeJobs.set(jobId, state);
    return state;
  }

  private async onPipeMessage(from: string, msg: PipeMessage): Promise<void> {
    switch (msg.kind) {
      case "ping":
        this.sendPipe(from, { kind: "pong", nonce: msg.nonce });
        return;
      case "pong": {
        const sent = this.pingSentAt.get(msg.nonce);
        if (sent != null) {
          this.rtt.set(from, performance.now() - sent);
          this.pingSentAt.delete(msg.nonce);
        }
        return;
      }
      case "abort": {
        const job = this.pipeJobs.get(msg.jobId);
        if (job) job.stopped = true;
        this.clearStepTimer(msg.jobId);
        return;
      }
      case "start": {
        const job = this.ensurePipeJob(msg.planId, msg.jobId);
        if (!job) return;
        await this.runShardStep(job, { kind: "ids", ids: msg.tokenIds }, 0);
        return;
      }
      case "activation": {
        const job = this.pipeJobs.get(msg.jobId);
        if (!job || job.stopped) return;
        await this.runShardStep(
          job,
          { kind: "hidden", dims: msg.dims, data: msg.data },
          msg.step,
        );
        return;
      }
      case "token": {
        if (msg.to === "head") {
          // I am shard 0: embed this token and run the next step.
          const job = this.pipeJobs.get(msg.jobId);
          if (!job || job.stopped) return;
          await this.runShardStep(
            job,
            { kind: "ids", ids: [msg.tokenId] },
            msg.step,
          );
        } else {
          // I am the requester: stream the sampled token.
          await this.onTokenSink(msg.jobId, msg.tokenId, msg.done);
        }
        return;
      }
      default:
        return;
    }
  }

  private async runShardStep(
    job: PipeJobState,
    input: { kind: "ids"; ids: number[] } | { kind: "hidden"; dims: number[]; data: ArrayBuffer },
    step: number,
  ): Promise<void> {
    if (job.stopped) return;
    this.armStepTimer(job.jobId);
    try {
      const result = await this.inference.shardRun(input, job.isLast, {
        temperature: job.options.temperature,
        topP: job.options.topP,
      });
      if (result.kind === "hidden") {
        if (job.nextPeerId) {
          this.sendPipe(job.nextPeerId, {
            kind: "activation",
            jobId: job.jobId,
            step,
            dtype: "f32",
            dims: result.dims,
            data: result.data,
          });
        }
      } else {
        // Last shard: a token was sampled.
        job.generated += 1;
        const desc = await buildModelDescriptor(
          this.pipelines.get(job.planId)!.plan.repo,
        );
        const done =
          job.generated >= job.options.maxTokens ||
          isEos(result.tokenId, desc.eosTokenId);
        // Stream to requester.
        this.sendPipe(job.requester, {
          kind: "token",
          jobId: job.jobId,
          step,
          tokenId: result.tokenId,
          done,
          to: "sink",
        });
        // Feed back to the head for the next step, unless finished.
        if (!done && job.firstPeerId) {
          this.sendPipe(job.firstPeerId, {
            kind: "token",
            jobId: job.jobId,
            step: step + 1,
            tokenId: result.tokenId,
            done: false,
            to: "head",
          });
        } else {
          this.clearStepTimer(job.jobId);
        }
      }
    } catch (err) {
      this.abortPipeline(
        job.planId,
        err instanceof Error ? err.message : String(err),
      );
    }
  }

  private async onTokenSink(
    jobId: string,
    tokenId: number,
    done: boolean,
  ): Promise<void> {
    const record = [...this.pipelines.values()].find(
      (r) => r.plan.jobId === jobId,
    );
    if (!record) return;
    const job =
      this.pipeJobs.get(jobId) ?? this.ensurePipeJob(record.plan.planId, jobId);
    if (!job) return;

    job.outIds.push(tokenId);
    job.generated += 1;
    const elapsed = (performance.now() - job.startedAt) / 1000;
    this.pipeTps.set(jobId, elapsed > 0 ? job.outIds.length / elapsed : null);

    try {
      const tok = await loadTokenizer(record.plan.repo);
      this.pipeText.set(jobId, decodeIds(tok, job.outIds));
    } catch {
      // best-effort detokenization
    }

    if (done) {
      job.stopped = true;
      this.clearStepTimer(jobId);
      const cur = this.pipelines.get(record.plan.planId);
      if (cur) {
        this.pipelines.set(record.plan.planId, {
          ...cur,
          status: "done",
          result: this.pipeText.get(jobId) ?? "",
          updatedAt: Date.now(),
        });
      }
      if (this.warmedPlan === record.plan.planId) this.warmedPlan = null;
    }
    this.recompute();
  }

  // --- pipeline fault handling ----------------------------------------------

  private armStepTimer(jobId: string): void {
    this.clearStepTimer(jobId);
    this.pipeStepTimers.set(
      jobId,
      setTimeout(() => {
        const job = this.pipeJobs.get(jobId);
        if (job && !job.stopped) {
          this.abortPipeline(job.planId, "pipeline step timed out");
        }
      }, PIPE_STEP_TIMEOUT_MS),
    );
  }

  private clearStepTimer(jobId: string): void {
    const t = this.pipeStepTimers.get(jobId);
    if (t) {
      clearTimeout(t);
      this.pipeStepTimers.delete(jobId);
    }
  }

  private abortPipeline(planId: string, reason: string): void {
    const record = this.pipelines.get(planId);
    if (!record) return;
    const job = this.pipeJobs.get(record.plan.jobId);
    if (job) job.stopped = true;
    this.clearStepTimer(record.plan.jobId);
    // Tell the other shard peers to stop.
    for (const s of record.plan.shards) {
      if (s.peerId !== this.peerId) {
        this.sendPipe(s.peerId, {
          kind: "abort",
          jobId: record.plan.jobId,
          reason,
        });
      }
    }
    if (record.status !== "done") {
      this.pipelines.set(planId, {
        ...record,
        status: "error",
        error: reason,
        updatedAt: Date.now(),
      });
    }
    if (this.warmedPlan === planId) this.warmedPlan = null;
    this.recompute();
  }

  /** Detect shards on peers that have gone stale and abort their jobs. */
  private pipelineWatchdog(): void {
    const now = Date.now();
    for (const [planId, record] of this.pipelines.entries()) {
      if (record.status !== "running" && record.status !== "warming") continue;
      for (const s of record.plan.shards) {
        if (s.peerId === this.peerId) continue;
        const p = this.presence.get(s.peerId);
        const alive = p && now - p.ts < PEER_STALE_MS;
        if (!alive) {
          this.abortPipeline(planId, `shard peer ${s.peerId} dropped`);
          break;
        }
      }
    }
  }

  private pipelineViews(): PipelineView[] {
    const views: PipelineView[] = [];
    for (const record of this.pipelines.values()) {
      const jobId = record.plan.jobId;
      const readyCount = record.plan.shards.filter(
        (s) => record.ready[s.peerId],
      ).length;
      views.push({
        planId: record.plan.planId,
        modelId: record.plan.modelId,
        status: record.status,
        numShards: record.plan.numShards,
        readyCount,
        shards: record.plan.shards.map((s) => ({
          peerId: s.peerId,
          layerStart: s.layerStart,
          layerEnd: s.layerEnd,
        })),
        text: this.pipeText.get(jobId) ?? record.result ?? "",
        tokensPerSec: this.pipeTps.get(jobId) ?? null,
        error: record.error,
      });
    }
    return views.sort((a, b) => a.planId.localeCompare(b.planId));
  }

  // --- snapshot persistence (cold start for late joiners) -------------------

  private snapshotDebounce: ReturnType<typeof setTimeout> | null = null;
  private maybePersistSnapshot(): void {
    if (this.snapshotDebounce) clearTimeout(this.snapshotDebounce);
    this.snapshotDebounce = setTimeout(() => void this.persistSnapshot(), 2000);
  }

  private async persistSnapshot(): Promise<void> {
    try {
      const update = Y.encodeStateAsUpdate(this.doc);
      const b64 = bytesToBase64(update);
      await fetch("/api/signal", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          action: "snapshot:save",
          roomId: this.roomId,
          snapshot: b64,
        }),
      });
    } catch {
      // best-effort
    }
  }

  private async loadSnapshot(): Promise<void> {
    try {
      const res = await fetch("/api/signal", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          action: "snapshot:load",
          roomId: this.roomId,
        }),
      });
      const { snapshot } = (await res.json()) as { snapshot: string | null };
      if (snapshot) Y.applyUpdate(this.doc, base64ToBytes(snapshot));
    } catch {
      // best-effort
    }
  }

  // --- snapshot / notification ---------------------------------------------

  private broadcastApp(msg: AppMessage): void {
    this.mesh.broadcast(CHANNEL_APP, encodeApp(msg));
  }

  private recompute(): void {
    const tasks = [...this.tasks.values()].sort(
      (a, b) => b.createdAt - a.createdAt,
    );
    const streams: Record<string, string> = {};
    for (const [id, text] of this.streams) streams[id] = text;
    this.snapshot = {
      selfId: this.peerId,
      caps: this.caps,
      peers: [...this.presence.values()].sort((a, b) =>
        a.peerId === this.peerId ? -1 : b.peerId === this.peerId ? 1 : 0,
      ),
      tasks,
      streams,
      connected: this.mesh.connectedPeers.length > 0,
      activeModelId: this.activeModelId,
      modelLoad: this.modelLoad,
      pipelines: this.pipelineViews(),
    };
    for (const l of this.listeners) l();
  }

  private emptySnapshot(): GridSnapshot {
    return {
      selfId: this.peerId,
      caps: null,
      peers: [],
      tasks: [],
      streams: {},
      connected: false,
      activeModelId: null,
      modelLoad: null,
      pipelines: [],
    };
  }
}

function encodeApp(msg: AppMessage): Uint8Array {
  return enc.encode(JSON.stringify(msg));
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export const FALLBACK_MODEL_ID = DEFAULT_MODEL_ID;
