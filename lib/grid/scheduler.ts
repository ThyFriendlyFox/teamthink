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
  ProvisionedView,
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
  options: { temperature: number; topP: number; maxTokens: number };
  isFirst: boolean;
  isLast: boolean;
  nextPeerId: string | null;
  firstPeerId: string | null;
  requester: string;
  isRequester: boolean;
  /** Last-shard generated-token counter (for maxTokens stop). */
  generated: number;
  stopped: boolean;
}

/** Requester-side per-prompt display + accumulation state (the chat bubbles). */
interface JobView {
  jobId: string;
  planId: string;
  modelId: string;
  prompt: string;
  status: "queued" | "running" | "done" | "error";
  text: string;
  outIds: number[];
  tokensPerSec: number | null;
  error?: string;
  startedAt: number | null;
  createdAt: number;
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
  private pipeStepTimers = new Map<string, ReturnType<typeof setTimeout>>();

  // provisioned (selected) model + per-prompt jobs (requester side)
  private provisionedPlanId: string | null = null;
  private provisionedRepo: string | null = null;
  private jobs = new Map<string, JobView>();
  private jobMessages = new Map<string, ChatMessage[]>();
  private jobQueue: string[] = [];
  private currentJobId: string | null = null;

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

  async start(invite?: string | null): Promise<void> {
    if (this.started) return;
    this.started = true;

    this.caps = await detectCapabilities();

    this.mesh.on(CHANNEL_APP, (peerId, payload) =>
      this.onAppMessage(peerId, payload),
    );
    this.mesh.on(CHANNEL_PIPE, (peerId, payload) =>
      this.onPipeFrame(peerId, payload),
    );

    await this.mesh.start(invite);

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

  /** Mint an offer-in-link invite blob (embed after `#` in a session link). */
  createInvite(): Promise<string> {
    return this.mesh.createInvite();
  }

  /** Submit an inference request to the grid. Returns the task/job id. */
  submit(modelId: string, messages: ChatMessage[]): string {
    const model = getModel(modelId);
    if (model?.hfRepo) {
      void this.provision(model.id, model.hfRepo);
      return this.runPrompt(messages) ?? "";
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
   * Select a distributed model and warm it on the grid. Loading begins now (the
   * model's layers are partitioned across the pool and each peer range-fetches
   * its slice), so prompts run against an already-warm model. `modelId` is the
   * registry id (or the repo itself for custom repos).
   */
  async provision(modelId: string, repo: string): Promise<void> {
    if (this.provisionedRepo === repo && this.provisionedPlanId) return;
    const prev = this.provisionedPlanId;
    this.provisionedRepo = repo;
    const planId = `pl_${slug(modelId)}_${slug(repo)}`;
    this.provisionedPlanId = planId;
    // Cancel anything queued against the previous model.
    for (const jobId of this.jobQueue) {
      const v = this.jobs.get(jobId);
      if (v) {
        v.status = "error";
        v.error = "model changed";
      }
    }
    this.jobQueue = [];
    this.jobMessages.clear();
    this.recompute();
    if (prev && prev !== planId) {
      this.pipelines.delete(prev);
      if (this.warmedPlan === prev) this.warmedPlan = null;
    }
    await this.planAndPublish(modelId, repo, planId);
  }

  /** Convenience for an arbitrary HF repo (custom repo input). */
  provisionRepo(repo: string): Promise<void> {
    return this.provision(repo, repo);
  }

  /**
   * Run a prompt against the currently provisioned (warmed) model. Queues if the
   * model is still warming or another prompt is in flight. Returns the job id.
   */
  runPrompt(messages: ChatMessage[]): string | null {
    const planId = this.provisionedPlanId;
    if (!planId) return null;
    const jobId = `j_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    const modelId = this.pipelines.get(planId)?.plan.modelId ?? planId;
    this.jobs.set(jobId, {
      jobId,
      planId,
      modelId,
      prompt: messages.at(-1)?.content ?? "",
      status: "queued",
      text: "",
      outIds: [],
      tokensPerSec: null,
      startedAt: null,
      createdAt: Date.now(),
    });
    this.jobMessages.set(jobId, messages);
    this.jobQueue.push(jobId);
    this.recompute();
    void this.maybeRunNext();
    return jobId;
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

  /** Build and publish a warm plan for the selected model (no prompt yet). */
  private async planAndPublish(
    modelId: string,
    repo: string,
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
        jobId: planId,
        planId,
      });
      if (!result.ok) {
        this.publishPipelineError(planId, modelId, repo, result.error);
        return;
      }
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

  /** Observe plan changes: warm our assigned shard; mark ready when warm. */
  private onPipelinesChanged(): void {
    this.recompute();
    for (const [planId, record] of this.pipelines.entries()) {
      if (record.status === "error") continue;
      const role = roleFor(record.plan, this.peerId);

      // Warm our shard if assigned and not already warming/warmed.
      if (
        role.shardIndex != null &&
        this.warmedPlan !== planId &&
        !record.ready[this.peerId]
      ) {
        void this.warmShard(planId, record);
      }

      // Requester: once every shard is warm, mark ready and run queued prompts.
      if (
        role.isRequester &&
        record.status === "warming" &&
        this.allShardsReady(record)
      ) {
        const cur = this.pipelines.get(planId);
        if (cur) {
          this.pipelines.set(planId, {
            ...cur,
            status: "ready",
            updatedAt: Date.now(),
          });
        }
        void this.maybeRunNext();
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

  /** Dequeue and run the next prompt when the model is warm and idle. */
  private async maybeRunNext(): Promise<void> {
    if (this.currentJobId) return;
    const planId = this.provisionedPlanId;
    if (!planId) return;
    const record = this.pipelines.get(planId);
    if (!record || record.status !== "ready") return;
    const jobId = this.jobQueue.shift();
    if (!jobId) return;
    const messages = this.jobMessages.get(jobId);
    this.jobMessages.delete(jobId);
    if (!messages) return;
    this.currentJobId = jobId;
    await this.startJob(planId, jobId, messages);
  }

  private async startJob(
    planId: string,
    jobId: string,
    messages: ChatMessage[],
  ): Promise<void> {
    const record = this.pipelines.get(planId);
    const view = this.jobs.get(jobId);
    if (!record) {
      if (view) {
        view.status = "error";
        view.error = "model not provisioned";
      }
      this.currentJobId = null;
      this.recompute();
      return;
    }
    if (view) view.status = "running";
    this.recompute();
    try {
      const tok = await loadTokenizer(record.plan.repo);
      const ids = await encodePrompt(tok, messages);
      const head = record.plan.shards.find((s) => s.shardIndex === 0)!.peerId;
      this.sendPipe(head, {
        kind: "start",
        jobId,
        planId,
        tokenIds: ids,
        options: record.plan.options,
      });
    } catch (err) {
      if (view) {
        view.status = "error";
        view.error = err instanceof Error ? err.message : String(err);
      }
      this.currentJobId = null;
      this.recompute();
      void this.maybeRunNext();
    }
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
      options: record.plan.options,
      isFirst: role.isFirst,
      isLast: role.isLast,
      nextPeerId: role.nextPeerId,
      firstPeerId: role.firstPeerId,
      requester: record.plan.requester,
      isRequester: role.isRequester,
      generated: 0,
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
        // A mid/tail shard may not have a job yet; derive its plan from the
        // single plan it warmed.
        const job =
          this.pipeJobs.get(msg.jobId) ??
          (this.warmedPlan
            ? this.ensurePipeJob(this.warmedPlan, msg.jobId)
            : null);
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
    // Each prompt reuses the warm shard runner, so clear the KV cache at the
    // start of every job (step 0) before processing the prefill.
    if (step === 0) {
      try {
        await this.inference.shardReset();
      } catch {
        // best-effort; a fresh runner starts empty anyway
      }
      job.generated = 0;
    }
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
        job.jobId,
      );
    }
  }

  private async onTokenSink(
    jobId: string,
    tokenId: number,
    done: boolean,
  ): Promise<void> {
    const view = this.jobs.get(jobId);
    if (!view) return;
    if (view.startedAt == null) view.startedAt = performance.now();

    view.outIds.push(tokenId);
    const elapsed = (performance.now() - view.startedAt) / 1000;
    view.tokensPerSec = elapsed > 0 ? view.outIds.length / elapsed : null;

    const record = this.pipelines.get(view.planId);
    if (record) {
      try {
        const tok = await loadTokenizer(record.plan.repo);
        view.text = decodeIds(tok, view.outIds);
      } catch {
        // best-effort detokenization
      }
    }

    if (done) {
      view.status = "done";
      this.clearStepTimer(jobId);
      this.pipeJobs.delete(jobId);
      if (this.currentJobId === jobId) this.currentJobId = null;
      void this.maybeRunNext();
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
          this.abortPipeline(job.planId, "pipeline step timed out", jobId);
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

  private abortPipeline(planId: string, reason: string, jobId?: string): void {
    const record = this.pipelines.get(planId);
    const failedJob = jobId ?? this.currentJobId;
    if (failedJob) {
      const job = this.pipeJobs.get(failedJob);
      if (job) job.stopped = true;
      this.pipeJobs.delete(failedJob);
      this.clearStepTimer(failedJob);
      const view = this.jobs.get(failedJob);
      if (view && view.status !== "done") {
        view.status = "error";
        view.error = reason;
      }
      if (this.currentJobId === failedJob) this.currentJobId = null;
    }
    if (record) {
      // Tell the other shard peers to stop this job.
      for (const s of record.plan.shards) {
        if (s.peerId !== this.peerId) {
          this.sendPipe(s.peerId, {
            kind: "abort",
            jobId: failedJob ?? record.plan.jobId,
            reason,
          });
        }
      }
      if (record.status !== "error") {
        this.pipelines.set(planId, {
          ...record,
          status: "error",
          error: reason,
          updatedAt: Date.now(),
        });
      }
      if (this.warmedPlan === planId) this.warmedPlan = null;
    }
    this.recompute();
    void this.maybeRunNext();
  }

  /** Detect shards on peers that have gone stale and abort their jobs. */
  private pipelineWatchdog(): void {
    const now = Date.now();
    for (const [planId, record] of this.pipelines.entries()) {
      if (record.status === "error") continue;
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

  /** The model warmed on the grid (selected in the console), if any. */
  private provisionedView(): ProvisionedView | null {
    const planId = this.provisionedPlanId;
    if (!planId) return null;
    const record = this.pipelines.get(planId);
    if (!record) {
      return {
        modelId: this.provisionedRepo ?? planId,
        repo: this.provisionedRepo ?? "",
        status: "planning",
        numShards: 0,
        readyCount: 0,
        shards: [],
        progress: this.modelLoad,
      };
    }
    const readyCount = record.plan.shards.filter(
      (s) => record.ready[s.peerId],
    ).length;
    return {
      modelId: record.plan.modelId,
      repo: record.plan.repo,
      status: record.status,
      numShards: record.plan.numShards,
      readyCount,
      shards: record.plan.shards.map((s) => ({
        peerId: s.peerId,
        layerStart: s.layerStart,
        layerEnd: s.layerEnd,
      })),
      error: record.error,
      progress: this.modelLoad,
    };
  }

  /** One view per submitted prompt (chat history). */
  private pipelineViews(): PipelineView[] {
    const planId = this.provisionedPlanId;
    const record = planId ? this.pipelines.get(planId) : null;
    const numShards = record?.plan.numShards ?? 0;
    const readyCount = record
      ? record.plan.shards.filter((s) => record.ready[s.peerId]).length
      : 0;
    const shards =
      record?.plan.shards.map((s) => ({
        peerId: s.peerId,
        layerStart: s.layerStart,
        layerEnd: s.layerEnd,
      })) ?? [];
    return [...this.jobs.values()]
      .sort((a, b) => a.createdAt - b.createdAt)
      .map((j) => ({
        planId: j.jobId,
        modelId: j.modelId,
        status: j.status as PipelineView["status"],
        numShards,
        readyCount,
        shards,
        text: j.text,
        tokensPerSec: j.tokensPerSec,
        error: j.error,
      }));
  }

  // --- notification ---------------------------------------------------------

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
      provisioned: this.provisionedView(),
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
      provisioned: null,
      pipelines: [],
    };
  }
}

/** Filesystem-safe short id from an arbitrary string (for stable plan ids). */
function slug(s: string): string {
  return s.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
}

function encodeApp(msg: AppMessage): Uint8Array {
  return enc.encode(JSON.stringify(msg));
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export const FALLBACK_MODEL_ID = DEFAULT_MODEL_ID;
