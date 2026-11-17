import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { readFile, realpath } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { contained } from "./catalog.ts";
import { acceptResponse, assertRequest, bytesDigest, contextDigest, digest, errorCode, fail, isRetrieval, parseStrict, readBundle, readRetrievalBundle,
  RouterError, type InferenceBundle, type Catalog, type Digest, type Reason, type Request } from "./protocol.ts";
import type { Scorer, Scorers } from "./router.ts";

export interface WorkerPaths { taskRoot: string; pythonPath: string; workerPath: string; bundlePath: string }
export type Admission = () => Promise<boolean>;
export async function memoryAdmission(minAvailableMiB = 1024): Promise<boolean> {
  const info = await readFile("/proc/meminfo", "utf8");
  const available = /^MemAvailable:\s+(\d+) kB$/m.exec(info)?.[1];
  return available !== undefined && Number(available) >= minAvailableMiB * 1024;
}
async function privatePaths(paths: WorkerPaths): Promise<WorkerPaths> {
  if (Object.values(paths).some(path => !isAbsolute(path))) fail("backend_unavailable");
  const taskRoot = await realpath(paths.taskRoot);
  // Preserve the configured venv entry point. Resolving its python symlink before
  // exec would bypass pyvenv.cfg and lose the task-private site-packages.
  const pythonPath = resolve(paths.pythonPath);
  const pythonDirectory = await realpath(dirname(pythonPath));
  await realpath(pythonPath);
  const venvConfig = await realpath(resolve(pythonDirectory, "..", "pyvenv.cfg"));
  if (!contained(taskRoot, pythonDirectory) || !contained(taskRoot, venvConfig)) fail("backend_unavailable");
  const workerPath = await realpath(paths.workerPath);
  const bundlePath = await realpath(paths.bundlePath);
  if ([workerPath, bundlePath].some(path => !contained(taskRoot, path))) fail("backend_unavailable");
  return { taskRoot, pythonPath, workerPath, bundlePath };
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) fail("invalid_response");
  return value as Record<string, unknown>;
}
async function provenancePayload(root: string, bundle: InferenceBundle, name: string, expected: Digest): Promise<Record<string, unknown>> {
  const entry = bundle.files.find(file => file.path === name);
  if (!entry || entry.sha256 !== expected) fail("invalid_response");
  const path = await realpath(resolve(root, name));
  if (!contained(root, path)) fail("invalid_response");
  const bytes = await readFile(path);
  if (bytes.length !== entry.bytes || bytesDigest(bytes) !== expected) fail("invalid_response");
  return object(parseStrict(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
}
async function verifyProvenance(root: string, bundle: InferenceBundle): Promise<void> {
  const recipe = await provenancePayload(root, bundle, "recipe.json", bundle.provenance.recipe_sha256);
  if (isRetrieval(bundle)) {
    if (recipe.role !== "retrieval" || object(recipe.export).variant !== "fp32") fail("invalid_response");
    const fields = ["weights", "fit_split", "training_data_used", "contextual_binding", "dataset_sha256",
      "split_sha256", "training_recipe_sha256"] as const;
    for (const field of fields) if (digest(recipe[field]) !== digest(bundle.provenance[field])) fail("invalid_response");
    if (bundle.provenance.weights === "fine_tuned") {
      const training = await provenancePayload(root, bundle, "training.json", bundle.provenance.training_report_sha256);
      if (typeof training.chosen_epoch !== "number" || ![1, 2, 3].includes(training.chosen_epoch)) fail("invalid_response");
      for (const field of fields) if (digest(training[field]) !== digest(bundle.provenance[field])) fail("invalid_response");
    } else if (bundle.files.some(file => file.path === "training.json")) fail("invalid_response");
    for (const name of ["validation.json", "evaluation.json", "parity.json"]) {
      const entry = bundle.files.find(file => file.path === name);
      if (!entry) fail("invalid_response");
      const report = await provenancePayload(root, bundle, name, entry.sha256);
      if (report.role !== "retrieval") fail("invalid_response");
      if (name !== "parity.json") {
        for (const field of fields) if (digest(report[field]) !== digest(bundle.provenance[field])) fail("invalid_response");
      }
    }
    return;
  }
  const report = await provenancePayload(root, bundle, "validation.json", bundle.selection.validation_sha256);
  const provenance = object(recipe.provenance);
  for (const key of ["dataset_sha256", "split_sha256", "contextual_binding"] as const) {
    if (digest(provenance[key]) !== digest(bundle.provenance[key]) ||
        digest(report[key]) !== digest(bundle.provenance[key])) fail("invalid_response");
  }
  if (recipe.runtime_revision !== bundle.provenance.runtime_revision ||
      recipe.role !== "deployment_candidate" || report.role !== recipe.role || recipe.weights !== "fine_tuned" ||
      report.fit_split !== bundle.selection.fit_split ||
      digest(report.thresholds) !== digest([bundle.selection.lower_logit, bundle.selection.upper_logit])) fail("invalid_response");
}
interface Pending {
  resolve(value: unknown): void; reject(error: RouterError): void;
  timer: NodeJS.Timeout; signal: AbortSignal; abort(): void;
}

export interface StartupDiagnostics {
  stage: "retrieval" | "decider" | "complete"; elapsed_ms: number; limit_ms: number;
  admission: "not_checked" | "checking" | "passed" | "rejected" | "error"; worker_started: boolean;
  completed: { stage: "retrieval" | "decider"; elapsed_ms: number; reason_code: Reason; score_count: number }[];
  failure: null | { source: "memory_admission" | "admission_error" | "process_exit" | "process_error" | "stdin_error" |
    "request_write" | "invalid_response" | "missing_scores" | "deadline" | "cancelled" | "closed" | "host_validation";
    code: Reason; response_reason?: Reason; exit_code?: number | null; signal?: NodeJS.Signals | null };
}

/** One private process and one outstanding JSONL request. No sockets or cloud fallback. */
export class WorkerClient implements Scorer {
  #child: ChildProcessWithoutNullStreams | undefined;
  #pending: Pending | undefined;
  #busy = false;
  #closed = false;
  #buffer = Buffer.alloc(0);
  #epoch = 0;
  #decision: WorkerClient | undefined;
  #warming = false;
  #ready = false;
  #catalogDigest: Digest | undefined;
  #startup: StartupDiagnostics | undefined;
  get startupDiagnostics(): StartupDiagnostics | undefined { return this.#startup && structuredClone(this.#startup); }
  private constructor(readonly bundle: InferenceBundle, private readonly paths: WorkerPaths, private readonly admit: Admission,
    private readonly manifestDigest: Digest) {}
  static async create(paths: WorkerPaths, admit: Admission): Promise<WorkerClient> {
    try {
      const checked = await privatePaths(paths);
      const manifestPath = await realpath(resolve(checked.bundlePath, "manifest.json"));
      if (!contained(checked.bundlePath, manifestPath)) fail("invalid_response");
      const bytes = await readFile(manifestPath);
      if (bytes.length > 131072) fail("invalid_response");
      const value = parseStrict(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
      const manifest = object(value).role === "retrieval" ? readRetrievalBundle(value) : readBundle(value);
      if (bytesDigest(await readFile(checked.workerPath)) !== manifest.provenance.runtime_revision) fail("invalid_response");
      if (isRetrieval(manifest)) {
        for (const [name, expected] of Object.entries(manifest.provenance.runtime_files)) {
          const path = await realpath(resolve(dirname(checked.workerPath), name));
          if (!contained(checked.taskRoot, path) || bytesDigest(await readFile(path)) !== expected) fail("invalid_response");
        }
      }
      await verifyProvenance(checked.bundlePath, manifest);
      return new WorkerClient(manifest, checked, admit, bytesDigest(bytes));
    } catch (error) {
      if (error instanceof RouterError) throw error;
      return fail("backend_unavailable");
    }
  }
  /** Admit without launching; callers finish heavyweight authorization before ready(). */
  static async createScorers(paths: WorkerPaths, decisionBundlePath: string, admit: Admission,
    signal: AbortSignal = new AbortController().signal): Promise<Scorers & { shortlister: WorkerClient }> {
    if (signal.aborted) fail("stale_result");
    const decision = await WorkerClient.create({ ...paths, workerPath: resolve(dirname(paths.workerPath), "router-worker.py"),
      bundlePath: decisionBundlePath }, admit);
    const shared = await WorkerClient.create(paths, admit);
    if (!isRetrieval(shared.bundle) || isRetrieval(decision.bundle) ||
        decision.bundle.model.base !== "cross-encoder/ms-marco-MiniLM-L6-v2" ||
        decision.bundle.model.revision !== "233902d25c440f23af6f7d6e94d2946bac0bee0a" ||
        digest(shared.bundle.provenance.contextual_binding) !== digest(decision.bundle.provenance.contextual_binding)) fail("invalid_response");
    if (signal.aborted) fail("stale_result");
    shared.#decision = decision;
    return { shortlister: shared, ready: (signal, catalog) => shared.ready(signal, catalog), decider: { bundle: decision.bundle, verify: () => decision.verify(),
      score: (request, signal) => shared.score(request, signal), close: () => shared.close() } };
  }
  /** One combined startup budget, repeated on every cold launch, with no private case input. */
  async ready(signal: AbortSignal, catalog?: Catalog): Promise<void> {
    if (this.#closed) fail("backend_unavailable");
    if (signal.aborted) fail("stale_result");
    if (this.#busy || this.#warming) fail("busy");
    if (this.#ready && (!catalog || this.#catalogDigest === catalog.catalog_sha256)) return;
    const cold = !this.#ready;
    this.#warming = true;
    const started = performance.now();
    const controller = new AbortController();
    this.#startup = { stage: "retrieval", elapsed_ms: 0, limit_ms: this.bundle.limits.startup_timeout_ms,
      admission: "not_checked", worker_started: false, completed: [], failure: null };
    const timer = setTimeout(() => controller.abort(new RouterError("timeout")), this.bundle.limits.startup_timeout_ms);
    const startupSignal = AbortSignal.any([signal, controller.signal]);
    const intent = "test ".repeat(254);
    const description = "test ".repeat(224);
    const contentDigest = bytesDigest("nonprivate startup fixture");
    try {
      for (const bundle of cold ? [this.bundle, ...(this.#decision ? [this.#decision.bundle] : [])] : []) {
        const retrieval = isRetrieval(bundle);
        const stage = retrieval ? "retrieval" : "decider";
        this.#startup.stage = stage;
        const stageStarted = performance.now();
        // "test" is one native token in both pinned BERT vocabularies. The decider keeps
        // 96 intent + 224 description tokens + 3 specials, exercising two full microbatches.
        const candidates = Array.from({ length: retrieval ? 32 : 8 }, (_, index) => {
          const id = `startup-${String(index).padStart(2, "0")}`;
          return { id, name: id, description, contentDigest };
        });
        const warm: Request = { version: 1, requestId: retrieval ? "startup-retrieval" : "startup-decider",
          sessionKey: "startup", generation: 0, event: "context",
          catalogDigest: bundle.provenance.contextual_binding.catalog_sha256, policyDigest: digest({}), contextDigest: digest({}),
          selectorDigest: bundle.selector_sha256, intent, action: null, candidates, candidateCoverage: "full",
          requiredIds: [], excludedIds: [], maxSelections: retrieval ? 0 : bundle.selection.max_optional };
        warm.contextDigest = contextDigest(warm);
        const result = acceptResponse(await this.score(warm, startupSignal), warm, bundle);
        if (!result.scores) {
          this.#startup.failure = { source: "missing_scores", code: "backend_unavailable", response_reason: result.reasonCode };
          fail("backend_unavailable");
        }
        this.#startup.completed.push({ stage, elapsed_ms: performance.now() - stageStarted,
          reason_code: result.reasonCode, score_count: result.scores.values.length });
      }
      // Public bundles contain no owner's catalog. Populate the live-description
      // cache under the bounded readiness budget, before the 750 ms route timer.
      if (catalog && isRetrieval(this.bundle)) {
        this.#startup.stage = "retrieval";
        const entries = catalog.entries.filter(entry => entry.automatic_eligible);
        for (let start = 0; start < entries.length; start += this.bundle.limits.max_candidates) {
          const request: Request = { version: 1, requestId: `catalog-startup-${start}`, sessionKey: "catalog-startup",
            generation: 0, event: "context", catalogDigest: catalog.catalog_sha256, policyDigest: digest({}),
            contextDigest: digest({}), selectorDigest: this.bundle.selector_sha256, intent: "", action: null,
            candidates: entries.slice(start, start + this.bundle.limits.max_candidates).map(entry => ({
              id: entry.runtime_id, name: entry.runtime_id, description: entry.description, contentDigest: entry.content_sha256 })),
            candidateCoverage: "shortlist", requiredIds: [], excludedIds: [], maxSelections: 0 };
          request.contextDigest = contextDigest(request);
          const result = acceptResponse(await this.score(request, startupSignal), request, this.bundle);
          if (!result.scores) fail("backend_unavailable");
        }
        this.#catalogDigest = catalog.catalog_sha256;
      }
      if (signal.aborted) fail("stale_result");
      if (controller.signal.aborted || performance.now() - started >= this.bundle.limits.startup_timeout_ms) fail("timeout");
      this.#ready = true;
      this.#startup.stage = "complete";
    } catch (error) {
      const code = errorCode(error);
      this.#startup.failure ??= { source: code === "timeout" ? "deadline" : code === "stale_result" ? "cancelled" : "host_validation", code };
      this.#stop("backend_unavailable");
      throw error;
    } finally {
      this.#startup.elapsed_ms = performance.now() - started;
      clearTimeout(timer); this.#warming = false;
    }
  }
  async verify(): Promise<void> {
    try {
      const path = await realpath(resolve(this.paths.bundlePath, "manifest.json"));
      if (!contained(this.paths.bundlePath, path) || bytesDigest(await readFile(path)) !== this.manifestDigest) fail("stale_result");
    } catch {
      this.#stop("stale_result");
      fail("stale_result");
    }
  }
  #stop(code: "timeout" | "invalid_response" | "stale_result" | "backend_unavailable",
    failure?: NonNullable<StartupDiagnostics["failure"]>): void {
    if (this.#warming && this.#startup) this.#startup.failure ??= failure ?? {
      source: code === "timeout" ? "deadline" : code === "stale_result" ? "cancelled" : "invalid_response", code };
    this.#epoch++;
    this.#ready = false;
    const child = this.#child;
    this.#child = undefined;
    this.#buffer = Buffer.alloc(0);
    const pending = this.#pending;
    this.#pending = undefined;
    if (pending) {
      clearTimeout(pending.timer);
      pending.signal.removeEventListener("abort", pending.abort);
      pending.reject(new RouterError(code));
    }
    if (child && child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
      const timer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      }, 250);
      timer.unref();
      child.once("exit", () => clearTimeout(timer));
    }
  }
  #launch(): void {
    const command = ["-I", "-B", this.paths.workerPath, "--artifact", this.paths.bundlePath];
    if (this.#decision) command.push("--decision-artifact", this.#decision.paths.bundlePath);
    const child = spawn(this.paths.pythonPath, command, {
      cwd: this.paths.taskRoot, stdio: ["pipe", "pipe", "pipe"], shell: false,
      env: { PATH: "/usr/bin:/bin", HOME: this.paths.taskRoot, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1",
        TOKENIZERS_PARALLELISM: "false", OMP_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1", MKL_NUM_THREADS: "1", PYTHONUNBUFFERED: "1" },
    });
    this.#child = child;
    if (this.#warming && this.#startup) this.#startup.worker_started = child.pid !== undefined;
    // Discard diagnostic bytes; raw subprocess exceptions must never enter host logs.
    child.stderr.on("data", () => {});
    child.stdin.on("error", () => { if (this.#child === child) this.#stop("backend_unavailable", { source: "stdin_error", code: "backend_unavailable" }); });
    child.once("error", () => { if (this.#child === child) this.#stop("backend_unavailable", { source: "process_error", code: "backend_unavailable" }); });
    child.once("exit", (exit_code, signal) => { if (this.#child === child) this.#stop("backend_unavailable", {
      source: "process_exit", code: "backend_unavailable", exit_code, signal }); });
    child.stdout.on("data", (chunk: Buffer) => {
      if (this.#child !== child) return;
      if (!this.#pending || this.#buffer.length + chunk.length > this.bundle.limits.max_line_bytes) {
        this.#stop("invalid_response"); return;
      }
      this.#buffer = Buffer.concat([this.#buffer, chunk]);
      const newline = this.#buffer.indexOf(10);
      if (newline < 0) return;
      if (newline !== this.#buffer.length - 1) { this.#stop("invalid_response"); return; }
      try {
        const response = parseStrict(new TextDecoder("utf-8", { fatal: true }).decode(this.#buffer.subarray(0, newline)));
        const pending = this.#pending;
        this.#pending = undefined;
        this.#buffer = Buffer.alloc(0);
        clearTimeout(pending.timer);
        pending.signal.removeEventListener("abort", pending.abort);
        pending.resolve(response);
      } catch { this.#stop("invalid_response"); }
    });
  }
  async score(request: Request, signal: AbortSignal): Promise<unknown> {
    if (this.#closed) fail("backend_unavailable");
    if (this.#busy) fail("busy");
    assertRequest(request);
    const line = Buffer.from(`${JSON.stringify(request)}\n`);
    if (line.length > this.bundle.limits.max_line_bytes || request.candidates.length > this.bundle.limits.max_candidates) fail("context_overflow");
    this.#busy = true;
    const epoch = this.#epoch;
    try {
      if (signal.aborted) fail("stale_result");
      const startup = !this.#child;
      // Readiness shares its outer startup deadline across both populated requests.
      const { promise, resolve: resolveResponse, reject: rejectResponse } = Promise.withResolvers<unknown>();
      const abort = () => this.#stop(signal.reason instanceof RouterError && signal.reason.code === "timeout" ? "timeout" : "stale_result");
      const timer = setTimeout(() => this.#stop("timeout"), startup || this.#warming ? this.bundle.limits.startup_timeout_ms :
        this.bundle.limits.request_timeout_ms);
      this.#pending = { resolve: resolveResponse, reject: rejectResponse, timer, signal, abort };
      signal.addEventListener("abort", abort, { once: true });
      try {
        // Cancellation or the deadline must also release a stalled admission check.
        if (startup) {
          if (this.#warming && this.#startup) this.#startup.admission = "checking";
          let admitted: boolean;
          try { admitted = await Promise.race([this.admit(), promise.then(() => false)]); }
          catch (error) {
            if (this.#warming && this.#startup && epoch === this.#epoch) {
              this.#startup.admission = "error";
              this.#startup.failure = { source: "admission_error", code: "backend_unavailable" };
            }
            throw error;
          }
          if (this.#warming && this.#startup) this.#startup.admission = admitted ? "passed" : "rejected";
          if (!admitted) {
            this.#stop("backend_unavailable", { source: "memory_admission", code: "backend_unavailable" });
            fail("backend_unavailable");
          }
        }
        if (signal.aborted || this.#closed || epoch !== this.#epoch) fail("stale_result");
        if (!this.#child) this.#launch();
        this.#child!.stdin.write(line);
      } catch { if (epoch === this.#epoch) this.#stop("backend_unavailable", { source: "request_write", code: "backend_unavailable" }); }
      return await promise;
    } finally { this.#busy = false; }
  }
  close(): void { this.#closed = true; this.#stop("stale_result", { source: "closed", code: "stale_result" }); }
}
