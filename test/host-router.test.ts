import { afterEach, expect, test } from "bun:test";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import type { ContextEvent, Skill, ToolCallEvent } from "@oh-my-pi/pi-coding-agent";
import { LiveCatalog, type HostContext, type Owner } from "../src/catalog.ts";
import { acceptResponse, bytesDigest, contextDigest, digest, parseStrict, readBundle, RouterError, uncertain,
  type Bundle, type Request, type Response } from "../src/protocol.ts";
import { resolvePolicy, route, SHORTLIST_SIZE, validatePolicy, type PolicyArtifact, type Scorer, type Scorers } from "../src/router.ts";
import { BLOCK_TYPE, compileIntent, explicitCommands, normalAgentHandler, SessionRouter } from "../src/session.ts";
import { WorkerClient } from "../src/worker.ts";
import bundleSchema from "../schemas/model-bundle.schema.json";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const hash = bytesDigest("fixture");
const memberIdentity = {
  target_sha256: hash, rubric_sha256: bytesDigest("independent-rubric"), schema_sha256: hash,
  annotation_sha256: hash, receipt_sha256: hash, catalog_sha256: hash, source_dataset_sha256: hash,
  dataset_schema_sha256: hash, source_contract_sha256: hash, role_plan_sha256: hash,
  identity_map_sha256: hash, projection_sha256: hash,
};
const bundle: Bundle = {
  schema_version: 1, model: { type: "window_bag", base: "fixture", revision: "fixture" },
  graphs: bundleSchema.$defs.decision.properties.graphs.const, tokenizer_file: "tokenizer.json", model_config_file: "config.json",
  files: [], pair_format: "intent-action-window-bag-v2", geometry: bundleSchema.$defs.decision.properties.geometry.const,
  selection: { lower_logit: "0", upper_logit: "1", max_optional: 6, validation_sha256: hash, fit_split: "validation" },
  limits: { max_candidates: 32, max_line_bytes: 131072, microbatch: 4, request_timeout_ms: 750, startup_timeout_ms: 10000 },
  calibration: null,
  provenance: {
    dataset_sha256: hash, split_sha256: hash, recipe_sha256: hash, runtime_revision: "authored-fixture",
    contextual_binding: {
      version: 1, target_sha256: hash, catalog_sha256: hash, runtime_revision: hash, approval_sha256: hash,
      purpose: "local-canonical-integration", fixture_only: true,
      source_set_sha256: digest({ purpose: "local-canonical-integration", fixture_only: true, members: [memberIdentity] }),
    },
  },
  scorer_sha256: hash, selector_sha256: hash, calibration_sha256: null, publication_allowed: true,
};

test("bundle admission retains contextual authority and rejects missing or conflicting targets", async () => {
  const root = await mkdtemp(join(tmpdir(), "contextual-bundle-"));
  roots.push(root);
  // Hash-only artifacts exercise both real manifest readers, not model inference.
  const process = spawnSync("python3", ["-B", "-c",
    "from pathlib import Path; import sys; from test.test_router_worker import ArtifactTests; ArtifactTests().make_bundle(Path(sys.argv[1]))",
    root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8" });
  expect(process.status).toBe(0);
  const manifest = readBundle(parseStrict(await readFile(join(root, "manifest.json"), "utf8")));
  const { contextual_binding, ...historicalProvenance } = manifest.provenance;
  expect("members" in contextual_binding).toBe(false);
  expect(() => readBundle({ ...manifest, provenance: historicalProvenance })).toThrow("invalid_response");
  expect(() => readBundle({ ...manifest, provenance: { ...manifest.provenance,
    contextual_binding: { ...contextual_binding, target_sha256: bytesDigest("different-target") },
  } })).toThrow("invalid_response");
  expect(() => readBundle({ ...manifest, provenance: { ...manifest.provenance,
    contextual_binding: { ...contextual_binding, members: [] },
  } })).toThrow("invalid_response");
});

test("host creation admits only hash-verified coherent provenance payloads", async () => {
  const root = await mkdtemp(join(tmpdir(), "runtime-binding-"));
  roots.push(root);
  // Real private paths and producer-built inert files. No scoring or model startup.
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import json, shutil, sys, venv
from test.test_router_worker import ArtifactTests
root = Path(sys.argv[1])
venv.create(root / "venv", with_pip=False, symlinks=True)
shutil.copyfile("python/router-worker.py", root / "router-worker.py")
fixture = ArtifactTests()
for case in ("coherent", *fixture.REJECTIONS):
    directory = root / case
    directory.mkdir()
    if case == "coherent":
        fixture.make_bundle(directory)
    else:
        fixture.rejection_bundle(directory, case)
print(json.dumps(fixture.REJECTIONS))
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  const cases: string[] = JSON.parse(child.stdout);
  const paths = (name: string) => ({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
    workerPath: join(root, "router-worker.py"), bundlePath: join(root, name) });
  const client = await WorkerClient.create(paths("coherent"), async () => true);
  try {
    const expected = readBundle(parseStrict(await readFile(join(root, "coherent/manifest.json"), "utf8")));
    expect(client.bundle).toEqual(expected);
    await client.verify();
  } finally { client.close(); }
  for (const name of cases) {
    await expect(WorkerClient.create(paths(name), async () => true)).rejects.toThrow(
      name === "missing_recipe_bytes" ? "backend_unavailable" : "invalid_response");
  }
}, 30000);

test("paired scorers recover cold workers before the route deadline and cancel readiness", async () => {
  // Real process startup must cross the real route deadline; fake clocks cannot exercise this boundary.
  const root = await mkdtemp(join(tmpdir(), "paired-recovery-"));
  roots.push(root);
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import runpy, sys
runpy.run_path('test/retrieval-fixtures.py')['make_pair'](Path(sys.argv[1]))
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  let admissions = 0;
  const pair = await WorkerClient.createScorers({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
    workerPath: join(root, "python/retrieval-worker.py"), bundlePath: join(root, "retrieval") },
    join(root, "decision"), async () => { admissions++; return true; });
  // Authorization can finish here without an inference process or memory admission.
  expect(admissions).toBe(0);
  await expect(readFile(join(root, "owned-pids.jsonl"))).rejects.toThrow();
  const f = await fixture(["optional"]);
  const catalog = (await f.catalog.getCatalog(f.ctx)).catalog;
  const run = (intent: string, signal = new AbortController().signal) => route({ catalog,
    policy: resolvePolicy(emptyPolicy, "Main", null, []), intent, action: null,
    sessionKey: "owned-recovery", generation: 0, event: "context", scorers: pair, signal, isCurrent: () => true });
  try {
    await pair.ready!(new AbortController().signal);
    const startup: Request[] = (await readFile(join(root, "requests.jsonl"), "utf8")).trim().split("\n").map(line => JSON.parse(line));
    expect(startup.map(request => [request.selectorDigest, request.candidates.length])).toEqual([
      [pair.shortlister.bundle.selector_sha256, 32], [pair.decider.bundle.selector_sha256, 8],
    ]);
    expect(startup[0]!.intent.trim().split(" ")).toHaveLength(254);
    expect(startup[1]!.candidates.every(candidate => candidate.description.trim().split(" ").length === 224)).toBe(true);
    await pair.ready!(new AbortController().signal);
    expect(admissions).toBe(1);
    expect((await run("warm")).response.reasonCode).toBe("no_match");
    const cancel = new AbortController();
    const timer = setTimeout(() => cancel.abort(), 30);
    try { expect((await run("cancel", cancel.signal)).response.reasonCode).toBe("stale_result"); }
    finally { clearTimeout(timer); }
    const cancelStartup = new AbortController();
    const startupTimer = setTimeout(() => cancelStartup.abort(), 100);
    const startCancel = performance.now();
    try { expect((await run("recovery", cancelStartup.signal)).response.reasonCode).toBe("stale_result"); }
    finally { clearTimeout(startupTimer); }
    expect(performance.now() - startCancel).toBeLessThan(750);
    const startRecovery = performance.now();
    expect((await run("recovery")).response.reasonCode).toBe("no_match");
    expect(performance.now() - startRecovery).toBeGreaterThan(750);
    expect((await run("warm-again")).response.reasonCode).toBe("no_match");
    expect((await run("crash")).response.reasonCode).toBe("backend_unavailable");
    expect((await run("recover-crash")).response.reasonCode).toBe("no_match");
  } finally { pair.shortlister.close(); pair.decider.close(); }
  // close() bounds owned-child TERM/KILL cleanup to 250 ms.
  await Bun.sleep(350);
  const pids: number[] = (await readFile(join(root, "owned-pids.jsonl"), "utf8")).trim().split("\n").map(Number);
  for (const pid of pids) expect(() => process.kill(pid, 0)).toThrow();
}, 20000);

test("startup identifies rejected memory admission without launching or blaming inference", async () => {
  const root = await mkdtemp(join(tmpdir(), "paired-memory-admission-"));
  roots.push(root);
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import runpy, sys
runpy.run_path('test/retrieval-fixtures.py')['make_pair'](Path(sys.argv[1]))
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  let admission: "reject" | "error" | "admit" = "reject";
  const pair = await WorkerClient.createScorers({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
    workerPath: join(root, "python/retrieval-worker.py"), bundlePath: join(root, "retrieval") },
    join(root, "decision"), async () => {
      if (admission === "error") throw new Error("private admission detail must not be copied");
      return admission === "admit";
    });
  try {
    await expect(pair.ready!(new AbortController().signal)).rejects.toThrow("backend_unavailable");
    expect(pair.shortlister.startupDiagnostics).toEqual(expect.objectContaining({ stage: "retrieval", admission: "rejected",
      worker_started: false, completed: [], failure: { source: "memory_admission", code: "backend_unavailable" } }));
    expect(pair.shortlister.startupDiagnostics!.elapsed_ms).toBeLessThan(750);
    await expect(readFile(join(root, "owned-pids.jsonl"))).rejects.toThrow();
    admission = "error";
    await expect(pair.ready!(new AbortController().signal)).rejects.toThrow("backend_unavailable");
    expect(pair.shortlister.startupDiagnostics).toEqual(expect.objectContaining({ admission: "error", worker_started: false,
      failure: { source: "admission_error", code: "backend_unavailable" } }));
    expect(JSON.stringify(pair.shortlister.startupDiagnostics)).not.toContain("private admission detail");
    admission = "admit";
    await pair.ready!(new AbortController().signal);
    expect(pair.shortlister.startupDiagnostics).toEqual(expect.objectContaining({ stage: "complete", admission: "passed",
      worker_started: true, failure: null }));
    expect(pair.shortlister.startupDiagnostics!.completed.map(stage => stage.stage)).toEqual(["retrieval", "decider"]);
  } finally { pair.shortlister.close(); }
}, 10000);

test("readiness rejects a decider response without populated scores and starts cleanly afterward", async () => {
  const root = await mkdtemp(join(tmpdir(), "paired-warmup-failure-"));
  roots.push(root);
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import runpy, sys
runpy.run_path('test/retrieval-fixtures.py')['make_pair'](Path(sys.argv[1]))
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  const pair = await WorkerClient.createScorers({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
    workerPath: join(root, "python/retrieval-worker.py"), bundlePath: join(root, "retrieval") },
    join(root, "decision"), async () => true);
  try {
    await writeFile(join(root, "reject-startup-decider"), "");
    await expect(pair.ready!(new AbortController().signal)).rejects.toThrow("backend_unavailable");
    expect(pair.shortlister.startupDiagnostics).toEqual(expect.objectContaining({ stage: "decider", admission: "passed",
      worker_started: true, failure: { source: "missing_scores", code: "backend_unavailable", response_reason: "backend_unavailable" } }));
    expect(pair.shortlister.startupDiagnostics!.completed.map(stage => [stage.stage, stage.score_count])).toEqual([["retrieval", 32]]);
    await rm(join(root, "reject-startup-decider"));
    await writeFile(join(root, "exit-startup"), "");
    await expect(pair.ready!(new AbortController().signal)).rejects.toThrow("backend_unavailable");
    expect(pair.shortlister.startupDiagnostics).toEqual(expect.objectContaining({ stage: "retrieval", worker_started: true,
      completed: [], failure: { source: "process_exit", code: "backend_unavailable", exit_code: 7, signal: null } }));
    await rm(join(root, "exit-startup"));
    await pair.ready!(new AbortController().signal);
    expect((await readFile(join(root, "owned-pids.jsonl"), "utf8")).trim().split("\n")).toHaveLength(3);
  } finally { pair.shortlister.close(); }
}, 10000);

test("both readiness stages share one ten-second budget", async () => {
  const root = await mkdtemp(join(tmpdir(), "paired-warmup-budget-"));
  roots.push(root);
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import runpy, sys
runpy.run_path('test/retrieval-fixtures.py')['make_pair'](Path(sys.argv[1]))
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  await writeFile(join(root, "stall-startup-decider"), "");
  const pair = await WorkerClient.createScorers({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
    workerPath: join(root, "python/retrieval-worker.py"), bundlePath: join(root, "retrieval") },
    join(root, "decision"), async () => true);
  try {
    const start = performance.now();
    await expect(pair.ready!(new AbortController().signal)).rejects.toThrow("timeout");
    expect(performance.now() - start).toBeLessThan(10500);
    expect(pair.shortlister.startupDiagnostics).toEqual(expect.objectContaining({ stage: "decider",
      failure: { source: "deadline", code: "timeout" } }));
    expect(pair.shortlister.startupDiagnostics!.completed.map(stage => stage.stage)).toEqual(["retrieval"]);
  } finally { pair.shortlister.close(); }
}, 15000);

test("paired startup rejects TinyBERT and unpinned MiniLM revisions before launching", async () => {
  const root = await mkdtemp(join(tmpdir(), "paired-admission-"));
  roots.push(root);
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import runpy, sys
from test.test_router_worker import ArtifactTests
root = Path(sys.argv[1])
runpy.run_path('test/retrieval-fixtures.py')['make_pair'](root)
artifacts = ArtifactTests()
for name, base, revision in (
    ('tinybert', 'cross-encoder/ms-marco-TinyBERT-L2-v2', '81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc'),
    ('wrong-revision', 'cross-encoder/ms-marco-MiniLM-L6-v2', '0' * 40),
):
    directory = root / name
    directory.mkdir()
    manifest = artifacts.make_bundle(directory)
    manifest['model'] = {'type': 'window_bag', 'base': base, 'revision': revision}
    artifacts.save(directory, manifest)
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  for (const name of ["tinybert", "wrong-revision"]) {
    await expect(WorkerClient.createScorers({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
      workerPath: join(root, "python/retrieval-worker.py"), bundlePath: join(root, "retrieval") },
      join(root, name), async () => { throw new Error("must reject before process admission"); }))
      .rejects.toThrow("invalid_response");
  }
  await expect(readFile(join(root, "owned-pids.jsonl"))).rejects.toThrow();
}, 20000);
const emptyPolicy: PolicyArtifact = { version: 1, rules: [], unsupported: ["Natural-language rules are not compiled"] };
const action: ToolCallEvent = { type: "tool_call", toolName: "fixture-write", toolCallId: "call", input: {} };
const messages: ContextEvent["messages"] = [{ role: "user", content: "review files without deployment", timestamp: 1 }];
function policy(requiredIds: string[], toolName: string | null = null, excludedIds: string[] = []): PolicyArtifact {
  return { version: 1, unsupported: [], rules: [{ id: "fixture-rule", provenance: "fixture-reviewed-policy",
    scope: { actorId: null, toolName }, requiredIds, excludedIds }] };
}
class FixtureScorer implements Scorer {
  readonly bundle = bundle;
  calls = 0; closed = false;
  constructor(readonly positives: string[] = [], readonly failure = false) {}
  async verify(): Promise<void> {}
  async score(request: Request): Promise<Response> {
    this.calls++;
    if (this.failure) throw new RouterError("backend_unavailable");
    const selectedIds = [...new Set([...request.requiredIds, ...this.positives])].filter(id => request.candidates.some(c => c.id === id)).sort();
    const response = uncertain(request, "insufficient_evidence");
    // Like the worker: an incomplete candidate list can never prove "none".
    const status = selectedIds.length ? "selected" : request.candidateCoverage === "full" ? "none" : "uncertain";
    return { ...response, status, selectedIds,
      reasonCode: selectedIds.length ? (this.positives.some(id => !request.requiredIds.includes(id) && selectedIds.includes(id)) ? "matched" : "policy_only")
        : status === "none" ? "no_match" : "inadequate_candidates",
      scores: { kind: "ranking", scorerDigest: hash, calibrationDigest: null,
        values: request.candidates.map(candidate => ({ id: candidate.id, score: this.positives.includes(candidate.id) ? 2 : -1 })) } };
  }
  close(): void { this.closed = true; }
}

const decisionBundle: Bundle = { ...bundle, selection: { ...bundle.selection, lower_logit: "-1" } };
/** Exercise the real worker decision and JSON boundary without importing ML dependencies. */
function pythonDecision(request: Request, values: number[]): unknown {
  const child = spawnSync("python3", ["-I", "-B", "-c", `
import json, runpy, sys
worker = runpy.run_path(sys.argv[1])
payload = json.load(sys.stdin)
request = payload["request"]
worker["validate_request"](request)
candidates = worker["preflight"](request)
response = worker["decide"](request, candidates, payload["values"], -1, 1, payload["scorerDigest"])
print(json.dumps(response, allow_nan=False))
`, fileURLToPath(new URL("../python/router-worker.py", import.meta.url))], {
    input: JSON.stringify({ request, values, scorerDigest: decisionBundle.scorer_sha256 }),
    encoding: "utf8", timeout: 5000,
  });
  if (child.error) throw child.error;
  if (child.status !== 0) throw new Error(`Python decision failed: ${child.stderr}`);
  return parseStrict(child.stdout);
}

function decisionRequest(candidateCoverage: Request["candidateCoverage"], requiredIds: string[] = []): Request {
  const request: Request = { version: 1, requestId: "cross-language", sessionKey: "fixture", generation: 0,
    event: "context", catalogDigest: hash, contextDigest: hash, policyDigest: hash, selectorDigest: hash,
    intent: "review files", action: null, candidateCoverage, requiredIds, excludedIds: [], maxSelections: 6,
    candidates: ["required", "positive", "gray"].map(id => ({ id, name: id, description: `review ${id}`, contentDigest: hash })) };
  request.contextDigest = contextDigest(request);
  return request;
}
async function fixture(names: string[] = ["required", "optional"]) {
  const root = await mkdtemp(join(tmpdir(), "host-router-")); roots.push(root);
  const skills: Skill[] = [];
  for (const [index, name] of names.entries()) {
    const baseDir = join(root, String(index)); await mkdir(baseDir);
    const filePath = join(baseDir, "SKILL.md");
    await writeFile(filePath, `---\nname: ${name}\n---\nExact ${name} instructions.\n`);
    skills.push({ name, description: `review files ${name}`, filePath, baseDir, source: "fixture" });
  }
  let sessionId = "fixture-session";
  const manager = { getSessionId: () => sessionId };
  const owner: Owner = { skills, isDisposed: false, sessionManager: manager };
  let current: object | null = owner;
  const catalog = new LiveCatalog({ get: id => id === "Main" ? { session: current } : undefined });
  const ctx: HostContext = { agent: { id: "Main", kind: "main", name: "main", depth: 0 }, sessionManager: manager };
  return { root, skills, owner, catalog, ctx, setOwner: (next: object | null) => { current = next; },
    switchSession: () => { sessionId = "other-session"; } };
}
function scorers(decider: Scorer, shortlister: Scorer = new FixtureScorer()): Scorers {
  return { shortlister, decider };
}
function session(f: { catalog: LiveCatalog }, artifact = emptyPolicy, scorer: Scorer = new FixtureScorer()) {
  return new SessionRouter({ catalog: f.catalog, createScorers: () => scorers(scorer), policy: async () => artifact, bodyBudgetBytes: 65536 });
}

test("canonical hashes preserve bytes, Unicode and property order", () => {
  expect(digest({ z: "é", a: 2 })).toBe(bytesDigest('{"a":2,"z":"é"}'));
  expect(bytesDigest("body\n")).not.toBe(bytesDigest("body"));
  expect(() => digest("\ud800")).toThrow("invalid_response");
  expect(() => parseStrict('{"id":1,"id":2}')).toThrow("invalid_response");
  expect(() => parseStrict('{"score":1e999}')).toThrow("invalid_response");
  expect(() => parseStrict('{"x":"\\ud800"}')).toThrow("invalid_response");
});
test("live catalog excludes disabled entries and preserves namespaces and hidden admission", async () => {
  const f = await fixture(["plugin/review", "hidden"]); f.skills[1]!.hide = true;
  const lease = await f.catalog.getCatalog(f.ctx);
  expect(lease.catalog.entries.map(entry => [entry.runtime_id, entry.automatic_eligible])).toEqual([["hidden", false], ["plugin/review", true]]);
  const resolved = resolvePolicy(policy(["hidden"]), "Main", null, []);
  const decider = new FixtureScorer();
  const { request } = await route({ catalog: lease.catalog, policy: resolved, intent: "review", action: null, sessionKey: "fixture",
    generation: 0, event: "context", scorers: scorers(decider), signal: new AbortController().signal, isCurrent: () => true });
  expect(request.candidates.map(item => item.id)).toEqual(["hidden", "plugin/review"]);
  f.skills.splice(0, 1);
  expect((await f.catalog.getCatalog(f.ctx)).catalog.entries.map(entry => entry.runtime_id)).toEqual(["hidden"]);
  expect(f.catalog.current(f.ctx, lease)).toBe(false);
});
test("missing registry and owner-session mismatch are not empty catalogs", async () => {
  const f = await fixture(); f.setOwner(null);
  await expect(f.catalog.getCatalog(f.ctx)).rejects.toThrow("backend_unavailable");
  f.setOwner({ ...f.owner, sessionManager: { getSessionId: () => "wrong-session" } });
  await expect(f.catalog.getCatalog(f.ctx)).rejects.toThrow("backend_unavailable");
});
test("a registry session handle without skills or disposal state fails closed", async () => {
  // OMP 18.18 worker proxies expose AgentSessionHandle, which lacks both fields.
  const f = await fixture();
  f.setOwner({ sessionManager: f.owner.sessionManager, isStreaming: false });
  await expect(f.catalog.getCatalog(f.ctx)).rejects.toThrow("backend_unavailable");
  f.setOwner({ skills: f.skills, sessionManager: f.owner.sessionManager });
  await expect(f.catalog.getCatalog(f.ctx)).rejects.toThrow("backend_unavailable");
});
test("advisor resolves documented parent without borrowing another owner", async () => {
  const f = await fixture();
  const ctx: HostContext = { ...f.ctx, agent: { id: "advisor", kind: "sub", name: "advisor", depth: 0, parentId: "Main" } };
  expect((await f.catalog.getCatalog(ctx)).owner).toBe(f.owner);
  await expect(f.catalog.getCatalog({ ...ctx, agent: { ...ctx.agent, parentId: "other" } })).rejects.toThrow("backend_unavailable");
});
test("native advisor dispatch preserves parent injection, pending actions and decision cache", async () => {
  const f = await fixture();
  const scorer = new FixtureScorer();
  const artifact = policy(["required"]);
  artifact.rules.push({ id: "advisor-only", provenance: "fixture", scope: { actorId: "advisor", toolName: action.toolName },
    requiredIds: ["optional"], excludedIds: [] });
  const router = session(f, artifact, scorer);
  const context = normalAgentHandler((event: ContextEvent, ctx: HostContext) => router.context(event.messages, ctx));
  const tool = normalAgentHandler((event: ToolCallEvent, ctx: HostContext) => router.toolCall(event, ctx));
  const invalidate = normalAgentHandler((_event: unknown, _ctx: HostContext) => router.invalidate());
  const shutdown = normalAgentHandler((_event: unknown, _ctx: HostContext) => router.shutdown());
  const advisor: HostContext = { ...f.ctx, agent: { kind: "sub", id: "advisor", name: "advisor", depth: 0, parentId: "Main" } };
  expect((await tool(action, f.ctx))?.block).toBe(true);
  await context({ type: "context", messages }, f.ctx);
  expect(await tool(action, advisor)).toBeUndefined();
  expect(await context({ type: "context", messages }, advisor)).toBeUndefined();
  await invalidate({}, advisor); await shutdown({}, advisor);
  expect(await tool(action, f.ctx)).toBeUndefined();
  const next = await context({ type: "context", messages }, f.ctx);
  expect(JSON.stringify(next?.messages)).toContain("Exact required instructions");
  expect(JSON.stringify(next?.messages)).not.toContain("Exact optional instructions");
  expect(scorer.calls).toBe(1);
});
test("ordinary children and partial advisor identities still enter the prerequisite guard", async () => {
  const f = await fixture();
  const router = session(f, policy(["required"]));
  const tool = normalAgentHandler((event: ToolCallEvent, ctx: HostContext) => router.toolCall(event, ctx));
  for (const agent of [
    { kind: "sub" as const, id: "Child", name: "advisor", depth: 0, parentId: "Main" },
    { kind: "sub" as const, id: "advisor", name: "task", depth: 0, parentId: "Main" },
    { kind: "sub" as const, id: "advisor", name: "advisor", depth: 1, parentId: "Main" },
    { kind: "sub" as const, id: "advisor", name: "advisor", depth: 0 },
  ]) expect((await tool(action, { ...f.ctx, agent }))?.block).toBe(true);
});
test("containment rejects a symlink to another skill root", async () => {
  const f = await fixture();
  await rm(f.skills[0]!.filePath); await symlink(f.skills[1]!.filePath, f.skills[0]!.filePath);
  await expect(f.catalog.getCatalog(f.ctx)).rejects.toThrow("body_unavailable");
});
test("body and metadata changes invalidate a lease", async () => {
  const f = await fixture(); const lease = await f.catalog.getCatalog(f.ctx);
  const entry = lease.catalog.entries.find(entry => entry.runtime_id === "required")!;
  await writeFile(f.skills[0]!.filePath, "changed body");
  await expect(f.catalog.getSkillBody(f.ctx, "required", lease.catalog.catalog_sha256, entry.content_sha256, lease)).rejects.toThrow("stale_result");
  f.skills[0]!.description = "new description";
  expect(f.catalog.current(f.ctx, lease)).toBe(false);
});
test("private export is authoritative and never overwrites a snapshot", async () => {
  const f = await fixture(); const path = join(f.root, "catalog.json");
  const snapshot = await f.catalog.exportPrivate(f.ctx, path, f.root);
  expect(JSON.parse(await readFile(path, "utf8"))).toEqual(snapshot);
  await expect(f.catalog.exportPrivate(f.ctx, path, f.root)).rejects.toThrow();
});
test("model shortlist keeps the top-scored skills across chunks and only claims full coverage when complete", async () => {
  const names = Array.from({ length: 40 }, (_, i) => `skill-${String(i).padStart(2, "0")}`);
  const f = await fixture(names);
  const catalog = (await f.catalog.getCatalog(f.ctx)).catalog;
  // The best skills sit in the second chunk, beyond any first-N or name-order cut.
  const shortlister = new FixtureScorer(["skill-39", "skill-35"]);
  const seen: Request[] = [];
  const decider: Scorer = { bundle, async verify() {}, close() {},
    async score(request) { seen.push(request); return new FixtureScorer(["skill-39"]).score(request); } };
  const decision = await route({ catalog, policy: resolvePolicy(emptyPolicy, "Main", null, []), intent: "review", action: null,
    sessionKey: "fixture", generation: 0, event: "context", scorers: scorers(decider, shortlister),
    signal: new AbortController().signal, isCurrent: () => true });
  expect(shortlister.calls).toBe(2);
  expect(seen).toHaveLength(1);
  expect(seen[0]!.candidateCoverage).toBe("shortlist");
  expect(seen[0]!.candidates).toHaveLength(SHORTLIST_SIZE);
  expect(seen[0]!.candidates.map(item => item.id)).toContain("skill-39");
  expect(seen[0]!.candidates.map(item => item.id)).toContain("skill-35");
  expect(decision.response.selectedIds).toEqual(["skill-39"]);
  const small = await fixture(names.slice(0, SHORTLIST_SIZE));
  const unused = new FixtureScorer();
  const full = await route({ catalog: (await small.catalog.getCatalog(small.ctx)).catalog, policy: resolvePolicy(emptyPolicy, "Main", null, []),
    intent: "review", action: null, sessionKey: "fixture", generation: 0, event: "context", scorers: scorers(new FixtureScorer(), unused),
    signal: new AbortController().signal, isCurrent: () => true });
  expect(unused.calls).toBe(0);
  expect(full.request.candidateCoverage).toBe("full");
});
test("a failed shortlist stage fails closed without calling the decider", async () => {
  const f = await fixture(Array.from({ length: 20 }, (_, i) => `skill-${i}`));
  const decider = new FixtureScorer(["skill-1"]);
  const decision = await route({ catalog: (await f.catalog.getCatalog(f.ctx)).catalog, policy: resolvePolicy(emptyPolicy, "Main", null, []),
    intent: "review", action: null, sessionKey: "fixture", generation: 0, event: "context",
    scorers: scorers(decider, new FixtureScorer([], true)), signal: new AbortController().signal, isCurrent: () => true });
  expect([decision.response.status, decision.response.reasonCode]).toEqual(["uncertain", "backend_unavailable"]);
  expect(decider.calls).toBe(0);
});
test("required and optional bodies occur exactly once before the first action", async () => {
  const f = await fixture(); const scorer = new FixtureScorer(["optional"]);
  const router = session(f, policy(["required"]), scorer);
  expect((await router.toolCall(action, f.ctx))?.block).toBe(true);
  const outgoing = await router.context(messages, f.ctx);
  const block = outgoing.messages.at(-1)!;
  expect(block.role).toBe("custom");
  if (block.role !== "custom" || typeof block.content !== "string") throw new Error("missing block");
  expect(block.content).toContain("Exact required instructions.\\n");
  expect(block.content).toContain("Exact optional instructions.\\n");
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
  const retry = await router.context(outgoing.messages, f.ctx);
  expect(retry.messages.filter(message => message.role === "custom" && message.customType === BLOCK_TYPE)).toHaveLength(1);
  expect(scorer.calls).toBe(1);
});
test("new action requirements block first, then load on the next context", async () => {
  const f = await fixture(); const router = session(f, policy(["required"], action.toolName));
  await router.context(messages, f.ctx);
  expect((await router.toolCall(action, f.ctx))?.reason).toContain("missing_required");
  const next = await router.context(messages, f.ctx);
  expect(JSON.stringify(next.messages)).toContain("Exact required instructions");
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
});
test("a policy change invalidates local injection before another action", async () => {
  const f = await fixture();
  let artifact = policy(["required"]);
  const scorer = new FixtureScorer();
  const router = new SessionRouter({ catalog: f.catalog, createScorers: () => scorers(scorer),
    policy: async () => artifact, bodyBudgetBytes: 65536 });
  await router.context(messages, f.ctx);
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
  artifact = policy(["optional"]);
  expect((await router.toolCall(action, f.ctx))?.block).toBe(true);
  const next = await router.context(messages, f.ctx);
  expect(JSON.stringify(next.messages)).toContain("Exact optional instructions");
  expect(JSON.stringify(next.messages)).not.toContain("Exact required instructions");
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
});
test("optional backend uncertainty does not block actions, required bodies still load", async () => {
  const f = await fixture(); const scorer = new FixtureScorer([], true);
  const router = session(f, policy(["required"]), scorer);
  expect((await router.context(messages, f.ctx)).reason).toBe("backend_unavailable");
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
  const optionalOnly = session(f, emptyPolicy, scorer);
  await optionalOnly.context(messages, f.ctx);
  expect(await optionalOnly.toolCall(action, f.ctx)).toBeUndefined();
});
test("lifecycle invalidation and body mutation invalidate local prerequisites", async () => {
  const f = await fixture(); const scorer = new FixtureScorer(); const router = session(f, policy(["required"]), scorer);
  await router.context(messages, f.ctx); router.invalidate();
  expect((await router.toolCall(action, f.ctx))?.block).toBe(true);
  await router.context(messages, f.ctx);
  await writeFile(f.skills[0]!.filePath, "new body");
  expect((await router.toolCall(action, f.ctx))?.reason).toContain("stale_result");
  router.shutdown(); expect(scorer.closed).toBe(true);
});
test("a pending response from an invalidated generation never injects bodies", async () => {
  const f = await fixture(); const started = Promise.withResolvers<void>(); const gate = Promise.withResolvers<void>();
  const scorer: Scorer = { bundle, close() {}, async verify() {}, async score(request) {
    started.resolve(); await gate.promise; return new FixtureScorer(["optional"]).score(request);
  } };
  const router = session(f, emptyPolicy, scorer); const pending = router.context(messages, f.ctx);
  await started.promise; router.invalidate(); gate.resolve();
  const outgoing = await pending;
  expect(outgoing.reason).toBe("stale_result");
  expect(JSON.stringify(outgoing.messages)).not.toContain("Exact optional instructions");
});
test("all mandatory candidates survive retrieval and overflow is explicit before scoring", async () => {
  const f = await fixture(Array.from({ length: 33 }, (_, i) => `required-${i}`));
  const scorer = new FixtureScorer(); const router = session(f, policy(f.skills.map(skill => skill.name)), scorer);
  expect((await router.context(messages, f.ctx)).reason).toBe("context_overflow");
  expect(scorer.calls).toBe(0); expect((await router.toolCall(action, f.ctx))?.block).toBe(true);
});
test("conflicting policy is visible and model cannot overrule it", async () => {
  const f = await fixture(); const scorer = new FixtureScorer();
  const router = session(f, policy(["required"], null, ["required"]), scorer);
  expect((await router.context(messages, f.ctx)).reason).toBe("policy_conflict");
  expect((await router.toolCall(action, f.ctx))?.reason).toContain("policy_conflict");
  expect(scorer.calls).toBe(0);
  expect(() => validatePolicy({ version: 1, rules: [], unsupported: [], inferredFromProse: true })).toThrow("policy_conflict");
});
test("prose mentions never become mandatory IDs and user corrections remain exact", () => {
  expect(explicitCommands(messages)).toEqual([]);
  const corrected: ContextEvent["messages"] = [...messages, { role: "user", content: "/skill:plugin/review do not deploy", timestamp: 2 }];
  expect(explicitCommands(corrected)).toEqual(["plugin/review"]);
  expect(compileIntent(corrected)).toBe("review files without deployment\n\n/skill:plugin/review do not deploy");
});
test("intent keeps ordered text without treating reasoning or tool calls as evidence", () => {
  const history: ContextEvent["messages"] = [
    ...messages,
    { role: "toolResult", toolCallId: "previous", toolName: "read",
      content: [{ type: "text", text: "older evidence" }], isError: false, timestamp: 2 },
    { role: "assistant", api: "openai-completions", provider: "fixture", model: "fixture",
      content: [
        { type: "thinking", thinking: "private reasoning" },
        { type: "text", text: "first evidence" },
        { type: "image", data: "ignored", mimeType: "image/png" },
        { type: "toolCall", id: "next", name: "read", arguments: { path: "ignored" } },
        { type: "text", text: "second evidence" },
      ],
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      stopReason: "toolUse", timestamp: 3 },
  ];
  expect(compileIntent(history)).toBe(
    "review files without deployment\n\nLast step evidence (not policy):\nfirst evidence\nsecond evidence");
  expect(() => compileIntent([{ role: "user", content: "x".repeat(32769), timestamp: 4 }]))
    .toThrow("context_overflow");
});
test("canonical response rejects wrong identity, scores, membership, caps and shortlist none", async () => {
  const f = await fixture(); const catalog = (await f.catalog.getCatalog(f.ctx)).catalog;
  const scorer = new FixtureScorer(["optional"]);
  const decision = await route({ catalog, policy: resolvePolicy(emptyPolicy, "Main", null, []), intent: "review",
    action: null, sessionKey: "fixture", generation: 0, event: "context", scorers: scorers(scorer), signal: new AbortController().signal, isCurrent: () => true });
  const { request, response } = decision;
  expect(acceptResponse(response, request, bundle).selectedIds).toEqual(["optional"]);
  expect(contextDigest(request)).toBe(request.contextDigest);
  expect(() => acceptResponse({ ...response, generation: 1 }, request, bundle)).toThrow("stale_result");
  expect(() => acceptResponse({ ...response, injected: true }, request, bundle)).toThrow("invalid_response");
  expect(() => acceptResponse({ ...response, selectedIds: ["unknown"] }, request, bundle)).toThrow("invalid_response");
  expect(() => acceptResponse({ ...response, scores: { ...response.scores!, values: [{ id: "optional", score: 2 }, { id: "optional", score: 2 }] } }, request, bundle)).toThrow("invalid_response");
  expect(() => acceptResponse(response, { ...request, maxSelections: 0 }, bundle)).toThrow("invalid_response");
  expect(() => acceptResponse({ ...response, status: "none", reasonCode: "no_match", selectedIds: [] }, { ...request, candidateCoverage: "shortlist" }, bundle)).toThrow("invalid_response");
});

for (const coverage of ["full", "shortlist"] as const) {
  test(`Python decision accepts confident positives beside gray scores with ${coverage} coverage`, () => {
    for (const requiredIds of [[], ["required"]]) {
      const request = decisionRequest(coverage, requiredIds);
      const response = acceptResponse(pythonDecision(request, [-2, 2, 0]), request, decisionBundle);
      expect(response.status).toBe("selected");
      expect(response.reasonCode).toBe("matched");
      expect(response.selectedIds).toEqual([...requiredIds, "positive"]);
      // A valid mixed-score result must not relax exact selection or policy checks.
      for (const selectedIds of [[], ["gray"], ["positive", "gray"], ["unknown"]]) {
        expect(() => acceptResponse({ ...response, selectedIds }, request, decisionBundle)).toThrow("invalid_response");
      }
      expect(() => acceptResponse(response, { ...request, excludedIds: ["positive"] }, decisionBundle)).toThrow("invalid_response");
      expect(() => acceptResponse({ ...response, selectedIds: ["positive", "positive"] }, request, decisionBundle)).toThrow("invalid_response");
      const multiple = acceptResponse(pythonDecision(request, [2, 2, 0]), request, decisionBundle);
      expect(multiple.selectedIds).toEqual(["required", "positive"]);
      expect(() => acceptResponse({ ...multiple, selectedIds: ["positive"] }, request, decisionBundle)).toThrow("invalid_response");
    }
  });

  test(`Python decision preserves abstention, caps and threshold boundaries with ${coverage} coverage`, () => {
    const request = decisionRequest(coverage);
    for (const requiredIds of [[], ["required"]]) {
      const grayRequest = { ...request, requiredIds };
      const gray = acceptResponse(pythonDecision(grayRequest, [0, 0, 0]), grayRequest, decisionBundle);
      expect([gray.status, gray.reasonCode, gray.selectedIds]).toEqual(["uncertain", "insufficient_evidence", []]);
      expect(() => acceptResponse({ ...gray, status: requiredIds.length ? "selected" : "none",
        reasonCode: requiredIds.length ? "policy_only" : "no_match", selectedIds: requiredIds }, grayRequest, decisionBundle)).toThrow("invalid_response");
    }
    const capped = { ...request, maxSelections: 1 };
    const overflow = acceptResponse(pythonDecision(capped, [-2, 2, 2]), capped, decisionBundle);
    expect([overflow.status, overflow.reasonCode, overflow.selectedIds]).toEqual(["uncertain", "selection_overflow", []]);
    expect(() => acceptResponse({ ...overflow, status: "selected", reasonCode: "matched", selectedIds: ["positive"] }, capped, decisionBundle)).toThrow("invalid_response");
    const boundary = acceptResponse(pythonDecision(request, [-1, 1, 0]), request, decisionBundle);
    expect([boundary.status, boundary.selectedIds]).toEqual(["selected", ["positive"]]);
    const negative = acceptResponse(pythonDecision(request, [-1, -2, -1]), request, decisionBundle);
    expect([negative.status, negative.reasonCode, negative.selectedIds]).toEqual(
      coverage === "full" ? ["none", "no_match", []] : ["uncertain", "inadequate_candidates", []]);
    if (coverage === "shortlist") {
      expect(() => acceptResponse({ ...negative, status: "none", reasonCode: "no_match" }, request, decisionBundle)).toThrow("invalid_response");
    }
    const requiredRequest = { ...request, requiredIds: ["required"] };
    const policyOnly = acceptResponse(pythonDecision(requiredRequest, [0, -1, -2]), requiredRequest, decisionBundle);
    expect([policyOnly.status, policyOnly.reasonCode, policyOnly.selectedIds]).toEqual(["selected", "policy_only", ["required"]]);
    for (const values of [
      [{ id: "required", score: -2 }, { id: "positive", score: 2 }],
      [{ id: "required", score: -2 }, { id: "positive", score: 2 }, { id: "gray", score: Infinity }],
    ]) {
      expect(() => acceptResponse({ ...boundary, scores: { ...boundary.scores!, values } }, request, decisionBundle)).toThrow("invalid_response");
    }
  });

  test(`Python decision loads only required and positive bodies through context with ${coverage} coverage`, async () => {
    const names = ["required", "positive", "gray"];
    if (coverage === "shortlist") names.push(...Array.from({ length: 16 }, (_, index) => `z-other-${index}`));
    const f = await fixture(names);
    const scorer: Scorer = {
      bundle: decisionBundle,
      async verify() {},
      async score(request) {
        expect(request.candidateCoverage).toBe(coverage);
        return pythonDecision(request, request.candidates.map(candidate => candidate.id === "positive" ? 2 : 0));
      },
      close() {},
    };
    const router = session(f, policy(["required"]), scorer);
    const outgoing = await router.context(messages, f.ctx);
    const block = outgoing.messages.at(-1)!;
    if (block.role !== "custom" || typeof block.content !== "string") throw new Error("missing block");
    expect(block.content).toContain("Exact required instructions.\\n");
    expect(block.content).toContain("Exact positive instructions.\\n");
    expect(block.content).not.toContain("Exact gray instructions.");
    expect(block.content).not.toContain("Exact z-other-");
    expect(block.content).not.toContain("invalid_response");
    expect(await router.toolCall(action, f.ctx)).toBeUndefined();
  });
}

test("missing model artifacts do not prevent independently verified mandatory loading", async () => {
  const f = await fixture();
  const router = new SessionRouter({ catalog: f.catalog, createScorers: () => { throw new RouterError("backend_unavailable"); },
    policy: async () => policy(["required"]), bodyBudgetBytes: 65536 });
  expect((await router.context(messages, f.ctx)).reason).toBe("backend_unavailable");
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
});

test("body budget overflow never silently truncates required instructions", async () => {
  const f = await fixture();
  await writeFile(f.skills[1]!.filePath, "optional body ".repeat(300));
  const router = new SessionRouter({ catalog: f.catalog, createScorers: () => scorers(new FixtureScorer(["optional"])),
    policy: async () => policy(["required"]), bodyBudgetBytes: 1024 });
  const outgoing = await router.context(messages, f.ctx);
  expect(outgoing.reason).toBe("context_overflow");
  expect(JSON.stringify(outgoing.messages)).toContain("Exact required instructions");
  expect(JSON.stringify(outgoing.messages)).not.toContain("optional body");
  expect(await router.toolCall(action, f.ctx)).toBeUndefined();
  await writeFile(f.skills[0]!.filePath, "mandatory body ".repeat(300));
  expect((await router.context(messages, f.ctx)).reason).toBe("context_overflow");
  expect((await router.toolCall(action, f.ctx))?.block).toBe(true);
});

test("restricted child catalog never inherits parent skills", async () => {
  const f = await fixture();
  const child: Owner = { skills: [f.skills[1]!], isDisposed: false, sessionManager: { getSessionId: () => "child-session" } };
  const catalog = new LiveCatalog({ get: id => id === "child" ? { session: child } : { session: f.owner } });
  const ctx: HostContext = { agent: { kind: "sub", id: "child", parentId: "Main", name: "task", depth: 1 }, sessionManager: child.sessionManager };
  expect((await catalog.getCatalog(ctx)).catalog.entries.map(entry => entry.runtime_id)).toEqual(["optional"]);
  f.skills[0]!.description = "parent changed";
  expect((await catalog.getCatalog(ctx)).catalog.entries.map(entry => entry.runtime_id)).toEqual(["optional"]);
});

test("native user skill messages admit hidden names, not agent-authored claims", () => {
  const native: ContextEvent["messages"] = [{ role: "custom", customType: "skill-prompt", attribution: "user",
    content: "native skill body", details: { name: "plugin/hidden" }, display: true, timestamp: 1 }];
  expect(explicitCommands(native)).toEqual(["plugin/hidden"]);
  expect(explicitCommands([{ ...native[0]!, attribution: "agent" } as ContextEvent["messages"][number]])).toEqual([]);
});

test("the whole route times out across individually bounded retrieval chunks", async () => {
  const f = await fixture(Array.from({ length: 40 }, (_, index) => `skill-${index}`));
  // Deliberately exercise the real platform deadline without mocked clocks or timers.
  const delayed: Scorer = { bundle, async verify() {}, close() {}, async score(request) {
    await Bun.sleep(400);
    return new FixtureScorer().score(request);
  } };
  const decider = new FixtureScorer();
  const started = performance.now();
  const result = await route({ catalog: (await f.catalog.getCatalog(f.ctx)).catalog,
    policy: resolvePolicy(emptyPolicy, "Main", null, []), intent: "review", action: null,
    sessionKey: "deadline", generation: 0, event: "context", scorers: scorers(decider, delayed),
    signal: new AbortController().signal, isCurrent: () => true });
  expect(result.response.reasonCode).toBe("timeout");
  expect(performance.now() - started).toBeLessThan(1000);
  expect(decider.calls).toBe(0);
});


test("readiness caches only live catalog descriptions before timed routing", async () => {
  const root = await mkdtemp(join(tmpdir(), "live-catalog-readiness-"));
  roots.push(root);
  const child = spawnSync("python3", ["-B", "-c", `
from pathlib import Path
import runpy, sys
runpy.run_path('test/retrieval-fixtures.py')['make_pair'](Path(sys.argv[1]))
`, root], { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 20000 });
  expect(child.status).toBe(0);
  const pair = await WorkerClient.createScorers({ taskRoot: root, pythonPath: join(root, "venv/bin/python"),
    workerPath: join(root, "python/retrieval-worker.py"), bundlePath: join(root, "retrieval") },
    join(root, "decision"), async () => true);
  const f = await fixture(["review", "images"]);
  const catalog = (await f.catalog.getCatalog(f.ctx)).catalog;
  try {
    await pair.ready!(new AbortController().signal, catalog);
    const requests = async (): Promise<Request[]> => (await readFile(join(root, "requests.jsonl"), "utf8"))
      .trim().split("\n").map(line => JSON.parse(line));
    const first = await requests();
    expect(first.map(request => request.requestId)).toEqual(["startup-retrieval", "startup-decider", "catalog-startup-0"]);
    expect(first[2]!.candidates.map(candidate => candidate.description)).toEqual(
      catalog.entries.filter(entry => entry.automatic_eligible).map(entry => entry.description));
    await pair.ready!(new AbortController().signal, catalog);
    expect(await requests()).toEqual(first);
    await pair.ready!(new AbortController().signal, { ...catalog, catalog_sha256: bytesDigest("changed-live-catalog") });
    expect((await requests()).map(request => request.requestId)).toEqual([
      "startup-retrieval", "startup-decider", "catalog-startup-0", "catalog-startup-0"]);
  } finally { pair.shortlister.close(); }
}, 20000);
