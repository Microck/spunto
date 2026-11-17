import { expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { compactSourceSetBinding, digest, bytesDigest as digestBytes, uncertain, RouterError, isRetrieval, type Bundle, type InferenceBundle, type RetrievalBundle, type Catalog, type Reason, type Request, type Response } from "../src/protocol.ts";
import type { Scorer } from "../src/router.ts";
import schema from "../schemas/model-bundle.schema.json";
import { authorizeCollection, evaluationOutcome, observeRoute, validateObservation, summarize, scoreObservations,
  type BoundCase, type Case, type Design, type Plan, type Observation } from "../src/router-evaluation.ts";

const hash = digest("inert-fixture-not-authority");
const identity = { target_sha256: hash, rubric_sha256: hash, schema_sha256: hash,
  annotation_sha256: hash, receipt_sha256: hash, catalog_sha256: hash, source_dataset_sha256: hash,
  dataset_schema_sha256: hash, source_contract_sha256: hash, role_plan_sha256: hash,
  identity_map_sha256: hash, projection_sha256: hash };
const memberHash = digest(identity);
const sourceSetHash = digest({ purpose: "local-canonical-integration", fixture_only: true, members: [identity] });
const bundle: Bundle = {
  schema_version: 1, model: { type: "window_bag", base: "fixture", revision: "fixture" },
  graphs: schema.$defs.decision.properties.graphs.const, tokenizer_file: "tokenizer.json", model_config_file: "config.json",
  files: [], pair_format: "intent-action-window-bag-v2", geometry: schema.$defs.decision.properties.geometry.const,
  selection: { lower_logit: "0", upper_logit: "1", max_optional: 6, validation_sha256: hash, fit_split: "validation" },
  limits: { max_candidates: 32, max_line_bytes: 131072, microbatch: 4, request_timeout_ms: 750, startup_timeout_ms: 10000 },
  calibration: null, provenance: { dataset_sha256: hash, split_sha256: hash, recipe_sha256: hash, runtime_revision: "fixture",
    contextual_binding: { version: 1, target_sha256: hash, catalog_sha256: hash, source_set_sha256: sourceSetHash,
      runtime_revision: hash, approval_sha256: hash, purpose: "local-canonical-integration", fixture_only: true } },
  scorer_sha256: hash, selector_sha256: hash, calibration_sha256: null, publication_allowed: true,
};
type FixtureScores = Partial<Record<string, number>>;
class FixedScorer implements Scorer {
  readonly bundle: InferenceBundle;
  constructor(readonly values: FixtureScores, readonly failure: Reason | null = null, manifest: InferenceBundle = bundle) {
    this.bundle = manifest;
  }
  async verify() {}
  close() {}
  async score(request: Request): Promise<Response> {
    if (this.failure) throw new RouterError(this.failure);
    const scores = request.candidates.map(c => ({ id: c.id, score: this.values[c.id] ?? -1 }));
    if (isRetrieval(this.bundle)) return { ...uncertain(request, "inadequate_candidates"), scores: {
      kind: "ranking", scorerDigest: this.bundle.scorer_sha256, calibrationDigest: null, values: scores } };
    const selectedIds = [...new Set([...request.requiredIds, ...scores.filter(s => s.score >= 1).map(s => s.id)])].sort();
    const status = selectedIds.length ? "selected" : request.candidateCoverage === "full" ? "none" : "uncertain";
    return { ...uncertain(request, "insufficient_evidence"), status, selectedIds,
      reasonCode: status === "selected" ? "matched" : status === "none" ? "no_match" : "inadequate_candidates",
      scores: { kind: "ranking", scorerDigest: hash, calibrationDigest: null, values: scores } };
  }
}
function fixture(size = 40) {
  const ids = [...Array.from({ length: size }, (_, i) => `a${String(i).padStart(2, "0")}`), "unslop", "emil-unslop-writing"].sort();
  const catalog: Catalog = { schema_version: 1, catalog_sha256: hash, captured_at: "2026-01-01T00:00:00Z",
    loader_contract: "fixture", eligibility_sha256: hash, entries: ids.map(id => ({ runtime_id: id,
      declared_name: null, description: `Guide ${id}`, content_sha256: hash, routing_sha256: hash,
      resource_tree_sha256: null, source_kind: "fixture", source_ref: "fixture", hidden: false,
      automatic_eligible: true, explicit_eligible: true, family_id: null })) };
  const item: BoundCase = { id: "fixture", source_member_sha256: memberHash, record_sha256: hash, explicit_ids: [], runtime: {
    version: 1, catalogDigest: hash, policyDigest: hash, event: "context", intent: "Repair the requested component", action: null,
    requiredIds: [], excludedIds: [], candidateCoverage: "full",
    candidates: catalog.entries.map(e => ({ id: e.runtime_id, name: e.runtime_id, description: e.description, contentDigest: hash })) } };
  const label: Case = { id: item.id, raw_role: "test", role: "temporal_test", stratum: "general", fit: false,
    diagnostic: false, membership: "known-needed", contextual_ids: ["a00"], negative_ids: ["a01"],
    unknown_ids: ids.filter(id => !["a00", "a01", "unslop", "emil-unslop-writing"].includes(id)),
    family_mask_ids: [], intent_group_authority: null, source_member_sha256: memberHash };
  return { catalog, item, label };
}
async function run(values: FixtureScores, failure: Reason | null = null) {
  const f = fixture();
  const observation = await observeRoute({ ...f, scorers: { shortlister: new FixedScorer(values), decider: new FixedScorer(values, failure) },
    baseline: "candidate", planSha256: hash, startupMs: 0, signal: new AbortController().signal });
  const complete = await validateObservation(f.catalog, f.item, observation);
  return { ...f, observation, complete, outcome: evaluationOutcome(observation.response, f.label, [], complete) };
}

test("Python and TypeScript preserve semantic identities across JSON key order", () => {
  const value = { text: "café", nested: { z: [0, null, true], a: "original" } };
  const pythonDigest = execFileSync("python3", ["-B", "-c",
    "import importlib.util,json,sys; s=importlib.util.spec_from_file_location('binder',sys.argv[1]); " +
    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.canonical_hash(json.load(sys.stdin)))",
    `${import.meta.dir}/../python/router-worker.py`],
    { input: JSON.stringify(value), encoding: "utf8" }).trim();
  expect(pythonDigest).toBe(digest(value));
  expect(pythonDigest).toBe(digest({ nested: { a: "original", z: [0, null, true] }, text: "café" }));
});

test("fractional scores replay faithfully and changed scores reject", async () => {
  const r = await run({ a00: 1.25, a02: 1.5, a01: -0 });
  expect(r.outcome.rankedTop1Id).toBe("a02");
  expect(r.outcome.correctness).toBe("unknown");
  const roundtrip = JSON.parse(JSON.stringify(r.observation));
  expect(await validateObservation(r.catalog, r.item, roundtrip)).toBe(true);
  const changed = structuredClone(r.observation);
  changed.response.scores!.values.find(value => value.id === "a00")!.score = 1.375;
  await expect(validateObservation(r.catalog, r.item, changed)).rejects.toThrow("route_replay_mismatch");
});

test("production stages cover the full universe rather than annotated proposals", async () => {
  const r = await run({ a00: 2 });
  const retrieval = r.observation.stages.filter(s => s.stage === "shortlister" && s.operation === "score");
  expect(retrieval.map(s => s.request!.candidates.length)).toEqual([32, 10]);
  expect(r.observation.request.candidates.length).toBe(8);
  expect(r.outcome.correctness).toBe("correct");
  expect(r.complete).toBe(true);
  const incomplete = structuredClone(r.item); incomplete.runtime.candidates.pop();
  await expect(validateObservation(r.catalog, incomplete, r.observation)).rejects.toThrow("incomplete_runtime_universe");
});

test("ranked identity and ties cannot become selectedIds array order", async () => {
  const ranked = await run({ a00: 2, a02: 3 });
  expect(ranked.observation.response.selectedIds).toEqual(["a00", "a02"]);
  expect(ranked.outcome.rankedTop1Id).toBe("a02");
  expect(ranked.outcome.correctness).toBe("unknown");
  const tied = await run({ a00: 3, a02: 3 });
  expect(tied.outcome.rankedTop1Id).toBe("a00");
});

test("unknown top1 lowers precision and unknown memberships do not disappear", async () => {
  const r = await run({ a02: 2 });
  const missing: Case = { ...r.label, id: "unknown-abstention", membership: "unknown", contextual_ids: [] };
  const summary = summarize([r.label, missing], new Map([[r.label.id, r.outcome]]));
  expect(summary.precisionBounds).toEqual([0, 1]);
  expect(summary.coverageBounds).toEqual([0.5, 1]);
  expect(summary.knownNeeded).toBe(1); expect(summary.unknownMembership).toBe(1);
  expect(summary.routingGate).toBe(false);
});

test("neutral-only selection is neither learned firing nor needed coverage", async () => {
  const r = await run({ unslop: 3, "emil-unslop-writing": 2 });
  expect(r.outcome.rankedTop1Id).toBe("unslop");
  expect(r.outcome.learnedTop1Id).toBeNull();
  expect(r.outcome.fired).toBe(false);
  expect(summarize([r.label], new Map([[r.label.id, r.outcome]])).knownNeededFireCoverage).toBe(0);
});

test("missing a mandatory runtime ID makes an otherwise correct response unhealthy", async () => {
  const r = await run({ a00: 2 });
  const outcome = evaluationOutcome(r.observation.response, r.label, ["a03"], true);
  expect(outcome.healthy).toBe(false);
});

test.each(["backend_unavailable", "timeout"] as const)("%s cannot pass a no-skill case", async failure => {
  const r = await run({}, failure);
  const none: Case = { ...r.label, membership: "known-none", contextual_ids: [], unknown_ids: [] };
  const outcome = evaluationOutcome(r.observation.response, none, [], r.complete);
  expect(outcome.rawReason).toBe(failure); expect(outcome.noSkillPass).toBe(false);
  const summary = summarize([none], new Map([[none.id, outcome]]));
  expect(summary.noSkillFailures).toBe(1); expect(summary.routingGate).toBe(false);
});

test("a complete shortlist abstention retains its raw reason", async () => {
  const r = await run({});
  const none: Case = { ...r.label, membership: "known-none", contextual_ids: [], unknown_ids: [] };
  const outcome = evaluationOutcome(r.observation.response, none, [], r.complete);
  expect(outcome.rawStatus).toBe("uncertain"); expect(outcome.rawReason).toBe("inadequate_candidates");
  expect(outcome.noSkillPass).toBe(true);
});

test("a scored gray-band abstention passes a no-skill case and misses a needed one", async () => {
  const r = await run({});
  const response = { ...r.observation.response, status: "uncertain" as const, reasonCode: "insufficient_evidence" as const, selectedIds: [] };
  const none: Case = { ...r.label, membership: "known-none", contextual_ids: [], unknown_ids: [] };
  const abstained = evaluationOutcome(response, none, [], r.complete);
  expect(abstained.healthy).toBe(true); expect(abstained.noSkillPass).toBe(true);
  const needed = evaluationOutcome(response, r.label, [], r.complete);
  expect(needed.fired).toBe(false);
  expect(summarize([r.label], new Map([[r.label.id, needed]])).knownNeededFireCoverage).toBe(0);
  expect(evaluationOutcome({ ...response, scores: null }, none, [], r.complete).noSkillPass).toBe(false);
});

test("missing stages, altered context and hidden retrieval truncation reject", async () => {
  const r = await run({ a00: 2 });
  for (const mutate of [
    (o: typeof r.observation) => { o.stages.splice(0, 1); },
    (o: typeof r.observation) => { o.stages[0]!.request!.intent = "foreign task"; },
    (o: typeof r.observation) => { o.stages[0]!.request!.candidates.pop(); },
  ]) {
    const changed = structuredClone(r.observation); mutate(changed);
    await expect(validateObservation(r.catalog, r.item, changed)).rejects.toThrow();
  }
});

test("explicit diagnostics cannot inflate automatic coverage", async () => {
  const r = await run({ a00: 2 });
  const summary = summarize([{ ...r.label, diagnostic: true, membership: "diagnostic" }], new Map([[r.label.id, r.outcome]]));
  expect(summary.diagnostics).toBe(1); expect(summary.fires).toBe(0); expect(summary.knownNeeded).toBe(0);
});


function plans(r: { catalog: Catalog; label: Case; item: BoundCase }) {
  const source: Design["source_set"] = {
    declaration_sha256: hash, source_set_sha256: sourceSetHash,
    declaration: { format: "contextual-source-set-v2", purpose: "local-canonical-integration", fixture_only: true,
      members: [{ identity, reference: { receipt: "fixture-only" } }] },
    members: [{ member_sha256: memberHash, identity, source_hashes: { fixture: hash } }],
    records: [r.label, { ...r.label, id: "train-fixture", role: "train" }].map(c => ({
      id: c.id, member_sha256: memberHash, role: c.role, record_sha256: hash,
      source: { fixture_only: true }, authored: { fixture_only: true }, pair_ids: ["fixture-pair"] })),
  };
  const controls = { source_hashes: { fixture: hash }, member_sha256: digest({ fixture: hash }), records: [] };
  const design: Design = { format: "prospective-contextual-source-set-evaluation-v2", source_set: source,
    cases: [r.label], neutral_ids: ["unslop", "emil-unslop-writing"],
    required_roles: ["fresh_test"],
    controls, fresh: structuredClone(controls), sensitivity: { excluded_ids: [], confirmed: [], uncertain: [], source_hashes: {} } };
  const plan: Plan = { format: "bound-contextual-source-set-evaluation-v2", design_sha256: digest(design),
    contextual_binding: { ...bundle.provenance.contextual_binding,
      members: [{ member_sha256: memberHash, approval_sha256: hash, identity }] }, source_set: source, controls, fresh: design.fresh,
    catalog: r.catalog, cases: [r.item], execution_authorized: false };
  return { design, plan };
}

test("composed membership, training exclusion and missing metadata cannot be hidden", async () => {
  for (const mutation of ["member", "train", "fit", "metadata", "proposal"] as const) {
    const { design, plan } = structuredClone(plans(fixture()));
    if (mutation === "member") plan.contextual_binding.members[0]!.identity.rubric_sha256 = digest("foreign");
    else if (mutation === "train") design.cases.push({ ...design.cases[0]!, id: "train-fixture", role: "train" });
    else if (mutation === "fit") design.cases[0]!.fit = true;
    else if (mutation === "metadata") plan.source_set.records[0]!.authored = null;
    else plan.format = "contextual-source-set-evaluation-proposal-v1";
    plan.design_sha256 = digest(design);
    await expect(scoreObservations(design, plan, [])).rejects.toThrow();
  }
  expect(() => summarize([{ ...fixture().label, role: "train" }], new Map())).toThrow("train_not_evaluation");
});

test("missing execution authority rejects before any scorer access", async () => {
  class UnusedScorer extends FixedScorer {
    accesses = 0;
    override async verify() { this.accesses++; }
    override async score(request: Request) { this.accesses++; return super.score(request); }
  }
  const scorer = new UnusedScorer({});
  await expect(authorizeCollection({
    design: `${import.meta.dir}/design.json`, plan: `${import.meta.dir}/absent-plan.json`,
    designSha256: hash,
    reference: `${import.meta.dir}/absent-reference.json`, approvalSha256: hash,
    catalog: `${import.meta.dir}/absent-catalog.json`, proofRoot: `${import.meta.dir}/absent-proof`,
    executionApproval: `${import.meta.dir}/absent-approval.json`, executionApprovalSha256: hash,
  }, "candidate", { shortlister: scorer, decider: scorer })).rejects.toThrow();
  expect(scorer.accesses).toBe(0);
});

test("an unbound proposal cannot enter the saved-observation evaluator", async () => {
  const { design, plan } = plans(fixture());
  const proposal = { ...plan, format: "unadmitted-contextual-proposal-v1" };
  await expect(scoreObservations(design, proposal as Plan, [])).rejects.toThrow("plan_design_binding");
});

async function qualificationFixture(values?: (label: Case, baseline: "installed" | "candidate") => FixtureScores, neededCount = 1) {
  const f = fixture();
  const { design, plan } = plans(f);
  design.cases = ["fresh_test", "validation", "unseen_family_test", "temporal_test"].flatMap(role => [
    ...Array.from({ length: neededCount }, (_, index) => ({ ...f.label, id: role + "-needed-" + index, role })),
    { ...f.label, id: role + "-none", role, membership: "known-none" as const,
      contextual_ids: [], negative_ids: f.catalog.entries.map(e => e.runtime_id), unknown_ids: [] },
  ]);
  design.source_set.records = design.cases.map(label => ({ ...design.source_set.records[0]!, id: label.id, role: label.role }));
  plan.cases = design.cases.map(label => ({ ...f.item, id: label.id }));
  plan.design_sha256 = digest(design);
  const payloads = {
    "parity.json": JSON.stringify({ comparison: { passed: true, tolerance: .0001, max_absolute_logit_error: .00001,
      decision_changes: 0, ranking_pair_changes: 0 } }),
    "evaluation.json": JSON.stringify({ drift: { validation: { max_absolute_logit_error: .00001, decision_changes: 0 } } }),
    "recipe.json": JSON.stringify({ export: { variant: "fp32" } }),
    "validation.json": JSON.stringify({ fit_split: "validation" }),
  };
  const comparison = { passed: true, tolerance: .0001, max_absolute_embedding_error: .00001,
    max_absolute_pooled_error: .00001, ranking_pair_changes: 0 };
  const retrievalTraining = { role: "retrieval", weights: "unchanged_pretrained", fit_split: null,
    training_data_used: false, contextual_binding: compactSourceSetBinding(plan.contextual_binding),
    dataset_sha256: hash, split_sha256: hash, training_recipe_sha256: null };
  const retrievalReports = {
    "parity.json": JSON.stringify({ role: "retrieval", nonprivate: true, cases: [{}, {}, {}, {}], comparison }),
    "recipe.json": JSON.stringify({ ...retrievalTraining, export: { variant: "fp32" } }),
    "evaluation.json": JSON.stringify({ ...retrievalTraining, comparison }),
    "validation.json": JSON.stringify({ ...retrievalTraining, comparison }),
  };
  const retrievalBundle: RetrievalBundle = { schema_version: 1, role: "retrieval",
    model: schema.$defs.retrieval.properties.model.const, graphs: schema.$defs.retrieval.properties.graphs.const,
    pooling: schema.$defs.retrieval.properties.pooling.const, tokenizer_file: "tokenizer.json", model_config_file: "config.json",
    embedding_table: "embeddings.json", limits: bundle.limits, selection: { top_k: 8, tie_break: "id" },
    provenance: { contextual_binding: compactSourceSetBinding(plan.contextual_binding), catalog_sha256: hash, recipe_sha256: hash,
      runtime_revision: hash, runtime_files: {}, dataset_sha256: hash, split_sha256: hash,
      weights: "unchanged_pretrained", training_recipe_sha256: null, training_report_sha256: null, fit_split: null, training_data_used: false },
    scorer_sha256: hash, selector_sha256: hash, publication_allowed: true,
    files: Object.entries(retrievalReports).map(([path, content]) => ({ path, sha256: digestBytes(content), bytes: Buffer.byteLength(content) })) };
  const candidateBundle = { ...bundle, files: Object.entries(payloads).map(([path, content]) =>
    ({ path, sha256: digestBytes(content), bytes: Buffer.byteLength(content) })) };
  const observations = [];
  for (const baseline of ["installed", "candidate"] as const) for (const label of design.cases) {
    const scores = values?.(label, baseline) ?? (label.membership === "known-needed" ? { a00: 2 } : {});
    const scorer = new FixedScorer(scores, null, candidateBundle);
    observations.push(await observeRoute({ catalog: f.catalog, item: plan.cases.find(c => c.id === label.id)!,
      scorers: { shortlister: new FixedScorer(scores, null, retrievalBundle), decider: scorer }, baseline, planSha256: digest(plan),
      startupMs: 0, signal: new AbortController().signal }));
  }
  return { design, plan, observations, evidence: { candidateExports: { shortlister: retrievalReports, decider: payloads } } };
}

test("a perfect router qualifies against the same-plan installed baseline", async () => {
  const f = await qualificationFixture();
  const report = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(report.qualified).toBe(true);
  expect(report.failingItems).toEqual([]);
});

test("development errors, false fires and missing observations do not gate fresh qualification", async () => {
  const f = await qualificationFixture((label, baseline) =>
    baseline === "candidate" && label.role !== "fresh_test" || label.membership === "known-needed" ? { a00: 2 } : {});
  const observations = f.observations.filter(o => o.id !== "temporal_test-needed-0");
  for (const observation of observations) if (observation.id === "validation-none") observation.elapsed_ms = 751;
  const report = await scoreObservations(f.design, f.plan, observations, f.evidence);
  expect(report.qualified).toBe(true);
  expect(report.failingItems).toEqual([]);
  for (const view of report.baselines[1]!.views) {
    expect(view.roles.map(role => role.role)).toEqual(["fresh_test"]);
    expect(view.developmentDiagnostics.find(role => role.role === "validation")).toMatchObject({ falseFires: 1, failures: 1 });
    expect(view.developmentDiagnostics.find(role => role.role === "temporal_test")).toMatchObject({ missing: 1 });
  }
});

test("development roles cannot replace the fresh qualification role", async () => {
  const f = await qualificationFixture();
  f.design.required_roles = ["validation", "unseen_family_test", "temporal_test"];
  f.plan.design_sha256 = digest(f.design);
  await expect(scoreObservations(f.design, f.plan, [])).rejects.toThrow("required_roles");
});

test("a false fire reports the exact failing gates in both views", async () => {
  const f = await qualificationFixture((label, baseline) =>
    label.membership === "known-needed" || baseline === "candidate" && label.id === "fresh_test-none" ? { a00: 2 } : {});
  const report = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(report.qualified).toBe(false);
  expect(report.failingItems.map(item => [item.view, item.role, item.check])).toEqual(
    ["full", "sensitivity"].flatMap(view => ["precision_lower_bound", "false_fires", "no_skill_failures",
      "installed_precision_comparison", "installed_false_fires_comparison"].map(check => [view, "fresh_test", check])));
});

test("sensitivity excludes only pinned reused cases and still requires known-none evidence", async () => {
  const f = await qualificationFixture((label, baseline) => label.membership === "known-none" ? {} :
    baseline === "candidate" && label.id === "fresh_test-needed-1" ? { a01: 2 } : { a00: 2 }, 2);
  f.design.sensitivity.excluded_ids = ["fresh_test-needed-1"];
  f.plan.design_sha256 = digest(f.design);
  for (const observation of f.observations) observation.plan_sha256 = digest(f.plan);
  const report = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  const candidate = report.baselines[1]!;
  expect(candidate.views[0]!.roles[0]!.precisionBounds[0]).toBe(.5);
  expect(candidate.views[1]!.roles[0]!.precisionBounds[0]).toBe(1);
  expect(report.qualified).toBe(false);
  f.design.sensitivity.excluded_ids = ["fresh_test-none"];
  f.plan.design_sha256 = digest(f.design);
  for (const observation of f.observations) observation.plan_sha256 = digest(f.plan);
  const noControls = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(noControls.failingItems).toContainEqual({ view: "sensitivity", role: "fresh_test",
    check: "known_none_presence", actual: 0, expected: ">0" });
});

test("passing absolute precision and coverage cannot hide an installed-router regression", async () => {
  const f = await qualificationFixture((label, baseline) => label.membership === "known-none" ? {} :
    baseline === "candidate" && label.id.endsWith("-0") ? { a01: 2 } : { a00: 2 }, 20);
  const report = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(report.baselines[1]!.views.every(view => view.roles.every(role => role.routingGate))).toBe(true);
  expect(report.failingItems.every(item => item.check === "installed_precision_comparison")).toBe(true);
  expect(report.failingItems).toHaveLength(2);
  expect(report.qualified).toBe(false);
});

test("supplemental unknown labels apply to both routers but cannot replace existing gold", async () => {
  const f = await qualificationFixture(label => label.membership === "known-needed" ? { a02: 2 } : {});
  const before = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(before.qualified).toBe(false);
  const labels = { format: "contextual-supplemental-labels-v1", design_sha256: digest(f.design),
    purpose: "evaluation_only", fit_allowed: false, train_allowed: false,
    labels: f.design.cases.filter(c => c.membership === "known-needed").map(c => ({ case_id: c.id,
      runtime_id: "a02", status: "contextual", opinion_sha256: [hash, digest("second independent opinion")],
      adjudication_sha256: digest("blinded adjudication") })) };
  const content = JSON.stringify(labels);
  const evidence = { ...f.evidence, supplementalLabels: { content, sha256: digestBytes(content) } };
  const after = await scoreObservations(f.design, f.plan, f.observations, evidence);
  expect(after.qualified).toBe(true);
  expect(after.baselines.every(baseline => baseline.views.every(view => view.roles.every(role => role.precisionBounds[0] === 1)))).toBe(true);
  expect(f.design.cases[0]!.contextual_ids).toEqual(["a00"]);
  labels.labels[0]!.runtime_id = "a01";
  const conflicting = JSON.stringify(labels);
  await expect(scoreObservations(f.design, f.plan, f.observations, { ...f.evidence,
    supplementalLabels: { content: conflicting, sha256: digestBytes(conflicting) } })).rejects.toThrow("precedence");
  await expect(scoreObservations(f.design, f.plan, f.observations, { ...f.evidence,
    supplementalLabels: { content, sha256: hash } })).rejects.toThrow("supplemental_label_pin");
});

test("missing observations and candidate parity remain explicit qualification failures", async () => {
  const f = await qualificationFixture();
  const observations = f.observations.filter(o => o.baseline !== "candidate" || o.id !== "fresh_test-needed-0");
  const report = await scoreObservations(f.design, f.plan, observations);
  expect(report.qualified).toBe(false);
  expect(report.failingItems).toContainEqual({ check: "missing_observations", baseline: "candidate", caseIds: ["fresh_test-needed-0"] });
  expect(report.failingItems.filter(item => item.check === "missing_candidate_export_report").map(item => item.stage)).toEqual(["shortlister", "decider"]);
  const missing = { ...f.plan, cases: [] };
  await expect(scoreObservations(f.design, missing, [])).rejects.toThrow("all_original_cases_required");
});

test("FP32 parity must meet the tolerance even when a pinned report claims passed", async () => {
  const f = await qualificationFixture();
  const content = JSON.stringify({ role: "retrieval", nonprivate: true, cases: [{}, {}, {}, {}], comparison: { passed: true,
    tolerance: .0001, max_absolute_embedding_error: .00011, max_absolute_pooled_error: .00001, ranking_pair_changes: 0 } });
  f.evidence.candidateExports.shortlister["parity.json"] = content;
  await expect(scoreObservations(f.design, f.plan, f.observations, f.evidence)).rejects.toThrow("candidate_export_report_pin");
  for (const observation of f.observations) {
    const entry = observation.bundles.shortlister.files.find(file => file.path === "parity.json")!;
    entry.sha256 = digestBytes(content); entry.bytes = Buffer.byteLength(content);
  }
  const report = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(report.qualified).toBe(false);
  expect(report.failingItems).toEqual([
    { check: "fp32_parity_gate", stage: "shortlister" },
  ]);
});

test("lower coverage than installed passes when absolute gates hold; per-request latency stays enforced", async () => {
  const f = await qualificationFixture((label, baseline) => label.membership === "known-none" ||
    baseline === "candidate" && label.id.endsWith("-0") ? {} : { a00: 2 }, 4);
  const report = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(report.baselines[1]!.views.every(view => view.roles.every(role => role.routingGate))).toBe(true);
  expect(report.failingItems).toEqual([]);
  const observation = f.observations.find(o => o.baseline === "candidate" && o.id === "fresh_test-none")!;
  observation.elapsed_ms = 750.01;
  const slow = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(slow.failingItems).toContainEqual({ check: "unhealthy_observations", baseline: "candidate", caseIds: ["fresh_test-none"] });
  observation.elapsed_ms = 750; observation.startup_ms = 10000;
  const boundary = await scoreObservations(f.design, f.plan, f.observations, f.evidence);
  expect(boundary.failingItems.some(item => item.check === "unhealthy_observations")).toBe(false);
});

test("a retrieval bundle bound to a different source plan is refused", async () => {
  const f = await qualificationFixture();
  for (const observation of f.observations.filter(o => o.baseline === "candidate")) {
    const shortlister = observation.bundles.shortlister;
    if (!isRetrieval(shortlister)) throw new Error("expected retrieval fixture");
    observation.bundles.shortlister = { ...shortlister,
      provenance: { ...shortlister.provenance,
        contextual_binding: { ...f.plan.contextual_binding, approval_sha256: digest("different-plan") } } };
  }
  await expect(scoreObservations(f.design, f.plan, f.observations, f.evidence)).rejects.toThrow("bundle_binding_or_limits");
});

test("whole-route deadline failures preserve timeout stage evidence for qualification replay", async () => {
  const f = fixture();
  const delayed: Scorer = { bundle, async verify() {}, close() {}, async score(request, signal) {
    // Exercise the platform timer and AbortSignal contract without mocked clocks.
    const pending = Promise.withResolvers<unknown>();
    const timer = setTimeout(() => { void new FixedScorer({}).score(request).then(pending.resolve, pending.reject); }, 400);
    const abort = () => pending.reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    try { return await pending.promise; }
    finally { clearTimeout(timer); signal.removeEventListener("abort", abort); }
  } };
  const observation = await observeRoute({ ...f, scorers: { shortlister: delayed, decider: new FixedScorer({}) },
    baseline: "candidate", planSha256: hash, startupMs: 0, signal: new AbortController().signal });
  expect(observation.response.reasonCode).toBe("timeout");
  expect(observation.stages.at(-1)!.error).toBe("timeout");
  expect(await validateObservation(f.catalog, f.item, observation)).toBe(false);
});

test("a deadline that expires during final verification replays as an unhealthy terminal result", async () => {
  const f = await run({ a00: 2 });
  f.observation.elapsed_ms = 750.01;
  f.observation.response = uncertain(f.observation.request, "timeout");
  expect(await validateObservation(f.catalog, f.item, f.observation)).toBe(false);
});

