// Current composed evaluation. Training permission never authorizes task evaluation.
import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import { join } from "node:path";
import { route, type RouteContract, type Scorer, type Scorers } from "./router.ts";
import { assertRequest, bytesDigest, compare, compactSourceSetBinding, contextDigest, digest, errorCode, isRetrieval, uncertain,
  parseStrict, RouterError, type ContextualBinding, type Bundle, type InferenceBundle, type Catalog, type Digest, type Reason, type Request, type Response } from "./protocol.ts";

export const NEUTRAL = ["unslop", "emil-unslop-writing"];
export interface Case {
  id: string; raw_role: string; role: string; stratum: string; fit: boolean; diagnostic: boolean;
  membership: "known-needed" | "known-none" | "unknown" | "diagnostic";
  contextual_ids: string[]; negative_ids: string[]; unknown_ids: string[];
  family_mask_ids: string[]; intent_group_authority: null;
  source_member_sha256: Digest; evaluation_only?: true;
}
export interface Design {
  format: "prospective-contextual-source-set-evaluation-v2"; source_set: SourceInventory;
  cases: Case[]; neutral_ids: string[]; required_roles: string[];
  controls: EvaluationSource; fresh: EvaluationSource;
  sensitivity: { excluded_ids: string[]; confirmed: unknown[]; uncertain: unknown[]; source_hashes: Record<string, Digest> };
}
export interface EvaluationSource {
  source_hashes: Record<string, Digest>; member_sha256: Digest;
  records: { example_id: string; split: string; purpose: "evaluation_only"; fit_allowed: false; train_allowed: false;
    catalog_sha256: Digest; policy_sha256: Digest; native_messages: unknown[];
    input: { request: string; action: null; candidate_ids: string[]; candidate_coverage: "full";
      required_ids: string[]; excluded_ids: string[]; explicit_ids: string[] } }[];
}
/** Raw export payloads are checked against the candidate manifests, not trusted summary flags. */
export interface QualificationEvidence {
  candidateExports?: Record<Stage["stage"], Record<"parity.json" | "evaluation.json" | "recipe.json" | "validation.json", string>>;
  supplementalLabels?: { sha256: Digest; content: string };
}
interface SupplementalLabels {
  format: "contextual-supplemental-labels-v1"; design_sha256: Digest; purpose: "evaluation_only";
  fit_allowed: false; train_allowed: false;
  labels: { case_id: string; runtime_id: string; status: "contextual" | "nonapplicable" | "response_only" | "unresolved";
    opinion_sha256: [Digest, Digest]; adjudication_sha256: Digest }[];
}
export type Runtime = Pick<Request, "version" | "catalogDigest" | "policyDigest" | "event" | "intent" |
  "action" | "candidates" | "candidateCoverage" | "requiredIds" | "excludedIds">;
export interface BoundCase {
  id: string; runtime: Runtime; explicit_ids: string[]; record_sha256: Digest; source_member_sha256: Digest;
}
type Binding = ContextualBinding;
export interface SourceInventory {
  declaration_sha256: Digest; source_set_sha256: Digest;
  declaration: { format: string; purpose: string; fixture_only: boolean;
    members: { identity: Binding["members"][number]["identity"]; reference: Record<string, unknown> }[] };
  members: { member_sha256: Digest; identity: Binding["members"][number]["identity"];
    source_hashes: Record<string, Digest> }[];
  records: { id: string; member_sha256: Digest; role: string; record_sha256: Digest;
    source: unknown; authored: unknown; pair_ids: string[] }[];
}
export interface Plan {
  format: string; design_sha256: Digest; contextual_binding: ContextualBinding;
  source_set: SourceInventory; controls: EvaluationSource; fresh: EvaluationSource; catalog: Catalog; cases: BoundCase[]; execution_authorized: false;
}
export interface Stage {
  stage: "shortlister" | "decider"; operation: "score" | "verify"; request: Request | null;
  raw_response: unknown; error: Reason | null;
}
export interface Observation {
  id: string; baseline: "installed" | "candidate"; plan_sha256: Digest;
  request: Request; response: Response; stages: Stage[];
  elapsed_ms: number; startup_ms: number; aborted: boolean; current: boolean;
  bundles: { shortlister: InferenceBundle; decider: InferenceBundle };
}
export interface Outcome {
  rawStatus: Response["status"]; rawReason: Reason; selectedIds: string[];
  rankedSelectedIds: string[]; rankedTop1Id: string | null; learnedSelectedIds: string[];
  learnedTop1Id: string | null; neutralSelectedIds: string[]; rawScores: Response["scores"];
  fired: boolean; correctness: "not-fired" | "correct" | "incorrect" | "unknown";
  healthy: boolean; unknownSelectedIds: string[]; noSkillPass: boolean; missingRequiredIds: string[];
}
function requireContract(value: unknown, code: string): asserts value {
  if (!value) throw new Error(`contextual_evaluation:${code}`);
}
/** Structural equality for validated JSON, including fractional scores and signed zero. */
function equal(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length &&
      a.every((value, index) => equal(value, b[index]));
  }
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length &&
    keys.every(key => Object.hasOwn(right, key) && equal(left[key], right[key]));
}

export interface AuthorizationInputs {
  design: string; plan: string; reference: string; approvalSha256: string; catalog: string;
  designSha256: string;
  proofRoot: string; executionApproval: string; executionApprovalSha256: string;
}
/** Re-admit the actual source records, not a claimed verification receipt. */
export function loadAuthorized(args: AuthorizationInputs): { design: Design; plan: Plan } {
  const verified = execFileSync("python3", ["-B", `${import.meta.dir}/../python/contextual-evaluation.py`, "verify", "--design", args.design,
    "--plan", args.plan, "--reference", args.reference, "--approval-sha256", args.approvalSha256,
    "--design-sha256", args.designSha256,
    "--catalog", args.catalog, "--proof-root", args.proofRoot, "--execution-approval", args.executionApproval,
    "--execution-approval-sha256", args.executionApprovalSha256], { stdio: ["ignore", "pipe", "pipe"] });
  const plan = parseStrict(readFileSync(args.plan, "utf8")) as Plan;
  const design = parseStrict(readFileSync(args.design, "utf8")) as Design;
  requireContract(JSON.parse(verified.toString()).plan_sha256 === digest(plan) &&
    plan.design_sha256 === digest(design), "verified_bytes_changed");
  return { design, plan };
}

// Evaluation-only reference, pinned to the installed observation collection.
const INSTALLED_ROUTE: RouteContract = Object.freeze({ shortlistSize: 16, timeoutMs: null });

function routeInput(catalog: Catalog, item: BoundCase, scorers: Scorers, signal: AbortSignal, current: boolean) {
  const runtime = item.runtime;
  return { catalog, policy: { requiredIds: runtime.requiredIds, excludedIds: runtime.excludedIds,
    explicitIds: item.explicit_ids, policyDigest: runtime.policyDigest, conflict: false, unsupported: false },
    intent: runtime.intent, action: runtime.action, event: runtime.event,
    sessionKey: `contextual-evaluation:${item.id}`, generation: 0, scorers, signal, isCurrent: () => current };
}
function validateUniverse(catalog: Catalog, item: BoundCase) {
  const runtime = item.runtime;
  const eligible = catalog.entries.filter(e => (e.automatic_eligible && !runtime.excludedIds.includes(e.runtime_id)) ||
    runtime.requiredIds.includes(e.runtime_id));
  const candidates = eligible.map(e => ({ id: e.runtime_id, name: e.runtime_id,
    description: e.description, contentDigest: e.content_sha256 })).sort((a, b) => compare(a.id, b.id));
  requireContract(runtime.candidateCoverage === "full" && runtime.catalogDigest === catalog.catalog_sha256 &&
    equal(runtime.candidates, candidates), "incomplete_runtime_universe");
}

/** Transparent recorder around real HostRouter Scorers; raw replies and failures survive. */
export async function observeRoute(args: {
  catalog: Catalog; item: BoundCase; scorers: Scorers; baseline: Observation["baseline"];
  planSha256: Digest; startupMs: number; signal: AbortSignal; current?: boolean;
}): Promise<Observation> {
  validateUniverse(args.catalog, args.item);
  const stages: Stage[] = [];
  const wrap = (stage: Stage["stage"]): Scorer => {
    const scorer = args.scorers[stage];
    return { bundle: scorer.bundle, close: () => scorer.close(),
      async verify() {
        const record: Stage = { stage, operation: "verify", request: null, raw_response: null, error: null };
        stages.push(record);
        try { await scorer.verify(); } catch (error) { record.error = errorCode(error); throw error; }
      },
      async score(request, signal) {
        const record: Stage = { stage, operation: "score", request: structuredClone(request), raw_response: null, error: null };
        stages.push(record);
        try { const raw = await scorer.score(request, signal); record.raw_response = structuredClone(raw); return raw; }
        catch (error) { record.error = errorCode(error); throw error; }
      } };
  };
  const readyStart = performance.now();
  await args.scorers.ready?.(args.signal);
  const startupMs = args.startupMs + (performance.now() - readyStart);
  const start = performance.now();
  const current = args.current ?? true;
  const result = await route(routeInput(args.catalog, args.item,
    { shortlister: wrap("shortlister"), decider: wrap("decider") }, args.signal, current), args.baseline === "installed" ? INSTALLED_ROUTE : undefined);
  return { id: args.item.id, baseline: args.baseline, plan_sha256: args.planSha256, ...result, stages,
    elapsed_ms: performance.now() - start, startup_ms: startupMs, aborted: args.signal.aborted, current,
    bundles: { shortlister: args.scorers.shortlister.bundle, decider: args.scorers.decider.bundle } };
}

function withoutRequestId<T extends { requestId: string }>(value: T) {
  const { requestId, ...rest } = value;
  return rest;
}

/** Replay only saved stage evidence through route(). No model, subprocess or network. */
export async function validateObservation(catalog: Catalog, item: BoundCase, observation: Observation) {
  validateUniverse(catalog, item);
  requireContract(observation.id === item.id && Number.isFinite(observation.elapsed_ms) && observation.elapsed_ms >= 0 &&
    Number.isFinite(observation.startup_ms) && observation.startup_ms >= 0 &&
    typeof observation.aborted === "boolean" && typeof observation.current === "boolean", "observation_identity");
  const identities = new Set<string>();
  let cursor = 0;
  let invalidStage = false;
  function stageCheck(value: unknown, code: string): asserts value {
    if (!value) invalidStage = true;
    requireContract(value, code);
  }
  const replay = (stage: Stage["stage"]): Scorer => ({
    bundle: observation.bundles[stage], close() {},
    async verify() {
      const record = observation.stages[cursor++];
      stageCheck(record?.stage === stage && record.operation === "verify" && record.request === null &&
        record.raw_response === null, "verify_stage");
      if (record.error) throw new RouterError(record.error);
    },
    async score(request) {
      const record = observation.stages[cursor++];
      stageCheck(record?.stage === stage && record.operation === "score" && record.request, "score_stage");
      try { assertRequest(record.request); } catch (error) { invalidStage = true; throw error; }
      stageCheck(record.request.contextDigest === contextDigest(record.request) &&
        !identities.has(record.request.requestId) && equal(withoutRequestId(record.request), withoutRequestId(request)), "stage_request");
      identities.add(record.request.requestId);
      if (record.error) {
        stageCheck(record.raw_response === null, "failure_with_response");
        throw new RouterError(record.error);
      }
      // Preserve malformed raw evidence: production acceptResponse makes the same failure.
      if (typeof record.raw_response !== "object" || record.raw_response === null) return record.raw_response;
      const raw = record.raw_response as Record<string, unknown>;
      if (raw.requestId !== record.request.requestId) return raw;
      return { ...raw, requestId: request.requestId };
    },
  });
  const controller = new AbortController();
  if (observation.aborted) controller.abort();
  const result = await route(routeInput(catalog, item,
    { shortlister: replay("shortlister"), decider: replay("decider") }, controller.signal, observation.current), observation.baseline === "installed" ? INSTALLED_ROUTE : undefined);
  // Replay has no wall-clock delays. A deadline can expire after the last stage
  // succeeds; validate that terminal failure against observed time and the exact
  // canonical timeout response, after replay has checked the entire stage trace.
  const terminalDeadline = observation.baseline === "candidate" && observation.elapsed_ms >= 750 && observation.response.reasonCode === "timeout" &&
    equal(observation.response, uncertain(observation.request, "timeout"));
  requireContract(!invalidStage && cursor === observation.stages.length &&
    equal(withoutRequestId(result.request), withoutRequestId(observation.request)) &&
    (equal(withoutRequestId(result.response), withoutRequestId(observation.response)) || terminalDeadline), "route_replay_mismatch");
  requireContract(observation.response.requestId === observation.request.requestId &&
    observation.request.contextDigest === contextDigest(observation.request), "final_request_binding");
  // A successful replay must actually have reached a successful decider verification.
  const complete = !terminalDeadline && observation.stages.at(-1)?.operation === "verify" && observation.stages.at(-1)?.error === null;
  return complete;
}

export function evaluationOutcome(response: Response, label: Case, requiredIds: string[], complete: boolean): Outcome {
  const scores = new Map(response.scores?.values.map(item => [item.id, item.score]) ?? []);
  const ranked = [...response.selectedIds].sort((a, b) =>
    (scores.get(b) ?? -Infinity) - (scores.get(a) ?? -Infinity) || compare(a, b));
  const learned = ranked.filter(id => !NEUTRAL.includes(id) && !requiredIds.includes(id));
  const top = learned[0] ?? null;
  const correctness = top === null ? "not-fired" : label.contextual_ids.includes(top) ? "correct" :
    label.negative_ids.includes(top) ? "incorrect" : "unknown";
  // Health means the router produced a scored decision in time. A scored gray-band abstention
  // (insufficient_evidence) injects nothing in OMP, so it is judged by its effect: a no-skill pass on
  // known-none cases and a miss (lower coverage) on needed cases. Errors and timeouts stay unhealthy.
  const healthy = complete && requiredIds.every(id => response.selectedIds.includes(id)) &&
    (response.reasonCode === "matched" || response.reasonCode === "no_match" || response.reasonCode === "policy_only" ||
    (response.reasonCode === "inadequate_candidates" || response.reasonCode === "insufficient_evidence") && response.scores !== null);
  return { rawStatus: response.status, rawReason: response.reasonCode, selectedIds: response.selectedIds,
    rankedSelectedIds: ranked, rankedTop1Id: ranked[0] ?? null, learnedSelectedIds: learned, learnedTop1Id: top,
    neutralSelectedIds: ranked.filter(id => NEUTRAL.includes(id)), rawScores: response.scores,
    fired: learned.length > 0, correctness, healthy,
    unknownSelectedIds: learned.filter(id => !label.contextual_ids.includes(id) && !label.negative_ids.includes(id)),
    missingRequiredIds: requiredIds.filter(id => !response.selectedIds.includes(id)),
    noSkillPass: label.membership === "known-none" && learned.length === 0 && healthy };
}
const ratio = (a: number, b: number) => b ? a / b : null;

export function summarize(cases: Case[], outcomes: Map<string, Outcome>) {
  requireContract(cases.every(c => c.role !== "train"), "train_not_evaluation");
  const automatic = cases.filter(c => !c.diagnostic);
  const needed = automatic.filter(c => c.membership === "known-needed");
  const none = automatic.filter(c => c.membership === "known-none");
  const unknown = automatic.filter(c => c.membership === "unknown");
  const fires = automatic.filter(c => outcomes.get(c.id)?.fired);
  const correct = fires.filter(c => outcomes.get(c.id)?.correctness === "correct").length;
  const unknownFires = fires.filter(c => outcomes.get(c.id)?.correctness === "unknown").length;
  const neededFires = needed.filter(c => outcomes.get(c.id)?.fired).length;
  const unknownMemberFires = unknown.filter(c => outcomes.get(c.id)?.fired).length;
  const precision: [number | null, number | null] = [ratio(correct, fires.length), ratio(correct + unknownFires, fires.length)];
  const coverage: [number | null, number | null] = [ratio(neededFires, needed.length + unknown.length - unknownMemberFires),
    ratio(neededFires + unknownMemberFires, needed.length + unknownMemberFires)];
  const missing = automatic.filter(c => !outcomes.has(c.id)).length;
  const failures = automatic.filter(c => outcomes.has(c.id) && !outcomes.get(c.id)!.healthy).length;
  const falseFires = none.filter(c => outcomes.get(c.id)?.fired).length;
  const noSkillFailures = none.filter(c => !outcomes.get(c.id)?.noSkillPass).length;
  return { requests: cases.length, automatic: automatic.length, diagnostics: cases.length - automatic.length,
    knownNeeded: needed.length, knownNone: none.length, unknownMembership: unknown.length,
    fires: fires.length, knownCorrect: correct, unknownTop1: unknownFires, precisionBounds: precision,
    neededFires, knownNeededFireCoverage: ratio(neededFires, needed.length), coverageBounds: coverage,
    missing, failures, falseFires, noSkillFailures,
    routingGate: missing === 0 && failures === 0 && none.length > 0 && noSkillFailures === 0 &&
      precision[0] !== null && precision[0] >= .95 && coverage[0] !== null && coverage[0] >= .75 };
}

function validateSourceInventory(design: Design, plan: Plan) {
  const source = plan.source_set, binding = plan.contextual_binding;
  requireContract(source && equal(source, design.source_set) &&
    source.source_set_sha256 === binding.source_set_sha256 &&
    source.declaration.format === "contextual-source-set-v2" &&
    source.declaration.purpose === binding.purpose && source.declaration.fixture_only === binding.fixture_only &&
    equal(source.members.map(m => ({ member_sha256: m.member_sha256, identity: m.identity })),
      binding.members.map(m => ({ member_sha256: m.member_sha256, identity: m.identity }))) &&
    equal(source.declaration.members.map(m => m.identity), source.members.map(m => m.identity)) &&
    digest({ purpose: binding.purpose, fixture_only: binding.fixture_only,
      members: source.members.map(m => m.identity) }) === binding.source_set_sha256,
    "source_set_binding");
  const members = new Set(binding.members.map(m => m.member_sha256));
  const records = new Map(source.records.map(r => [r.id, r]));
  requireContract(records.size === source.records.length && source.records.every(r =>
    members.has(r.member_sha256) && r.source && r.authored && Array.isArray(r.pair_ids)) &&
    source.members.every(m => digest(m.identity) === m.member_sha256 &&
      Object.keys(m.source_hashes).length > 0), "incomplete_source_metadata");
  const evaluationSources = [plan.controls, plan.fresh];
  requireContract(equal(plan.controls, design.controls) && equal(plan.fresh, design.fresh) &&
    evaluationSources.every(source => digest(source.source_hashes) === source.member_sha256),
    "evaluation_source_binding");
  const evaluationRecords = new Map(evaluationSources.flatMap(source => source.records.map(record =>
    [record.example_id, { record, source }] as const)));
  requireContract(evaluationRecords.size === evaluationSources.reduce((sum, source) => sum + source.records.length, 0) &&
    [...evaluationRecords.values()].every(({ record }) => !records.has(record.example_id) &&
      record.purpose === "evaluation_only" && record.fit_allowed === false && record.train_allowed === false),
    "evaluation_never_fit_or_train");
  requireContract(equal([...source.records.filter(r => r.role !== "train").map(r => r.id), ...evaluationRecords.keys()].sort(),
    design.cases.map(c => c.id).sort()) && design.cases.every(c => {
      if (c.evaluation_only) {
        const entry = evaluationRecords.get(c.id);
        if (!entry || entry.record.split !== c.role || c.fit || c.diagnostic ||
          c.source_member_sha256 !== entry.source.member_sha256) return false;
        return entry.source === plan.fresh ? c.role === "fresh_test" && c.family_mask_ids.length === 0 :
          c.membership === "known-none" && c.contextual_ids.length === 0 && c.unknown_ids.length === 0;
      }
      const record = records.get(c.id);
      return record && c.role !== "train" && record.role === c.role &&
        record.member_sha256 === c.source_member_sha256 && (!c.fit || c.role === "validation");
    }) && plan.cases.every(c => {
      const entry = evaluationRecords.get(c.id);
      if (!entry) return c.source_member_sha256 === records.get(c.id)?.member_sha256;
      const { record } = entry;
      return c.source_member_sha256 === entry.source.member_sha256 && c.record_sha256 === digest(record) &&
        c.runtime.intent === record.input.request && c.runtime.catalogDigest === record.catalog_sha256 &&
        c.runtime.policyDigest === record.policy_sha256 && c.runtime.action === null && c.runtime.event === "context" &&
        equal(c.runtime.requiredIds, []) && equal(c.runtime.excludedIds, []) && equal(c.explicit_ids, []);
    }), "fit_evaluation_role_separation");
}

function supplementalCases(design: Design, pin: QualificationEvidence["supplementalLabels"]) {
  if (!pin) return design.cases;
  requireContract(bytesDigest(pin.content) === pin.sha256, "supplemental_label_pin");
  const supplemental = parseStrict(pin.content) as SupplementalLabels;
  requireContract(supplemental.format === "contextual-supplemental-labels-v1" &&
    supplemental.design_sha256 === digest(design) && supplemental.purpose === "evaluation_only" &&
    supplemental.fit_allowed === false && supplemental.train_allowed === false && Array.isArray(supplemental.labels),
    "supplemental_label_scope");
  const cases = new Map(design.cases.map(c => [c.id, structuredClone(c)]));
  const seen = new Set<string>();
  const isDigest = (value: unknown) => typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
  for (const label of supplemental.labels) {
    const item = cases.get(label.case_id), pair = JSON.stringify([label.case_id, label.runtime_id]);
    requireContract(item && !seen.has(pair) && item.unknown_ids.includes(label.runtime_id) &&
      !item.contextual_ids.includes(label.runtime_id) && !item.negative_ids.includes(label.runtime_id) &&
      !item.family_mask_ids.includes(label.runtime_id) && !NEUTRAL.includes(label.runtime_id) &&
      ["contextual", "nonapplicable", "response_only", "unresolved"].includes(label.status) &&
      label.opinion_sha256?.length === 2 && label.opinion_sha256.every(isDigest) &&
      label.opinion_sha256[0] !== label.opinion_sha256[1] && isDigest(label.adjudication_sha256),
      "supplemental_label_adjudication_or_precedence");
    seen.add(pair);
    if (label.status === "unresolved") continue;
    item.unknown_ids = item.unknown_ids.filter(id => id !== label.runtime_id);
    (label.status === "contextual" ? item.contextual_ids : item.negative_ids).push(label.runtime_id);
  }
  return [...cases.values()];
}

function candidateParity(bundles: Observation["bundles"] | undefined, exports: QualificationEvidence["candidateExports"]) {
  return (["shortlister", "decider"] as const).map(stage => {
    const payloads = exports?.[stage], bundle = bundles?.[stage];
    if (!payloads || !bundle) return { stage, passed: false, reason: "missing_candidate_export_report" };
    for (const path of ["parity.json", "evaluation.json", "recipe.json", "validation.json"] as const) {
      const entry = bundle.files.find(file => file.path === path), content = payloads[path];
      requireContract(entry && typeof content === "string" && bytesDigest(content) === entry.sha256 &&
        Buffer.byteLength(content) === entry.bytes, "candidate_export_report_pin");
    }
    if (isRetrieval(bundle)) {
      const parity = parseStrict(payloads["parity.json"]) as { role: string; nonprivate: boolean; cases: unknown[]; comparison: {
        passed: boolean; tolerance: number; max_absolute_embedding_error: number; max_absolute_pooled_error: number; ranking_pair_changes: number } };
      const recipe = parseStrict(payloads["recipe.json"]) as { role: string; weights: string; fit_split: string | null;
        training_data_used: boolean; export: { variant: string }; contextual_binding: unknown;
        dataset_sha256: Digest; split_sha256: Digest; training_recipe_sha256: Digest | null };
      const reports = ["evaluation.json", "validation.json"].map(name => parseStrict(payloads[name as "evaluation.json" | "validation.json"]) as
        typeof recipe & { comparison: unknown });
      const bindingMatches = (report: typeof recipe) =>
        (["weights", "fit_split", "training_data_used", "contextual_binding", "dataset_sha256", "split_sha256", "training_recipe_sha256"] as const)
          .every(key => equal(report[key], bundle.provenance[key]));
      const comparison = parity.comparison;
      const passed = stage === "shortlister" && recipe.role === "retrieval" &&
        recipe.export?.variant === "fp32" &&
        bindingMatches(recipe) && parity.role === "retrieval" &&
        parity.nonprivate === true && Array.isArray(parity.cases) && parity.cases.length >= 4 && comparison?.passed === true &&
        Number.isFinite(comparison.tolerance) && comparison.tolerance > 0 && comparison.tolerance <= .0001 &&
        Number.isFinite(comparison.max_absolute_embedding_error) && comparison.max_absolute_embedding_error >= 0 &&
        comparison.max_absolute_embedding_error <= comparison.tolerance && comparison.ranking_pair_changes === 0 &&
        Number.isFinite(comparison.max_absolute_pooled_error) && comparison.max_absolute_pooled_error >= 0 &&
        comparison.max_absolute_pooled_error <= comparison.tolerance &&
        reports.every(report => report.role === "retrieval" &&
          bindingMatches(report) && equal(report.comparison, comparison));
      return { stage, passed, reason: passed ? null : "fp32_parity_gate", comparison };
    }
    if (stage === "shortlister") return { stage, passed: false, reason: "retrieval_role_required" };
    const parity = parseStrict(payloads["parity.json"]) as { comparison: {
      passed: boolean; tolerance: number; max_absolute_logit_error: number; decision_changes: number; ranking_pair_changes: number } };
    const evaluation = parseStrict(payloads["evaluation.json"]) as {
      drift: Record<string, { max_absolute_logit_error: number; decision_changes: number }> };
    const recipe = parseStrict(payloads["recipe.json"]) as { export: { variant: string } };
    const validation = parseStrict(payloads["validation.json"]) as { fit_split: string };
    const withinTolerance = (value: number) => Number.isFinite(value) && value >= 0 && value <= .0001;
    const comparison = parity.comparison;
    const passed = recipe.export?.variant === "fp32" && validation.fit_split === "validation" &&
      bundle.selection.fit_split === "validation" && comparison?.passed === true &&
      withinTolerance(comparison.tolerance) && withinTolerance(comparison.max_absolute_logit_error) &&
      comparison.decision_changes === 0 && comparison.ranking_pair_changes === 0 && evaluation.drift &&
      Object.hasOwn(evaluation.drift, "validation") && Object.values(evaluation.drift).every(report =>
        withinTolerance(report.max_absolute_logit_error) && report.decision_changes === 0);
    return { stage, passed: Boolean(passed), reason: passed ? null : "fp32_parity_gate", comparison };
  });
}

/** Qualification is an evidence result, never execution, installation or deployment permission. */
export async function scoreObservations(design: Design, plan: Plan, observations: Observation[], evidence: QualificationEvidence = {}) {
  requireContract(equal(design.neutral_ids, NEUTRAL) && plan.design_sha256 === digest(design) &&
    design.format === "prospective-contextual-source-set-evaluation-v2" &&
    plan.format === "bound-contextual-source-set-evaluation-v2" && plan.execution_authorized === false,
    "plan_design_binding");
  validateSourceInventory(design, plan);
  requireContract(equal(design.required_roles, ["fresh_test"]), "required_roles");
  const labels = supplementalCases(design, evidence.supplementalLabels);
  const cases = new Map(labels.map(c => [c.id, c]));
  const inputs = new Map(plan.cases.map(c => [c.id, c]));
  requireContract(cases.size === design.cases.length && inputs.size === plan.cases.length &&
    equal([...cases.keys()].sort(), [...inputs.keys()].sort()), "all_original_cases_required");
  const excluded = new Set(design.sensitivity.excluded_ids);
  requireContract(excluded.size === design.sensitivity.excluded_ids.length && [...excluded].every(id => cases.has(id)),
    "sensitivity_unknown_or_duplicate_case");
  const groups = new Map<string, Map<string, Outcome>>();
  const manifests = new Map<string, Observation["bundles"]>();
  const planDigest = digest(plan);
  for (const observation of observations) {
    requireContract(["installed", "candidate"].includes(observation.baseline) && cases.has(observation.id) &&
      observation.plan_sha256 === planDigest, "observation_plan_binding");
    for (const bundle of Object.values(observation.bundles)) {
      // The installed router predates this source set. Its own manifest remains pinned by the observation set.
      requireContract((observation.baseline === "installed" || equal(bundle.provenance.contextual_binding, compactSourceSetBinding(plan.contextual_binding))) &&
        bundle.limits.max_candidates <= 32 && bundle.limits.request_timeout_ms === 750 &&
        bundle.limits.startup_timeout_ms === 10000, "bundle_binding_or_limits");
    }
    const prior = manifests.get(observation.baseline);
    requireContract(!prior || equal(prior, observation.bundles), "router_bundle_identity_changed");
    manifests.set(observation.baseline, observation.bundles);
    const outcomes = groups.get(observation.baseline) ?? new Map<string, Outcome>();
    requireContract(!outcomes.has(observation.id), "duplicate_observation");
    const item = inputs.get(observation.id)!;
    const complete = await validateObservation(plan.catalog, item, observation);
    const outcome = evaluationOutcome(observation.response, cases.get(observation.id)!, item.runtime.requiredIds, complete);
    if (observation.elapsed_ms > 750 || observation.startup_ms > 10000) {
      outcome.healthy = false; outcome.noSkillPass = false;
    }
    outcomes.set(observation.id, outcome);
    groups.set(observation.baseline, outcomes);
  }
  const baselines = (["installed", "candidate"] as const).map(baseline => {
    const outcomes = groups.get(baseline) ?? new Map<string, Outcome>();
    return { baseline, bundleSha256: manifests.has(baseline) ? digest(manifests.get(baseline)) : null,
      views: (["full", "sensitivity"] as const).map(view => ({ view,
        excludedIds: view === "sensitivity" ? [...excluded] : [],
        roles: design.required_roles.map(role => ({ role,
          ...summarize(labels.filter(c => c.role === role && (view === "full" || !excluded.has(c.id))), outcomes) })),
        developmentDiagnostics: ["validation", "unseen_family_test", "temporal_test"].map(role => ({ role,
          ...summarize(labels.filter(c => c.role === role && (view === "full" || !excluded.has(c.id))), outcomes) })) })),
      diagnostics: labels.filter(c => c.diagnostic || c.role === "challenge").map(c => ({ id: c.id, outcome: outcomes.get(c.id) ?? null })),
      outcomes: Object.fromEntries(outcomes) };
  });
  const failingItems: { check: string; baseline?: string; view?: string; role?: string; stage?: string;
    actual?: unknown; expected?: unknown; caseIds?: string[] }[] = [];
  const parity = candidateParity(manifests.get("candidate"), evidence.candidateExports);
  for (const item of parity) if (!item.passed) failingItems.push({ check: item.reason!, stage: item.stage });
  // Development observations remain visible but cannot gate the one-shot fresh qualification.
  const qualificationCases = labels.filter(c => design.required_roles.includes(c.role));
  for (const baseline of ["installed", "candidate"]) {
    const outcomes = groups.get(baseline) ?? new Map<string, Outcome>();
    for (const [check, affected] of [
      ["missing_observations", qualificationCases.filter(c => !outcomes.has(c.id)).map(c => c.id)],
      ["unhealthy_observations", qualificationCases.filter(c => outcomes.has(c.id) && !outcomes.get(c.id)!.healthy).map(c => c.id)],
      ["missing_required_ids", qualificationCases.filter(c => outcomes.get(c.id)?.missingRequiredIds.length).map(c => c.id)],
    ] as const) if (affected.length) failingItems.push({ check, baseline, caseIds: affected });
  }
  const installed = baselines[0]!, candidate = baselines[1]!;
  for (const [index, view] of candidate.views.entries()) for (const [roleIndex, summary] of view.roles.entries()) {
    const baseline = installed.views[index]!.roles[roleIndex]!;
    const context = { view: view.view, role: summary.role };
    const tests: [string, boolean, unknown, unknown][] = [
      ["known_none_presence", summary.knownNone > 0, summary.knownNone, ">0"],
      ["precision_lower_bound", summary.precisionBounds[0] !== null && summary.precisionBounds[0] >= .95, summary.precisionBounds[0], ">=0.95"],
      ["coverage_lower_bound", summary.coverageBounds[0] !== null && summary.coverageBounds[0] >= .75, summary.coverageBounds[0], ">=0.75"],
      ["false_fires", summary.falseFires === 0, summary.falseFires, 0],
      ["no_skill_failures", summary.noSkillFailures === 0, summary.noSkillFailures, 0],
      ["installed_precision_comparison", baseline.precisionBounds[0] !== null && summary.precisionBounds[0] !== null &&
        summary.precisionBounds[0] >= baseline.precisionBounds[0], summary.precisionBounds[0], baseline.precisionBounds[0]],
      ["installed_false_fires_comparison", summary.falseFires <= baseline.falseFires, summary.falseFires, `<=${baseline.falseFires}`],
    ];
    for (const [check, passed, actual, expected] of tests) if (!passed) failingItems.push({ ...context, check, actual, expected });
  }
  return { format: "contextual-qualification-report-v1", qualified: failingItems.length === 0, failingItems,
    planSha256: planDigest, designSha256: digest(design), supplementalLabelsSha256: evidence.supplementalLabels?.sha256 ?? null,
    scope: "qualification-protocol-v1-amendment-2; fresh_test only; no execution, installation or deployment authorization; intent groups diagnostic only",
    sensitivity: design.sensitivity, parity, baselines };
}

function authorizeBundles(args: AuthorizationInputs, baseline: Observation["baseline"], bundles: Observation["bundles"]) {
  const approvalBytes = readFileSync(args.executionApproval);
  requireContract(bytesDigest(approvalBytes) === args.executionApprovalSha256, "execution_approval_changed");
  const authority = parseStrict(approvalBytes.toString()) as {
    bundle_sha256: Record<Observation["baseline"], { shortlister: Digest; decider: Digest }>;
  };
  for (const stage of ["shortlister", "decider"] as const) {
    requireContract(authority.bundle_sha256[baseline]?.[stage] === digest(bundles[stage]), "execution_bundle_pin");
  }
}

/** Public saved-observation entrypoint. Pure reducers above remain deterministic test seams. */
export async function scoreAuthorized(args: AuthorizationInputs, observations: Observation[], evidence: QualificationEvidence = {}) {
  const { design, plan } = loadAuthorized(args);
  for (const observation of observations) authorizeBundles(args, observation.baseline, observation.bundles);
  return scoreObservations(design, plan, observations, evidence);
}

/**
 * Admits one baseline's WorkerClients for a whole collection, never a ranking proxy.
 * Source re-admission (loadAuthorized) is a full Python verification of the proof, about a
 * minute and ~800 MB; it runs once here, and the returned observer reuses the verified plan.
 */
export async function authorizeCollection(args: AuthorizationInputs, baseline: Observation["baseline"], scorers: Scorers) {
  const { design, plan } = loadAuthorized(args);
  authorizeBundles(args, baseline, { shortlister: scorers.shortlister.bundle, decider: scorers.decider.bundle });
  await scorers.shortlister.verify();
  await scorers.decider.verify();
  const planSha256 = digest(plan);
  const items = new Map(plan.cases.map(c => [c.id, c]));
  return {
    design, plan,
    observe(id: string, startupMs: number, signal: AbortSignal) {
      const item = items.get(id);
      requireContract(item, "unknown_case");
      return observeRoute({ catalog: plan.catalog, item, scorers, baseline, planSha256, startupMs, signal });
    },
  };
}

/** Saved replay only. External byte pins identify the already-bound plan; no model is loaded. */
async function main() {
  const names = ["design", "design-sha256", "plan", "plan-sha256", "installed-observations", "candidate-observations",
    "candidate-shortlister", "candidate-decider", "supplemental-labels", "supplemental-labels-sha256", "output"];
  const { values, positionals } = parseArgs({ args: process.argv.slice(2), allowPositionals: true,
    options: Object.fromEntries(names.map(name => [name, { type: "string" as const }])) });
  requireContract(equal(positionals, ["score"]), "score_command_required");
  const arg = (name: string) => { const value = values[name]; requireContract(value, `missing_${name}`); return value; };
  function pinned<T>(path: string, pin: string): T {
    const content = readFileSync(path, "utf8");
    requireContract(bytesDigest(content) === pin, "saved_input_pin");
    return parseStrict(content) as T;
  }
  const design = pinned<Design>(arg("design"), arg("design-sha256"));
  const plan = pinned<Plan>(arg("plan"), arg("plan-sha256"));
  const observations = (["installed", "candidate"] as const).flatMap(baseline => {
    const rows = parseStrict(readFileSync(arg(`${baseline}-observations`), "utf8")) as Observation[];
    requireContract(Array.isArray(rows) && rows.every(row => row.baseline === baseline), "observation_file_baseline");
    return rows;
  });
  const payloads = (stage: Stage["stage"]) => Object.fromEntries(
    ["parity.json", "evaluation.json", "recipe.json", "validation.json"].map(path =>
      [path, readFileSync(join(arg(`candidate-${stage}`), path), "utf8")])) as NonNullable<QualificationEvidence["candidateExports"]>[Stage["stage"]];
  const evidence: QualificationEvidence = { candidateExports: { shortlister: payloads("shortlister"), decider: payloads("decider") } };
  if (values["supplemental-labels"] || values["supplemental-labels-sha256"]) evidence.supplementalLabels = {
    content: readFileSync(arg("supplemental-labels"), "utf8"), sha256: arg("supplemental-labels-sha256") as Digest };
  const report = await scoreObservations(design, plan, observations, evidence);
  writeFileSync(arg("output"), JSON.stringify(report, null, 2) + "\n", { mode: 0o600, flag: "wx" });
  console.log(JSON.stringify({ qualified: report.qualified, failingItems: report.failingItems, output: values.output }));
}
if (import.meta.main) main().catch(error => { console.error(error.message); process.exitCode = 2; });
