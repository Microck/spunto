import { randomUUID } from "node:crypto";
import { acceptResponse, assertRequest, compare, contextDigest, digest, errorCode, fail, isRetrieval, RouterError, sortedIds, uncertain,
  type Action, type InferenceBundle, type Catalog, type Digest, type Request, type Response } from "./protocol.ts";

export interface Rule {
  id: string; provenance: string;
  scope: { actorId: string | null; toolName: string | null };
  requiredIds: string[]; excludedIds: string[];
}
export interface PolicyArtifact { version: 1; rules: Rule[]; unsupported: string[] }
export interface Policy {
  requiredIds: string[]; excludedIds: string[]; explicitIds: string[];
  policyDigest: Digest; conflict: boolean; unsupported: boolean;
}
export function validatePolicy(value: unknown): PolicyArtifact {
  const record = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
  const keys = (v: Record<string, unknown>, expected: string[]): boolean => Object.keys(v).sort().join() === expected.sort().join();
  const ids = (v: unknown): v is string[] => Array.isArray(v) && v.length <= 256 &&
    v.every(id => typeof id === "string" && id.length > 0 && id.length <= 256) && new Set(v).size === v.length;
  if (!record(value) || !keys(value, ["version", "rules", "unsupported"]) || value.version !== 1 ||
      !Array.isArray(value.rules) || value.rules.length > 256 || !ids(value.unsupported)) fail("policy_conflict");
  for (const rule of value.rules) {
    if (!record(rule) || !keys(rule, ["id", "provenance", "scope", "requiredIds", "excludedIds"]) ||
        typeof rule.id !== "string" || !rule.id || typeof rule.provenance !== "string" || !rule.provenance ||
        rule.provenance.length > 1024 || !record(rule.scope) || !keys(rule.scope, ["actorId", "toolName"]) ||
        ![rule.scope.actorId, rule.scope.toolName].every(v => v === null || (typeof v === "string" && v.length > 0 && v.length <= 256)) ||
        !ids(rule.requiredIds) || !ids(rule.excludedIds)) fail("policy_conflict");
  }
  if (new Set(value.rules.map(rule => (rule as Rule).id)).size !== value.rules.length) fail("policy_conflict");
  digest(value); // Reject unsupported Unicode and noncanonical JSON values at the boundary.
  return value as unknown as PolicyArtifact;
}
export function resolvePolicy(artifact: PolicyArtifact, actorId: string, action: Action, explicitIds: string[]): Policy {
  const active = artifact.rules.filter(rule => (rule.scope.actorId === null || rule.scope.actorId === actorId) &&
    (rule.scope.toolName === null || rule.scope.toolName === action?.toolName));
  const requiredIds = sortedIds([...explicitIds, ...active.flatMap(rule => rule.requiredIds)]);
  const excludedIds = sortedIds(active.flatMap(rule => rule.excludedIds));
  const explicit = sortedIds(explicitIds);
  return { requiredIds, excludedIds, explicitIds: explicit,
    policyDigest: digest({ schema_version: 1, rules_sha256: digest(artifact), required_ids: requiredIds,
      excluded_ids: excludedIds, explicit_ids: explicit }),
    conflict: requiredIds.some(id => excludedIds.includes(id)), unsupported: artifact.unsupported.length > 0 };
}
export function pairText(intent: string, action: Action): string {
  return action === null ? intent : `${intent}\nAction: ${action.toolName}\n${action.summary}`;
}
export interface Scorer {
  readonly bundle: InferenceBundle;
  verify(): Promise<void>;
  score(request: Request, signal: AbortSignal): Promise<unknown>;
  close(): void;
}
/** Frozen bi-encoder retrieval followed by the unchanged FP32 window-bag decider. */
export interface Scorers { shortlister: Scorer; decider: Scorer; ready?(signal: AbortSignal, catalog?: Catalog): Promise<void> }
export const SHORTLIST_SIZE = 8;
export interface RouteContract { readonly shortlistSize: number; readonly timeoutMs: number | null }
const PRODUCTION_ROUTE: RouteContract = Object.freeze({ shortlistSize: SHORTLIST_SIZE, timeoutMs: 750 });
type Entry = Catalog["entries"][number];
type Base = Pick<Request, "version" | "sessionKey" | "generation" | "event" | "catalogDigest" | "policyDigest" | "intent" | "action">;

function candidates(entries: Entry[]): Request["candidates"] {
  return [...entries].sort((a, b) => compare(a.runtime_id, b.runtime_id)).map(entry => ({ id: entry.runtime_id,
    name: entry.runtime_id, description: entry.description, contentDigest: entry.content_sha256 }));
}
function build(base: Base, scorer: Scorer, entries: Entry[], candidateCoverage: Request["candidateCoverage"],
  requiredIds: string[], excludedIds: string[]): Request {
  const fields = { ...base, requestId: randomUUID(), selectorDigest: scorer.bundle.selector_sha256,
    candidates: candidates(entries), candidateCoverage, requiredIds, excludedIds,
    maxSelections: isRetrieval(scorer.bundle) ? 0 : scorer.bundle.selection.max_optional };
  return { ...fields, contextDigest: contextDigest(fields) };
}
async function call(scorer: Scorer, request: Request, signal: AbortSignal): Promise<Response> {
  if (signal.aborted) fail("stale_result");
  if (request.candidates.length > scorer.bundle.limits.max_candidates) fail("context_overflow");
  assertRequest(request);
  if (Buffer.byteLength(JSON.stringify(request)) + 1 > scorer.bundle.limits.max_line_bytes) fail("context_overflow");
  return acceptResponse(await scorer.score(request, signal), request, scorer.bundle);
}
/** Score all optional entries with the shortlister in bundle-sized chunks and keep the
 * best contract-sized shortlist (ties by runtime ID). A chunk without scores fails closed. */
async function shortlist(base: Base, optional: Entry[], scorer: Scorer, signal: AbortSignal, size: number): Promise<Entry[]> {
  const scores = new Map<string, number>();
  for (let start = 0; start < optional.length; start += scorer.bundle.limits.max_candidates) {
    const chunk = optional.slice(start, start + scorer.bundle.limits.max_candidates);
    const response = await call(scorer, build(base, scorer, chunk, "shortlist", [], []), signal);
    if (!response.scores) fail(response.reasonCode);
    for (const item of response.scores.values) scores.set(item.id, item.score);
  }
  return [...optional].sort((a, b) => scores.get(b.runtime_id)! - scores.get(a.runtime_id)! ||
    compare(a.runtime_id, b.runtime_id)).slice(0, size);
}
export async function route(input: {
  catalog: Catalog; policy: Policy; intent: string; action: Action; sessionKey: string; generation: number;
  event: Request["event"]; scorers: Scorers; signal: AbortSignal; isCurrent: () => boolean;
}, contract: RouteContract = PRODUCTION_ROUTE): Promise<{ request: Request; response: Response }> {
  let started = performance.now();
  const { catalog, policy, intent, action, event, sessionKey, generation } = input;
  const controller = new AbortController();
  const signal = AbortSignal.any([input.signal, controller.signal]);
  const { shortlister, decider } = input.scorers;
  const base: Base = { version: 1, sessionKey, generation, event, catalogDigest: catalog.catalog_sha256,
    policyDigest: policy.policyDigest, intent, action };
  const requiredSet = new Set(policy.requiredIds);
  const excludedSet = new Set(policy.excludedIds);
  const required = catalog.entries.filter(entry => requiredSet.has(entry.runtime_id));
  const optional = catalog.entries.filter(entry => entry.automatic_eligible && !requiredSet.has(entry.runtime_id) && !excludedSet.has(entry.runtime_id));
  const complete = optional.length <= contract.shortlistSize;
  let request = build(base, decider, complete ? [...required, ...optional] : required,
    complete ? "full" : "shortlist", policy.requiredIds, policy.excludedIds);
  const finish = (response: Response) => ({ request, response });
  if (policy.conflict) return finish(uncertain(request, "policy_conflict"));
  if (policy.requiredIds.some(id => !catalog.entries.some(entry => entry.runtime_id === id && entry.explicit_eligible))) {
    return finish(uncertain(request, "missing_required"));
  }
  const readyStart = performance.now();
  try { await input.scorers.ready?.(input.signal, catalog); }
  catch (error) { return finish(uncertain(request, errorCode(error))); }
  // Exclude only readiness, not catalog filtering or request construction.
  started += performance.now() - readyStart;
  const deadline = Promise.withResolvers<{ request: Request; response: Response }>();
  const timer = contract.timeoutMs === null ? undefined : setTimeout(() => {
    controller.abort(new RouterError("timeout"));
    deadline.resolve(finish(uncertain(request, "timeout")));
  }, Math.max(0, contract.timeoutMs - (performance.now() - started)));
  const run = async () => {
    try {
      if (!complete) {
        const top = await shortlist(base, optional, shortlister, signal, contract.shortlistSize);
        request = build(base, decider, [...required, ...top], "shortlist", policy.requiredIds, policy.excludedIds);
      }
      const response = await call(decider, request, signal);
      await decider.verify();
      if (contract.timeoutMs !== null && performance.now() - started >= contract.timeoutMs) return finish(uncertain(request, "timeout"));
      if (signal.aborted || !input.isCurrent()) return finish(uncertain(request, "stale_result"));
      return finish(response);
    } catch (error) {
      return finish(uncertain(request, controller.signal.aborted ? "timeout" : errorCode(error)));
    }
  };
  try { return await Promise.race([run(), deadline.promise]); }
  finally { clearTimeout(timer); }
}
