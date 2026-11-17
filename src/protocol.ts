import { createHash } from "node:crypto";
import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { visit } from "jsonc-parser";
import catalogSchema from "../schemas/catalog.schema.json";
import routingSchema from "../schemas/routing.schema.json";
import bundleSchema from "../schemas/model-bundle.schema.json";

export type Digest = `sha256:${string}`;
export type Action = { toolName: string; summary: string } | null;
export interface Candidate { id: string; name: string; description: string; contentDigest: Digest }
export interface Request {
  version: 1; requestId: string; sessionKey: string; generation: number;
  event: "context" | "tool_call"; catalogDigest: Digest; contextDigest: Digest;
  policyDigest: Digest; selectorDigest: Digest; intent: string; action: Action;
  candidates: Candidate[]; candidateCoverage: "full" | "shortlist";
  requiredIds: string[]; excludedIds: string[]; maxSelections: number;
}
export type Reason = "matched" | "policy_only" | "no_match" | "insufficient_evidence" |
  "inadequate_candidates" | "policy_conflict" | "missing_required" | "selection_overflow" |
  "context_overflow" | "busy" | "timeout" | "backend_unavailable" | "invalid_response" |
  "stale_result" | "body_unavailable";
export interface Response extends Pick<Request, "version" | "requestId" | "sessionKey" | "generation" |
  "catalogDigest" | "contextDigest" | "policyDigest" | "selectorDigest"> {
  status: "selected" | "none" | "uncertain"; selectedIds: string[]; reasonCode: Reason;
  scores: null | { kind: "ranking" | "calibrated_probability"; scorerDigest: Digest;
    calibrationDigest: Digest | null; values: { id: string; score: number }[] };
}
export interface Entry {
  runtime_id: string; declared_name: string | null; description: string;
  content_sha256: Digest; routing_sha256: Digest; resource_tree_sha256: Digest | null;
  source_kind: string; source_ref: string; hidden: boolean; automatic_eligible: boolean;
  explicit_eligible: boolean; family_id: string | null;
}
export interface Catalog {
  schema_version: 1; catalog_sha256: Digest; captured_at: string; loader_contract: string;
  eligibility_sha256: Digest; entries: Entry[];
}
export interface CompactContextualBinding {
  version: 1; target_sha256: Digest; catalog_sha256: Digest; source_set_sha256: Digest;
  runtime_revision: Digest; approval_sha256: Digest;
  purpose: "local-canonical-integration"; fixture_only: boolean;
}
export interface ContextualBinding extends CompactContextualBinding {
  members: { member_sha256: Digest; approval_sha256: Digest;
    identity: Record<"target_sha256" | "rubric_sha256" | "schema_sha256" | "annotation_sha256" |
      "receipt_sha256" | "catalog_sha256" | "source_dataset_sha256" | "dataset_schema_sha256" |
      "source_contract_sha256" | "role_plan_sha256" | "identity_map_sha256" | "projection_sha256", Digest> }[];
}
/** Admission and plans retain members; serving binds their committed source identity. */
export function compactSourceSetBinding(binding: ContextualBinding): CompactContextualBinding {
  const { members, ...compact } = binding;
  return compact;
}

export interface Bundle {
  schema_version: 1; model: { type: "window_bag"; base: string; revision: string };
  graphs: Record<"encoder" | "aggregator", { file: string; inputs: unknown[]; outputs: unknown[] }>;
  tokenizer_file: string;
  model_config_file: string; files: { path: string; sha256: Digest; bytes: number }[];
  pair_format: "intent-action-window-bag-v2"; geometry: typeof bundleSchema.$defs.decision.properties.geometry.const;
  selection: { lower_logit: string; upper_logit: string; max_optional: number; validation_sha256: Digest; fit_split: "validation" };
  limits: { max_candidates: number; max_line_bytes: number; microbatch: number; request_timeout_ms: number; startup_timeout_ms: number };
  calibration: null | { method: "platt"; slope: string; intercept: string; validation_sha256: Digest };
  provenance: {
    dataset_sha256: Digest; split_sha256: Digest; recipe_sha256: Digest; runtime_revision: string;
    contextual_binding: CompactContextualBinding;
  };
  scorer_sha256: Digest; selector_sha256: Digest;
  calibration_sha256: Digest | null; publication_allowed: true;
}
export interface RetrievalBundle extends Pick<Bundle, "schema_version" | "tokenizer_file" | "model_config_file" |
  "files" | "limits" | "scorer_sha256" | "selector_sha256" | "publication_allowed"> {
  role: "retrieval";
  model: typeof bundleSchema.$defs.retrieval.properties.model.const;
  graphs: typeof bundleSchema.$defs.retrieval.properties.graphs.const;
  pooling: typeof bundleSchema.$defs.retrieval.properties.pooling.const;
  embedding_table: "embeddings.json";
  selection: { top_k: 8; tie_break: "id" };
  provenance: { contextual_binding: Bundle["provenance"]["contextual_binding"]; catalog_sha256: Digest;
    recipe_sha256: Digest; runtime_revision: Digest; runtime_files: Record<string, Digest>;
    dataset_sha256: Digest; split_sha256: Digest } & (
      { weights: "unchanged_pretrained"; training_report_sha256: null; training_recipe_sha256: null;
        fit_split: null; training_data_used: false } |
      { weights: "fine_tuned"; training_report_sha256: Digest; training_recipe_sha256: Digest;
        fit_split: "validation"; training_data_used: true });
}
export type InferenceBundle = Bundle | RetrievalBundle;
export function isRetrieval(bundle: InferenceBundle): bundle is RetrievalBundle { return "role" in bundle; }

const ajv = new Ajv2020({ strict: false, allErrors: false });
addFormats(ajv);
ajv.addSchema(catalogSchema).addSchema(routingSchema).addSchema(bundleSchema);
const validateRequest = ajv.compile<Request>({ $ref: `${routingSchema.$id}#/$defs/request` });
const validateResponse = ajv.compile<Response>({ $ref: `${routingSchema.$id}#/$defs/response` });
const validateCatalog = ajv.compile<Catalog>({ $ref: catalogSchema.$id });
const validateBundle = ajv.compile<Bundle>({ $ref: `${bundleSchema.$id}#/$defs/decision` });
const validateRetrieval = ajv.compile<RetrievalBundle>({ $ref: `${bundleSchema.$id}#/$defs/retrieval` });

export class RouterError extends Error {
  constructor(readonly code: Reason) { super(`skill_router:${code}`); }
}
export function fail(code: Reason): never { throw new RouterError(code); }
export function errorCode(error: unknown): Reason {
  return error instanceof RouterError ? error.code : "backend_unavailable";
}
export function compare(a: string, b: string): number {
  let left = 0, right = 0;
  while (left < a.length && right < b.length) {
    const x = a.codePointAt(left)!, y = b.codePointAt(right)!;
    if (x !== y) return x - y;
    left += x > 0xffff ? 2 : 1; right += y > 0xffff ? 2 : 1;
  }
  return (a.length - left) - (b.length - right);
}
export const sortedIds = (ids: Iterable<string>): string[] => [...new Set(ids)].sort(compare);
export function bytesDigest(bytes: Uint8Array | string): Digest {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}
function validString(value: string): void {
  for (let i = 0; i < value.length; i++) {
    const unit = value.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) fail("invalid_response");
    } else if (unit >= 0xdc00 && unit <= 0xdfff) fail("invalid_response");
  }
}
function canonical(value: unknown): string {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") { validString(value); return JSON.stringify(value); }
  if (typeof value === "number" && Number.isSafeInteger(value)) return String(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    return `{${Object.keys(value).sort(compare).map(key => `${canonical(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(",")}}`;
  }
  return fail("invalid_response");
}
export const digest = (value: unknown): Digest => bytesDigest(canonical(value));

/** JSON.parse alone silently accepts duplicate properties and lone surrogates. */
export function parseStrict(text: string): unknown {
  const stack: Set<string>[] = [];
  let invalid = false;
  visit(text, {
    onObjectBegin: () => { stack.push(new Set()); },
    onObjectProperty: key => {
      const keys = stack.at(-1)!;
      if (keys.has(key)) invalid = true;
      keys.add(key); validString(key);
    },
    onObjectEnd: () => { stack.pop(); },
    onLiteralValue: value => {
      if (typeof value === "string") validString(value);
      if (typeof value === "number" && !Number.isFinite(value)) invalid = true;
    },
    onError: () => { invalid = true; },
    onComment: () => { invalid = true; },
  }, { disallowComments: true, allowTrailingComma: false });
  if (invalid) fail("invalid_response");
  try { return JSON.parse(text) as unknown; } catch { return fail("invalid_response"); }
}
export function assertCatalog(value: unknown): asserts value is Catalog {
  if (!validateCatalog(value)) fail("invalid_response");
}
export function assertRequest(value: unknown): asserts value is Request {
  if (!validateRequest(value)) fail("context_overflow");
}
export function contextDigest(request: Omit<Request, "contextDigest"> | Request): Digest {
  const { version, event, intent, action, candidates, candidateCoverage, requiredIds,
    excludedIds, maxSelections, policyDigest, selectorDigest } = request;
  return digest({ version, event, intent, action, candidates, candidateCoverage, requiredIds,
    excludedIds, maxSelections, policyDigest, selectorDigest });
}
/** Manifest-only validation. WorkerClient.create also admits verified provenance payloads. */
export function readBundle(value: unknown): Bundle {
  if (!validateBundle(value)) fail("invalid_response");
  const b = value;
  const { model, graphs, tokenizer_file, model_config_file, files, pair_format, geometry, provenance } = b;
  if (b.scorer_sha256 !== digest({ model, graphs, tokenizer_file, model_config_file, files, pair_format, geometry, provenance }) ||
      b.calibration !== null || b.calibration_sha256 !== null ||
      b.selection.max_optional > 6 || !Number.isFinite(Number(b.selection.lower_logit)) ||
      !Number.isFinite(Number(b.selection.upper_logit)) || Number(b.selection.lower_logit) >= Number(b.selection.upper_logit) ||
      b.selector_sha256 !== digest({ scorer_sha256: b.scorer_sha256, selection: b.selection,
        limits: b.limits, calibration_sha256: b.calibration_sha256 })) fail("invalid_response");
  const limits = { max_candidates: 32, max_line_bytes: 131072, microbatch: 4, request_timeout_ms: 750, startup_timeout_ms: 10000 };
  if (digest(b.limits) !== digest(limits) ||
      !files.some(file => file.path === "recipe.json" && file.sha256 === provenance.recipe_sha256) ||
      !files.some(file => file.path === "validation.json" && file.sha256 === b.selection.validation_sha256) ||
      files.some((file, index) => index > 0 && compare(files[index - 1]!.path, file.path) >= 0)) fail("invalid_response");
  const payloads = [graphs.encoder.file, graphs.aggregator.file, tokenizer_file, model_config_file];
  if (new Set(payloads).size !== 4 || payloads.some(path => !files.some(file => file.path === path))) fail("invalid_response");
  return b;
}
/** Retrieval binds one frozen trained encoder and validation-only epoch evidence. */
export function readRetrievalBundle(value: unknown): RetrievalBundle {
  if (!validateRetrieval(value)) fail("invalid_response");
  const b = value;
  const { role, model, graphs, tokenizer_file, model_config_file, files, pooling, embedding_table, provenance } = b;
  const binding = provenance.contextual_binding;
  if (binding.catalog_sha256 !== provenance.catalog_sha256 ||
      b.scorer_sha256 !== digest({ role, model, graphs, tokenizer_file, model_config_file, files, pooling, embedding_table, provenance }) ||
      b.selector_sha256 !== digest({ scorer_sha256: b.scorer_sha256, selection: b.selection, limits: b.limits }) ||
      files.some((file, index) => index > 0 && compare(files[index - 1]!.path, file.path) >= 0) ||
      !files.some(file => file.path === "recipe.json" && file.sha256 === provenance.recipe_sha256) ||
      files.some(file => file.path === "training.json") !== (provenance.weights === "fine_tuned") ||
      ["encoder.onnx", "tokenizer.json", "config.json", "embeddings.json", "parity.json", "evaluation.json", "validation.json"]
        .some(name => !files.some(file => file.path === name))) fail("invalid_response");
  return b;
}
export function uncertain(request: Request, reasonCode: Reason): Response {
  const { version, requestId, sessionKey, generation, catalogDigest, contextDigest, policyDigest, selectorDigest } = request;
  return { version, requestId, sessionKey, generation, catalogDigest, contextDigest, policyDigest, selectorDigest,
    status: "uncertain", selectedIds: [], scores: null, reasonCode };
}
export function acceptResponse(value: unknown, request: Request, bundle: InferenceBundle): Response {
  if (!validateResponse(value)) fail("invalid_response");
  const response = value;
  for (const key of ["version", "requestId", "sessionKey", "generation", "catalogDigest", "contextDigest", "policyDigest", "selectorDigest"] as const) {
    if (response[key] !== request[key]) fail("stale_result");
  }
  const ids = new Set(request.candidates.map(candidate => candidate.id));
  const required = new Set(request.requiredIds);
  const selected = new Set(response.selectedIds);
  if (response.selectedIds.some(id => !ids.has(id) || request.excludedIds.includes(id)) ||
      response.selectedIds.filter(id => !required.has(id)).length > request.maxSelections ||
      (response.status === "selected" && request.requiredIds.some(id => !selected.has(id))) ||
      (response.status === "none" && (required.size > 0 || request.candidateCoverage !== "full"))) fail("invalid_response");
  if (response.scores) {
    const scores = response.scores;
    if (scores.kind !== "ranking" || scores.scorerDigest !== bundle.scorer_sha256 || scores.calibrationDigest !== null ||
        scores.values.length !== ids.size || new Set(scores.values.map(item => item.id)).size !== ids.size ||
        scores.values.some(item => !ids.has(item.id) || !Number.isFinite(item.score))) fail("invalid_response");
    if (isRetrieval(bundle)) {
      if (response.status !== "uncertain" || response.reasonCode !== "inadequate_candidates") fail("invalid_response");
      return response;
    }
    const optional = scores.values.filter(item => !required.has(item.id));
    const positive = optional.filter(item => item.score >= Number(bundle.selection.upper_logit)).map(item => item.id);
    const gray = optional.some(item => item.score > Number(bundle.selection.lower_logit) && item.score < Number(bundle.selection.upper_logit));
    if (response.status !== "uncertain" && ((positive.length === 0 && gray) || positive.length > request.maxSelections ||
        digest(sortedIds([...required, ...positive])) !== digest(sortedIds(selected)))) fail("invalid_response");
  } else if (response.status !== "uncertain" && request.candidates.some(candidate => !required.has(candidate.id))) {
    fail("invalid_response");
  }
  if (response.status === "selected" && ((response.reasonCode === "policy_only") !== (selected.size === required.size))) fail("invalid_response");
  if (response.status === "none" && !request.intent.trim() && !request.action?.summary.trim()) fail("invalid_response");
  return response;
}
