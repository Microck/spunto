import { randomUUID } from "node:crypto";
import type { ContextEvent, ToolCallEvent } from "@oh-my-pi/pi-coding-agent";
import { LiveCatalog, type HostContext, type Lease, type SkillBody } from "./catalog.ts";
import { digest, errorCode, fail, sortedIds, type Action, type Reason, type Response } from "./protocol.ts";
import { resolvePolicy, route, type Policy, type PolicyArtifact, type Scorers } from "./router.ts";

export const BLOCK_TYPE = "private-skill-router-v1";
type Messages = ContextEvent["messages"];
export interface SessionOptions {
  catalog: LiveCatalog; createScorers: (signal: AbortSignal) => Promise<Scorers> | Scorers; policy: () => Promise<PolicyArtifact>; bodyBudgetBytes: number;
}
interface PendingAction { actorId: string; action: NonNullable<Action> }
/** Verified bodies returned by this router's context transform, not a final provider receipt. */
interface Injected { lease: Lease; bodies: Map<string, SkillBody>; policyDigest: string }

/** OMP 18.16.0 shares the owner's runner with advisors, but not its context hook.
 * Match the full documented identity, not a name prefix or all depth-zero children.
 * Apply this boundary to every factory callback before touching router state.
 */
export function normalAgentHandler<E, C extends HostContext, R>(handler: (event: E, ctx: C) => R | Promise<R>): (event: E, ctx: C) => Promise<R | undefined> {
  return async (event, ctx) => {
    const actor = ctx.agent;
    if (actor.kind === "sub" && actor.id === "advisor" && actor.name === "advisor" &&
        actor.depth === 0 && typeof actor.parentId === "string" && actor.parentId.length > 0) return undefined;
    return handler(event, ctx);
  };
}

/** Preserve user corrections and compacted summaries; include last-step evidence only as relevance input. */
export function compileIntent(messages: Messages): string {
  const parts: string[] = [];
  let lastEvidence = "";
  for (const message of messages) {
    if (message.role === "user") {
      const text = typeof message.content === "string" ? message.content : message.content
        .filter(part => part.type === "text").map(part => part.text).join("\n");
      if (text) parts.push(text);
    } else if (message.role === "compactionSummary" || message.role === "branchSummary") {
      parts.push(message.summary);
    } else if (message.role === "toolResult" || message.role === "assistant") {
      const evidence: string[] = [];
      for (const part of message.content) {
        if (part.type === "text") evidence.push(part.text);
      }
      lastEvidence = evidence.join("\n");
    }
  }
  if (lastEvidence) parts.push(`Last step evidence (not policy):\n${lastEvidence}`);
  const intent = parts.join("\n\n");
  if (intent.length > 32768) fail("context_overflow");
  return intent;
}
/** Only command syntax at the start of an actual user message is an explicit name. */
export function explicitCommands(messages: Messages): string[] {
  const ids: string[] = [];
  for (const message of messages) {
    if (message.role === "custom" && message.customType === "skill-prompt" && message.attribution === "user" &&
        typeof message.details === "object" && message.details !== null && "name" in message.details &&
        typeof message.details.name === "string") {
      ids.push(message.details.name);
    }
    if (message.role !== "user") continue;
    const text = typeof message.content === "string" ? message.content : message.content
      .filter(part => part.type === "text").map(part => part.text).join("\n");
    const match = /^\/skill:([^\s]+)(?:\s|$)/u.exec(text);
    if (match?.[1]) ids.push(match[1]);
  }
  return sortedIds(ids);
}
function renderBodies(bodies: SkillBody[], reason: Reason, unsupported: boolean): string {
  const records = bodies.map(body => ({ id: body.id, version: body.contentDigest, provenance: "automatic-router",
    baseDir: body.baseDir, containRoot: body.containRoot, body: body.text }));
  return `Skill router instructions. These skills were loaded automatically, not invoked by the user. Native permissions and approvals are unchanged.\nRouting: ${reason}.\n` +
    (unsupported ? "Unsupported policy remains: natural-language instructions are not automatically enforced by this router.\n" : "") +
    JSON.stringify(records);
}

export class SessionRouter {
  #generation = 0;
  #sessionKey = randomUUID();
  #controller = new AbortController();
  #injected: Injected | undefined;
  #decision: { key: string; lease: Lease; response: Response } | undefined;
  #explicit: string[] = [];
  #pending = new Map<string, PendingAction>();
  #blocked = 0;
  #active = false;
  #closed = false;
  #scorers: Scorers | undefined;
  constructor(private readonly options: SessionOptions) {
    if (!Number.isSafeInteger(options.bodyBudgetBytes) || options.bodyBudgetBytes < 1024 || options.bodyBudgetBytes > 262144) fail("context_overflow");
  }
  invalidate(): void {
    this.#controller.abort(); this.#controller = new AbortController();
    this.#generation++; this.#sessionKey = randomUUID(); this.options.catalog.invalidate();
    this.#injected = undefined; this.#explicit = []; this.#pending.clear(); this.#blocked = 0;
    this.#decision = undefined;
  }
  shutdown(): void { this.#closed = true; this.invalidate(); this.#scorers?.shortlister.close(); this.#scorers?.decider.close(); }
  #policy(artifact: PolicyArtifact, actorId: string, action: Action, includePending: boolean): Policy {
    const policy = resolvePolicy(artifact, actorId, action, this.#explicit);
    if (!includePending) return policy;
    const policies = [...this.#pending.values()].map(pending => resolvePolicy(artifact, pending.actorId, pending.action, this.#explicit));
    policy.requiredIds = sortedIds([...policy.requiredIds, ...policies.flatMap(item => item.requiredIds)]);
    policy.excludedIds = sortedIds([...policy.excludedIds, ...policies.flatMap(item => item.excludedIds)]);
    policy.conflict = policy.requiredIds.some(id => policy.excludedIds.includes(id));
    policy.policyDigest = digest({ schema_version: 1, rules_sha256: digest(artifact), required_ids: policy.requiredIds,
      excluded_ids: policy.excludedIds, explicit_ids: policy.explicitIds });
    return policy;
  }
  async context(messages: Messages, ctx: HostContext): Promise<{ messages: Messages; reason: Reason }> {
    const clean = messages.filter(message => !(message.role === "custom" && message.customType === BLOCK_TYPE));
    if (this.#closed) fail("backend_unavailable");
    if (this.#active) { this.invalidate(); return { messages: clean, reason: "busy" }; }
    this.#active = true;
    this.#injected = undefined;
    const generation = this.#generation;
    const signal = this.#controller.signal;
    const bodies: SkillBody[] = [];
    let reason: Reason = "backend_unavailable";
    let unsupported = true;
    try {
      this.#explicit = explicitCommands(clean);
      const artifact = await this.options.policy();
      const action = this.#pending.values().next().value?.action ?? null;
      const policy = this.#policy(artifact, ctx.agent.id, action, true);
      unsupported = policy.unsupported;
      const lease = await this.options.catalog.getCatalog(ctx);
      if (policy.conflict) fail("policy_conflict");
      if (policy.requiredIds.length > 32) fail("context_overflow");
      if (policy.requiredIds.some(id => !lease.catalog.entries.some(entry => entry.runtime_id === id && entry.explicit_eligible))) fail("missing_required");
      // Required bodies are independent of optional inference and remain available on abstention.
      for (const id of policy.requiredIds) {
        const entry = lease.catalog.entries.find(entry => entry.runtime_id === id)!;
        bodies.push(await this.options.catalog.getSkillBody(ctx, id, lease.catalog.catalog_sha256, entry.content_sha256, lease));
      }
      let intent: string;
      try { intent = compileIntent(clean); } catch { intent = ""; reason = "context_overflow"; }
      if (reason !== "context_overflow") {
        try {
          const scorers = this.#scorers ??= await this.options.createScorers(signal);
          if (this.#closed) { scorers.shortlister.close(); scorers.decider.close(); fail("stale_result"); }
          if (signal.aborted || generation !== this.#generation) fail("stale_result");
          await scorers.shortlister.verify();
          await scorers.decider.verify();
          const key = digest({ intent, action, generation, catalog: lease.catalog.catalog_sha256, policy: policy.policyDigest,
            shortlister: scorers.shortlister.bundle.selector_sha256, selector: scorers.decider.bundle.selector_sha256 });
          const cached = this.#decision;
          const response = cached?.key === key && cached.lease.owner === lease.owner && this.options.catalog.current(ctx, cached.lease)
            ? cached.response
            : (await route({ catalog: lease.catalog, policy, intent, action, event: "context", sessionKey: this.#sessionKey,
              generation, scorers, signal, isCurrent: () => generation === this.#generation && this.options.catalog.current(ctx, lease) })).response;
          if (!signal.aborted && generation === this.#generation && response.scores) this.#decision = { key, lease, response };
          reason = response.reasonCode;
          if (response.status === "selected") {
            for (const id of response.selectedIds.filter(id => !policy.requiredIds.includes(id))) {
              const entry = lease.catalog.entries.find(entry => entry.runtime_id === id)!;
              bodies.push(await this.options.catalog.getSkillBody(ctx, id, lease.catalog.catalog_sha256, entry.content_sha256, lease));
            }
          }
          await scorers.decider.verify();
          await scorers.shortlister.verify();
        } catch (error) {
          bodies.splice(policy.requiredIds.length);
          reason = errorCode(error);
        }
      }
      let content = renderBodies(bodies, reason, unsupported);
      if (Buffer.byteLength(content) > this.options.bodyBudgetBytes) {
        bodies.splice(policy.requiredIds.length);
        reason = "context_overflow";
        content = renderBodies(bodies, reason, unsupported);
        if (Buffer.byteLength(content) > this.options.bodyBudgetBytes) fail("context_overflow");
      }
      await this.options.catalog.recheck(ctx, lease);
      if (signal.aborted || generation !== this.#generation || digest(await this.options.policy()) !== digest(artifact)) fail("stale_result");
      this.#injected = { lease, bodies: new Map(bodies.map(body => [body.id, body])), policyDigest: digest(artifact) };
      return { messages: [...clean, { role: "custom", customType: BLOCK_TYPE, content, display: false,
        attribution: "agent", timestamp: Date.now() }], reason };
    } catch (error) {
      reason = errorCode(error);
      // Do not return any old body when ownership or a digest changed during this request.
      if (generation === this.#generation) this.#injected = undefined;
      return { messages: [...clean, { role: "custom", customType: BLOCK_TYPE,
        content: `Skill router: ${reason}. No bodies injected. Unsupported natural-language policy is not automatically enforced. Native permissions are unchanged.`,
        display: false, attribution: "agent", timestamp: Date.now() }], reason };
    } finally { this.#active = false; }
  }
  async toolCall(event: ToolCallEvent, ctx: HostContext): Promise<{ block: true; reason: string } | undefined> {
    try {
      // The deterministic predicate needs only tool/actor identity, never opaque input programs.
      const action: NonNullable<Action> = { toolName: event.toolName, summary: "" };
      const artifact = await this.options.policy();
      const policy = this.#policy(artifact, ctx.agent.id, action, false);
      if (policy.requiredIds.length === 0 && !policy.conflict) return undefined;
      if (this.#closed) fail("backend_unavailable");
      if (policy.conflict) fail("policy_conflict");
      const key = `${ctx.agent.id}\u0000${event.toolName}`;
      if (!this.#pending.has(key) && this.#pending.size >= 16) fail("context_overflow");
      this.#pending.set(key, { actorId: ctx.agent.id, action: { toolName: action.toolName, summary: "" } });
      // Local ordering/freshness only. Later context transforms must preserve bodies;
      // compatibility is measured at provider entry, not certified by this state.
      const injected = this.#injected;
      if (!injected || injected.policyDigest !== digest(artifact)) fail("missing_required");
      await this.options.catalog.recheck(ctx, injected.lease);
      for (const id of policy.requiredIds) {
        const body = injected.bodies.get(id);
        if (!body) fail("missing_required");
        await this.options.catalog.getSkillBody(ctx, id, injected.lease.catalog.catalog_sha256, body.contentDigest, injected.lease);
      }
      if (this.#injected !== injected || !this.options.catalog.current(ctx, injected.lease) ||
          digest(await this.options.policy()) !== digest(artifact)) fail("stale_result");
      this.#blocked = 0;
      return undefined;
    } catch (error) {
      // Failure to resolve trusted policy is itself an unresolved prerequisite. Optional
      // model failure never reaches this guard, and never creates a permission denial.
      const code = errorCode(error);
      this.#blocked++;
      return { block: true, reason: `skill_router:${code}. ${this.#blocked > 2 ?
        "Repeated prerequisite failure; stop retrying and resolve the reported policy or body failure." :
        "Required instructions have not been injected by this router or are no longer current. Regenerate after the next context request."}` };
    }
  }
}
