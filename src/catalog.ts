import { constants } from "node:fs";
import { open, realpath, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import type { ExtensionContext, Skill } from "@oh-my-pi/pi-coding-agent";
import { assertCatalog, bytesDigest, compare, digest, fail, type Catalog, type Digest, type Entry } from "./protocol.ts";

export interface HostContext {
  agent: ExtensionContext["agent"];
  sessionManager: Pick<ExtensionContext["sessionManager"], "getSessionId">;
}
export interface Owner {
  readonly skills: readonly Skill[];
  readonly isDisposed: boolean;
  sessionManager: Pick<ExtensionContext["sessionManager"], "getSessionId">;
}
/** OMP 18.18 types registry entries as `AgentSessionHandle`, which does not declare
 * `skills` or `isDisposed`. Main-thread entries are the live session that has both; a
 * worker proxy does not. `owner()` accepts only the full shape and fails closed otherwise. */
export interface Registry { get(id: string): { session: object | null } | undefined }
function isOwner(session: object): session is Owner {
  const candidate = session as Partial<Owner>;
  return Array.isArray(candidate.skills) && typeof candidate.isDisposed === "boolean" &&
    typeof candidate.sessionManager?.getSessionId === "function";
}
export interface Lease {
  readonly owner: Owner; readonly sessionId: string; readonly actorId: string;
  readonly generation: number; readonly records: ReadonlyMap<string, Skill>;
  readonly catalog: Catalog;
}
export interface SkillBody { id: string; contentDigest: Digest; text: string; baseDir: string; containRoot: string }
const LOADER = "omp-18.16.0-injected-registry-v1";
const MAX_BODY_BYTES = 262144;
export const contained = (root: string, path: string): boolean => {
  const rel = relative(root, path);
  return rel === "" || (!isAbsolute(rel) && rel !== ".." && !rel.startsWith(`..${sep}`));
};
function recordDigest(skill: Skill): Digest {
  return digest({ name: skill.name, description: skill.description, filePath: skill.filePath,
    baseDir: skill.baseDir, source: skill.source, hide: skill.hide === true, containRoot: skill.containRoot ?? null });
}
/** Bound reads and verify the opened inode, not merely a pathname checked before open. */
async function readSkill(skill: Skill): Promise<{ bytes: Buffer; baseDir: string; containRoot: string }> {
  try {
    const baseDir = await realpath(skill.baseDir);
    const containRoot = await realpath(skill.containRoot ?? skill.baseDir);
    const path = await realpath(skill.filePath);
    if (!contained(containRoot, baseDir) || !contained(containRoot, path)) fail("body_unavailable");
    const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      // This adapter targets the verified Linux host. /proc resolves the opened descriptor,
      // catching parent-directory replacement between realpath and open.
      const openedPath = await realpath(`/proc/self/fd/${file.fd}`);
      if (!contained(containRoot, openedPath)) fail("body_unavailable");
      const before = await file.stat();
      if (!before.isFile() || before.size > MAX_BODY_BYTES) fail("context_overflow");
      const bytes = Buffer.alloc(before.size + 1);
      let offset = 0;
      while (offset < bytes.length) {
        const chunk = await file.read(bytes, offset, bytes.length - offset, offset);
        if (!chunk.bytesRead) break;
        offset += chunk.bytesRead;
      }
      const after = await file.stat();
      if (offset !== before.size || before.size !== after.size || before.mtimeMs !== after.mtimeMs ||
          before.ctimeMs !== after.ctimeMs || await realpath(skill.filePath) !== openedPath) fail("stale_result");
      return { bytes: bytes.subarray(0, offset), baseDir, containRoot };
    } finally { await file.close(); }
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("skill_router:")) throw error;
    return fail("body_unavailable");
  }
}

export class LiveCatalog {
  #generation = 0;
  constructor(private readonly registry: Registry) {}
  invalidate(): void { this.#generation++; }
  owner(ctx: HostContext): Owner {
    const actor = ctx.agent;
    const id = actor.kind === "sub" && actor.id === "advisor" ? actor.parentId : actor.id;
    const owner = id ? this.registry.get(id)?.session : null;
    if (!owner || !isOwner(owner) || owner.isDisposed || owner.sessionManager.getSessionId() !== ctx.sessionManager.getSessionId()) fail("backend_unavailable");
    return owner;
  }
  current(ctx: HostContext, lease: Lease): boolean {
    try {
      const owner = this.owner(ctx);
      return owner === lease.owner && lease.generation === this.#generation &&
        owner.sessionManager.getSessionId() === lease.sessionId &&
        owner.skills.length === lease.records.size && owner.skills.every(skill => {
          const old = lease.records.get(skill.name);
          return old !== undefined && recordDigest(old) === recordDigest(skill);
        });
    } catch { return false; }
  }
  async getCatalog(ctx: HostContext): Promise<Lease> {
    const owner = this.owner(ctx);
    const sessionId = owner.sessionManager.getSessionId();
    const records = new Map<string, Skill>();
    for (const skill of owner.skills) {
      if (records.has(skill.name)) fail("invalid_response");
      records.set(skill.name, Object.freeze({ ...skill }));
    }
    const generation = this.#generation;
    const entries: Entry[] = [];
    for (const skill of [...records.values()].sort((a, b) => compare(a.name, b.name))) {
      const { bytes } = await readSkill(skill);
      const content_sha256 = bytesDigest(bytes);
      const flags = { runtime_id: skill.name, description: skill.description, hidden: skill.hide === true,
        automatic_eligible: skill.hide !== true && skill.description.length > 0, explicit_eligible: true, content_sha256 };
      entries.push(Object.freeze({ ...flags, declared_name: null, routing_sha256: digest(flags),
        resource_tree_sha256: null, source_kind: skill.source || "host", source_ref: digest({ source: skill.source, path: skill.filePath }), family_id: null }));
    }
    const eligibility_sha256 = digest({ adapter: LOADER, authority: "owner.session.skills", entries: entries.map(entry => ({
      runtime_id: entry.runtime_id, hidden: entry.hidden, automatic_eligible: entry.automatic_eligible, explicit_eligible: entry.explicit_eligible,
    })) });
    const projection = { schema_version: 1 as const, loader_contract: LOADER, eligibility_sha256, entries };
    const catalog = { ...projection, catalog_sha256: digest(projection), captured_at: new Date().toISOString() };
    assertCatalog(catalog);
    Object.freeze(entries); Object.freeze(catalog);
    const lease = Object.freeze({ owner, sessionId, actorId: ctx.agent.id, generation, records, catalog });
    if (!this.current(ctx, lease)) fail("stale_result");
    return lease;
  }
  async getSkillBody(ctx: HostContext, id: string, catalogDigest: Digest, contentDigest: Digest, lease: Lease): Promise<SkillBody> {
    if (!this.current(ctx, lease) || lease.catalog.catalog_sha256 !== catalogDigest) fail("stale_result");
    const skill = lease.records.get(id);
    const entry = lease.catalog.entries.find(entry => entry.runtime_id === id);
    if (!skill || !entry || entry.content_sha256 !== contentDigest) fail("stale_result");
    const { bytes, baseDir, containRoot } = await readSkill(skill);
    if (bytesDigest(bytes) !== contentDigest || !this.current(ctx, lease)) fail("stale_result");
    // Keep exact verified bytes, including frontmatter. Never parse a second catalog from it.
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return { id, contentDigest, text, baseDir, containRoot };
  }
  async recheck(ctx: HostContext, lease: Lease): Promise<void> {
    if (!this.current(ctx, lease)) fail("stale_result");
    const fresh = await this.getCatalog(ctx);
    if (fresh.owner !== lease.owner || fresh.catalog.catalog_sha256 !== lease.catalog.catalog_sha256 ||
        fresh.generation !== lease.generation) fail("stale_result");
  }
  /** The caller chooses a private task path. Never overwrite an earlier dataset snapshot. */
  async exportPrivate(ctx: HostContext, path: string, privateRoot: string): Promise<Catalog> {
    const root = await realpath(privateRoot);
    const parent = await realpath(dirname(resolve(path)));
    if (!contained(root, parent)) fail("body_unavailable");
    const lease = await this.getCatalog(ctx);
    await this.recheck(ctx, lease);
    await writeFile(resolve(parent, path.split(sep).at(-1)!), `${JSON.stringify(lease.catalog)}\n`, { flag: "wx", mode: 0o600 });
    return lease.catalog;
  }
}
