import { readFile, realpath } from "node:fs/promises";
import { isAbsolute } from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { contained, LiveCatalog, type Registry } from "../src/catalog.ts";
import { fail, parseStrict } from "../src/protocol.ts";
import { validatePolicy, type PolicyArtifact, type Scorers } from "../src/router.ts";
import { normalAgentHandler, SessionRouter } from "../src/session.ts";
import { memoryAdmission, WorkerClient, type WorkerPaths } from "../src/worker.ts";

export interface ExtensionOptions {
  createScorers: (signal: AbortSignal) => Promise<Scorers> | Scorers; policy: () => Promise<PolicyArtifact>; bodyBudgetBytes: number; registry?: Registry;
  exportRoot?: string;
}
/** Each host invocation creates fresh session state; no parent session is captured. */
export function createRouterExtension(options: ExtensionOptions): (pi: ExtensionAPI) => Promise<void> {
  return async pi => {
    if (pi.pi.VERSION !== "18.18.0") fail("backend_unavailable");
    const registry = options.registry ?? pi.pi.AgentRegistry.global();
    const catalog = new LiveCatalog(registry);
    const router = new SessionRouter({ ...options, catalog });
    if (options.exportRoot) {
      const root = options.exportRoot;
      pi.registerCommand("skill-router-export-catalog", {
        description: "Export the current authoritative catalog to a new private task file",
        handler: normalAgentHandler(async (args, ctx) => {
          try {
            if (!isAbsolute(args.trim())) fail("body_unavailable");
            await catalog.exportPrivate(ctx, args.trim(), root);
            ctx.ui.notify("skill_router:catalog_exported", "info");
          } catch { ctx.ui.notify("skill_router:catalog_export_failed", "error"); }
        }),
      });
    }
    let scopeAnnounced = false;
    pi.on("session_start", normalAgentHandler((_event, ctx) => {
      router.invalidate();
      if (!scopeAnnounced) {
        ctx.ui.notify("Skill router autoload covers main and supported child agents only. Built-in advisors are excluded; their configuration and native permissions are unchanged.", "info");
        scopeAnnounced = true;
      }
    }));
    pi.on("session_switch", normalAgentHandler(() => router.invalidate()));
    pi.on("session_branch", normalAgentHandler(() => router.invalidate()));
    pi.on("session_tree", normalAgentHandler(() => router.invalidate()));
    pi.on("session_compact", normalAgentHandler(() => router.invalidate()));
    pi.on("before_agent_start", normalAgentHandler(() => router.invalidate()));
    pi.on("session_shutdown", normalAgentHandler(() => router.shutdown()));
    pi.on("context", normalAgentHandler(async (event, ctx) => ({ messages: (await router.context(event.messages, ctx)).messages })));
    pi.on("tool_call", normalAgentHandler((event, ctx) => router.toolCall(event, ctx)));
  };
}
/** Retrieval and decision share one private process, admitted and started once. */
interface Configuration extends WorkerPaths { shortlistBundlePath: string; policyPath: string; bodyBudgetBytes: number; minAvailableMiB: number }
function configuration(value: unknown): Configuration {
  if (typeof value !== "object" || value === null || Array.isArray(value)) fail("backend_unavailable");
  const config = value as Record<string, unknown>;
  const names = ["taskRoot", "pythonPath", "workerPath", "bundlePath", "shortlistBundlePath", "policyPath", "bodyBudgetBytes", "minAvailableMiB"];
  if (Object.keys(config).sort().join() !== names.sort().join() ||
      names.slice().filter(name => name.endsWith("Path") || name === "taskRoot").some(name => typeof config[name] !== "string" || !isAbsolute(config[name])) ||
      !Number.isSafeInteger(config.bodyBudgetBytes) || Number(config.bodyBudgetBytes) < 1024 || Number(config.bodyBudgetBytes) > 262144 ||
      !Number.isSafeInteger(config.minAvailableMiB) || Number(config.minAvailableMiB) < 1024) fail("backend_unavailable");
  return config as unknown as Configuration;
}

/** Load the router from an absolute config file. An installed loader in
 * `~/.omp/agent/extensions/` passes its fixed path, because OMP started by another app
 * does not inherit a shell's environment variables. */
export function loadSkillRouter(configPath: string): (pi: ExtensionAPI) => Promise<void> {
  return async pi => {
    if (!isAbsolute(configPath)) fail("backend_unavailable");
    try {
      const bytes = await readFile(configPath);
      if (bytes.length > 16384) fail("context_overflow");
      const config = configuration(parseStrict(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
      const root = await realpath(config.taskRoot);
      if (!contained(root, await realpath(configPath)) || !contained(root, await realpath(config.policyPath))) fail("backend_unavailable");
      const policy = async (): Promise<PolicyArtifact> => {
        if (!contained(root, await realpath(config.policyPath))) fail("policy_conflict");
        const content = await readFile(config.policyPath);
        if (content.length > 65536) fail("policy_conflict");
        return validatePolicy(parseStrict(new TextDecoder("utf-8", { fatal: true }).decode(content)));
      };
      await policy();
      const admit = () => memoryAdmission(config.minAvailableMiB);
      // WorkerClient checks that every supplied value is an absolute path, so pass only
      // the four worker paths, never the whole configuration with its numeric limits.
      const { taskRoot, pythonPath, workerPath } = config;
      const createScorers = (signal: AbortSignal): Promise<Scorers> => WorkerClient.createScorers(
        { taskRoot, pythonPath, workerPath, bundlePath: config.shortlistBundlePath }, config.bundlePath, admit, signal);
      await createRouterExtension({ createScorers, policy, bodyBudgetBytes: config.bodyBudgetBytes, exportRoot: root })(pi);
    } catch { fail("backend_unavailable"); }
  };
}

/** `-e` entry for disposable runs: the config path comes from OMP_SKILL_ROUTER_CONFIG.
 * Explicit opt-in only; importing this module does not start a process or discover skills. */
export default async function skillRouter(pi: ExtensionAPI): Promise<void> {
  await loadSkillRouter(process.env.OMP_SKILL_ROUTER_CONFIG ?? "")(pi);
}
