import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const target = process.argv[2];
if (!target) throw new Error("Usage: bun scripts/verify-install.ts /path/to/prepared-installation");
const root = resolve(target);
const { digest } = await import(`${root}/src/protocol.ts`);
const { route, resolvePolicy } = await import(`${root}/src/router.ts`);
const { WorkerClient, memoryAdmission } = await import(`${root}/src/worker.ts`);
const probes: { catalog: [string, string][]; requests: string[] } = JSON.parse(
  await readFile(new URL("../fixtures/evaluation/release-probes.json", import.meta.url), "utf8"));
const catalog = {
  schema_version: 1, catalog_sha256: digest(probes.catalog), captured_at: "2026-01-01T00:00:00Z",
  loader_contract: "authored-release-probes-v1", eligibility_sha256: digest([]),
  entries: probes.catalog.map(([id, description]) => ({
    runtime_id: id, declared_name: id, description, content_sha256: digest(description), routing_sha256: digest([id, description]),
    resource_tree_sha256: null, source_kind: "fixture", source_ref: "authored-release-probes", hidden: false,
    automatic_eligible: true, explicit_eligible: true, family_id: null,
  })),
};
const scorers = await WorkerClient.createScorers({ taskRoot: root, pythonPath: `${root}/venv/bin/python`,
  workerPath: `${root}/python/retrieval-worker.py`, bundlePath: `${root}/shortlist-bundle` },
  `${root}/decision-bundle`, () => memoryAdmission(1024));
const policy = resolvePolicy({ version: 1, rules: [], unsupported: [] }, "Main", null, []);
try {
  const results = [];
  for (const [generation, intent] of probes.requests.entries()) {
    const result = await route({ catalog, policy, intent, action: null, sessionKey: "authored-release-check", generation,
      event: "context", scorers, signal: new AbortController().signal, isCurrent: () => true });
    if (!result.response.scores) throw new Error(`Authored workflow has no scores: ${result.response.reasonCode}`);
    results.push({ request: intent, selected_ids: result.response.selectedIds, status: result.response.status,
      reason: result.response.reasonCode, scores: result.response.scores.values });
  }
  console.log(JSON.stringify(results, null, 2));
} finally {
  scorers.shortlister.close();
}
