"""Exercise both stages on authored requests. Model weights are required, but no network is used."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROBES = json.loads((Path(__file__).resolve().parents[1] / "fixtures/evaluation/release-probes.json").read_text())
CATALOG = PROBES["catalog"]
REQUESTS = PROBES["requests"]


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()


def check(args):
    retrieval = json.loads((args.model_root / "shortlist-bundle/manifest.json").read_text())
    decision = json.loads((args.model_root / "decision-bundle/manifest.json").read_text())
    candidates = [{"id": name, "name": name, "description": description,
                   "contentDigest": digest(description)} for name, description in CATALOG]
    command = [str(args.python), "-I", "-B", str(args.runtime_root / "python/retrieval-worker.py"),
               "--artifact", str(args.model_root / "shortlist-bundle"),
               "--decision-artifact", str(args.model_root / "decision-bundle")]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as worker:
        def score(index, intent, manifest, supplied):
            request = {"version": 1, "requestId": str(index), "sessionKey": "authored-release-check",
                       "generation": index, "event": "context", "catalogDigest": digest(CATALOG),
                       "policyDigest": digest({}), "contextDigest": digest(intent),
                       "selectorDigest": manifest["selector_sha256"], "intent": intent, "action": None,
                       "candidates": sorted(supplied, key=lambda candidate: candidate["id"]),
                       "candidateCoverage": "shortlist", "requiredIds": [], "excludedIds": [],
                       "maxSelections": 0 if manifest.get("role") == "retrieval" else 6}
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError("Worker exited before returning an authored check result")
            response = json.loads(line)
            if response.get("scores") is None:
                raise RuntimeError("Authored workflow returned no model scores: " + response["reasonCode"])
            return response
        results = []
        for index, intent in enumerate(REQUESTS):
            ranking = score(index, intent, retrieval, candidates)
            ranked = sorted(ranking["scores"]["values"], key=lambda value: (-value["score"], value["id"]))
            shortlist = ranked[:8]
            ids = {value["id"] for value in shortlist}
            response = score(index, intent, decision, [candidate for candidate in candidates if candidate["id"] in ids])
            results.append({"request": intent, "shortlist": shortlist, "selected_ids": response["selectedIds"],
                            "status": response["status"], "reason": response["reasonCode"],
                            "scores": response["scores"]["values"]})
        worker.stdin.close()
        if worker.wait(timeout=10) != 0:
            raise RuntimeError("Worker did not exit cleanly")
    return results


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-root", type=Path, required=True)
    cli.add_argument("--runtime-root", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument("--python", type=Path, default=Path(sys.executable))
    print(json.dumps(check(cli.parse_args()), indent=2))
