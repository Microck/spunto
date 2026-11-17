#!/usr/bin/env python3
"""Synthetic, bounded Linux profile. Starts and terminates only its own worker."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("router_worker", ROOT / "python/router-worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def system_evidence():
    evidence = {}
    for name in ("pressure/memory", "meminfo"):
        evidence[name] = (Path("/proc") / name).read_text()
    for row in Path("/proc/self/cgroup").read_text().splitlines():
        if row.startswith("0::"):
            group = Path("/sys/fs/cgroup") / row[3:].lstrip("/")
            for name in ("memory.peak", "memory.events", "memory.max", "memory.swap.max", "cpu.max"):
                path = group / name
                evidence[name] = path.read_text() if path.exists() else None
    return evidence


def synthetic_requests(manifest, tokenizer):
    digest = "sha256:" + hashlib.sha256(b"synthetic-profile-only").hexdigest()
    requests = []
    for count in (1, 8, 16, 32):
        for tokens in (64, 256, 512, 768):
            intent = "Review code."
            # A real tokenizer, not a word-count approximation, defines each pair length.
            for words in range(1, 769):
                description = " ".join(["test"] * words)
                if len(tokenizer.encode(intent, description).ids) >= tokens:
                    break
            worker.require(len(tokenizer.encode(intent, description).ids) == tokens)
            bag = worker.windows.build_bag(tokenizer, intent, None, description)
            worker.windows.admit_request([bag] * count)
            request = {"version": 1, "requestId": f"synthetic-{count}-{tokens}",
                       "sessionKey": "synthetic-profile", "generation": 0, "event": "context",
                       "catalogDigest": digest, "contextDigest": digest, "policyDigest": digest,
                       "selectorDigest": manifest["selector_sha256"], "intent": intent, "action": None,
                       "candidateCoverage": "full", "requiredIds": [], "excludedIds": [],
                       "maxSelections": manifest["selection"]["max_optional"], "candidates": [
                           {"id": f"candidate-{index}", "name": f"candidate-{index}",
                            "description": description, "contentDigest": digest} for index in range(count)]}
            requests.append((request, tokens))
    return requests


def validate_reply(request, reply, scorer):
    worker.require(isinstance(reply, dict))
    for key in worker.IDENTITY:
        worker.require(reply.get(key) == request[key])
    worker.require(reply.get("status") in ("selected", "none", "uncertain"))
    reasons = {"selected": {"matched", "policy_only"}, "none": {"no_match"},
               "uncertain": {"insufficient_evidence", "inadequate_candidates", "policy_conflict",
                             "missing_required", "selection_overflow", "context_overflow", "busy",
                             "timeout", "backend_unavailable", "invalid_response", "stale_result",
                             "body_unavailable"}}
    worker.require(reply.get("reasonCode") in reasons[reply["status"]])
    selected = reply.get("selectedIds")
    worker.ids(selected)
    allowed = {candidate["id"] for candidate in request["candidates"]}
    worker.require(set(selected) <= allowed and not set(selected) & set(request["excludedIds"]))
    worker.require(reply["status"] == "selected" or not selected)
    if reply["status"] == "selected":
        worker.require(bool(selected) and set(request["requiredIds"]) <= set(selected))
    scores = reply.get("scores")
    if scores is not None:
        worker.require(scores["kind"] == "ranking" and scores["calibrationDigest"] is None)
        worker.require(scores["scorerDigest"] == scorer)
        worker.ids([item["id"] for item in scores["values"]])
        worker.require({item["id"] for item in scores["values"]} ==
                       allowed - set(request["excludedIds"]))
        worker.require(all(item["id"] in allowed and type(item["score"]) in (float, int)
                           and math.isfinite(item["score"]) for item in scores["values"]))
    return reply


async def profile(args, manifest, fixtures, report):
    start = time.monotonic()
    end = start + args.deadline_seconds
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(args.worker), "--artifact", str(args.artifact),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, limit=worker.MAX_LINE + 1)
    peak_rss_kib = 0
    samples = []

    async def exchange(request, timeout):
        async def transaction():
            process.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
            await process.stdin.drain()
            line = await process.stdout.readline()
            worker.require(bool(line) and len(line) <= worker.MAX_LINE and line.endswith(b"\n"))
            return validate_reply(request, worker.decode_json(line), manifest["scorer_sha256"])
        remaining = min(timeout, end - time.monotonic())
        if remaining <= 0:
            raise TimeoutError
        started = time.perf_counter_ns()
        reply = await asyncio.wait_for(transaction(), remaining)
        return reply, (time.perf_counter_ns() - started) / 1_000_000

    try:
        first, first_ms = await exchange(fixtures[0][0], args.startup_timeout_seconds)
        worker.require(first["scores"] is not None)
        report["startup_to_first_response_ms"] = (time.monotonic() - start) * 1000
        report["first_exchange_ms"] = first_ms
        for index in range(args.warmup + args.requests):
            request, tokens = fixtures[index % len(fixtures)]
            reply, elapsed = await exchange(request, args.request_timeout_ms / 1000)
            worker.require(reply["scores"] is not None and reply["reasonCode"] not in (
                "backend_unavailable", "invalid_response", "stale_result", "context_overflow"))
            status = Path(f"/proc/{process.pid}/status").read_text()
            peak_rss_kib = max(peak_rss_kib, next(int(line.split()[1]) for line in status.splitlines()
                                               if line.startswith("VmHWM:")))
            if index >= args.warmup:
                samples.append({"candidates": len(request["candidates"]), "pair_tokens": tokens,
                                "elapsed_ms": elapsed, "status": reply["status"],
                                "reasonCode": reply["reasonCode"]})
        overflow = json.loads(json.dumps(fixtures[-1][0]))
        overflow["candidates"][0]["description"] = "test " * 1100
        reply, _ = await exchange(overflow, args.request_timeout_ms / 1000)
        worker.require(reply["status"] == "uncertain" and reply["reasonCode"] == "context_overflow")
        report["overflow_rejection"] = True
    finally:
        # Never enumerate or signal other processes. Cleanup is bounded even after timeout.
        if process.stdin is not None:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 1)
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 1)
            except asyncio.TimeoutError:
                if process.returncode is None:
                    process.kill()
                await process.wait()
        report["child_returncode"] = process.returncode
        report["worker_peak_rss_kib"] = peak_rss_kib
        report["samples"] = samples
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        report["children"] = {"maxrss_kib": usage.ru_maxrss, "minor_faults": usage.ru_minflt,
                              "major_faults": usage.ru_majflt, "user_seconds": usage.ru_utime,
                              "system_seconds": usage.ru_stime}
    worker.require(process.returncode == 0)
    durations = sorted(sample["elapsed_ms"] for sample in samples)
    report["warm"] = {"p50_ms": statistics.median(durations),
                      "p95_ms": durations[math.ceil(len(durations) * .95) - 1],
                      "max_ms": max(durations),
                      "requests_per_second": 1000 * len(durations) / sum(durations)}


def main():
    parser = argparse.ArgumentParser(description="Profile only a newly owned offline worker with synthetic pairs.")
    parser.add_argument("--worker", type=Path, default=ROOT / "python/router-worker.py")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON evidence file; never overwritten.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--deadline-seconds", type=float, default=45)
    parser.add_argument("--request-timeout-ms", type=float, default=750)
    parser.add_argument("--startup-timeout-seconds", type=float, default=10)
    args = parser.parse_args()
    if not (0 <= args.warmup <= 12 and 16 <= args.requests <= 120
            and 0 < args.deadline_seconds <= 45 and 0 < args.request_timeout_ms <= 750
            and 0 < args.startup_timeout_seconds <= 10):
        parser.error("profile bounds exceeded")
    worker.configure_environment()
    report = {"status": "failed", "synthetic_only": True}
    # Reserve the output before launching anything, so existing evidence is never overwritten.
    with args.output.open("x", encoding="utf-8") as output:
        try:
            report["before"] = system_evidence()
            manifest, artifacts = worker.load_manifest(args.artifact)
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_str(artifacts[manifest["tokenizer_file"]].decode("utf-8"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            fixtures = synthetic_requests(manifest, tokenizer)
            del artifacts, tokenizer
            asyncio.run(profile(args, manifest, fixtures, report))
            report["status"] = "passed"
        except Exception as exc:
            report["failure"] = "timeout" if isinstance(exc, TimeoutError) else "profile_failed"
        finally:
            report["after"] = system_evidence()
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
