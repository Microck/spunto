#!/usr/bin/env python3
"""Private offline evaluation. No fitting, source-history parsing, or network calls."""
import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import os
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("no-router", "lexical", "pretrained", "finetuned", "oracle")
STRATA = set("natural-frequency challenge renamed unseen-description negation review-only no-deploy multi-skill stage-change false-friend overlap explicit hidden disabled insufficient-info restore".split())
MEASURES = ("compliance", "completed_task", "corrections", "input_tokens", "output_tokens", "latency_ms")
IDENTITY = ("version", "requestId", "sessionKey", "generation", "catalogDigest", "contextDigest", "policyDigest", "selectorDigest")
CONTEXT = ("version", "event", "intent", "action", "candidates", "candidateCoverage", "requiredIds", "excludedIds", "maxSelections", "policyDigest", "selectorDigest")
MAX_RECORD = 8 * 1024 * 1024


def require(condition):
    if not condition:
        raise ValueError("evaluation_contract")


def keys(value, names):
    require(isinstance(value, dict) and set(value) == set(names))


def digest(value):
    require(isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            and all(c in "0123456789abcdef" for c in value[7:]))


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result)
        result[key] = value
    return result


def decode(payload):
    def reject(_):
        raise ValueError("invalid_json")
    value = json.loads(payload, object_pairs_hook=unique, parse_constant=reject)
    # Encoding rejects unpaired surrogates; allow_nan also rejects numeric overflow.
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def hash_value(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def hash_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def validators():
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource
    schemas = [decode(path.read_bytes()) for path in sorted((ROOT / "schemas").glob("*.schema.json"))]
    registry = Registry().with_resources((s["$id"], Resource.from_contents(s)) for s in schemas)
    result = {}
    for schema in schemas:
        name = schema["$id"].rsplit("/", 1)[1].split(".")[0]
        result[name] = Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
        if name == "routing":
            for definition in ("request", "response"):
                result[definition] = Draft202012Validator(
                    {"$ref": schema["$id"] + "#/$defs/" + definition},
                    registry=registry, format_checker=FormatChecker())
    return result


def validate_contextual_design(plan, design, design_sha256):
    """An operator pins the separately reviewed evaluation design, not a payload flag."""
    if design is None or design_sha256 is None:
        raise ValueError("contextual_evaluation_design_required")
    digest(design_sha256)
    require(hash_file(Path(design)) == design_sha256)
    approved = decode(Path(design).read_bytes())
    keys(approved, ("version", "target_sha256", "contextual_binding", "plan_sha256", "status"))
    require(approved["version"] == 1 and approved["status"] == "authorized_contextual_evaluation")
    require(approved["plan_sha256"] == hash_value(plan))
    if "contextual_binding" not in plan:
        raise ValueError("missing_contextual_target")
    require(approved["contextual_binding"] == plan["contextual_binding"] and
            approved["target_sha256"] == plan["contextual_binding"]["target_sha256"])
    return approved


def validate_contextual_gold(case, admission):
    """Check new-target gold against reviewed pairs, never the historical broad label."""
    if "contextual_gold" not in case:
        raise ValueError("contextual_evaluation_gold_required")
    gold = case["contextual_gold"]
    keys(gold, ("decision", "selected_ids", "verified_negative_ids", "unknown_ids",
                "acceptable_alternative_sets", "activate_now", "earliest_supported_event",
                "activation_deadline_event"))
    record = case["record"]
    candidates = set(record["input"]["candidate_ids"])
    for name in ("selected_ids", "verified_negative_ids", "unknown_ids", "activate_now"):
        require(isinstance(gold[name], list) and all(isinstance(key, str) for key in gold[name]) and
                len(gold[name]) == len(set(gold[name])) and set(gold[name]) <= candidates)
    reviewed = [pair for pair in admission.by_example[record["example_id"]]
                if pair["pair_id"] in admission.approved]
    positive = {pair["runtime_id"] for pair in reviewed if pair["status"] == "contextual"}
    negative = {pair["runtime_id"] for pair in reviewed if pair["status"] in ("response_only", "nonapplicable")}
    unknown = candidates - positive - negative
    require(set(gold["selected_ids"]) == positive and set(gold["verified_negative_ids"]) == negative and
            set(gold["unknown_ids"]) == unknown)
    require(gold["decision"] == ("selected" if positive else "uncertain" if unknown else "none"))
    loaded = {item["runtime_id"]: item["content_sha256"] for item in record["input"]["already_loaded"]}
    entries = {entry["runtime_id"]: entry for entry in case["catalog"]["entries"]}
    require(set(gold["activate_now"]) == {skill for skill in positive
            if loaded.get(skill) != entries[skill]["content_sha256"]})
    alternatives = gold["acceptable_alternative_sets"]
    require(isinstance(alternatives, list) and (positive or not alternatives))
    for alternative in alternatives:
        require(isinstance(alternative, list) and alternative and
                all(isinstance(skill, str) for skill in alternative) and
                len(alternative) == len(set(alternative)) and set(alternative) <= candidates - negative)
    for key in ("earliest_supported_event", "activation_deadline_event"):
        require(gold[key] is None or isinstance(gold[key], str))


def validate_plan(plan, validate, dataset, audit_dir, index, *, design=None, design_sha256=None, admission=None):
    validate_contextual_design(plan, design, design_sha256)
    keys(plan, ("version", "dataset_sha256", "split_sha256", "admission_sha256", "baselines", "cases", "contextual_binding"))
    require(type(plan["version"]) is int and plan["version"] == 2)
    require(admission is not None and plan["contextual_binding"] == admission.binding)
    for field in ("dataset_sha256", "split_sha256", "admission_sha256"):
        digest(plan[field])
    require(plan["baselines"] == list(BASELINES) and isinstance(plan["cases"], list) and plan["cases"])
    require(len(plan["cases"]) <= 10000)
    seen, groups = set(), {}
    for case in plan["cases"]:
        keys(case, ("record", "catalog", "requests", "strata", "timing", "contextual_gold"))
        record, catalog = case["record"], case["catalog"]
        dataset.validate_record(record, catalog, audit_dir=admission.owners[record['example_id']], index=index, admission=admission)
        validate_contextual_gold(case, admission)
        mapped = dataset.dataset_to_runtime(record, catalog)
        require(record["split"] not in ("train", "quarantine"))
        require(record["example_id"] not in seen)
        seen.add(record["example_id"])
        for group in (record["source"]["conversation_group"], record["source"]["template_group"]):
            if group is not None:
                require(groups.setdefault(group, record["split"]) == record["split"])
        entries = catalog["entries"]
        by_id = {e["runtime_id"]: e for e in entries}
        inputs, label = record["input"], case["contextual_gold"]
        full_candidates = {c["id"]: c for c in mapped["candidates"]}
        require(record["audit"]["full_catalog_reviewed"] is True)
        require(inputs["candidate_coverage"] == "full")
        required, excluded = set(inputs["required_ids"]), set(inputs["excluded_ids"])
        require(isinstance(case["strata"], list) and case["strata"] and len(set(case["strata"])) == len(case["strata"]) and set(case["strata"]) <= STRATA)
        keys(case["requests"], BASELINES)
        for request in case["requests"].values():
            validate["request"].validate(request)
            require(all(request[k] == value for k, value in mapped.items()
                        if k not in ("candidates", "candidateCoverage")))
            require(request["contextDigest"] == hash_value({k: request[k] for k in CONTEXT}))
            candidates = request["candidates"]
            ids = [c["id"] for c in candidates]
            require(ids == sorted(set(ids)) and len(ids) <= 32)
            eligible = {e["runtime_id"] for e in entries if e["automatic_eligible"] and e["runtime_id"] not in excluded} | required
            require(set(ids) <= eligible and required <= set(ids))
            require(set(inputs["candidate_ids"]) == eligible)
            if request["candidateCoverage"] == "full":
                require(set(ids) == eligible)
            else:
                require(len(set(ids) - required) <= 16)
            for candidate in candidates:
                entry = by_id[candidate["id"]]
                require(candidate == full_candidates[candidate["id"]])
                require(entry["automatic_eligible"] or candidate["id"] in required and entry["explicit_eligible"])
        timing = case["timing"]
        if timing is not None:
            keys(timing, ("support_event", "deadline_event", "support_sequence", "deadline_sequence", "audit_sha256"))
            require(timing["support_event"] is not None and timing["support_event"] == label["earliest_supported_event"])
            require(timing["deadline_event"] is not None and timing["deadline_event"] == label["activation_deadline_event"])
            require(all(type(timing[k]) is int and timing[k] >= 0 for k in ("support_sequence", "deadline_sequence")))
            require(timing["support_sequence"] <= timing["deadline_sequence"])
            digest(timing["audit_sha256"])
            sidecar = dataset.blob(audit_dir, timing["audit_sha256"])
            require(sidecar == {k: v for k, v in timing.items() if k != "audit_sha256"})
    return {c["record"]["example_id"]: c for c in plan["cases"]}


def accepted_sets(case):
    label = case["contextual_gold"]
    return [set(label["selected_ids"]), *(set(x) for x in label["acceptable_alternative_sets"])]


def validate_observation(observation, case, manifests, validate, worker):
    keys(observation, ("version", "example_id", "baseline", "request", "response", "loaded_ids", "trace"))
    baseline = observation["baseline"]
    require(type(observation["version"]) is int and observation["version"] == 1 and baseline in BASELINES)
    require(observation["example_id"] == case["record"]["example_id"])
    request, reply = observation["request"], observation["response"]
    require(request == case["requests"][baseline])
    required, excluded = set(request["requiredIds"]), set(request["excludedIds"])
    entries = {e["runtime_id"]: e for e in case["catalog"]["entries"]}
    loaded = observation["loaded_ids"]
    if loaded is not None:
        require(isinstance(loaded, list) and all(isinstance(x, str) for x in loaded))
        require(len(loaded) == len(set(loaded)) and set(loaded) <= entries.keys() and not set(loaded) & excluded)
    if baseline in ("no-router", "oracle"):
        require(reply is None)
        if baseline == "oracle" and loaded is not None:
            require(case["contextual_gold"]["decision"] != "uncertain")
            require(any(set(loaded) == required | accepted for accepted in accepted_sets(case)))
    else:
        validate["response"].validate(reply)
        require(all(reply[k] == request[k] for k in IDENTITY))
        candidate_ids = {c["id"] for c in request["candidates"]}
        selected = set(reply["selectedIds"])
        require(selected <= candidate_ids and not selected & excluded)
        if reply["status"] == "selected":
            require(required <= selected and len(selected - required) <= request["maxSelections"])
            require((reply["reasonCode"] == "policy_only") == (selected == required))
        if reply["status"] == "none":
            require(not required and request["candidateCoverage"] == "full")
            require(request["intent"].strip() or request["action"] is not None and request["action"]["summary"].strip())
        scores = reply["scores"]
        if scores is not None:
            score_ids = [x["id"] for x in scores["values"]]
            require(len(score_ids) == len(set(score_ids)) and set(score_ids) <= candidate_ids - excluded)
            require(all(type(x["score"]) in (int, float) and math.isfinite(x["score"]) for x in scores["values"]))
        if baseline in manifests:
            manifest = manifests[baseline]
            require(request["selectorDigest"] == manifest["selector_sha256"])
            require(request["maxSelections"] <= manifest["selection"]["max_optional"])
            if scores is not None:
                require(scores["kind"] == "ranking" and scores["scorerDigest"] == manifest["scorer_sha256"] and scores["calibrationDigest"] is None)
                candidates = worker.preflight(request)
                by_id = {x["id"]: x["score"] for x in scores["values"]}
                require(set(by_id) == {c["id"] for c in candidates})
                expected = worker.decide(request, candidates, [by_id[c["id"]] for c in candidates],
                    float(manifest["selection"]["lower_logit"]), float(manifest["selection"]["upper_logit"]),
                    manifest["scorer_sha256"], manifest["selection"]["max_optional"])
                require(reply == expected)
            else:
                require(reply["status"] == "uncertain")
    trace = observation["trace"]
    if trace is not None:
        keys(trace, ("run_id", "fixture_sha256", "environment_sha256", "seed", "evidence_sha256", "measurements", "activations"))
        require(isinstance(trace["run_id"], str) and 0 < len(trace["run_id"]) <= 256)
        require(type(trace["seed"]) is int and trace["seed"] >= 0)
        for field in ("fixture_sha256", "environment_sha256", "evidence_sha256"):
            digest(trace[field])
        keys(trace["measurements"], MEASURES)
        for field, value in trace["measurements"].items():
            if value is None:
                continue
            if field in ("compliance", "completed_task"):
                require(type(value) is bool)
            elif field == "latency_ms":
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0)
            else:
                require(type(value) is int and value >= 0)
        if trace["activations"] is not None:
            require(isinstance(trace["activations"], list))
            for activation in trace["activations"]:
                keys(activation, ("runtime_id", "content_sha256", "event_sequence"))
                require(activation["runtime_id"] in entries)
                require(activation["content_sha256"] == entries[activation["runtime_id"]]["content_sha256"])
                require(type(activation["event_sequence"]) is int and activation["event_sequence"] >= 0)


def ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def timing_counts(case, observation):
    timing, trace = case["timing"], observation["trace"]
    if timing is None or trace is None or trace["activations"] is None:
        return {"unobserved": 1}
    counts = Counter()
    for identifier in case["contextual_gold"]["activate_now"]:
        events = [x["event_sequence"] for x in trace["activations"] if x["runtime_id"] == identifier]
        if not events:
            counts["missing_activation"] += 1
        elif min(events) < timing["support_sequence"]:
            counts["early"] += 1
        elif min(events) > timing["deadline_sequence"]:
            counts["late"] += 1
        else:
            counts["on_time"] += 1
    return dict(counts)


def conditional_prediction(request, reply, manifest):
    if reply["reasonCode"] != "policy_only":
        return reply["status"]
    # Policy membership says nothing about optional relevance. Require the same
    # full-catalog, all-negative evidence as a worker none decision.
    if manifest is None or request.get("candidateCoverage") != "full" or reply.get("scores") is None:
        return "uncertain"
    required, excluded = set(request["requiredIds"]), set(request["excludedIds"])
    optional = {candidate["id"] for candidate in request["candidates"]} - required - excluded
    scores = {item["id"]: item["score"] for item in reply["scores"]["values"]}
    adequate = request["intent"].strip() or request["action"] is not None and request["action"]["summary"].strip()
    lower = float(manifest["selection"]["lower_logit"])
    return "none" if adequate and optional <= scores.keys() and all(scores[key] <= lower for key in optional) else "uncertain"


def summarize(rows, expected, manifests=None):
    counts, confusion, timing = Counter(), Counter(), Counter()
    measurements = {name: [] for name in MEASURES}
    for case, observation in rows:
        record, request, reply = case["record"], observation["request"], observation["response"]
        label = case["contextual_gold"]
        positives, negatives, unknown = (set(label[k]) for k in ("selected_ids", "verified_negative_ids", "unknown_ids"))
        retrieved = {c["id"] for c in request["candidates"]}
        counts.update(known_pairs=len(positives | negatives), unknown_pairs=len(unknown),
                      all_pairs=len(positives | negatives | unknown), positives=len(positives),
                      negatives=len(negatives), retrieved_positive=len(positives & retrieved))
        if label["decision"] == "selected":
            counts["retrieval_sets"] += 1
            counts["retrieved_complete_set"] += any(s <= retrieved for s in accepted_sets(case))
        selected = None
        if reply is not None:
            selected = set(reply["selectedIds"])
            counts["responses"] += 1
            counts["abstentions"] += reply["status"] == "uncertain"
            predicted = conditional_prediction(request, reply, (manifests or {}).get(observation["baseline"]))
            confusion[label["decision"] + ":" + predicted] += 1
        elif observation["loaded_ids"] is not None:
            selected = set(observation["loaded_ids"])
        if selected is not None:
            counts["selection_observed"] += 1
            counts["observed_negatives"] += len(negatives)
            counts["tp"] += len(selected & positives)
            counts["fp"] += len(selected & negatives)
            counts["fn"] += len(positives - selected)
            counts["selected_unknown"] += len(selected & unknown)
            counts["selected_pairs"] += len(selected & (positives | negatives | unknown))
            if label["decision"] != "uncertain":
                counts["set_evaluable"] += 1
                # Exact-set credit is defined only on verified labels and explicit
                # accepted alternatives. Unknown selections are reported separately.
                accepted = accepted_sets(case)
                evaluable = positives | negatives | set().union(*accepted)
                conditional = selected & evaluable
                predicted = conditional_prediction(request, reply, (manifests or {}).get(observation["baseline"])) if reply else None
                counts["exact_set"] += any(conditional == s for s in accepted) and (reply is None or predicted != "uncertain")
        timing.update(timing_counts(case, observation))
        if observation["trace"] is not None:
            for name, value in observation["trace"]["measurements"].items():
                if value is not None:
                    measurements[name].append(value)
    counts["expected"] = expected
    counts["observed"] = len(rows)
    counts["missing"] = expected - len(rows)
    return {"counts": dict(counts), "known_pair_coverage": ratio(counts["known_pairs"], counts["all_pairs"]),
            "unknown_pair_coverage": ratio(counts["unknown_pairs"], counts["all_pairs"]),
            "selected_unknown_rate": ratio(counts["selected_unknown"], counts["selected_pairs"]),
            "precision": ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "verified_negative_activation_rate": ratio(counts["fp"], counts["observed_negatives"]),
            "shortlist_recall": ratio(counts["retrieved_positive"], counts["positives"]),
            "complete_bundle_retrieval": ratio(counts["retrieved_complete_set"], counts["retrieval_sets"]),
            "exact_set_accuracy": ratio(counts["exact_set"], counts["set_evaluable"]),
            "abstention_rate": ratio(counts["abstentions"], counts["responses"]),
            "confusion": dict(confusion), "activation_timing": dict(timing),
            "measurements": {name: {"measured": len(values), "missing": expected - len(values),
                "mean": ratio(sum(values), len(values)),
                "p95": sorted(values)[math.ceil(.95 * len(values)) - 1] if values and name == "latency_ms" else None}
                for name, values in measurements.items()}}


def score(plan, observations, manifests=None):
    cases = {case["record"]["example_id"]: case for case in plan["cases"]}
    lookup, run_ids, paired_identity = {}, set(), {}
    for observation in observations:
        key = (observation["example_id"], observation["baseline"])
        require(key[0] in cases and key[1] in BASELINES and key not in lookup)
        lookup[key] = observation
        trace = observation["trace"]
        if trace is not None:
            require(trace["run_id"] not in run_ids)
            run_ids.add(trace["run_id"])
            identity = tuple(trace[k] for k in ("fixture_sha256", "environment_sha256", "seed"))
            require(paired_identity.setdefault(key[0], identity) == identity)
    groups = defaultdict(list)
    for case in cases.values():
        split = case["record"]["split"]
        groups[split + "/all"].append(case)
        for stratum in case["strata"]:
            groups[split + "/" + stratum].append(case)
    results, paired = {}, {}
    for group, members in sorted(groups.items()):
        results[group] = {}
        paired[group] = {}
        for baseline in BASELINES:
            rows = [(case, lookup[(case["record"]["example_id"], baseline)]) for case in members if (case["record"]["example_id"], baseline) in lookup]
            results[group][baseline] = summarize(rows, len(members), manifests)
        for baseline in BASELINES:
            if baseline == "no-router":
                continue
            comparisons = {}
            for control in ("no-router", "lexical"):
                if baseline == control:
                    continue
                deltas = {name: [] for name in MEASURES}
                for case in members:
                    identifier = case["record"]["example_id"]
                    left, right = lookup.get((identifier, baseline)), lookup.get((identifier, control))
                    if left is None or right is None or left["trace"] is None or right["trace"] is None:
                        continue
                    for name in MEASURES:
                        a, b = left["trace"]["measurements"][name], right["trace"]["measurements"][name]
                        if a is not None and b is not None:
                            deltas[name].append(int(a) - int(b) if isinstance(a, bool) else a - b)
                comparisons[control] = {name: {"pairs": len(values), "missing_pairs": len(members) - len(values), "mean_difference": ratio(sum(values), len(values))} for name, values in deltas.items()}
            paired[group][baseline] = comparisons
    return {"version": 1, "plan_sha256": hash_value(plan), "label_authority": "machine_audited_not_human_gold", "admission": "not_decided_by_scorer", "results": results, "paired": paired}


def load_control(directory, worker):
    """Separate admission for the unchanged mean-logit control, never deployment."""
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    require(manifest_path.resolve(strict=True).is_relative_to(root) and manifest_path.stat().st_size <= worker.MAX_LINE)
    manifest = decode(manifest_path.read_bytes())
    if "contextual_binding" not in manifest.get("provenance", {}):
        raise ValueError("missing_contextual_target")
    keys(manifest, ("schema_version", "role", "control", "base", "revision", "pair_format", "geometry",
                    "files", "selection", "provenance", "limits", "publication_allowed", "scorer_sha256", "selector_sha256"))
    trainer = module(ROOT / "python/train-router.py")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1 and manifest["role"] == "evaluation_only"
            and manifest["control"] == "unchanged-pretrained-native-window-logit-arithmetic-mean"
            and worker.windows.MODEL_BASES.get(manifest["base"]) == manifest["revision"]
            and manifest["pair_format"] == trainer.PAIR_FORMAT and manifest["geometry"] == worker.windows.GEOMETRY
            and manifest["limits"] == trainer.LIMITS and manifest["publication_allowed"] is True)
    require(manifest["scorer_sha256"] == worker.canonical_hash({key: value for key, value in manifest.items()
            if key not in ("scorer_sha256", "selector_sha256")}))
    require(manifest["selector_sha256"] == worker.canonical_hash({"scorer_sha256": manifest["scorer_sha256"], "selection": manifest["selection"]}))
    paths, payloads = [], {}
    for entry in manifest["files"]:
        keys(entry, ("path", "sha256", "bytes"))
        name = entry["path"]
        require(isinstance(name, str) and Path(name).name == name and name not in (".", "..", "manifest.json"))
        path = root / name
        require(path.resolve(strict=True).is_relative_to(root) and path.is_file())
        raw = path.read_bytes()
        require(type(entry["bytes"]) is int and len(raw) == entry["bytes"] and
                "sha256:" + hashlib.sha256(raw).hexdigest() == entry["sha256"])
        paths.append(name)
        payloads[name] = raw
    require(paths == sorted(set(paths)) and {p.name for p in root.iterdir()} == set(paths) | {"manifest.json"})
    require({"control.onnx", "tokenizer.json", "config.json", "recipe.json", "validation.json", "evaluation.json", "parity.json", "training.lock"} == set(paths))
    selection = manifest["selection"]
    keys(selection, ("lower_logit", "upper_logit", "max_optional", "validation_sha256", "fit_split"))
    worker.verify_provenance_payloads(manifest, payloads, "evaluation_only")
    require(worker.finite_decimal(selection["lower_logit"]) < worker.finite_decimal(selection["upper_logit"]))
    require(selection["max_optional"] == 6 and selection["fit_split"] == "validation")
    return manifest, payloads


def execution_authority(path, pin):
    require(hash_file(path) == pin)
    authority = decode(Path(path).read_bytes())
    keys(authority, ("format", "status", "plan_sha256", "contextual_binding",
                     "evaluator_file_sha256", "worker_file_sha256", "bundle_file_sha256", "remote_export"))
    require(authority["format"] == "private-router-evaluation-execution-v1" and
            authority["status"] == "authorized_local_contextual_evaluation" and authority["remote_export"] is False)
    require(authority["evaluator_file_sha256"] == hash_file(__file__) and
            authority["worker_file_sha256"] == hash_file(ROOT / "python/router-worker.py"))
    require(authority["contextual_binding"]["fixture_only"] is False)
    keys(authority["bundle_file_sha256"], ("pretrained", "finetuned"))
    for value in authority["bundle_file_sha256"].values():
        digest(value)
    return authority


def control_worker(directory, approval, approval_sha256):
    worker = module(ROOT / "python/router-worker.py")
    worker.configure_environment()
    manifest, payloads = load_control(directory, worker)
    authority = execution_authority(approval, approval_sha256)
    require(authority["bundle_file_sha256"]["pretrained"] == hash_file(directory / "manifest.json") and
            worker.compact_source_set_binding(authority["contextual_binding"]) == manifest["provenance"]["contextual_binding"])
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    trainer = module(ROOT / "python/train-router.py")
    tokenizer = Tokenizer.from_str(payloads["tokenizer.json"].decode())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    pad_id = tokenizer.token_to_id("[PAD]")
    require(pad_id is not None)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(payloads["control.onnx"], sess_options=options, providers=["CPUExecutionProvider"])
    require([(n.name, n.type, n.shape) for n in session.get_inputs()] ==
            [(name, "tensor(int64)", ["batch", "sequence"]) for name in worker.windows.SIGNATURE])
    require([(n.name, n.type, n.shape) for n in session.get_outputs()] == [("logits", "tensor(float)", ["batch", 1])])
    while (request := worker.read_request(sys.stdin.buffer)) is not None:
        worker.validate_request(request)
        try:
            if request["selectorDigest"] != manifest["selector_sha256"]:
                raise worker.RoutingFailure("stale_result")
            candidates = worker.preflight(request)
            bags = worker.windows.admit_request(worker.windows.build_bags(tokenizer, request["intent"], request["action"], [c["description"] for c in candidates]))
            values = trainer.score_control_onnx(session, pad_id, bags, np)
            reply = worker.decide(request, candidates, values, float(manifest["selection"]["lower_logit"]),
                                  float(manifest["selection"]["upper_logit"]), manifest["scorer_sha256"])
        except worker.windows.ContextOverflow:
            reply = worker.response(request, "uncertain", "context_overflow")
        except worker.RoutingFailure as error:
            reply = worker.response(request, "uncertain", error.reason)
        print(json.dumps(reply, allow_nan=False), flush=True)
    return 0


async def collect(args, plan, manifest):
    requests = [(case, case["requests"][args.baseline]) for case in plan["cases"]]
    require(len(requests) <= args.max_requests)
    limits = manifest["limits"]
    started = time.monotonic()
    command = ([str(Path(__file__)), "control-worker", str(args.pretrained_bundle),
                str(args.execution_approval), args.execution_approval_sha256] if args.baseline == "pretrained" else
               [str(ROOT / "python/router-worker.py"), "--artifact", str(args.finetuned_bundle)])
    process = await asyncio.create_subprocess_exec(sys.executable, *command, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=limits["max_line_bytes"] + 1)
    observations = []
    try:
        for index, (case, request) in enumerate(requests):
            payload = canonical(request) + b"\n"
            require(len(payload) <= limits["max_line_bytes"])
            async def exchange():
                process.stdin.write(payload)
                await process.stdin.drain()
                line = await process.stdout.readline()
                require(line.endswith(b"\n") and len(line) <= limits["max_line_bytes"])
                return decode(line)
            timeout = min(args.deadline_seconds - (time.monotonic() - started),
                          limits["startup_timeout_ms" if index == 0 else "request_timeout_ms"] / 1000)
            require(timeout > 0)
            reply = await asyncio.wait_for(exchange(), timeout)
            observations.append({"version": 1, "example_id": case["record"]["example_id"], "baseline": args.baseline,
                "request": request, "response": reply, "loaded_ids": None, "trace": None})
    finally:
        # This PID is created here, never discovered from a live operator session.
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
    require(process.returncode == 0)
    return observations


def admit_bundle(directory, name, plan, worker, *, design=None, design_sha256=None):
    """Admit an integrity-checked control without claiming deployment quality."""
    validate_contextual_design(plan, design, design_sha256)
    manifest, payloads = load_control(directory, worker) if name == "pretrained" else worker.load_manifest(directory)
    require(manifest["provenance"].get("contextual_binding") == worker.compact_source_set_binding(plan["contextual_binding"]))
    require(manifest["provenance"]["dataset_sha256"] == plan["dataset_sha256"]
            and manifest["provenance"]["split_sha256"] == plan["split_sha256"])
    files = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    require(files.get("recipe.json") == manifest["provenance"]["recipe_sha256"])
    require(files.get("validation.json") == manifest["selection"]["validation_sha256"])
    require({"recipe.json", "validation.json", "evaluation.json", "parity.json"} <= files.keys())
    recipe = decode(payloads["recipe.json"])
    report = decode(payloads["validation.json"])
    exported = decode(payloads["evaluation.json"])
    parity = decode(payloads["parity.json"])
    role = "evaluation_only" if name == "pretrained" else "deployment_candidate"
    require(recipe["role"] == report["role"] == exported["role"] == role)
    require(recipe["weights"] == ("unchanged_pretrained" if name == "pretrained" else "fine_tuned"))
    require(report["fit_split"] == "validation" and report["dataset_sha256"] == plan["dataset_sha256"]
            and report["split_sha256"] == plan["split_sha256"]
            and report.get("contextual_binding") == worker.compact_source_set_binding(plan["contextual_binding"]))
    require(report["thresholds"] == [manifest["selection"]["lower_logit"], manifest["selection"]["upper_logit"]])
    require(type(report["targets_met"]) is bool and exported["quality_targets_met"] == report["targets_met"])
    require(exported["acceptance_passed"] is True and parity["comparison"]["passed"] is True)
    require(report["targets"] == recipe["targets"])
    require(report["artifact_validation_metrics"] == exported["metrics"]["validation"])
    fit = report["fit_metrics"]
    require(0 < fit["positives"] < fit["count"])
    trainer = module(ROOT / "python/train-router.py")
    require(report["targets_met"] == trainer.meets_targets(
        report["artifact_validation_metrics"], SimpleNamespace(**report["targets"])))
    if name == "pretrained":
        require(report["threshold_source"] == "pretrained_torch_validation")
        require(report["method"] in ("inclusive-observed-boundaries-max-coverage-v1",
                                    "validation-balanced-accuracy-full-coverage-v1"))
        require(recipe["baseline_control_method"] == report["method"])
    else:
        require(report["targets_met"] is True and report["threshold_source"] == "tuned_torch_validation")
    return manifest


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "control-worker":
        try:
            return control_worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        except Exception:
            print("evaluation_failed", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "collect"))
    for name in ("plan", "admission", "pretrained-bundle", "finetuned-bundle", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("frozen", "dataset-export", "audit-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--evaluation-design", type=Path, required=True)
    parser.add_argument("--evaluation-design-sha256", required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--execution-approval", type=Path, required=True)
    parser.add_argument("--execution-approval-sha256", required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--baseline", choices=("pretrained", "finetuned"))
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--max-requests", type=int, default=100)
    args = parser.parse_args()
    try:
        require(not args.output.exists())
        require(args.plan.stat().st_size <= 64 * 1024 * 1024)
        require(math.isfinite(args.deadline_seconds) and 0 < args.deadline_seconds <= 3600 and 0 < args.max_requests <= 10000)
        validate = validators()
        plan = decode(args.plan.read_bytes())
        dataset = module(ROOT / "python/build-dataset.py")
        _, _, contextual_admission, _ = dataset.verify_contextual_export(args.dataset_export, args.approval_sha256)
        index = dataset.open_index(args.source_index) if args.source_index else None
        try:
            cases = validate_plan(plan, validate, dataset, args.audit_dir, index,
                                  design=args.evaluation_design, design_sha256=args.evaluation_design_sha256,
                                  admission=contextual_admission)
        finally:
            if index is not None:
                index.close()
        frozen = dataset.verify_manifest(args.frozen)
        exported = dataset.verify_manifest(args.dataset_export)
        require(frozen["kind"] == "split" and exported["kind"] == "dataset")
        require(frozen["artifact_sha256"] in exported["parents"])
        require(exported["dataset_sha256"] == plan["dataset_sha256"])
        require(exported["split_sha256"] == frozen["split_sha256"] == plan["split_sha256"])
        require(hash_file(args.frozen / "dataset.jsonl") == frozen["dataset_sha256"])
        require(hash_file(args.frozen / "splits.jsonl") == frozen["split_sha256"])
        matched = set()
        for row in dataset.rows(args.frozen / "dataset.jsonl"):
            identifier = row["example_id"]
            if identifier in cases:
                require(identifier not in matched and cases[identifier]["record"] == row)
                matched.add(identifier)
        require(matched == cases.keys())
        require(hash_file(args.admission) == plan["admission_sha256"])
        worker = module(ROOT / "python/router-worker.py")
        manifests = {}
        for name in ("pretrained", "finetuned"):
            directory = getattr(args, name + "_bundle")
            manifest = admit_bundle(directory, name, plan, worker,
                                    design=args.evaluation_design, design_sha256=args.evaluation_design_sha256)
            manifests[name] = manifest
            for case in cases.values():
                request = case["requests"][name]
                require(request["selectorDigest"] == manifest["selector_sha256"] and request["maxSelections"] <= manifest["selection"]["max_optional"])
        authority = execution_authority(args.execution_approval, args.execution_approval_sha256)
        require(authority["plan_sha256"] == hash_value(plan) and
                authority["contextual_binding"] == contextual_admission.binding)
        for name in ("pretrained", "finetuned"):
            require(authority["bundle_file_sha256"][name] == hash_file(getattr(args, name + "_bundle") / "manifest.json"))
        if args.command == "collect":
            require(args.baseline is not None and args.observations is None)
            observations = asyncio.run(collect(args, plan, manifests[args.baseline]))
        else:
            require(args.observations is not None)
            observations = []
            with args.observations.open("rb") as source:
                while line := source.readline(MAX_RECORD + 1):
                    require(len(line) <= MAX_RECORD and line.endswith(b"\n"))
                    require(len(observations) < len(cases) * len(BASELINES))
                    observations.append(decode(line))
        for observation in observations:
            require(observation["example_id"] in cases)
            validate_observation(observation, cases[observation["example_id"]], manifests, validate, worker)
        report = score(plan, observations, manifests)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            if args.command == "collect":
                for observation in observations:
                    output.write(json.dumps(observation, ensure_ascii=True, allow_nan=False) + "\n")
            else:
                json.dump(report, output, ensure_ascii=True, allow_nan=False, indent=2)
                output.write("\n")
        print("evaluation_written")
        return 0
    except Exception:
        print("evaluation_failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
