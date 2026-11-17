#!/usr/bin/env python3
"""Offline, serialized JSONL relevance worker. The host owns all authorization."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import importlib.util

_spec = importlib.util.spec_from_file_location("pair_windows", Path(__file__).with_name("pair-windows.py"))
windows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(windows)

MAX_LINE = 128 * 1024
MAX_CANDIDATES = 32
MICROBATCH = 4
IDENTITY = ("version", "requestId", "sessionKey", "generation", "catalogDigest",
            "contextDigest", "policyDigest", "selectorDigest")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SURROGATE = re.compile(r"[\ud800-\udfff]")


class InvalidInput(ValueError):
    """A content-free boundary error, safe to classify without printing input."""


class RoutingFailure(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def require(condition):
    if not condition:
        raise InvalidInput("invalid_input")


def text(value, maximum, minimum=0):
    require(isinstance(value, str) and minimum <= len(value) <= maximum)
    require(SURROGATE.search(value) is None)


def digest(value):
    require(isinstance(value, str) and DIGEST.fullmatch(value) is not None)


def keys(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected))


def ids(value):
    require(isinstance(value, list))
    for item in value:
        text(item, 256, 1)
    require(len(value) == len(set(value)))


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj)
        obj[key] = value
    return obj


def reject_constant(_value):
    raise InvalidInput("invalid_json")


def decode_json(payload):
    try:
        return json.loads(payload, object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidInput("invalid_json") from exc


def read_request(stream):
    # A line includes its newline. Never drain an attacker-sized line into memory.
    line = stream.readline(MAX_LINE + 1)
    if not line:
        return None
    require(len(line) <= MAX_LINE and line.endswith(b"\n"))
    request = decode_json(line.decode("utf-8", errors="strict"))
    require(request is not None)
    return request


def validate_identity(request):
    require(isinstance(request, dict))
    require(type(request.get("version")) is int and request["version"] == 1)
    for key in ("requestId", "sessionKey"):
        text(request.get(key), 256, 1)
    require(type(request.get("generation")) is int and request["generation"] >= 0)
    for key in IDENTITY[4:]:
        digest(request.get(key))


def validate_request(request):
    validate_identity(request)
    keys(request, (*IDENTITY, "event", "intent", "action", "candidates",
                   "candidateCoverage", "requiredIds", "excludedIds", "maxSelections"))
    require(request["event"] in ("context", "tool_call"))
    require(request["candidateCoverage"] in ("full", "shortlist"))
    text(request["intent"], 32768)
    action = request["action"]
    if action is not None:
        keys(action, ("toolName", "summary"))
        text(action["toolName"], 256, 1)
        text(action["summary"], 2048)
    require(request["event"] != "tool_call" or action is not None)
    for key in ("requiredIds", "excludedIds"):
        ids(request[key])
    require(type(request["maxSelections"]) is int and request["maxSelections"] >= 0)
    require(isinstance(request["candidates"], list))
    seen = set()
    for candidate in request["candidates"]:
        keys(candidate, ("id", "name", "description", "contentDigest"))
        text(candidate["id"], 256, 1)
        text(candidate["name"], 256, 1)
        require(candidate["name"] == candidate["id"])
        text(candidate["description"], 16384)
        digest(candidate["contentDigest"])
        require(candidate["id"] not in seen)
        seen.add(candidate["id"])




def preflight(request):
    required, excluded = set(request["requiredIds"]), set(request["excludedIds"])
    available = {candidate["id"] for candidate in request["candidates"]}
    if required & excluded:
        raise RoutingFailure("policy_conflict")
    if not required <= available:
        raise RoutingFailure("missing_required")
    if len(request["candidates"]) > MAX_CANDIDATES:
        raise RoutingFailure("context_overflow")
    return [candidate for candidate in request["candidates"] if candidate["id"] not in excluded]


def response(request, status, reason, selected=(), scores=None):
    return {**{key: request[key] for key in IDENTITY}, "status": status,
            "selectedIds": list(selected), "scores": scores, "reasonCode": reason}


def decide(request, candidates, values, lower, upper, scorer_digest, max_optional=6):
    if request["maxSelections"] > max_optional:
        return response(request, "uncertain", "invalid_response")
    require(len(candidates) == len(values))
    require(all(type(value) in (int, float) and math.isfinite(value) for value in values))
    score_set = {"kind": "ranking", "scorerDigest": scorer_digest,
                 "calibrationDigest": None,
                 "values": [{"id": candidate["id"], "score": value}
                            for candidate, value in zip(candidates, values)]}
    required = set(request["requiredIds"])
    optional = [(candidate["id"], value) for candidate, value in zip(candidates, values)
                if candidate["id"] not in required]
    matched = [identifier for identifier, value in optional if value >= upper]
    if len(matched) > request["maxSelections"]:
        return response(request, "uncertain", "selection_overflow", scores=score_set)
    if not matched and any(lower < value < upper for _, value in optional):
        return response(request, "uncertain", "insufficient_evidence", scores=score_set)
    selected = required | set(matched)
    if selected:
        ordered = [candidate["id"] for candidate in candidates if candidate["id"] in selected]
        return response(request, "selected", "matched" if matched else "policy_only", ordered, score_set)
    if request["candidateCoverage"] != "full":
        return response(request, "uncertain", "inadequate_candidates", scores=score_set)
    if not request["intent"].strip() and (request["action"] is None or not request["action"]["summary"].strip()):
        return response(request, "uncertain", "insufficient_evidence", scores=score_set)
    return response(request, "none", "no_match", scores=score_set)


def configure_environment():
    for key, value in {
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value


class Router:
    def __init__(self, directory):
        configure_environment()
        self.manifest, artifacts = load_manifest(directory)
        # No imports with numerical thread pools happen before the environment is set.
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        self.tokenizer = Tokenizer.from_str(artifacts[self.manifest["tokenizer_file"]].decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        require(self.pad_id is not None)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 4
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        # ORT receives verified bytes for both self-contained graphs.
        self.sessions = {}
        for role, expected in self.manifest["graphs"].items():
            session = ort.InferenceSession(artifacts[expected["file"]], sess_options=options,
                                           providers=["CPUExecutionProvider"])
            require(session.get_providers() == ["CPUExecutionProvider"])
            for actual, declared in ((session.get_inputs(), expected["inputs"]),
                                     (session.get_outputs(), expected["outputs"])):
                require(len(actual) == len(declared))
                for node, spec in zip(actual, declared):
                    dtype = {"int64": "tensor(int64)", "float32": "tensor(float)"}[spec["dtype"]]
                    require(node.name == spec["name"] and node.type == dtype and node.shape == spec["shape"])
            self.sessions[role] = session

    def score(self, request, candidates):
        try:
            bags = windows.admit_request(windows.build_bags(self.tokenizer, request["intent"],
                request["action"], [candidate["description"] for candidate in candidates]))
        except windows.ContextOverflow:
            raise RoutingFailure("context_overflow") from None
        values = [0.] * len(bags)
        # Length bucketing reduces encoder padding without changing native bags
        # or outward candidate order. The microbatch and window bounds stay fixed.
        order = sorted(range(len(bags)), key=lambda i: len(bags[i].windows[0]["input_ids"]))
        # Geometry admits one window per candidate. Batch independent pairs at the
        # existing microbatch bound; every GRU row still starts from zero.
        for start in range(0, len(order), MICROBATCH):
            indices = order[start:start + MICROBATCH]
            batch = [bags[i] for i in indices]
            arrays = windows.padded_arrays([bag.windows[0] for bag in batch], self.pad_id, self.np)
            pooled = self.sessions["encoder"].run(["pooled"], arrays)[0]
            if pooled.shape != (len(batch), 128) or not self.np.isfinite(pooled).all():
                raise ValueError("invalid_encoder_output")
            hidden, logits = self.sessions["aggregator"].run(["hidden_out", "logit"], {
                "pooled": pooled,
                "positions": self.np.asarray([bag.positions[0] for bag in batch], dtype=self.np.float32),
                "hidden": self.np.zeros((len(batch), 32), dtype=self.np.float32)})
            if (hidden.shape != (len(batch), 32) or logits.shape != (len(batch), 1)
                    or not self.np.isfinite(hidden).all() or not self.np.isfinite(logits).all()):
                raise ValueError("invalid_aggregator_output")
            for index, value in zip(indices, logits[:, 0]):
                values[index] = float(value)
        return values

    def route(self, request):
        if request["selectorDigest"] != self.manifest["selector_sha256"]:
            return response(request, "uncertain", "stale_result")
        if request["maxSelections"] > self.manifest["selection"]["max_optional"]:
            return response(request, "uncertain", "invalid_response")
        try:
            candidates = preflight(request)
            values = self.score(request, candidates)
            selection = self.manifest["selection"]
            return decide(request, candidates, values, float(selection["lower_logit"]),
                          float(selection["upper_logit"]), self.manifest["scorer_sha256"],
                          selection["max_optional"])
        except RoutingFailure as exc:
            return response(request, "uncertain", exc.reason)
        except InvalidInput:
            return response(request, "uncertain", "invalid_response")
        except Exception:
            # Exceptions may contain inputs or runtime paths. Never send their text.
            return response(request, "uncertain", "backend_unavailable")


def main():
    parser = argparse.ArgumentParser(description="Serve one offline CPU model over bounded stdin/stdout JSONL.")
    parser.add_argument("--artifact", required=True, type=Path, help="Local immutable directory containing manifest.json.")
    args = parser.parse_args()
    try:
        router = Router(args.artifact)
    except Exception:
        print("backend_unavailable", file=sys.stderr)
        return 1
    while True:
        try:
            request = read_request(sys.stdin.buffer)
            if request is None:
                return 0
            validate_identity(request)
        except (InvalidInput, UnicodeError, OSError):
            print("invalid_response", file=sys.stderr)
            return 2
        try:
            validate_request(request)
        except InvalidInput:
            reply = response(request, "uncertain", "invalid_response")
        else:
            reply = router.route(request)
        try:
            print(json.dumps(reply, ensure_ascii=True, separators=(",", ":"), allow_nan=False), flush=True)
        except (BrokenPipeError, OSError):
            return 1


def canonical_hash(value):
    def validate(item):
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            require(abs(item) <= 9007199254740991)
        elif isinstance(item, str):
            text(item, len(item))
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                require(isinstance(key, str))
                validate(key)
                validate(child)
        else:
            raise InvalidInput("invalid_digest_input")
    validate(value)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def finite_decimal(value):
    require(isinstance(value, str) and re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", value) is not None)
    number = float(value)
    require(math.isfinite(number))
    return number


def scorer_digest(manifest):
    """Canonical deployment identity, shared by publication and admission."""
    return canonical_hash({key: manifest[key] for key in (
        "model", "graphs", "tokenizer_file", "model_config_file", "files",
        "pair_format", "geometry", "provenance")})


def compact_source_set_binding(binding):
    """Project admitted source provenance for serving; admission retains members."""
    return {key: value for key, value in binding.items() if key != "members"}


def validate_source_set_binding(binding):
    """Validate compact provenance, never an admission or execution grant."""
    keys(binding, ("version", "target_sha256", "catalog_sha256", "source_set_sha256",
                   "runtime_revision", "approval_sha256", "purpose", "fixture_only"))
    require(type(binding["version"]) is int and binding["version"] == 1 and
            binding["purpose"] == "local-canonical-integration" and
            type(binding["fixture_only"]) is bool)
    for key in ("target_sha256", "catalog_sha256", "source_set_sha256", "runtime_revision", "approval_sha256"):
        digest(binding[key])


def verify_provenance_payloads(manifest, payloads, role):
    """Bind exact inventoried bytes to source claims, not to external authority."""
    require(role in ("deployment_candidate", "evaluation_only"))
    provenance, selection = manifest["provenance"], manifest["selection"]
    keys(provenance, ("dataset_sha256", "split_sha256", "recipe_sha256", "runtime_revision", "contextual_binding"))
    binding = provenance["contextual_binding"]
    validate_source_set_binding(binding)
    for key in ("dataset_sha256", "split_sha256", "recipe_sha256"):
        digest(provenance[key])
    text(provenance["runtime_revision"], 4096, 1)
    require(provenance["runtime_revision"] == "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    documents = {}
    for name, expected in (("recipe.json", provenance["recipe_sha256"]),
                           ("validation.json", selection["validation_sha256"])):
        entries = [entry for entry in manifest["files"] if entry["path"] == name]
        require(len(entries) == 1 and name in payloads)
        raw = payloads[name]
        require(len(raw) == entries[0]["bytes"] and
                "sha256:" + hashlib.sha256(raw).hexdigest() == entries[0]["sha256"] == expected)
        documents[name] = decode_json(raw)
        require(isinstance(documents[name], dict))
    recipe, report = documents["recipe.json"], documents["validation.json"]
    require(isinstance(recipe.get("provenance"), dict))
    require(recipe.get("runtime_revision") == provenance["runtime_revision"])
    for key in ("dataset_sha256", "split_sha256", "contextual_binding"):
        require(recipe["provenance"].get(key) == report.get(key) == provenance[key])
    require(recipe.get("role") == report.get("role") == role)
    require(recipe.get("weights") == ("fine_tuned" if role == "deployment_candidate" else "unchanged_pretrained"))
    require(report.get("fit_split") == selection["fit_split"] == "validation")
    require(report.get("thresholds") == [selection["lower_logit"], selection["upper_logit"]])
    return recipe, report


def load_manifest(directory):
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    require(manifest_path.resolve().is_relative_to(root))
    require(manifest_path.stat().st_size <= MAX_LINE)
    manifest = decode_json(manifest_path.read_bytes())
    keys(manifest, ("schema_version", "model", "graphs", "tokenizer_file", "model_config_file",
                    "files", "pair_format", "geometry", "selection", "limits",
                    "calibration", "provenance", "scorer_sha256", "selector_sha256",
                    "calibration_sha256", "publication_allowed"))
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1)
    require(manifest["publication_allowed"] is True)
    require(manifest["pair_format"] == windows.PAIR_FORMAT)
    require(manifest["geometry"] == windows.GEOMETRY)
    require(canonical_hash(manifest["geometry"]) == canonical_hash(windows.GEOMETRY))
    keys(manifest["model"], ("type", "base", "revision"))
    require(manifest["model"]["type"] == "window_bag"
            and windows.MODEL_BASES.get(manifest["model"]["base"]) == manifest["model"]["revision"])
    require(manifest["graphs"] == windows.graph_contracts())
    require(canonical_hash(manifest["graphs"]) == canonical_hash(windows.graph_contracts()))
    graph_files = [graph["file"] for graph in manifest["graphs"].values()]
    selection = manifest["selection"]
    keys(selection, ("lower_logit", "upper_logit", "max_optional", "validation_sha256", "fit_split"))
    require(finite_decimal(selection["lower_logit"]) < finite_decimal(selection["upper_logit"]))
    require(type(selection["max_optional"]) is int and 0 <= selection["max_optional"] <= 6)
    require(selection["fit_split"] == "validation")
    digest(selection["validation_sha256"])
    expected_limits = {"max_candidates": 32, "max_line_bytes": 131072, "microbatch": 4,
                       "request_timeout_ms": 750, "startup_timeout_ms": 10000}
    require(manifest["limits"] == expected_limits)
    require(all(type(value) is int for value in manifest["limits"].values()))
    # This first deployment exposes raw logits, not probability calibration.
    require(manifest["calibration"] is None and manifest["calibration_sha256"] is None)
    # Provenance syntax and payload agreement are checked together below.
    entries = manifest["files"]
    require(isinstance(entries, list) and len(entries) >= 3)
    paths = []
    verified = {}
    for entry in entries:
        keys(entry, ("path", "sha256", "bytes"))
        relative = entry["path"]
        require(isinstance(relative, str) and re.fullmatch(
            r"[A-Za-z0-9_-]+(?:[./][A-Za-z0-9_-]+)*", relative) is not None)
        require(relative != "manifest.json" and relative not in paths)
        paths.append(relative)
        digest(entry["sha256"])
        require(type(entry["bytes"]) is int and 0 <= entry["bytes"] <= 9007199254740991)
        file_path = root / relative
        require(file_path.resolve(strict=True).is_relative_to(root))
        require(file_path.is_file() and file_path.stat().st_size == entry["bytes"])
        hasher = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        require("sha256:" + hasher.hexdigest() == entry["sha256"])
        if relative in (*graph_files, manifest["tokenizer_file"], "recipe.json", "validation.json",
                        "evaluation.json", "parity.json"):
            payload = file_path.read_bytes()
            require(len(payload) == entry["bytes"] and
                    "sha256:" + hashlib.sha256(payload).hexdigest() == entry["sha256"])
            verified[relative] = payload
    require(paths == sorted(paths))
    require(all(name in paths for name in (*graph_files, manifest["tokenizer_file"],
                                          manifest["model_config_file"])))
    require(len({*graph_files, manifest["tokenizer_file"], manifest["model_config_file"]}) == 4)
    verify_provenance_payloads(manifest, verified, "deployment_candidate")
    actual = set()
    for path in root.rglob("*"):
        require(path.resolve(strict=True).is_relative_to(root))
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(paths) | {"manifest.json"})
    digest(manifest["scorer_sha256"])
    digest(manifest["selector_sha256"])
    require(scorer_digest(manifest) == manifest["scorer_sha256"])
    selector = {key: manifest[key] for key in ("scorer_sha256", "selection", "limits", "calibration_sha256")}
    require(canonical_hash(selector) == manifest["selector_sha256"])
    return manifest, verified


if __name__ == "__main__":
    raise SystemExit(main())
