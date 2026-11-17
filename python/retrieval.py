"""Offline FP32 retrieval contract shared by bundle construction and serving."""
import hashlib
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("retrieval_boundary", Path(__file__).with_name("router-worker.py"))
boundary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boundary)
require = boundary.require
MODEL = {"type": "bi_encoder", "base": "sentence-transformers/all-MiniLM-L6-v2",
         "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}
DECISION_MODEL = {"type": "window_bag", "base": "cross-encoder/ms-marco-MiniLM-L6-v2",
                  "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a"}
POOLING = {"max_chunk_tokens": 256, "payload_tokens": 254, "overlap_tokens": 0,
           "context_payload_tokens": 254, "context_selection": "tail",
           "token_pooling": "attention-masked-mean-including-specials",
           "chunk_pooling": "payload-length-weighted-mean", "normalization": "l2", "dimensions": 384}
LIMITS = {"max_candidates": 32, "max_line_bytes": 131072, "microbatch": 4,
          "request_timeout_ms": 750, "startup_timeout_ms": 10000}
GRAPHS = {"encoder": {"file": "encoder.onnx", "inputs": [
    {"name": name, "dtype": "int64", "shape": ["batch", "sequence"]}
    for name in ("input_ids", "attention_mask", "token_type_ids")],
    "outputs": [{"name": "pooled", "dtype": "float32", "shape": ["batch", 384]}]}}


def file_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedding_identity(identifier, content_digest, routing_digest, description):
    identity = {"id": identifier, "content_sha256": content_digest, "routing_sha256": routing_digest,
                "description_sha256": "sha256:" + hashlib.sha256(description.encode("utf-8")).hexdigest()}
    return {**identity, "key": boundary.canonical_hash(identity)}


def scorer_digest(manifest):
    return boundary.canonical_hash({key: manifest[key] for key in (
        "role", "model", "graphs", "tokenizer_file", "model_config_file", "files", "pooling",
        "embedding_table", "provenance")})


def selector_digest(manifest):
    return boundary.canonical_hash({key: manifest[key] for key in ("scorer_sha256", "selection", "limits")})


def validate_parity(parity):
    comparison = parity["comparison"]
    require(parity["role"] == "retrieval" and parity["nonprivate"] is True and len(parity["cases"]) >= 4)
    require(comparison["passed"] is True and 0 < comparison["tolerance"] <= 1e-4 and
            0 <= comparison["max_absolute_embedding_error"] <= comparison["tolerance"] and
            0 <= comparison["max_absolute_pooled_error"] <= comparison["tolerance"] and
            comparison["ranking_pair_changes"] == 0)


def load_manifest(directory):
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    require(manifest_path.stat().st_size <= boundary.MAX_LINE and manifest_path.resolve().is_relative_to(root))
    manifest = boundary.decode_json(manifest_path.read_bytes())
    boundary.keys(manifest, ("schema_version", "role", "model", "graphs", "tokenizer_file", "model_config_file",
        "files", "pooling", "embedding_table", "selection", "limits", "provenance", "scorer_sha256",
        "selector_sha256", "publication_allowed"))
    require(manifest["schema_version"] == 1 and manifest["role"] == "retrieval" and manifest["publication_allowed"] is True)
    require(manifest["model"] == MODEL and manifest["pooling"] == POOLING and manifest["graphs"] == GRAPHS)
    require(manifest["limits"] == LIMITS and manifest["selection"] == {"top_k": 8, "tie_break": "id"})
    provenance = manifest["provenance"]
    boundary.keys(provenance, ("contextual_binding", "catalog_sha256", "recipe_sha256", "runtime_revision", "runtime_files",
        "dataset_sha256", "split_sha256", "weights", "training_report_sha256", "training_recipe_sha256", "fit_split", "training_data_used"))
    trained = provenance["weights"] == "fine_tuned"
    require(provenance["weights"] in ("unchanged_pretrained", "fine_tuned") and
            provenance["fit_split"] == ("validation" if trained else None) and
            provenance["training_data_used"] is trained)
    for name in ("dataset_sha256", "split_sha256"):
        boundary.digest(provenance[name])
    for name in ("training_report_sha256", "training_recipe_sha256"):
        if trained:
            boundary.digest(provenance[name])
        else:
            require(provenance[name] is None)
    boundary.validate_source_set_binding(provenance["contextual_binding"])
    require(provenance["catalog_sha256"] == provenance["contextual_binding"]["catalog_sha256"])
    require(provenance["runtime_revision"] == file_digest(Path(__file__).with_name("retrieval-worker.py")))
    require(provenance["runtime_files"] == {name: file_digest(Path(__file__).with_name(name)) for name in
                                          ("retrieval.py", "router-worker.py", "pair-windows.py")})
    verified = {}
    paths = [entry["path"] for entry in manifest["files"]]
    require(paths == sorted(set(paths)) and "manifest.json" not in paths)
    for entry in manifest["files"]:
        boundary.keys(entry, ("path", "sha256", "bytes"))
        name = entry["path"]
        require(isinstance(name, str) and boundary.re.fullmatch(r"[A-Za-z0-9_-]+(?:[./][A-Za-z0-9_-]+)*", name))
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file())
        data = path.read_bytes()
        require(len(data) == entry["bytes"] and "sha256:" + hashlib.sha256(data).hexdigest() == entry["sha256"])
        verified[name] = data
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(paths) | {"manifest.json"})
    require(all(name in verified for name in ("encoder.onnx", "tokenizer.json", "config.json", "embeddings.json",
                                             "recipe.json", "validation.json", "evaluation.json", "parity.json")))
    require(("training.json" in verified) is trained)
    require(manifest["tokenizer_file"] == "tokenizer.json" and manifest["model_config_file"] == "config.json" and
            manifest["embedding_table"] == "embeddings.json")
    reports = {name: boundary.decode_json(verified[name]) for name in
               ("recipe.json", "validation.json", "evaluation.json", "parity.json")}
    recipe = reports["recipe.json"]
    require(file_digest(root / "recipe.json") == provenance["recipe_sha256"] and recipe["export"]["variant"] == "fp32")
    bound_reports = [recipe, reports["validation.json"], reports["evaluation.json"]]
    if trained:
        require(file_digest(root / "training.json") == provenance["training_report_sha256"])
        training = boundary.decode_json(verified["training.json"])
        require(type(training["chosen_epoch"]) is int and training["chosen_epoch"] in (1, 2, 3))
        bound_reports.append(training)
    for report in bound_reports:
        require(report["role"] == "retrieval")
        for name in ("weights", "fit_split", "training_data_used", "dataset_sha256", "split_sha256",
                     "training_recipe_sha256", "contextual_binding"):
            require(report[name] == provenance[name])
    require(all(report["comparison"] == reports["parity.json"]["comparison"]
                for report in (reports["validation.json"], reports["evaluation.json"])))
    validate_parity(reports["parity.json"])
    require(manifest["scorer_sha256"] == scorer_digest(manifest) and manifest["selector_sha256"] == selector_digest(manifest))
    return manifest, verified


def load_decider_manifest(directory):
    """Pin the production deciding slot without changing the historical bundle reader."""
    manifest, _ = boundary.load_manifest(directory)
    require(manifest["model"] == DECISION_MODEL)
    return manifest


def embed(tokenizer, run_encoder, text, np):
    """Description embedding: every token participates, preserving export parity fixtures."""
    return _embed_ids(tokenizer, run_encoder, tokenizer.encode(text, add_special_tokens=False).ids, np)


def embed_context(tokenizer, run_encoder, text, np):
    """Request embedding: one chunk containing only the most recent native tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return _embed_ids(tokenizer, run_encoder, ids[-POOLING["context_payload_tokens"]:], np)


def _embed_ids(tokenizer, run_encoder, ids, np):
    total = np.zeros(384, dtype=np.float32)
    weight = 0
    for start in range(0, max(len(ids), 1), 254):
        chunk = ids[start:start + 254]
        tokens = np.asarray([[tokenizer.token_to_id("[CLS]"), *chunk, tokenizer.token_to_id("[SEP]")]], dtype=np.int64)
        pooled = run_encoder({"input_ids": tokens, "attention_mask": np.ones_like(tokens), "token_type_ids": np.zeros_like(tokens)})
        require(pooled.shape == (1, 384) and np.isfinite(pooled).all())
        n = max(len(chunk), 1)
        total += pooled[0] * n
        weight += n
    total /= weight
    norm = np.linalg.norm(total)
    require(np.isfinite(norm) and norm > 0)
    return total / norm


class Retriever:
    def __init__(self, directory):
        boundary.configure_environment()
        self.manifest, data = load_manifest(directory)
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        self.tokenizer = Tokenizer.from_str(data["tokenizer.json"].decode("utf-8"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.encoder = ort.InferenceSession(data["encoder.onnx"], sess_options=options, providers=["CPUExecutionProvider"])
        for nodes, specs in ((self.encoder.get_inputs(), GRAPHS["encoder"]["inputs"]),
                             (self.encoder.get_outputs(), GRAPHS["encoder"]["outputs"])):
            require([{ "name": n.name, "dtype": {"tensor(int64)": "int64", "tensor(float)": "float32"}.get(n.type),
                       "shape": n.shape} for n in nodes] == specs)
        table = boundary.decode_json(data["embeddings.json"])
        boundary.keys(table, ("format", "entries"))
        require(table["format"] == "content-addressed-skill-embeddings-v1" and isinstance(table["entries"], list))
        self.vectors = {}
        for entry in table["entries"]:
            boundary.keys(entry, ("id", "content_sha256", "routing_sha256", "description_sha256", "key", "embedding"))
            identity = {k: entry[k] for k in ("id", "content_sha256", "routing_sha256", "description_sha256")}
            require(entry["key"] == boundary.canonical_hash(identity))
            for key in ("content_sha256", "routing_sha256", "description_sha256", "key"):
                boundary.digest(entry[key])
            vector = np.asarray(entry["embedding"], dtype=np.float32)
            require(vector.shape == (384,) and np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1) < 1e-4)
            # A skill's embedding is a pure function of its description text (the builder embeds
            # exactly entry["description"]), so serving keys vectors by that text alone.
            self.vectors[entry["description_sha256"]] = vector
        # Live catalogs drift from the build-time table (new skills, edited descriptions). A
        # description absent from the table is embedded here with the same encoder and
        # embed() the builder uses, which yields the vector a rebuild would store, and is
        # kept for the worker's lifetime. About 12-25 ms per new description on CPU.
        # Only the last request context is cached across bounded 32-entry score chunks.
        # Identity includes session/generation and exact structured context, never candidates.
        self.context_key = None
        self.context_vector = None

    def vector(self, description):
        key = "sha256:" + hashlib.sha256(description.encode("utf-8")).hexdigest()
        if key not in self.vectors:
            self.vectors[key] = embed(self.tokenizer, lambda arrays: self.encoder.run(["pooled"], arrays)[0],
                                      description, self.np)
        return self.vectors[key]

    def route(self, request):
        if request["selectorDigest"] != self.manifest["selector_sha256"]:
            return boundary.response(request, "uncertain", "stale_result")
        try:
            candidates = boundary.preflight(request)
            key = (request["sessionKey"], request["generation"], request["intent"],
                   None if request["action"] is None else (request["action"]["toolName"], request["action"]["summary"]))
            if key != self.context_key:
                vector = embed_context(self.tokenizer, lambda arrays: self.encoder.run(["pooled"], arrays)[0],
                                       boundary.windows.render_first(request["intent"], request["action"]), self.np)
                self.context_vector, self.context_key = vector, key
            scores = [{"id": c["id"], "score": float(self.np.dot(self.vector(c["description"]), self.context_vector))}
                      for c in candidates]
            return boundary.response(request, "uncertain", "inadequate_candidates", scores={"kind": "ranking",
                "scorerDigest": self.manifest["scorer_sha256"], "calibrationDigest": None, "values": scores})
        except boundary.RoutingFailure as error:
            return boundary.response(request, "uncertain", error.reason)
