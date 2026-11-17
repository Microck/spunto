#!/usr/bin/env python3
"""Export the pinned pretrained MiniLM retriever offline, bound to current admission."""
import os
os.umask(0o077)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                  OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("retrieval", Path(__file__).with_name("retrieval.py"))
retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval)
boundary = retrieval.boundary
FIXTURES = ["", "Review keyboard access in a browser form.", "Do not deploy. Review the release plan only.",
            "Inspect a form.\nAction: read\nRead source without changes.", "review code " * 700]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def build(args):
    # Admission happens before importing numerical runtimes or loading weights.
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
    boundary.require(available >= 1.2 * 1024 * 1024)
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    # Export the pinned pretrained revision without an online fallback.
    source = retrieval.MODEL["base"]
    catalog = boundary.decode_json(args.catalog.read_bytes())
    if "catalog" in catalog:
        catalog = catalog["catalog"]
    plan = boundary.decode_json(args.plan.read_bytes())
    binding = boundary.compact_source_set_binding(plan["contextual_binding"])
    boundary.validate_source_set_binding(binding)
    boundary.require(catalog["catalog_sha256"] == binding["catalog_sha256"])
    training_identity = {"dataset_sha256": retrieval.file_digest(args.plan.parent / "audited-pairs.jsonl"),
                        "split_sha256": plan["split_sha256"], "weights": "unchanged_pretrained",
                        "training_recipe_sha256": None, "fit_split": None, "training_data_used": False}
    boundary.require(not args.output.exists())
    args.output.mkdir(parents=True, mode=0o700)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=retrieval.MODEL["revision"],
        local_files_only=True, trust_remote_code=False, token=False, use_fast=True).backend_tokenizer
    tokenizer.no_padding()
    tokenizer.no_truncation()
    model = AutoModel.from_pretrained(source, revision=retrieval.MODEL["revision"], local_files_only=True,
        trust_remote_code=False, token=False, attn_implementation="eager").eval().float()

    class MeanEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(1) / mask.sum(1)

    encoder = MeanEncoder().eval()
    example = torch.tensor([[tokenizer.token_to_id("[CLS]"), 2531, tokenizer.token_to_id("[SEP]")]], dtype=torch.long)
    with torch.inference_mode():
        torch.onnx.export(encoder, (example, torch.ones_like(example), torch.zeros_like(example)),
            str(args.output / "encoder.onnx"), input_names=list(boundary.windows.SIGNATURE), output_names=["pooled"],
            dynamic_axes={**{name: {0: "batch", 1: "sequence"} for name in boundary.windows.SIGNATURE}, "pooled": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False, external_data=False)
    graph = onnx.load(str(args.output / "encoder.onnx"))
    boundary.require(all(not tensor.external_data for tensor in graph.graph.initializer))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.output / "encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"])

    pooled_errors = []
    def torch_run(arrays):
        with torch.inference_mode():
            return encoder(**{key: torch.from_numpy(value) for key, value in arrays.items()}).numpy()

    def onnx_run(arrays):
        return session.run(["pooled"], arrays)[0]

    def parity_run(arrays):
        actual = onnx_run(arrays)
        pooled_errors.append(float(np.abs(torch_run(arrays) - actual).max()))
        return actual

    expected = [retrieval.embed(tokenizer, torch_run, text, np) for text in FIXTURES]
    actual = [retrieval.embed(tokenizer, parity_run, text, np) for text in FIXTURES]
    error = max(float(np.abs(a - b).max()) for a, b in zip(expected, actual))
    # Sort fixed fixture IDs on ties; compare both pooled graph outputs and final embeddings.
    ranks = lambda vectors: [sorted(range(len(vectors)), key=lambda j: (-float(np.dot(v, vectors[j])), j)) for v in vectors]
    rank_changes = sum(a != b for a, b in zip(ranks(expected), ranks(actual)))
    comparison = {"passed": error <= 1e-4 and max(pooled_errors) <= 1e-4 and rank_changes == 0, "tolerance": 1e-4,
                  "max_absolute_embedding_error": error, "max_absolute_pooled_error": max(pooled_errors), "ranking_pair_changes": rank_changes}
    parity = {"fixture_version": 1, "role": "retrieval", "nonprivate": True, "pooling": retrieval.POOLING,
              "comparison": comparison, "cases": [{"text": text, "torch_embedding": a.tolist(), "export_embedding": b.tolist()}
                                                     for text, a, b in zip(FIXTURES, expected, actual)]}
    retrieval.validate_parity(parity)
    write_json(args.output / "parity.json", parity)
    tokenizer.save(str(args.output / "tokenizer.json"))
    model.config.to_json_file(str(args.output / "config.json"))
    # Embeddings are rebuilt from the exact encoder exported in this bundle.
    entries = []
    for entry in sorted(catalog["entries"], key=lambda entry: entry["runtime_id"]):
        if not entry["automatic_eligible"]:
            continue
        identity = retrieval.embedding_identity(entry["runtime_id"], entry["content_sha256"], entry["routing_sha256"], entry["description"])
        vector = retrieval.embed(tokenizer, onnx_run, entry["description"], np).tolist()
        entries.append({**identity, "embedding": vector})
    write_json(args.output / "embeddings.json", {"format": "content-addressed-skill-embeddings-v1", "entries": entries})
    recipe = {"role": "retrieval", **training_identity,
              "model": retrieval.MODEL, "pooling": retrieval.POOLING,
              "contextual_binding": binding, "catalog_sha256": catalog["catalog_sha256"],
              "builder_sha256": retrieval.file_digest(__file__),
              "dependencies": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
              "export": {"variant": "fp32", "opset": 17, "external_data": False}}
    write_json(args.output / "recipe.json", recipe)
    for name in ("validation.json", "evaluation.json"):
        write_json(args.output / name, {**recipe, "comparison": comparison,
            "scope": "nonprivate FP32 export parity; no retriever fitting or epoch selection"})
    manifest = {"schema_version": 1, "role": "retrieval", "model": retrieval.MODEL, "graphs": retrieval.GRAPHS,
        "tokenizer_file": "tokenizer.json", "model_config_file": "config.json", "pooling": retrieval.POOLING,
        "embedding_table": "embeddings.json", "selection": {"top_k": 8, "tie_break": "id"}, "limits": retrieval.LIMITS,
        "provenance": {"contextual_binding": binding, "catalog_sha256": catalog["catalog_sha256"],
            **training_identity, "training_report_sha256": None,
            "recipe_sha256": retrieval.file_digest(args.output / "recipe.json"),
            "runtime_revision": retrieval.file_digest(Path(__file__).with_name("retrieval-worker.py")),
            "runtime_files": {name: retrieval.file_digest(Path(__file__).with_name(name)) for name in ("retrieval.py", "router-worker.py", "pair-windows.py")}},
        "files": [{"path": p.name, "sha256": retrieval.file_digest(p), "bytes": p.stat().st_size} for p in sorted(args.output.iterdir())],
        "publication_allowed": True}
    manifest["scorer_sha256"] = retrieval.scorer_digest(manifest)
    manifest["selector_sha256"] = retrieval.selector_digest(manifest)
    write_json(args.output / "manifest.json", manifest)
    retrieval.load_manifest(args.output)
    print(json.dumps({"entries": len(entries), "parity": comparison, "selector_sha256": manifest["selector_sha256"]}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "plan", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    build(cli.parse_args())
