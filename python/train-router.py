#!/usr/bin/env python3
"""Train the deciding window-bag model in a task-owned remote container.

No provisioning or publishing. ML dependencies load only in run(), so dataset
validation tests require only the standard library. See docs/retraining.md.
"""
from __future__ import annotations

import argparse
import contextlib
from decimal import Decimal
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import time
import tempfile
import unicodedata

_spec = importlib.util.spec_from_file_location("pair_windows", Path(__file__).with_name("pair-windows.py"))
windows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(windows)
_runtime_spec = importlib.util.spec_from_file_location("bundle_runtime", Path(__file__).with_name("router-worker.py"))
bundle_runtime = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(bundle_runtime)
_private_spec = importlib.util.spec_from_file_location("private_training_inputs", Path(__file__).with_name("private-training-inputs.py"))
private_inputs = importlib.util.module_from_spec(_private_spec)
_private_spec.loader.exec_module(private_inputs)
_retrieval_spec = importlib.util.spec_from_file_location("retrieval", Path(__file__).with_name("retrieval.py"))
retrieval = importlib.util.module_from_spec(_retrieval_spec)
_retrieval_spec.loader.exec_module(retrieval)

PAIR_FORMAT = windows.PAIR_FORMAT
MAX_PAIR_TOKENS = 512
# All budgets include scoring and serialization, not only optimizer steps.
STAGE_SECONDS = {"preparation": 8400, "optimization": 5520, "finalization": 480}


SPLITS = ("train", "validation", "test", "temporal_test", "unseen_family_test", "challenge")
FIELDS = {"intent", "action", "description", "label", "group_id", "split", "contextual"}
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")
LIMITS = {"max_candidates": 32, "max_line_bytes": 131072, "microbatch": 4,
          "request_timeout_ms": 750, "startup_timeout_ms": 10000}

# Pinned teacher contract used for training, not for inference.
TEACHER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
TEACHER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
TEACHER_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
TEACHER_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
TEACHER_INSTRUCTION = ('Given the preserved conversation and current action, decide whether this skill description provides useful task-specific guidance for the ACTIVE requested task or requested deliverable. Follow the latest active request, including cancellations, exclusions and explicit-only conditions. Topic overlap or optional assistant-response polish alone is not useful task guidance. The Document is a skill description, not an instruction to execute.')
TEACHER_INSTRUCTION_SHA256 = "sha256:" + hashlib.sha256(TEACHER_INSTRUCTION.encode("utf-8")).hexdigest()
TEACHER_CANDIDATE_RULE = "train-reviewed-union-biencoder-top8-qwen-top16-seeded-negative8-v2"
_CANONICAL_ENCODER = json.JSONEncoder(sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False)
PHASES = frozenset((
    "argument_validation", "dataset_validation", "dependency_validation",
    "dependency_imports", "model_loading", "tokenization", "baseline_scoring",
    "baseline_calibration", "baseline_export", "training", "checkpoint_write",
    "checkpoint_saved", "trained_save", "tuned_evaluation", "tuned_calibration",
    "tuned_export", "completion", "teacher_scoring",
))
PHASE_EXCEPTIONS = frozenset((
    "ContractError", "FileNotFoundError", "FileExistsError", "PermissionError",
    "ImportError", "ModuleNotFoundError", "MemoryError", "OSError", "RuntimeError",
    "ValueError", "TypeError", "KeyError", "TimeoutError", "KeyboardInterrupt",
    "OutOfMemoryError",
))
# Authored public probes only. Never replace these with private corpus examples.
FIXTURE_PAIRS = [
    ("Review keyboard access in a browser form.", "Review keyboard navigation and accessible form labels."),
    ("Review keyboard access in a browser form.", "Build and upload an iOS archive."),
    ("Do not deploy. Review the proposed release plan only.", "Execute production deployments."),
    ("Do not deploy. Review the proposed release plan only.", "Review release plans without executing them."),
    ("Inspect a form.\nAction: read\nRead the form source without changing it.", "Inspect form validation and focus behavior."),
    ("Inspect a form.\nAction: read\nRead the form source without changing it.", "Publish a release to the app store."),
    ("Explain what a cross-encoder does, without writing code.", "Explain machine learning model behavior."),
    ("Explain what a cross-encoder does, without writing code.", "Implement and deploy a model training service."),
]


class ContractError(ValueError):
    """Fixed, content-free diagnostic code safe for stderr."""

CONTRACT_ERROR_CODES = frozenset((
    'bundle_already_finalized',
    'bundle_schema_mismatch',
    'bundle_provenance_mismatch',
    'calibration_needs_both_labels',
    'contradictory_label',
    'control_thresholds_unavailable',
    'init_weights_mismatch',
    'cross_split_duplicate',
    'cross_split_group',
    'dataset_export_binding',
    'dataset_export_counts',
    'dataset_export_paths',
    'dependency_lock_incomplete_closure',
    'dependency_lock_missing_direct_pins',
    'dependency_lock_requires_exact_pins',
    'dependency_lock_requires_hashes',
    'dependency_lock_version_mismatch',
    'duplicate_json_key',
    'duplicate_pair',
    'empty_evaluation',
    'export_directory_exists',
    'external_onnx_tensor',
    'fit_split_needs_both_labels',
    'fp32_graph_unavailable',
    'invalid_binary_label',
    'invalid_dataset_export',
    'invalid_digest_value',
    'invalid_export_input',
    'invalid_export_input_names',
    'invalid_export_output',
    'invalid_export_score_shape',
    'invalid_json',
    'invalid_model_scores',
    'invalid_phase_counter',
    'invalid_phase_event',
    'invalid_positive_argument',
    'invalid_row_fields',
    'invalid_row_text',
    'invalid_run_argument',
    'invalid_split',
    'invalid_target',
    'invalid_unicode',
    'missing_fit_split',
    'missing_runtime_revision',
    'runtime_revision_mismatch',
    'missing_test_split',
    'nonfinite_json',
    'nonfinite_threshold',
    'nonfinite_training_loss',
    'nonfinite_training_weights',
    'pair_token_overflow',
    'resume_checkpoint_missing',
    'resume_identity_mismatch',
    'resume_recipe_mismatch',
    'tokenizer_parity_failed',
    'unsupported_logit_shape',
    'unsupported_tokenizer_signature',
    'validation_targets_unmet',
    "missing_contextual_target", "contextual_approval_pin_required", "contextual_approval_digest",
    "contextual_approval_binding", "contextual_receipt_binding", "contextual_target_binding",
    "contextual_sealed_evidence", "contextual_export_binding", "contextual_export_files",
    "contextual_export_contract", "contextual_input_source_mapping", "contextual_projection_counts",
    "contextual_approval_scope", "contextual_approved_pair_allowlist", "contextual_author_not_authority",
    "contextual_admission_reference_fields", "contextual_body_missing", "contextual_broad_provenance",
    "contextual_candidate_ineligible", "contextual_catalog_binding", "contextual_catalog_body",
    "contextual_catalog_eligibility", "contextual_cross_split_group", "contextual_cutoff_binding",
    "contextual_disposition", "contextual_duplicate_identity", "contextual_duplicate_projection",
    "contextual_duplicate_source", "contextual_eligibility", "contextual_exclusion_counts",
    "contextual_explicit_required", "contextual_family_mask_binding", "contextual_manifest_binding",
    "contextual_opinion_binding", "contextual_pair_identity", "contextual_pair_not_approved",
    "contextual_privacy_binding", "contextual_privacy_report_fields", "contextual_quote_binding",
    "contextual_receipt_pairs", "contextual_recipe_binding", "contextual_recipe_report_binding",
    "contextual_record_binding", "contextual_record_not_reviewed", "contextual_reference_digest",
    "contextual_request_digest", "contextual_reviewed_split_changed", "contextual_source_conflict",
    "contextual_source_coverage", "contextual_source_dataset", "contextual_source_dataset_binding",
    "contextual_source_identity", "contextual_source_snapshot", "contextual_split_binding",
    "contextual_status_evidence", "contextual_unknown_source",
    "invalid_teacher_settings", "teacher_train_only", "teacher_context_capacity",
    "teacher_candidate_binding", "teacher_empty_candidates",
    "retrieval_train_only", "retrieval_model_file_hash",
    "trainer_stage_deadline", "trainer_window_budget",
    "unknown_contract_error",
))


def reportable_error_code(value):
    """Allow fixed trainer codes and content-free input-verifier identifiers."""
    return type(value) is str and (value in CONTRACT_ERROR_CODES or re.fullmatch(r"private_[a-z0-9_]{1,64}", value) is not None)


def contract_error_code(error):
    """Never serialize arbitrary exception arguments, even from ContractError."""
    if not isinstance(error, ContractError):
        return None
    value = error.args[0] if len(error.args) == 1 else None
    return value if reportable_error_code(value) else "unknown_contract_error"


def emit_phase(args, phase, state="BEGIN", *, step=None, checkpoint_bytes=None,
               partition_index=0, pair_count=0, error=None):
    """Emit only the fixed metadata contract, outside redirected library output."""
    if phase not in PHASES or state not in ("BEGIN", "PROGRESS", "END", "ERROR"):
        raise ContractError("invalid_phase_event")
    counters = {
        "step": getattr(args, "_phase_step", 0) if step is None else step,
        "checkpoint_bytes": (getattr(args, "_phase_checkpoint_bytes", 0)
                             if checkpoint_bytes is None else checkpoint_bytes),
        "partition_index": partition_index, "pair_count": pair_count,
    }
    if any(type(value) is not int or not 0 <= value < 2**63 for value in counters.values()):
        raise ContractError("invalid_phase_counter")
    exception_type = None if error is None else type(error).__name__
    if exception_type is not None and exception_type not in PHASE_EXCEPTIONS:
        exception_type = "Other"
    args._stage = phase
    args._phase_sequence = getattr(args, "_phase_sequence", 0) + 1
    args._phase_step = counters["step"]
    args._phase_checkpoint_bytes = counters["checkpoint_bytes"]
    event = {"event": "PHASE", "schema_version": 1, "phase": phase, "state": state,
             "sequence": args._phase_sequence, **counters, "exception_type": exception_type}
    if isinstance(error, ContractError):
        event["error_code"] = contract_error_code(error)
    payload = json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
    stream = getattr(args, "_phase_stream", sys.stdout)
    stream.write(json.dumps({"event": "PHASE", "phase": phase}) + "\n")
    stream.flush()
    # Only run() enables persistence, after exclusive creation or resume validation.
    directory = getattr(args, "_phase_directory", None)
    if directory is not None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".trainer-phase-",
                                             dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / "trainer-phase.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def emit_failure_phase(args, error):
    # A failed diagnostic sink must not suppress the original private handler.
    try:
        emit_phase(args, args._stage, "ERROR", error=error)
    except OSError:
        print('{"status":"error","code":"diagnostic_write_failed"}', file=sys.stderr)


def canonical_bytes(value):
    def validate(item):
        if item is None or isinstance(item, (str, bool)):
            return
        if type(item) is int and abs(item) <= 9007199254740991:
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child)
            return
        raise ContractError("invalid_digest_value")
    validate(value)
    try:
        return _CANONICAL_ENCODER.encode(value).encode("utf-8")
    except UnicodeError:
        raise ContractError("invalid_unicode") from None


def digest(value):
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _partition_digest(rows):
    """Rows already passed load_rows; hash their canonical array without copying it."""
    checksum = hashlib.sha256()
    for chunk in _CANONICAL_ENCODER.iterencode(rows):
        checksum.update(chunk.encode("utf-8"))
    return "sha256:" + checksum.hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return "sha256:" + checksum.hexdigest()


def teacher_settings(args):
    """No implicit teacher defaults: a partial request must fail before importing ML."""
    values = {name: getattr(args, name, None) for name in private_inputs.TEACHER_ARGUMENTS}
    if all(value is None for value in values.values()):
        return None
    pins = {"teacher_model": TEACHER_MODEL, "teacher_revision": TEACHER_REVISION,
            "teacher_instruction_sha256": TEACHER_INSTRUCTION_SHA256,
            "teacher_candidate_rule": TEACHER_CANDIDATE_RULE}
    if (any(values[name] != value for name, value in pins.items())
            or any(type(values[name]) not in (int, float) or not math.isfinite(values[name]) or values[name] <= 0
                   for name in ("distillation_alpha", "distillation_temperature"))
            or args.device != "cuda"):
        raise ContractError("invalid_teacher_settings")
    return {name: str(value) if type(value) is float else value for name, value in values.items()}


def run_recipe(args, provenance, dependencies, lock_digest):
    """Run-defining recipe written by the native trainer.

    The resume check requires the same inputs, runtime and optimization settings.
    """
    recipe = {"model": args.base, "revision": windows.MODEL_BASES[args.base], "init_weights_sha256": args.init_sha256,
            "license": "Apache-2.0", "pair_format": PAIR_FORMAT,
            "pair_text": "intent if action is null else intent + '\\nAction: ' + action.toolName + '\\n' + action.summary",
            "baseline_control_objective": "precision-constrained-then-balanced-accuracy-full-coverage-v1",
            "geometry": windows.GEOMETRY, "training_window_limit": 64,
            "geometry_sha256": file_digest(Path(windows.__file__)), "overflow": "intent-tail-description-head-then-reject-action", "score_semantics": "raw_logit",
            "baseline_control": "unchanged-pretrained-native-window-logit-arithmetic-mean",
            "loss": "binary_cross_entropy_with_logits_mean_by_original_pair", "full_fine_tuning": True,
            "dependencies": dependencies, "dependency_lock_sha256": lock_digest,
            "trainer_sha256": file_digest(Path(__file__)), "provenance": {**provenance,
                "contextual_binding": bundle_runtime.compact_source_set_binding(provenance["contextual_binding"])},
            "python_version": sys.version, "platform": sys.platform,
            "training": {key: getattr(args, key) for key in ("seed", "device", "epochs", "max_steps", "max_seconds",
                         "batch_size", "checkpoint_steps", "learning_rate")},
            "optimizer": {"type": "AdamW", "weight_decay": 0.01, "clip_gradient_norm": 1.0, "scheduler": None},
            "targets": {key: getattr(args, key) for key in ("target_precision", "target_negative_precision",
                        "min_calibration_support", "min_coverage")},
            "parity_tolerances": {"fp32_atol": args.fp32_atol, "int8_atol": args.int8_atol,
                                  "max_ranking_pair_changes": 0, "max_decision_changes": 0},
            "runtime_revision": args.runtime_revision, "runtime_limits": LIMITS}
    teacher = teacher_settings(args)
    if teacher is not None:
        recipe["distillation"] = teacher
        recipe["loss"] = "reviewed-hard-bce-mean-plus-alpha-teacher-bce-mean"
        recipe["distillation_retrieval"] = {"model": retrieval.MODEL, "pooling": retrieval.POOLING,
            "weights": "unchanged_pretrained", "bundle_path": "retrieval-bundle",
            "top_k": 8, "tie_break": "id", "runtime_sha256": file_digest(Path(retrieval.__file__))}
    recipe["stage_budgets_seconds"] = STAGE_SECONDS
    return recipe


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ContractError("duplicate_json_key")
        value[key] = item
    return value


def reject_constant(_value):
    raise ContractError("nonfinite_json")


def parse_json(text):
    try:
        return json.loads(text, object_pairs_hook=strict_object, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeError):
        raise ContractError("invalid_json") from None


def write_json(path, value):
    # Reports can contain finite float metrics; only digest projections forbid floats.
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    temporary = Path(str(path) + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_pair(row):
    if not isinstance(row, dict) or "contextual" not in row:
        raise ContractError("missing_contextual_target")
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ContractError("invalid_row_fields")
    for field in ("intent", "description", "group_id", "split"):
        value = row[field]
        if not isinstance(value, str) or (field in ("group_id", "split") and not value.strip()):
            raise ContractError("invalid_row_text")
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ContractError("invalid_unicode") from None
    action = row["action"]
    if action is not None:
        if not isinstance(action, dict) or set(action) != {"toolName", "summary"}:
            raise ContractError("invalid_row_fields")
        for name, maximum, minimum in (("toolName", 256, 1), ("summary", 2048, 0)):
            value = action[name]
            if not isinstance(value, str) or not minimum <= len(value) <= maximum:
                raise ContractError("invalid_row_text")
            try:
                value.encode("utf-8")
            except UnicodeError:
                raise ContractError("invalid_unicode") from None
    if type(row["label"]) is not int or row["label"] not in (0, 1):
        raise ContractError("invalid_binary_label")
    if row["split"] not in SPLITS:
        raise ContractError("invalid_split")


def load_rows(path, *, approved_pairs=None):
    if approved_pairs is None:
        raise ContractError("missing_contextual_target")
    partitions = {name: [] for name in SPLITS}
    groups = {}
    pairs = {}
    # Decode incrementally: one non-BMP character otherwise widens the entire
    # capsule's JSONL to four-byte text. Keep StringIO's LF-only line semantics.
    with (io.TextIOWrapper(io.BytesIO(path), encoding="utf-8", newline="\n")
          if isinstance(path, bytes) else Path(path).open(encoding="utf-8")) as stream:
        for line in stream:
            row = parse_json(line)
            if not isinstance(row, dict) or "contextual" not in row:
                raise ContractError("missing_contextual_target")
            contextual = row["contextual"]
            fingerprint = contextual.get("row_sha256") if isinstance(contextual, dict) else None
            if (type(row.get("label")) is not int or not isinstance(fingerprint, str) or fingerprint not in approved_pairs or
                    row != private_inputs.compact_training_pair(approved_pairs[fingerprint])):
                raise ContractError("contextual_input_source_mapping")
            row = approved_pairs[fingerprint]
            validate_pair(row)
            label = row["label"]
            split = row["split"]
            if split not in partitions:
                raise ContractError("invalid_split")
            group = row["group_id"]
            if group in groups and groups[group] != split:
                raise ContractError("cross_split_group")
            groups[group] = split
            pair = (normalized(windows.render_first(row["intent"], row["action"])), normalized(row["description"]))
            if pair in pairs:
                previous_split, previous_label = pairs[pair]
                if previous_split != split:
                    raise ContractError("cross_split_duplicate")
                if previous_label != label:
                    raise ContractError("contradictory_label")
                raise ContractError("duplicate_pair")
            pairs[pair] = (split, label)
            # These exact rows already passed the hash and equality join above.
            # Share the admitted objects instead of retaining a second parsed corpus.
            partitions[split].append(row)
    if not partitions["train"] or not partitions["validation"]:
        raise ContractError("missing_fit_split")
    if not any(partitions[name] for name in SPLITS[2:]):
        raise ContractError("missing_test_split")
    for split in ("train", "validation"):
        if {row["label"] for row in partitions[split]} != {0, 1}:
            raise ContractError("fit_split_needs_both_labels")
    return {name: rows for name, rows in partitions.items() if rows}


def load_export(data_path, manifest_path, *, approval_sha256=None):
    """Bind training bytes to the approved export, not a new partition identity."""
    spec = importlib.util.spec_from_file_location("router_dataset", Path(__file__).with_name("build-dataset.py"))
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    root = manifest_path.parent
    if manifest_path.name != "manifest.json" or data_path.resolve() != (root / "audited-pairs.jsonl").resolve():
        raise ContractError("dataset_export_paths")
    try:
        manifest, scan, admission, approved_pairs = dataset.verify_contextual_export(root, approval_sha256)
    except dataset.ContractError as error:
        raise ContractError(str(error)) from None
    except (KeyError, TypeError, OSError):
        raise ContractError("invalid_dataset_export") from None
    partitions = load_rows(data_path, approved_pairs=approved_pairs)
    if scan["counts"] != {split: len(rows) for split, rows in partitions.items()}:
        raise ContractError("dataset_export_counts")
    provenance = {"dataset_sha256": file_digest(data_path),
                  "split_sha256": manifest["split_sha256"],
                  "source_manifest_sha256": file_digest(manifest_path),
                  "contextual_binding": admission.binding,
                  "partitions": {split: _partition_digest(rows) for split, rows in partitions.items()}}
    return partitions, provenance, scan, admission


def teacher_cases(train_rows, catalog, admission, holdout_families):
    """Project only admitted TRAIN; validation never enters teacher mining."""
    pair_cases = {pair["pair_id"]: case_id for case_id, pairs in admission.by_example.items() for pair in pairs}
    entries = sorted(catalog["entries"], key=lambda entry: entry["runtime_id"])
    holdout_families = set(holdout_families)
    cases = {}
    for row in train_rows:
        if row["split"] != "train":
            raise ContractError("teacher_train_only")
        for contributor in row["contextual"]["contributors"]:
            case_id = pair_cases[contributor["pair_id"]]
            source = admission.records[case_id]
            if source["split"] != "train" or case_id.startswith("control-"):
                raise ContractError("teacher_train_only")
            inputs = source["input"]
            action = inputs["action"]
            action = None if action is None else {"toolName": action["tool_name"], "summary": action["summary"]}
            if row["intent"] != inputs["request"] or row["action"] != action:
                raise ContractError("teacher_candidate_binding")
            if case_id not in cases:
                excluded = set(inputs["excluded_ids"]) | {"unslop", "emil-unslop-writing"}
                # The reservation is catalog-wide, including IDs absent from reviewed pairs.
                excluded.update(entry["runtime_id"] for entry in entries
                                if entry["family_id"] == "app-store-operations")
                excluded.update(pair["runtime_id"] for pair in admission.by_example[case_id]
                                if pair["family_mask"] or pair["disposition"] in ("ineligible", "diagnostic")
                                or pair.get("purposes", {}).get("diagnostic", False))
                # Deduplicate descriptions, as the student input cannot distinguish
                # two skills with identical descriptions. Stable IDs break rank ties.
                eligible, retrieval_eligible = {}, {}
                for entry in entries:
                    if (entry["runtime_id"] not in excluded and entry["family_id"] not in holdout_families
                            and (entry["automatic_eligible"] or entry["runtime_id"] in inputs["required_ids"])):
                        eligible.setdefault(entry["description"], entry["runtime_id"])
                        if entry["automatic_eligible"] and entry["runtime_id"] not in inputs["required_ids"]:
                            retrieval_eligible[entry["runtime_id"]] = entry["description"]
                if not eligible:
                    raise ContractError("teacher_empty_candidates")
                cases[case_id] = {"case_id": case_id, "split": "train", "intent": row["intent"],
                                  "group_id": row["group_id"], "action": action, "eligible": eligible,
                                  "reviewed": {}, "retrieval_eligible": retrieval_eligible}
            case = cases[case_id]
            if row["description"] not in case["eligible"]:
                raise ContractError("teacher_candidate_binding")
            case["reviewed"][row["description"]] = row["label"]
    return [cases[key] for key in sorted(cases)]


def retrieved_training_descriptions(case, scores):
    """Rank only the admitted eligible catalog, using serving's stable ID tie break."""
    if case["split"] != "train" or case["case_id"].startswith("control-"):
        raise ContractError("retrieval_train_only")
    if set(scores) != set(case["retrieval_eligible"]):
        raise ContractError("teacher_candidate_binding")
    finite_scores(scores.values(), len(scores))
    ranked = sorted(scores, key=lambda skill_id: (-scores[skill_id], skill_id))[:8]
    # Aliases occupy shortlist slots in serving; deduplicate only after top-k.
    return list(dict.fromkeys(case["retrieval_eligible"][skill_id] for skill_id in ranked))


def retrieval_pooled(model, arrays, torch):
    """FP32 CPU masked mean, identical to the serving retrieval graph's output."""
    with torch.inference_mode():
        inputs = {name: torch.from_numpy(value) for name, value in arrays.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return ((hidden * mask).sum(1) / mask.sum(1)).numpy()


def add_retrieved_training_candidates(cases, model_directory):
    """Load one frozen retrieval encoder, mine TRAIN, then release it before Qwen."""
    if not cases or any(case["split"] != "train" or case["case_id"].startswith("control-") for case in cases):
        raise ContractError("retrieval_train_only")
    import gc
    model = retrieval.Retriever(model_directory)
    try:
        run_encoder = lambda arrays: model.encoder.run(["pooled"], arrays)[0]
        for case in cases:
            vector = retrieval.embed_context(model.tokenizer, run_encoder,
                windows.render_first(case["intent"], case["action"]), model.np)
            scores = {skill_id: float(model.np.dot(model.vector(description), vector))
                      for skill_id, description in case["retrieval_eligible"].items()}
            case["retrieved"] = retrieved_training_descriptions(case, scores)
    finally:
        del model
        gc.collect()


def teacher_training_pairs(case, scores, *, seed):
    """Full-catalog ranking mines candidates; only the fixed union trains the student."""
    if case["split"] != "train" or case["case_id"].startswith("control-"):
        raise ContractError("teacher_train_only")
    eligible, reviewed = case["eligible"], case["reviewed"]
    if set(scores) != set(eligible):
        raise ContractError("teacher_candidate_binding")
    finite_scores(scores.values(), len(eligible))
    ranked = sorted(eligible, key=lambda description: (-scores[description], eligible[description]))
    retrieved = case["retrieved"]
    if len(retrieved) > 8 or len(set(retrieved)) != len(retrieved) or not set(retrieved) <= set(eligible):
        raise ContractError("teacher_candidate_binding")
    selected = set(reviewed) | set(ranked[:16]) | set(retrieved)
    negatives = sorted(set(eligible) - selected)
    rng = random.Random(digest({"seed": seed, "case_id": case["case_id"]}))
    selected.update(rng.sample(negatives, min(8, len(negatives))))
    return [{"case_id": case["case_id"], "split": "train", "intent": case["intent"], "action": case["action"],
             "description": description, "label": reviewed.get(description), "teacher_logit": scores[description]}
            for description in sorted(selected, key=lambda description: eligible[description])]


# Hashes of the pinned public retriever snapshot files.
RETRIEVER_MODEL_FILES = {
    "config.json": "sha256:953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "model.safetensors": "sha256:53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "tokenizer.json": "sha256:be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
}


def build_pretrained_retrieval(args, download_snapshot, run_builder, *, model_files=RETRIEVER_MODEL_FILES):
    """Acquire public pinned bytes in-job before the separate offline-only export."""
    # Snapshot acquisition executes no repository code and receives no private inputs.
    snapshot = Path(download_snapshot(repo_id=retrieval.MODEL["base"], revision=retrieval.MODEL["revision"],
                                      allow_patterns=list(model_files), token=False))
    try:
        for name, expected in model_files.items():
            if file_digest(snapshot / name) != expected:
                raise ContractError("retrieval_model_file_hash")
    except OSError as error:
        raise ContractError("retrieval_model_file_hash") from error
    # The preparation alarm also bounds acquisition and hashing; only its remainder
    # is available to the offline builder. Never start export with unverified files.
    remaining = STAGE_SECONDS["preparation"] - (time.monotonic() - args._budget_started)
    if remaining <= 0:
        raise ContractError("trainer_stage_deadline")
    run_builder([sys.executable, "-B", str(Path(__file__).with_name("build-retrieval-bundle.py")),
        "--catalog", str(args.dataset_manifest.parent / "catalog.json"),
        "--plan", str(args.dataset_manifest),
        "--output", str(args.output / "retrieval-bundle")], check=True, timeout=remaining)


TEACHER_BATCH = 16


def teacher_case_logits(model, tokenizer, head, case, descriptions, torch):
    """Exact contextual Qwen prompt and yes-minus-no final-token scores for one case, batched.

    Left padding plus position ids from the attention mask makes each padded row compute what the
    same prompt computes alone (causal attention never sees the pads), so scores match one-at-a-time
    scoring; one pair per forward pass made ~128k teacher pairs take hours on an A10G.
    """
    if case["split"] != "train":
        raise ContractError("teacher_train_only")
    query = windows.render_first(case["intent"], case["action"])
    texts = [TEACHER_PREFIX + '<Instruct>: ' + TEACHER_INSTRUCTION + '\n<Query>: ' + query
             + '\n<Document>: ' + description + TEACHER_SUFFIX for description in descriptions]
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(texts), TEACHER_BATCH):
        encoded = tokenizer(texts[start:start + TEACHER_BATCH], return_tensors="pt", add_special_tokens=False,
                            truncation=False, padding=True, return_token_type_ids=False)
        lengths = encoded.attention_mask.sum(dim=1).tolist()
        if max(lengths) > 32768:
            raise ContractError("teacher_context_capacity")
        positions = (encoded.attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.inference_mode():
            hidden = model.model(input_ids=encoded.input_ids.to(device), attention_mask=encoded.attention_mask.to(device),
                                 position_ids=positions.to(device), use_cache=False).last_hidden_state[:, -1, :]
            logits = torch.nn.functional.linear(hidden, head)
            scores = (logits[:, 1] - logits[:, 0]).float().cpu().tolist()
        results.extend(zip(finite_scores(scores, len(scores)), lengths))
    return results


def score_teacher_training(cases, args, torch):
    """Mine TRAIN in the private job, persist private intermediates, then release the teacher."""
    if not cases or any(case["split"] != "train" for case in cases):
        raise ContractError("teacher_train_only")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc
    settings = teacher_settings(args)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, revision=args.teacher_revision,
        padding_side="left", trust_remote_code=False, token=False)
    # A10G supports fp16. SDPA plus the two-row output head avoids both a quadratic
    # attention matrix and the unused sequence-by-vocabulary output allocation.
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, revision=args.teacher_revision,
        torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=False, token=False).to(args.device).eval()
    head = None
    pairs = []
    try:
        ids = [tokenizer.convert_tokens_to_ids(token) for token in ("no", "yes")]
        head = model.get_output_embeddings().weight[ids].detach()
        write_json(args.output / "teacher-settings.json", {**settings, "seed": args.seed,
            "dtype": "float16", "attention": "sdpa", "max_tokens": 32768, "truncation": False,
            "batch": TEACHER_BATCH, "padding": "left-with-mask-positions",
            "retrieval": {"model": retrieval.MODEL, "pooling": retrieval.POOLING, "top_k": 8, "tie_break": "id"},
            "score": "yes_logit - no_logit", "head_ids": ids,
            "prompt_sha256": digest({"prefix": TEACHER_PREFIX, "suffix": TEACHER_SUFFIX,
                                      "instruction": TEACHER_INSTRUCTION})})
        (args.output / "teacher-settings.json").chmod(0o600)
        with os.fdopen(os.open(args.output / "teacher-scores.jsonl", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                       "w", encoding="utf-8") as scores_file, \
                os.fdopen(os.open(args.output / "distillation-pairs.jsonl", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                          "w", encoding="utf-8") as pairs_file:
            for index, case in enumerate(cases, 1):
                scores, records = {}, []
                eligible = list(case["eligible"].items())
                scored = teacher_case_logits(model, tokenizer, head, case, [d for d, _ in eligible], torch)
                for (description, skill_id), (score, tokens) in zip(eligible, scored):
                    scores[description] = score
                    records.append({"id": skill_id, "description": description, "logit": score, "tokens": tokens})
                scores_file.write(json.dumps({"case_id": case["case_id"], "split": "train", "scores": records},
                                             ensure_ascii=False, allow_nan=False) + "\n")
                selected = teacher_training_pairs(case, scores, seed=args.seed)
                for pair in selected:
                    pairs_file.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
                pairs.extend(selected)
                scores_file.flush()
                pairs_file.flush()
                emit_phase(args, "teacher_scoring", "PROGRESS", partition_index=index, pair_count=len(pairs))
    finally:
        # Drop every reference to CUDA teacher storage before creating the student's
        # optimizer. No teacher weights or scores enter any serving bundle.
        del head, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return pairs


def checked_tokens(tokenizer, rows):
    native = tokenizer.backend_tokenizer
    native.no_padding()
    native.no_truncation()
    groups = {}
    for index, row in enumerate(rows):
        action = row["action"]
        key = (row["intent"], None if action is None else (action["toolName"], action["summary"]))
        groups.setdefault(key, []).append(index)
    encoded = [None] * len(rows)
    try:
        for indices in groups.values():
            row = rows[indices[0]]
            bags = windows.build_bags(native, row["intent"], row["action"],
                                     [rows[index]["description"] for index in indices])
            for index, bag in zip(indices, bags):
                encoded[index] = bag
    except windows.ContextOverflow:
        raise ContractError("pair_token_overflow") from None
    return encoded


def finite_scores(scores, expected):
    values = [float(value) for value in scores]
    if len(values) != expected or not all(math.isfinite(value) for value in values):
        raise ContractError("invalid_model_scores")
    return values


def decision(score, thresholds):
    lower, upper = thresholds
    return 0 if score <= lower else 1 if score >= upper else None


def metrics(scores, labels, thresholds):
    scores = finite_scores(scores, len(labels))
    if not labels:
        raise ContractError("empty_evaluation")
    bce = sum(max(s, 0) - s * y + math.log1p(math.exp(-abs(s)))
              for s, y in zip(scores, labels)) / len(labels)
    report = {"count": len(labels), "positives": sum(labels), "bce": bce}
    if thresholds is None:
        return report
    predicted = [decision(s, thresholds) for s in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(predicted, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predicted, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))
    accepted = tp + fp + tn + fn
    report.update({"true_positive": tp, "false_positive": fp,
                   "true_negative": tn, "false_negative": fn,
                   "positive_precision": tp / (tp + fp) if tp + fp else None,
                   "negative_precision": tn / (tn + fn) if tn + fn else None,
                   "recall": tp / sum(labels) if sum(labels) else None,
                   "coverage": accepted / len(labels),
                   "abstention": (len(labels) - accepted) / len(labels)})
    return report


def fit_thresholds(scores, labels, positive_target, negative_target, support, min_coverage):
    """Fit inclusive outer decisions on validation only; ties cannot be split.

    Enumerate observed score boundaries. For each upper boundary, the largest
    admissible lower boundary maximizes coverage. Prefix counts avoid rescoring
    every threshold pair. Move the selected cutoffs into their adjacent gaps so
    small export rounding does not change validation decisions. No sentinels,
    infinities, or invented fallback policy.
    """
    scores = finite_scores(scores, len(labels))
    if set(labels) != {0, 1}:
        raise ContractError("calibration_needs_both_labels")
    bins = {}
    for score, label in zip(scores, labels):
        positive, count = bins.get(score, (0, 0))
        bins[score] = (positive + label, count + 1)
    boundaries = sorted(bins)
    lower_options = []
    positive_prefix = count_prefix = 0
    upper_options = []
    total_positive = sum(labels)
    for boundary in boundaries:
        positives, count = bins[boundary]
        positive_above = total_positive - positive_prefix
        count_above = len(labels) - count_prefix
        if count_above >= support and positive_above / count_above >= positive_target:
            upper_options.append((boundary, count_above))
        positive_prefix += positives
        count_prefix += count
        if count_prefix >= support and (count_prefix - positive_prefix) / count_prefix >= negative_target:
            lower_options.append((boundary, count_prefix))
    best = None
    lower_index = 0
    eligible_lower = None
    for upper, accepted_positive in upper_options:
        while lower_index < len(lower_options) and lower_options[lower_index][0] < upper:
            eligible_lower = lower_options[lower_index]
            lower_index += 1
        if eligible_lower is None:
            continue
        lower, accepted_negative = eligible_lower
        coverage = (accepted_positive + accepted_negative) / len(labels)
        if coverage >= min_coverage and (best is None or coverage > best[0]):
            best = (coverage, lower, upper)
    if best is None:
        return None
    _, lower, upper = best
    next_score = boundaries[boundaries.index(lower) + 1]
    previous_score = boundaries[boundaries.index(upper) - 1]
    # Half-sums avoid overflow. For full coverage, straddle the shared midpoint
    # with neighboring floats to keep a distinct uncertainty band.
    lower_cutoff = lower / 2 + next_score / 2
    upper_cutoff = previous_score / 2 + upper / 2
    if lower_cutoff >= upper_cutoff:
        lower_cutoff = math.nextafter(lower_cutoff, -math.inf)
        upper_cutoff = math.nextafter(upper_cutoff, math.inf)
    # A gap may have too few representable floats for interior cutoffs. Clamp
    # toward the included scores, never across an excluded score or a tied bin.
    lower_cutoff = max(lower, min(lower_cutoff, math.nextafter(next_score, -math.inf)))
    upper_cutoff = min(upper, max(upper_cutoff, math.nextafter(previous_score, math.inf)))
    return lower_cutoff, upper_cutoff


def meets_targets(report, args):
    return (report.get("positive_precision") is not None
            and report.get("negative_precision") is not None
            and report["positive_precision"] >= args.target_precision
            and report["negative_precision"] >= args.target_negative_precision
            and report["true_positive"] + report["false_positive"] >= args.min_calibration_support
            and report["true_negative"] + report["false_negative"] >= args.min_calibration_support
            and report["coverage"] >= args.min_coverage)


def fit_baseline_control(scores, labels, args):
    """Validation-only evaluation control; never used for tuned admission."""
    thresholds = fit_thresholds(scores, labels, args.target_precision,
                                args.target_negative_precision, args.min_calibration_support,
                                args.min_coverage)
    if thresholds is not None:
        return thresholds, "validation-gap-midpoints-max-coverage-v2"
    # Full coverage, balanced accuracy, then lowest boundary. Ties stay together.
    scores = finite_scores(scores, len(labels))
    bins = {}
    for score, label in zip(scores, labels):
        positive, count = bins.get(score, (0, 0))
        bins[score] = (positive + label, count + 1)
    boundaries = sorted(bins)
    if len(boundaries) == 1:
        value = boundaries[0]
        candidates = [(value, math.nextafter(value, math.inf)),
                      (math.nextafter(value, -math.inf), value)]
        candidates = [pair for pair in candidates if all(math.isfinite(v) for v in pair)]
        if not candidates:
            raise ContractError("control_thresholds_unavailable")
        # With both classes present, either constant decision has balanced
        # accuracy 0.5. Prefer all-negative unless its upper bound is nonfinite.
        thresholds = candidates[0]
    else:
        positives, negatives = sum(labels), len(labels) - sum(labels)
        positive_below = count_below = 0
        best = None
        for lower, upper in zip(boundaries, boundaries[1:]):
            positive, count = bins[lower]
            positive_below += positive
            count_below += count
            balanced = ((count_below - positive_below) / negatives
                        + (positives - positive_below) / positives) / 2
            if best is None or balanced > best[0]:
                best = (balanced, lower, upper)
        thresholds = (best[1], best[2])
    return thresholds, "validation-balanced-accuracy-full-coverage-v1"


def decimal_string(value):
    if not math.isfinite(value):
        raise ContractError("nonfinite_threshold")
    return format(Decimal(str(value)), "f")


def verify_lock(path):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    text = Path(path).read_text(encoding="utf-8").replace("\\\n", " ")
    pinned = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement_text = line.split(" --hash=", 1)[0].strip()
        if not re.search(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)", line):
            raise ContractError("dependency_lock_requires_hashes")
        requirement = Requirement(requirement_text)
        versions = list(requirement.specifier)
        if requirement.url or len(versions) != 1 or versions[0].operator != "==" or "*" in versions[0].version:
            raise ContractError("dependency_lock_requires_exact_pins")
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = importlib.metadata.version(requirement.name)
        if installed != versions[0].version:
            raise ContractError("dependency_lock_version_mismatch")
        pinned[canonicalize_name(requirement.name)] = installed
    required = {"sentence-transformers", "torch", "transformers", "tokenizers", "numpy",
                "onnx", "onnxruntime", "packaging", "jsonschema", "referencing"}
    if not required <= pinned.keys() or pinned["sentence-transformers"] != "5.1.2":
        raise ContractError("dependency_lock_missing_direct_pins")
    # Every installed dependency of the locked packages must also have a pin.
    for name in list(pinned):
        for dependency in importlib.metadata.requires(name) or []:
            requirement = Requirement(dependency)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency_name = canonicalize_name(requirement.name)
            if dependency_name not in pinned or not requirement.specifier.contains(pinned[dependency_name]):
                raise ContractError("dependency_lock_incomplete_closure")
    return pinned


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def bag_model(bert, torch):
    # Lazy construction keeps admission and diagnostic commands free of Torch.
    class Encoder(torch.nn.Module):
        # The bundle contract fixes pooled vectors at 128 values (geometry, schema, ONNX
        # graphs, worker), so a learned projection maps a wider base (MiniLM, 384) down to
        # it. A base that is already 128 wide (TinyBERT) keeps its pooled output unchanged:
        # run 09 showed an extra randomly initialized 128->128 layer cost TinyBERT its
        # validation targets, which run 02 without it had met.
        def __init__(self):
            super().__init__()
            self.bert = bert
            size = windows.GEOMETRY["pooled_size"]
            self.project = (torch.nn.Identity() if bert.config.hidden_size == size
                            else torch.nn.Linear(bert.config.hidden_size, size))

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.project(self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids, return_dict=False)[1])

    class Aggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = torch.nn.GRUCell(132, 32)
            self.head = torch.nn.Linear(32, 1)

        def forward(self, pooled, positions, hidden):
            hidden_out = self.cell(torch.cat((pooled, positions), dim=1), hidden)
            return hidden_out, self.head(hidden_out)

    class BagModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.aggregator = Aggregator()

        def forward(self, bag, tokenizer):
            device = next(self.parameters()).device
            hidden = torch.zeros((1, 32), device=device)
            for start in range(0, len(bag.windows), 4):
                batch = tokenizer.pad(bag.windows[start:start + 4], padding=True, return_tensors="pt")
                pooled = self.encoder(**{key: value.to(device) for key, value in batch.items()})
                for offset in range(len(pooled)):
                    position = torch.tensor([bag.positions[start + offset]], dtype=torch.float32, device=device)
                    hidden, logit = self.aggregator(pooled[offset:offset + 1], position, hidden)
            return logit.reshape(())

    return BagModel().to(next(bert.parameters()).device)


def score_pretrained(model, tokenizer, encoded, torch):
    """Unchanged native-window arithmetic mean logits, evaluation only."""
    model.eval()
    device = next(model.parameters()).device
    values = []
    with torch.inference_mode():
        for bag in encoded:
            logits = []
            for start in range(0, len(bag.windows), 4):
                batch = tokenizer.pad(bag.windows[start:start + 4], padding=True, return_tensors="pt")
                logits.extend(model(**{key: value.to(device) for key, value in batch.items()}).logits[:, 0].cpu().tolist())
            values.append(sum(logits) / len(logits))
    return finite_scores(values, len(encoded))


def score_torch(model, tokenizer, encoded, batch_size, torch):
    model.eval()
    values = []
    with torch.inference_mode():
        for bag in encoded:
            values.append(model(bag, tokenizer).item())
    return finite_scores(values, len(encoded))


def batch_indices(encoded, order, offset, pair_limit, window_limit=64):
    indices, count = [], 0
    for index in order[offset:offset + pair_limit]:
        size = len(encoded[index].windows)
        if count + size > window_limit:
            break
        indices.append(index)
        count += size
    if not indices:
        raise ContractError("pair_token_overflow")
    return indices


def distillation_loss(logits, labels, teacher_logits, alpha, temperature, torch):
    """Hard BCE over reviewed pairs plus soft BCE over all selected pairs.

    Unknown hard labels use -1. Teacher targets are detached, softened by T;
    student logits stay unscaled and there is no extra T-squared multiplier.
    """
    reviewed = labels >= 0
    hard = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp_min(0), reduction="none")
    hard = (hard * reviewed).sum() / reviewed.sum().clamp_min(1)
    soft = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.sigmoid(teacher_logits.detach() / temperature))
    return hard + alpha * soft


# Pointwise BCE per original pair. A listwise per-request variant (softmax over a zero
# "none" logit plus candidates) was tried in run 05 and lost on every held-out ranking
# metric on the development corpus; use the unchanged pretrained retriever.
def train(encoder, encoded, labels, args, identity, torch, np, *, teacher_logits=None):
    model = encoder.model
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    if teacher_logits is not None:
        hard_targets = torch.tensor([label if label is not None else -1 for label in labels],
                                    dtype=torch.float32, device=next(model.parameters()).device)
        soft_logits = torch.tensor(teacher_logits, dtype=torch.float32, device=hard_targets.device)
    state = {"epoch": 0, "step": 0, "offset": 0, "order": [], "elapsed_seconds": 0.0}
    checkpoint = args.output / "checkpoint.pt"
    if args.resume:
        # Trusted task-owned checkpoints only: optimizer/RNG state uses pickle.
        saved = torch.load(io.BytesIO(args._resume_checkpoint_bytes), map_location="cpu", weights_only=False)
        if saved["identity"] != args._resume_identity:
            raise ContractError("resume_identity_mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    started = time.monotonic()
    previous_elapsed = state["elapsed_seconds"]
    interrupted = False

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def save():
        emit_phase(args, "checkpoint_write", step=state["step"])
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        saved = {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "state": state, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint)
        emit_phase(args, "checkpoint_saved", "END", step=state["step"],
                   checkpoint_bytes=checkpoint.stat().st_size)
        emit_phase(args, "training", "PROGRESS")

    device = next(model.parameters()).device
    try:
        while state["epoch"] < args.epochs and state["step"] < args.max_steps:
            if interrupted or previous_elapsed + time.monotonic() - started >= args.max_seconds:
                break
            if not state["order"]:
                state["order"] = list(range(len(encoded)))
                random.shuffle(state["order"])
            indices = batch_indices(encoded, state["order"], state["offset"], args.batch_size)
            targets = (torch.tensor([labels[index] for index in indices], dtype=torch.float32, device=device)
                       if teacher_logits is None else hard_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            logits = torch.stack([model(encoded[index], encoder.tokenizer) for index in indices])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                    if teacher_logits is None else distillation_loss(logits, targets, soft_logits[indices],
                        args.distillation_alpha, args.distillation_temperature, torch))
            if not torch.isfinite(loss).item():
                raise ContractError("nonfinite_training_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            # One device-to-host sync per step instead of one per parameter tensor.
            if not torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all().item():
                raise ContractError("nonfinite_training_weights")
            state["step"] += 1
            state["offset"] += len(indices)
            if state["offset"] == len(encoded):
                state["epoch"] += 1
                state["offset"] = 0
                state["order"] = []
            if state["step"] % args.checkpoint_steps == 0:
                save()
        save()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return {key: value for key, value in state.items() if key != "order"}, interrupted


def session_for(graph, ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(graph.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"])


def graph_signature(sessions):
    contracts = windows.graph_contracts()
    for role, session in sessions.items():
        expected = contracts[role]
        for nodes, specs in ((session.get_inputs(), expected["inputs"]), (session.get_outputs(), expected["outputs"])):
            actual = [{"name": node.name, "dtype": {"tensor(int64)": "int64", "tensor(float)": "float32"}.get(node.type),
                       "shape": node.shape} for node in nodes]
            if actual != specs:
                raise ContractError("invalid_export_input")
    if set(sessions) != set(contracts):
        raise ContractError("invalid_export_input")
    return contracts


def score_onnx(sessions, tokenizer, encoded, np):
    return finite_scores([windows.score_bag(sessions["encoder"], sessions["aggregator"], bag,
                                           tokenizer.pad_token_id, np) for bag in encoded], len(encoded))


def split_drift_report(reference, candidate, thresholds, tolerance):
    """Aggregate Torch-versus-ONNX drift on one private split.

    Thresholds sit exactly on observed validation scores (inclusive boundaries), so a
    pair within `tolerance` of a threshold can flip under any ONNX rounding that the
    max-error check already allows. Only flips outside that band are parity failures;
    boundary flips are counted separately so they stay visible. Authored fixtures keep
    the strict rule in `drift`, since they are not threshold-fit data."""
    flipped = [a for a, b in zip(reference, candidate) if decision(a, thresholds) != decision(b, thresholds)]
    boundary = sum(1 for a in flipped if min(abs(a - t) for t in thresholds) <= tolerance)
    return {"max_absolute_logit_error": max(abs(a - b) for a, b in zip(reference, candidate)),
            "decision_changes": len(flipped) - boundary, "boundary_decision_changes": boundary}


def drift(reference, candidate, thresholds, pairs, tolerance):
    candidate = finite_scores(candidate, len(reference))
    maximum = max(abs(left - right) for left, right in zip(reference, candidate))
    changed = sum(decision(left, thresholds) != decision(right, thresholds)
                  for left, right in zip(reference, candidate))
    comparisons = flips = 0
    for index, pair in enumerate(pairs):
        for other in range(index + 1, len(pairs)):
            if pair[0] != pairs[other][0]:
                continue
            comparisons += 1
            original_order = (reference[index] > reference[other]) - (reference[index] < reference[other])
            exported_order = (candidate[index] > candidate[other]) - (candidate[index] < candidate[other])
            flips += original_order != exported_order
    return {"max_absolute_logit_error": maximum, "tolerance": tolerance,
            "decision_changes": changed, "ranking_pair_changes": flips,
            "ranking_pair_comparisons": comparisons,
            "passed": maximum <= tolerance and changed == 0 and flips == 0}


def validate_manifest(manifest, schema_directory):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    resources = []
    for path in schema_directory.glob("*.schema.json"):
        schema = parse_json(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)
    schema = parse_json((schema_directory / "model-bundle.schema.json").read_text(encoding="utf-8"))
    if not Draft202012Validator(schema, registry=registry).is_valid(manifest):
        raise ContractError("bundle_schema_mismatch")


def validate_contextual_binding(provenance):
    try:
        bundle_runtime.validate_source_set_binding(provenance.get("contextual_binding"))
    except (bundle_runtime.InvalidInput, KeyError, TypeError):
        raise ContractError("contextual_target_binding") from None


def validate_bundle_payloads(directory, manifest, role):
    try:
        bundle_runtime.verify_provenance_payloads(manifest, {
            name: (directory / name).read_bytes() for name in ("recipe.json", "validation.json")
        }, role)
    except (bundle_runtime.InvalidInput, KeyError, TypeError, OSError):
        raise ContractError("bundle_provenance_mismatch") from None


def make_manifest(directory, signature, thresholds, provenance, runtime_revision, schema_directory, base):
    provenance = {**provenance, "contextual_binding": bundle_runtime.compact_source_set_binding(provenance["contextual_binding"])}
    validate_contextual_binding(provenance)
    files = [{"path": path.relative_to(directory).as_posix(), "sha256": file_digest(path),
              "bytes": path.stat().st_size} for path in sorted(directory.rglob("*")) if path.is_file()]
    if any(value["path"] == "manifest.json" for value in files):
        raise ContractError("bundle_already_finalized")
    manifest = {"schema_version": 1, "model": {"type": "window_bag", "base": base, "revision": windows.MODEL_BASES[base]},
                "graphs": signature, "tokenizer_file": "tokenizer.json", "model_config_file": "config.json",
                "files": files, "pair_format": PAIR_FORMAT, "geometry": windows.GEOMETRY,
                "selection": {"lower_logit": decimal_string(thresholds[0]),
                              "upper_logit": decimal_string(thresholds[1]), "max_optional": 6,
                              "validation_sha256": file_digest(directory / "validation.json"),
                              "fit_split": "validation"},
                "limits": LIMITS, "calibration": None, "calibration_sha256": None,
                "provenance": {"dataset_sha256": provenance["dataset_sha256"],
                               "contextual_binding": provenance["contextual_binding"],

                               "split_sha256": provenance["split_sha256"],
                               "recipe_sha256": file_digest(directory / "recipe.json"),
                               "runtime_revision": runtime_revision}, "publication_allowed": True}
    validate_bundle_payloads(directory, manifest, "deployment_candidate")
    manifest["scorer_sha256"] = bundle_runtime.scorer_digest(manifest)
    manifest["selector_sha256"] = digest({key: manifest[key] for key in
        ("scorer_sha256", "selection", "limits", "calibration_sha256")})
    validate_manifest(manifest, schema_directory)
    write_json(directory / "manifest.json", manifest)
    return manifest


def score_control_onnx(session, pad_id, bags, np):
    values = []
    for bag in bags:
        logits = []
        for start in range(0, len(bag.windows), 4):
            batch = bag.windows[start:start + 4]
            output = session.run(["logits"], windows.padded_arrays(batch, pad_id, np))[0]
            if output.shape != (len(batch), 1):
                raise ContractError("invalid_export_score_shape")
            logits.extend(output[:, 0].tolist())
        values.append(sum(logits) / len(logits))
    return finite_scores(values, len(bags))


def make_control_manifest(directory, thresholds, provenance, runtime_revision, base):
    # Deliberately not a model-bundle schema: the deployment worker rejects it.
    provenance = {**provenance, "contextual_binding": bundle_runtime.compact_source_set_binding(provenance["contextual_binding"])}
    validate_contextual_binding(provenance)
    manifest = {"schema_version": 1, "role": "evaluation_only",
        "control": "unchanged-pretrained-native-window-logit-arithmetic-mean",
        "base": base, "revision": windows.MODEL_BASES[base], "pair_format": PAIR_FORMAT, "geometry": windows.GEOMETRY,
        "files": [{"path": path.name, "sha256": file_digest(path), "bytes": path.stat().st_size}
                  for path in sorted(directory.iterdir()) if path.is_file()],
        "selection": {"lower_logit": decimal_string(thresholds[0]), "upper_logit": decimal_string(thresholds[1]),
                      "max_optional": 6, "validation_sha256": file_digest(directory / "validation.json"), "fit_split": "validation"},
        "provenance": {"dataset_sha256": provenance["dataset_sha256"], "split_sha256": provenance["split_sha256"],
                       "contextual_binding": provenance["contextual_binding"],

                       "recipe_sha256": file_digest(directory / "recipe.json"), "runtime_revision": runtime_revision},
        "limits": LIMITS, "publication_allowed": True}
    validate_bundle_payloads(directory, manifest, "evaluation_only")
    manifest["scorer_sha256"] = digest(manifest)
    manifest["selector_sha256"] = digest({"scorer_sha256": manifest["scorer_sha256"], "selection": manifest["selection"]})
    write_json(directory / "manifest.json", manifest)
    return manifest


def export_graphs(encoder, fixture_encoded, directory, torch, *, pretrained=False):
    """Export graph bytes only. This does not publish or admit a model bundle."""
    sample = encoder.tokenizer.pad([bag.windows[0] for bag in fixture_encoded[:2]], padding=True, return_tensors="pt")
    dynamic = {key: {0: "batch", 1: "sequence"} for key in INPUT_NAMES}
    if pretrained:
        class LogitsOnly(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids).logits
        exported_model, output, filename = LogitsOnly(encoder.model), "logits", "control.onnx"
    else:
        exported_model, output, filename = encoder.model.encoder, "pooled", "encoder.onnx"
    dynamic[output] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(exported_model.eval(), tuple(sample[key] for key in INPUT_NAMES),
                          str(directory / filename), input_names=list(INPUT_NAMES), output_names=[output],
                          dynamic_axes=dynamic, opset_version=17, dynamo=False, external_data=False)
        if not pretrained:
            names = ["pooled", "positions", "hidden", "hidden_out", "logit"]
            torch.onnx.export(encoder.model.aggregator.eval(),
                (torch.zeros(2, 128), torch.zeros(2, 4), torch.zeros(2, 32)),
                str(directory / "aggregator-step.onnx"), input_names=names[:3], output_names=names[3:],
                dynamic_axes={key: {0: "batch"} for key in names}, opset_version=17, dynamo=False, external_data=False)


def export_candidate(encoder, encoded, labels, thresholds, reference_scores, recipe,
                     provenance, args, torch, np, directory, name):
    """One shared export/admission path for unchanged and trained weights."""
    provenance = {**provenance, "contextual_binding": bundle_runtime.compact_source_set_binding(provenance["contextual_binding"])}
    recipe = {**recipe, "provenance": {**recipe["provenance"],
        "contextual_binding": bundle_runtime.compact_source_set_binding(recipe["provenance"]["contextual_binding"])}}
    import onnx
    import onnxruntime as ort
    from tokenizers import Tokenizer

    quantized = name == "int8-arm64"
    pretrained = name == "pretrained-fp32"
    tolerance = args.int8_atol if quantized else args.fp32_atol
    encoder.model.to("cpu").float().eval()
    fixture_rows = [{"intent": pair[0], "action": None, "description": pair[1]} for pair in FIXTURE_PAIRS]
    fixture_rows.extend([
        {"intent": "Review only. " * 180, "action": {"toolName": "bash", "summary": ""}, "description": "Deploy now. " * 180},
        {"intent": "Literal Action: é 中文", "action": None, "description": ""},
        {"intent": "", "action": {"toolName": "read", "summary": ""}, "description": "inspect " * 700}])
    fixture_encoded = checked_tokens(encoder.tokenizer, fixture_rows)
    reference = (score_pretrained(encoder.model, encoder.tokenizer, fixture_encoded, torch) if pretrained else
                 score_torch(encoder.model, encoder.tokenizer, fixture_encoded, 4, torch))
    if quantized:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for filename in ("encoder.onnx", "aggregator-step.onnx"):
            source = directory.parent / "fp32" / filename
            if not source.is_file():
                raise ContractError("fp32_graph_unavailable")
            quantize_dynamic(str(source), str(directory / filename),
                             weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                             op_types_to_quantize=["MatMul"], use_external_data_format=False,
                             extra_options={"WeightSymmetric": True, "MatMulConstBOnly": True})
    else:
        export_graphs(encoder, fixture_encoded, directory, torch, pretrained=pretrained)
    for filename in (("control.onnx",) if pretrained else ("encoder.onnx", "aggregator-step.onnx")):
        graph = onnx.load(str(directory / filename), load_external_data=False)
        if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
            raise ContractError("external_onnx_tensor")
        onnx.checker.check_model(graph)
    encoder.tokenizer.backend_tokenizer.no_truncation()
    encoder.tokenizer.backend_tokenizer.no_padding()
    encoder.tokenizer.backend_tokenizer.save(str(directory / "tokenizer.json"))
    (encoder.model.config if pretrained else encoder.model.encoder.bert.config).to_json_file(str(directory / "config.json"))
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    for row, expected in zip(fixture_rows, fixture_encoded):
        if expected != windows.build_bag(tokenizer, row["intent"], row["action"], row["description"]):
            raise ContractError("tokenizer_parity_failed")
    if pretrained:
        session = session_for(directory / "control.onnx", ort)
        signature = None
        scorer = lambda bags: score_control_onnx(session, encoder.tokenizer.pad_token_id, bags, np)
    else:
        sessions = {role: session_for(directory / spec["file"], ort) for role, spec in windows.graph_contracts().items()}
        signature = graph_signature(sessions)
        scorer = lambda bags: score_onnx(sessions, encoder.tokenizer, bags, np)
    scores = {split: scorer(rows) for split, rows in encoded.items() if split != "train"}
    fixture_scores = scorer(fixture_encoded)
    comparison = drift(reference, fixture_scores, thresholds, [(windows.render_first(row["intent"], row["action"]), row["description"]) for row in fixture_rows], tolerance)
    # Private split evidence is aggregate-only; parity cases remain nonprivate.
    split_drift = {split: split_drift_report(reference_scores[split], values, thresholds, tolerance)
                   for split, values in scores.items()}
    evaluated = {split: metrics(values, labels[split], thresholds) for split, values in scores.items()}
    quality_passed = meets_targets(evaluated["validation"], args)
    parity_passed = comparison["passed"] and all(
        report["max_absolute_logit_error"] <= tolerance and report["decision_changes"] == 0
        for report in split_drift.values())
    passed = parity_passed and (pretrained or quality_passed)
    validation = {"fit_split": "validation", "pair_format": PAIR_FORMAT,
                  "dataset_sha256": provenance["dataset_sha256"], "split_sha256": provenance["split_sha256"],
                  "contextual_binding": provenance["contextual_binding"],

                  "validation_split_sha256": provenance["partitions"]["validation"],
                  "method": recipe["baseline_control_method"] if pretrained else "validation-gap-midpoints-max-coverage-v2",
                  "threshold_source": "pretrained_torch_validation" if pretrained else "tuned_torch_validation",
                  "targets": recipe["targets"], "thresholds": [decimal_string(v) for v in thresholds],
                  "fit_metrics": metrics(reference_scores["validation"], labels["validation"], thresholds),
                  "artifact_validation_metrics": evaluated["validation"],
                  "targets_met": quality_passed}
    validation["role"] = "evaluation_only" if pretrained else "deployment_candidate"
    write_json(directory / "validation.json", validation)
    write_json(directory / "parity.json", {"fixture_version": 1, "nonprivate": True,
        "pair_format": PAIR_FORMAT, "comparison": comparison,
        "cases": [{**row, "windows": bag.windows, "positions": bag.positions,
                   "torch_logit": original, "export_logit": exported,
                   "expected_decision": decision(original, thresholds),
                   "export_decision": decision(exported, thresholds)}
                  for row, bag, original, exported in zip(fixture_rows, fixture_encoded, reference, fixture_scores)]})
    write_json(directory / "evaluation.json", {"metrics": evaluated, "drift": split_drift,
                                                "acceptance_passed": passed,
                                                "quality_targets_met": quality_passed,
                                                "role": validation["role"]})
    artifact_recipe = {**recipe, "full_fine_tuning": not pretrained,
                       "weights": "unchanged_pretrained" if pretrained else "fine_tuned",
                       "role": validation["role"],
                       "export": {"variant": name, "opset": [{"domain": op.domain, "version": op.version}
                       for op in graph.opset_import], "ir_version": graph.ir_version,
                       "quantization": {"target": "arm64", "mode": "dynamic",
                           "activations": "uint8", "weights": "int8", "per_channel": True,
                           "reduce_range": False, "weight_symmetric": True, "operators": ["MatMul"]} if quantized else None}}
    write_json(directory / "recipe.json", artifact_recipe)
    shutil.copyfile(args.dependency_lock, directory / "training.lock")
    # The manifest is last and covers every payload. Failures have no manifest.
    manifest = None
    if passed:
        manifest = (make_control_manifest(directory, thresholds, provenance, args.runtime_revision, args.base) if pretrained else
                    make_manifest(directory, signature, thresholds, provenance, args.runtime_revision, args.schema_directory, args.base))
    return {"passed": passed, "code": "export_gates_passed" if passed else "export_gates_failed",
            "role": validation["role"], "quality_targets_met": quality_passed,
            "parity": comparison, "metrics": evaluated, "split_drift": split_drift,
            "selector_sha256": manifest["selector_sha256"] if manifest else None}


def export_candidates(encoder, encoded, labels, thresholds, reference_scores, recipe,
                      provenance, args, torch, np, *, pretrained=False):
    reports = {}
    names = ("pretrained-fp32",) if pretrained else ("fp32", "int8-arm64")
    for name in names:
        directory = args.output / args.export_name / name
        # Never catch a directory collision and then write into old evidence.
        directory.mkdir()
        try:
            if thresholds is None:
                raise ContractError("validation_targets_unmet")
            report = export_candidate(encoder, encoded, labels, thresholds, reference_scores,
                                      recipe, provenance, args, torch, np, directory, name)
        except ContractError as error:
            if str(error) == "trainer_stage_deadline":
                raise
            report = {"passed": False, "code": str(error), "selector_sha256": None}
        except Exception as error:
            record_failure(directory, "export_" + name, error)
            report = {"passed": False, "code": "candidate_export_runtime_failure", "selector_sha256": None}
        if not report["passed"]:
            write_json(directory / "failure.json", report)
        reports[name] = report
        # Preserve each outcome even if a later export or training is interrupted.
        write_json(args.output / args.export_name / (name + ".json"), report)
    return reports


def export_outcome(exports, steps):
    usable = sorted(name for name, report in exports.items() if report["passed"])
    ready = "pretrained-fp32" in usable and "fp32" in usable
    return {"status": "candidate_ready" if ready else "rejected",
            "code": "required_export_gates_passed" if ready else "required_export_gates_failed",
            "steps": steps, "comparison_ready": ready, "usable_candidates": usable,
            "exports": exports, "deployment_verified": False}, 0 if ready else 3


def parser():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("--version", action="version", version="train-router 1")
    for name in ("data", "dataset-manifest", "dependency-lock", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--runtime-revision", required=True, help="Exact SHA-256 digest of the current worker source")
    cli.add_argument("--approval-sha256", required=True, help="Operator-pinned contextual approval digest, independent of the export payload")
    cli.add_argument("--init-weights", type=Path, help="Optional initial bag-model safetensors file")
    cli.add_argument("--base", required=True, choices=("cross-encoder/ms-marco-MiniLM-L6-v2",),
                     help="Pinned FP32 decider base; retrieval uses the pinned pretrained MiniLM")
    cli.add_argument("--init-sha256", default="none",
                     help="'none', or the SHA-256 digest of --init-weights, loaded "
                          "only after the unchanged pretrained control is exported")
    cli.add_argument("--schema-directory", type=Path, default=Path(__file__).resolve().parents[1] / "schemas")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--export-name", default="exports", help="New directory name within output; never overwritten")
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--max-steps", type=int, default=10000)
    cli.add_argument("--max-seconds", type=float, default=5400)
    cli.add_argument("--batch-size", type=int, default=16)
    cli.add_argument("--checkpoint-steps", type=int, default=50)
    cli.add_argument("--learning-rate", type=float, default=0.00002)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--target-precision", type=float, required=True)
    cli.add_argument("--target-negative-precision", type=float, required=True)
    cli.add_argument("--min-calibration-support", type=int, required=True)
    cli.add_argument("--min-coverage", type=float, required=True)
    cli.add_argument("--fp32-atol", type=float, default=0.0001)
    cli.add_argument("--int8-atol", type=float, default=0.25)
    for name in private_inputs.TEACHER_ARGUMENTS:
        cli.add_argument("--" + name.replace("_", "-"),
                         type=float if name.startswith("distillation_") else str)
    return cli


def validate_arguments(args):
    teacher_settings(args)
    if args.max_seconds > 5400:
        raise ContractError("trainer_window_budget")
    for name in ("epochs", "max_steps", "max_seconds", "batch_size", "checkpoint_steps",
                 "learning_rate", "min_calibration_support", "fp32_atol", "int8_atol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ContractError("invalid_positive_argument")
    if args.fp32_atol > 0.0001:
        raise ContractError("invalid_target")
    for name in ("target_precision", "target_negative_precision", "min_coverage"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ContractError("invalid_target")
    if not 0 <= args.seed < 2**32 or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.export_name):
        raise ContractError("invalid_run_argument")
    if args.runtime_revision != file_digest(Path(bundle_runtime.__file__)):
        raise ContractError("runtime_revision_mismatch")


def stage_budget(args, stage):
    """Linux trainer guard; the controller remains the hard native-call timeout."""
    now = time.monotonic()
    if hasattr(args, "_budget_stage"):
        args._stage_timings[args._budget_stage] = now - args._budget_started
    args._budget_stage, args._budget_started = stage, now
    signal.setitimer(signal.ITIMER_REAL, STAGE_SECONDS[stage])


@contextlib.contextmanager
def trainer_budget(args):
    # A stage overrun rejects the attempt; it must never become an optional export failure.
    def expired(_signum, _frame):
        raise ContractError("trainer_stage_deadline")
    previous = signal.signal(signal.SIGALRM, expired)
    args._stage_timings = {}
    try:
        stage_budget(args, "preparation")
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if getattr(args, "_phase_directory", None) is not None:
            args._stage_timings[args._budget_stage] = time.monotonic() - args._budget_started
            write_json(args.output / "stage-timings.json", {"limits_seconds": STAGE_SECONDS,
                       "elapsed_seconds": args._stage_timings})


def run(args):
    with trainer_budget(args):
        return run_training(args)


def run_training(args):
    validate_arguments(args)
    if (args.output / args.export_name).exists():
        raise ContractError("export_directory_exists")
    emit_phase(args, "dataset_validation")
    partitions, provenance, scan, admission = load_export(
        args.data, args.dataset_manifest, approval_sha256=args.approval_sha256)
    provenance = {**provenance, "contextual_binding":
        bundle_runtime.compact_source_set_binding(provenance["contextual_binding"])}
    emit_phase(args, "dependency_validation")
    dependencies = verify_lock(args.dependency_lock)
    recipe = run_recipe(args, provenance, dependencies, file_digest(args.dependency_lock))
    # Recipes contain floats and are hashed by exact file bytes, not H(value).
    if args.resume:
        if not args.output.is_dir() or not (args.output / "checkpoint.pt").is_file():
            raise ContractError("resume_checkpoint_missing")
        previous_raw = (args.output / "recipe.json").read_bytes()
        if parse_json(previous_raw.decode("utf-8")) != recipe:
            raise ContractError("resume_recipe_mismatch")
        args._resume_identity = file_digest(args.output / "recipe.json")
        args._resume_checkpoint_bytes = (args.output / "checkpoint.pt").read_bytes()
        parent = args.output / ("parent-recipe-" + args._resume_identity[7:] + ".json")
        if parent.exists():
            if parent.read_bytes() != previous_raw:
                raise ContractError("resume_recipe_mismatch")
        else:
            with os.fdopen(os.open(parent, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
                handle.write(previous_raw)
        write_json(args.output / "recipe.json", recipe)
    else:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        write_json(args.output / "recipe.json", recipe)
    args._phase_directory = args.output
    identity = file_digest(args.output / "recipe.json")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    emit_phase(args, "dependency_imports")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder
    from transformers.utils import logging as transformer_logging
    transformer_logging.set_verbosity_error()
    torch.set_num_threads(2)
    seed_all(args.seed, torch, np)
    # The constructor owns cache lookup/download and model/tokenizer loading.
    emit_phase(args, "model_loading")
    encoder = CrossEncoder(args.base, revision=windows.MODEL_BASES[args.base], num_labels=1, max_length=MAX_PAIR_TOKENS,
                           activation_fn=torch.nn.Identity(), device=args.device, trust_remote_code=False,
                           token=False, model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"},
                           tokenizer_kwargs={"use_fast": True})
    encoded = {}
    for index, (split, rows) in enumerate(partitions.items(), 1):
        emit_phase(args, "tokenization", partition_index=index, pair_count=len(rows))
        encoded[split] = checked_tokens(encoder.tokenizer, rows)
    labels = {split: [row["label"] for row in rows] for split, rows in partitions.items()}
    emit_phase(args, "baseline_scoring")
    baseline_scores = {split: score_pretrained(encoder.model, encoder.tokenizer, rows, torch)
                       for split, rows in encoded.items() if split != "train"}
    emit_phase(args, "baseline_calibration")
    baseline_thresholds, baseline_method = fit_baseline_control(
        baseline_scores["validation"], labels["validation"], args)
    recipe["baseline_control_method"] = baseline_method
    baseline_metrics = {split: metrics(scores, labels[split], baseline_thresholds)
                        for split, scores in baseline_scores.items()}
    baseline = {"fit_split": "validation", "role": "evaluation_only",
                "thresholds": [decimal_string(v) for v in baseline_thresholds],
                "targets_met": meets_targets(baseline_metrics["validation"], args),
                "dataset_sha256": provenance["dataset_sha256"],
                "split_sha256": provenance["split_sha256"],
                "contextual_binding": provenance["contextual_binding"],

                "validation_split_sha256": provenance["partitions"]["validation"],
                "method": baseline_method,
                "targets": recipe["targets"],
                "metrics": baseline_metrics}
    write_json(args.output / "baseline.json", baseline)
    # Export the actual immutable base before train() can restore tuned checkpoint
    # weights or take an optimizer step. Resume repeats this from the pinned base.
    (args.output / args.export_name).mkdir()
    emit_phase(args, "baseline_export")
    exports = export_candidates(encoder, encoded, labels, baseline_thresholds, baseline_scores,
                                recipe, provenance, args, torch, np, pretrained=True)
    encoder.model.to(args.device)
    teacher_logits = None
    if args.teacher_model is not None:
        emit_phase(args, "teacher_scoring")
        catalog = parse_json((args.dataset_manifest.parent / "catalog.json").read_text())
        cases = teacher_cases(partitions["train"], catalog, admission, scan["holdout_families"])
        # New admission gets a new bundle, but never new retriever weights.
        retrieval_bundle = args.output / "retrieval-bundle"
        # A transported resume carries checkpoint state, not this deterministic export.
        # Reuse an existing complete bundle only after the same admission below.
        if not retrieval_bundle.exists():
            import subprocess
            from huggingface_hub import snapshot_download
            build_pretrained_retrieval(args, snapshot_download, subprocess.run)
        manifest, _ = retrieval.load_manifest(retrieval_bundle)
        if (manifest["provenance"]["weights"] != recipe["distillation_retrieval"]["weights"] or
                any(manifest["provenance"][key] != provenance[key]
                    for key in ("contextual_binding", "dataset_sha256", "split_sha256"))):
            raise ContractError("teacher_candidate_binding")
        add_retrieved_training_candidates(cases, retrieval_bundle)
        derived = score_teacher_training(cases, args, torch)
        training_encoded = checked_tokens(encoder.tokenizer, derived)
        training_labels = [pair["label"] for pair in derived]
        teacher_logits = [pair["teacher_logit"] for pair in derived]
        emit_phase(args, "teacher_scoring", "END", pair_count=len(derived))
    # Initialization includes the complete projection and recurrent head.
    # Load it below only after constructing the canonical bag model.
    # Baseline evaluation must not advance the training RNG stream.
    seed_all(args.seed, torch, np)
    encoder.model = bag_model(encoder.model.bert, torch)
    if args.init_sha256 != "none":
        from safetensors.torch import load_file
        init = args.init_weights
        if init is None:
            raise ContractError("init_weights_mismatch")
        if file_digest(init) != args.init_sha256:
            raise ContractError("init_weights_mismatch")
        encoder.model.load_state_dict(load_file(str(init)), strict=True)
        encoder.model.to(args.device)
    write_json(args.output / "architecture.json", {"encoder_parameters": sum(p.numel() for p in encoder.model.encoder.parameters()),
               "aggregator_parameters": sum(p.numel() for p in encoder.model.aggregator.parameters()),
               "total_parameters": sum(p.numel() for p in encoder.model.parameters()), "geometry": windows.GEOMETRY})
    stage_budget(args, "optimization")
    emit_phase(args, "training")
    if teacher_logits is None:
        state, interrupted = train(encoder, encoded["train"], labels["train"], args, identity, torch, np)
    else:
        state, interrupted = train(encoder, training_encoded, training_labels, args, identity, torch, np,
                                   teacher_logits=teacher_logits)
    stage_budget(args, "finalization")
    write_json(args.output / "training-state.json", {**state, "recipe_sha256": identity, "provenance": provenance})
    if interrupted or state["step"] == 0:
        result, _ = export_outcome(exports, state["step"])
        result.update(status="stopped", code="interrupted" if interrupted else "no_training_steps")
        return result, 3
    emit_phase(args, "trained_save")
    trained = args.output / "trained"
    trained.mkdir(exist_ok=True)
    torch.save(encoder.model.state_dict(), trained / "bag-model.pt")
    encoder.tokenizer.save_pretrained(str(trained))
    encoder.model.encoder.bert.config.to_json_file(str(trained / "config.json"))
    emit_phase(args, "tuned_evaluation")
    tuned_scores = {split: score_torch(encoder.model, encoder.tokenizer, rows, args.batch_size, torch)
                    for split, rows in encoded.items() if split != "train"}
    emit_phase(args, "tuned_calibration")
    thresholds = fit_thresholds(tuned_scores["validation"], labels["validation"], args.target_precision,
        args.target_negative_precision, args.min_calibration_support, args.min_coverage)
    report = {"baseline": baseline, "training": state, "provenance": provenance, "exports": exports,
              "tuned": {"fit_split": "validation", "targets_met": thresholds is not None,
                        "thresholds": None if thresholds is None else [decimal_string(v) for v in thresholds],
                        "metrics": {split: metrics(scores, labels[split], thresholds)
                                    for split, scores in tuned_scores.items()}}}
    write_json(args.output / "report.json", report)
    emit_phase(args, "tuned_export")
    exports.update(export_candidates(encoder, encoded, labels, thresholds, tuned_scores, recipe,
                                     provenance, args, torch, np))
    write_json(args.output / "report.json", report)
    return export_outcome(exports, state["step"])


def record_failure(directory, stage, error):
    """Private metadata only: no exception arguments, source lines, or locals."""
    frames = []
    frame = error.__traceback__
    while frame is not None:
        code = frame.tb_frame.f_code
        # External paths and arbitrary source text never enter diagnostics.
        if code.co_filename == __file__:
            frames.append({"function": code.co_name, "line": frame.tb_lineno})
        frame = frame.tb_next
    metadata = {"stage": stage, "exception_type": type(error).__name__, "frames": frames}
    if isinstance(error, ContractError):
        metadata["error_code"] = contract_error_code(error)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="failure-diagnostic-",
                                     suffix=".json", dir=directory, delete=False) as stream:
        json.dump(metadata, stream, allow_nan=False)
        stream.write("\n")


def main():
    args = parser().parse_args()
    args._stage = "argument_validation"
    args._phase_stream = sys.stdout
    logging.disable(logging.CRITICAL)
    try:
        emit_phase(args, "argument_validation")
        # Third-party warnings may embed paths or inputs. Keep them out of job
        # logs rather than forwarding unreviewed exception messages.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            result, exit_code = run(args)
        emit_phase(args, "completion", "END")
    except ContractError as error:
        emit_failure_phase(args, error)
        try:
            if getattr(args, "_phase_directory", None) is not None:
                record_failure(args._phase_directory, args._stage, error)
        except OSError:
            print('{"status":"error","code":"diagnostic_write_failed"}', file=sys.stderr)
        print(json.dumps({"status": "error", "code": contract_error_code(error)}), file=sys.stderr)
        return 2
    except KeyboardInterrupt as error:
        emit_failure_phase(args, error)
        print('{"status":"error","code":"interrupted"}', file=sys.stderr)
        return 130
    except Exception as error:
        emit_failure_phase(args, error)
        try:
            if getattr(args, "_phase_directory", None) is not None:
                record_failure(args._phase_directory, args._stage, error)
        except OSError:
            print('{"status":"error","code":"diagnostic_write_failed"}', file=sys.stderr)
        print('{"status":"error","code":"training_runtime_failure"}', file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
