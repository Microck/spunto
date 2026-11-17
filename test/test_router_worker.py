"""Stdlib boundary tests; real ONNX execution is a separate artifact-required gate."""
import copy
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "router_worker", Path(__file__).resolve().parents[1] / "python/router-worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "bundle_trainer", Path(__file__).resolve().parents[1] / "python/train-router.py")
trainer = importlib.util.module_from_spec(TRAINER_SPEC)
TRAINER_SPEC.loader.exec_module(trainer)
D = "sha256:" + "1" * 64
REVISION = trainer.file_digest(Path(worker.__file__))


def source_binding():
    """Inert composed provenance only; never an actual source or execution grant."""
    identity = {key: D for key in (
        "target_sha256", "schema_sha256", "annotation_sha256", "receipt_sha256",
        "catalog_sha256", "source_dataset_sha256", "dataset_schema_sha256",
        "source_contract_sha256", "role_plan_sha256", "identity_map_sha256", "projection_sha256")}
    identity["rubric_sha256"] = trainer.digest("independent-fixture-rubric")
    return {"version": 1, "target_sha256": D, "catalog_sha256": D,
            "source_set_sha256": trainer.digest({"purpose": "local-canonical-integration",
                "fixture_only": True, "members": [identity]}),
            "runtime_revision": trainer.file_digest(Path(__file__).resolve().parents[1] / "python/build-dataset.py"),
            "approval_sha256": D, "purpose": "local-canonical-integration", "fixture_only": True,
            "members": [{"member_sha256": trainer.digest(identity), "identity": identity, "approval_sha256": D}]}


def fixture():
    return {"version": 1, "requestId": "synthetic-request", "sessionKey": "synthetic-session",
            "generation": 0, "event": "context", "catalogDigest": D, "contextDigest": D,
            "policyDigest": D, "selectorDigest": D, "intent": "Review a Python function.",
            "action": None, "candidates": [
                {"id": "review", "name": "review", "description": "Review code for defects.", "contentDigest": D},
                {"id": "images", "name": "images", "description": "Resize images.", "contentDigest": D}],
            "candidateCoverage": "full", "requiredIds": [], "excludedIds": [], "maxSelections": 6}


class ProtocolTests(unittest.TestCase):

    def test_framing_and_json_reject_ambiguity(self):
        for payload in (b'null\n', b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{}', b'\xff\n',
                        b' ' * worker.MAX_LINE + b'\n'):
            with self.subTest(payload_length=len(payload)):
                with self.assertRaises((worker.InvalidInput, UnicodeError)):
                    worker.read_request(io.BytesIO(payload))
        self.assertIsNone(worker.read_request(io.BytesIO(b"")))

    def test_valid_jsonl_identity_echo(self):
        request = worker.read_request(io.BytesIO((json.dumps(fixture()) + "\n").encode()))
        worker.validate_request(request)
        response = worker.response(request, "uncertain", "busy")
        for key in worker.IDENTITY:
            self.assertEqual(response[key], request[key])

    def test_valid_unicode_is_admitted_and_unpaired_surrogates_are_rejected(self):
        request = fixture()
        request["intent"] = "café e\u0301 中文 \U0001f642"
        worker.validate_request(request)
        for codepoint in (0xD800, 0xDBFF, 0xDC00, 0xDFFF):
            request["intent"] = "prefix" + chr(codepoint) + "suffix"
            with self.subTest(codepoint=codepoint), self.assertRaises(worker.InvalidInput):
                worker.validate_request(request)

    def test_invalid_identity_and_unknown_fields(self):
        for key, value in (("version", True), ("generation", True), ("requestId", ""),
                           ("catalogDigest", "not-a-digest"), ("action", {"toolName": "read", "intent": "old"})):
            request = fixture()
            request[key] = value
            with self.subTest(key=key), self.assertRaises(worker.InvalidInput):
                worker.validate_request(request)
        request = fixture()
        request["unknown"] = "not accepted"
        with self.assertRaises(worker.InvalidInput):
            worker.validate_request(request)

    def test_duplicate_ids_and_action_requirement(self):
        request = fixture()
        request["candidates"].append(copy.deepcopy(request["candidates"][0]))
        with self.assertRaises(worker.InvalidInput):
            worker.validate_request(request)
        request = fixture()
        request["event"] = "tool_call"
        with self.assertRaises(worker.InvalidInput):
            worker.validate_request(request)


class PolicyTests(unittest.TestCase):
    def decide(self, request, values):
        worker.validate_request(request)
        candidates = worker.preflight(request)
        return worker.decide(request, candidates, values, -1.0, 1.0, D)

    def test_threshold_boundaries_are_not_probabilities(self):
        request = fixture()
        self.assertEqual(self.decide(request, [1.0, -2.0])["selectedIds"], ["review"])
        for value in (-0.999, 0.0, 0.999):
            result = self.decide(request, [value, -2.0])
            self.assertEqual((result["status"], result["reasonCode"]),
                             ("uncertain", "insufficient_evidence"))
        result = self.decide(request, [-1.0, -2.0])
        self.assertEqual(result["status"], "none")
        self.assertEqual(result["scores"]["kind"], "ranking")
        self.assertIsNone(result["scores"]["calibrationDigest"])

    def test_shortlist_and_missing_evidence_cannot_certify_none(self):
        request = fixture()
        request["candidateCoverage"] = "shortlist"
        self.assertEqual(self.decide(request, [-3.0, -2.0])["reasonCode"], "inadequate_candidates")
        request["candidateCoverage"] = "full"
        request["intent"] = " "
        self.assertEqual(self.decide(request, [-3.0, -2.0])["reasonCode"], "insufficient_evidence")

    def test_required_ids_do_not_consume_optional_cap(self):
        request = fixture()
        request["requiredIds"] = ["review"]
        request["maxSelections"] = 1
        self.assertEqual(self.decide(request, [-5.0, 3.0])["selectedIds"], ["review", "images"])
        request["maxSelections"] = 0
        result = self.decide(request, [0.0, -5.0])
        self.assertEqual((result["selectedIds"], result["reasonCode"]), (["review"], "policy_only"))


    def test_confident_positive_survives_gray_without_promoting_it(self):
        request = fixture()
        for coverage in ("full", "shortlist"):
            request["candidateCoverage"] = coverage
            result = self.decide(request, [1.0, 0.0])
            self.assertEqual((result["status"], result["selectedIds"], result["reasonCode"]),
                             ("selected", ["review"], "matched"))
        request["requiredIds"] = ["images"]
        self.assertEqual(self.decide(request, [1.0, 0.0])["selectedIds"], ["review", "images"])
        request["requiredIds"] = ["review"]
        result = self.decide(request, [-2.0, 0.0])
        self.assertEqual((result["status"], result["selectedIds"]), ("uncertain", []))

    def test_overflow_precedes_other_candidates_gray_abstention(self):
        request = fixture()
        request["maxSelections"] = 0
        result = self.decide(request, [2.0, 0.0])
        self.assertEqual(result["reasonCode"], "selection_overflow")
    def test_optional_overflow_never_drops_a_selection(self):
        request = fixture()
        request["maxSelections"] = 1
        result = self.decide(request, [2.0, 3.0])
        self.assertEqual((result["status"], result["selectedIds"], result["reasonCode"]),
                         ("uncertain", [], "selection_overflow"))

    def test_caller_cap_above_manifest_is_invalid_not_clamped(self):
        request = fixture()
        request["maxSelections"] = 7
        result = self.decide(request, [-3.0, -2.0])
        self.assertEqual((result["status"], result["reasonCode"], result["selectedIds"]),
                         ("uncertain", "invalid_response", []))

    def test_excluded_candidate_cannot_be_selected(self):
        request = fixture()
        request["excludedIds"] = ["review"]
        result = self.decide(request, [2.0])
        self.assertEqual(result["selectedIds"], ["images"])
        self.assertEqual(result["scores"]["values"], [{"id": "images", "score": 2.0}])

    def test_required_conflicts_and_unknowns_are_explicit(self):
        for required, excluded, reason in ((["review"], ["review"], "policy_conflict"),
                                           (["unknown"], [], "missing_required")):
            request = fixture()
            request["requiredIds"], request["excludedIds"] = required, excluded
            with self.assertRaises(worker.RoutingFailure) as caught:
                worker.preflight(request)
            self.assertEqual(caught.exception.reason, reason)

    def test_candidate_overflow_is_not_a_negative_decision(self):
        request = fixture()
        request["candidates"] = [
            {**request["candidates"][0], "id": f"candidate-{index}", "name": f"candidate-{index}"}
            for index in range(33)]
        worker.validate_request(request)
        with self.assertRaises(worker.RoutingFailure) as caught:
            worker.preflight(request)
        self.assertEqual(caught.exception.reason, "context_overflow")

    def test_nonfinite_scores_are_invalid(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(worker.InvalidInput):
                self.decide(fixture(), [value, -2.0])


class ArtifactTests(unittest.TestCase):
    """These test the bundle boundary, not inference on synthetic graph bytes."""

    REJECTIONS = (
        "binding_without_identity", "binding_with_stale_payloads", "runtime_without_identity",
        "runtime_with_stale_recipe", "recipe_runtime_revision", "recipe_runtime_missing",
        "missing_recipe", "missing_recipe_bytes", "binding_members",
        "recipe_digest", "recipe_provenance", "validation_contextual", "unverified_recipe_bytes",
        "recipe_role", "recipe_weights", "validation_thresholds", "validation_fit_split",
        *(f"{location}_{field}_null" for location in ("manifest", "recipe", "validation")
          for field in ("dataset_sha256", "split_sha256")),
    )

    def make_bundle(self, directory, role="deployment_candidate"):
        provenance = {"dataset_sha256": D, "split_sha256": D, "contextual_binding": worker.compact_source_set_binding(source_binding())}
        recipe = {"role": role, "weights": "fine_tuned" if role == "deployment_candidate" else "unchanged_pretrained",
                  "runtime_revision": REVISION, "provenance": provenance}
        report = {"role": role, "fit_split": "validation", "thresholds": ["-1", "1"], **provenance}
        payloads = {"tokenizer.json": b"{}", "config.json": b"{}",
                    "recipe.json": json.dumps(recipe).encode(),
                    "validation.json": json.dumps(report).encode(),
                    "evaluation.json": b"{}", "parity.json": b"{}", "training.lock": b"inert-fixture"}
        if role == "deployment_candidate":
            payloads.update({"encoder.onnx": b"hash-check-only-not-an-onnx-model",
                             "aggregator-step.onnx": b"hash-check-only-aggregator"})
        else:
            payloads["control.onnx"] = b"hash-check-only-not-an-onnx-model"
        for name, payload in payloads.items():
            (directory / name).write_bytes(payload)
        base = "cross-encoder/ms-marco-MiniLM-L6-v2"
        if role == "evaluation_only":
            return trainer.make_control_manifest(directory, (-1, 1), provenance, REVISION, base)
        return trainer.make_manifest(directory, worker.windows.graph_contracts(), (-1, 1),
                                     provenance, REVISION,
                                     Path(__file__).resolve().parents[1] / "schemas", base)

    def save(self, directory, manifest):
        if manifest.get("role") == "evaluation_only":
            manifest["scorer_sha256"] = worker.canonical_hash({
                key: value for key, value in manifest.items() if key not in ("scorer_sha256", "selector_sha256")})
            manifest["selector_sha256"] = worker.canonical_hash({
                key: manifest[key] for key in ("scorer_sha256", "selection")})
        else:
            manifest["scorer_sha256"] = worker.scorer_digest(manifest)
            manifest["selector_sha256"] = worker.canonical_hash({key: manifest[key] for key in (
                "scorer_sha256", "selection", "limits", "calibration_sha256")})
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def rejection_bundle(self, directory, case, role="deployment_candidate"):
        """Inert admission fixtures, never runnable graphs or quality evidence."""
        manifest = self.make_bundle(directory, role)
        changed = "sha256:" + "2" * 64
        if case in ("binding_without_identity", "binding_with_stale_payloads"):
            manifest["provenance"]["contextual_binding"]["source_set_sha256"] = changed
        elif case == "binding_members":
            manifest["provenance"]["contextual_binding"] = source_binding()
        elif case in ("runtime_without_identity", "runtime_with_stale_recipe"):
            manifest["provenance"]["runtime_revision"] = "different-source-revision"
        elif case == "missing_recipe_bytes":
            (directory / "recipe.json").unlink()
        elif case == "missing_recipe":
            (directory / "recipe.json").unlink()
            manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "recipe.json"]
        elif case == "recipe_digest":
            manifest["provenance"]["recipe_sha256"] = changed
        elif case.startswith("manifest_"):
            manifest["provenance"][case.removeprefix("manifest_").removesuffix("_null")] = None
        else:
            name = "validation.json" if case.startswith("validation_") else "recipe.json"
            document = json.loads((directory / name).read_bytes())
            if case.endswith("_null"):
                field = case.split("_", 1)[1].removesuffix("_null")
                (document["provenance"] if name == "recipe.json" else document)[field] = None
            elif case in ("recipe_provenance", "unverified_recipe_bytes"):
                document["provenance"]["dataset_sha256"] = changed
            elif case == "validation_contextual":
                document["contextual_binding"]["approval_sha256"] = changed
            elif case == "recipe_role":
                document["role"] = "other"
            elif case == "recipe_weights":
                document["weights"] = "other"
            elif case == "recipe_runtime_revision":
                document["runtime_revision"] = "different-source-revision"
            elif case == "recipe_runtime_missing":
                document.pop("runtime_revision")
            elif case == "validation_thresholds":
                document["thresholds"] = ["-2", "2"]
            elif case == "validation_fit_split":
                document["fit_split"] = "test"
            else:
                raise AssertionError(case)
            raw = json.dumps(document).encode()
            (directory / name).write_bytes(raw)
            if case != "unverified_recipe_bytes":
                entry = next(entry for entry in manifest["files"] if entry["path"] == name)
                entry.update(bytes=len(raw), sha256="sha256:" + hashlib.sha256(raw).hexdigest())
                if name == "recipe.json":
                    manifest["provenance"]["recipe_sha256"] = entry["sha256"]
                else:
                    manifest["selection"]["validation_sha256"] = entry["sha256"]
        if case in ("binding_without_identity", "runtime_without_identity"):
            (directory / "manifest.json").write_text(json.dumps(manifest))
        else:
            self.save(directory, manifest)
        return manifest

    def test_producer_and_both_runtime_roles_bind_provenance(self):
        spec = importlib.util.spec_from_file_location(
            "binding_evaluator", Path(__file__).resolve().parents[1] / "python/evaluate-router.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        for role in ("deployment_candidate", "evaluation_only"):
            def admit(path):
                return worker.load_manifest(path) if role == "deployment_candidate" else evaluator.load_control(path, worker)
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest = self.make_bundle(directory, role)
                admitted, payloads = admit(directory)
                self.assertEqual(admitted, manifest)
                self.assertEqual(payloads["recipe.json"], (directory / "recipe.json").read_bytes())
            for case in self.REJECTIONS:
                with self.subTest(role=role, case=case), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    self.rejection_bundle(directory, case, role)
                    with self.assertRaises((ValueError, OSError)):
                        admit(directory)

    def test_publishers_reject_incoherent_payloads(self):
        for role in ("deployment_candidate", "evaluation_only"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest = self.make_bundle(directory, role)
                (directory / "manifest.json").unlink()
                report = json.loads((directory / "validation.json").read_bytes())
                report["contextual_binding"]["source_set_sha256"] = "sha256:" + "2" * 64
                trainer.write_json(directory / "validation.json", report)
                with self.assertRaisesRegex(trainer.ContractError, "bundle_provenance_mismatch"):
                    if role == "evaluation_only":
                        trainer.make_control_manifest(directory, (-1, 1), manifest["provenance"], REVISION, manifest["base"])
                    else:
                        trainer.make_manifest(directory, manifest["graphs"], (-1, 1), manifest["provenance"],
                                              REVISION, Path(__file__).resolve().parents[1] / "schemas", manifest["model"]["base"])

    def test_producer_rejects_conflicting_or_noncurrent_revision(self):
        for role in ("deployment_candidate", "evaluation_only"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                before = self.make_bundle(directory, role)
                (directory / "manifest.json").unlink()

                def publish():
                    if role == "evaluation_only":
                        return trainer.make_control_manifest(directory, (-1, 1), before["provenance"],
                                                             "new-source-revision", before["base"])
                    return trainer.make_manifest(directory, before["graphs"], (-1, 1), before["provenance"],
                                                 "new-source-revision", Path(__file__).resolve().parents[1] / "schemas",
                                                 before["model"]["base"])

                with self.assertRaisesRegex(trainer.ContractError, "bundle_provenance_mismatch"):
                    publish()
                self.assertFalse((directory / "manifest.json").exists())
                recipe = json.loads((directory / "recipe.json").read_bytes())
                recipe["runtime_revision"] = "new-source-revision"
                trainer.write_json(directory / "recipe.json", recipe)
                with self.assertRaisesRegex(trainer.ContractError, "bundle_provenance_mismatch"):
                    publish()
                self.assertFalse((directory / "manifest.json").exists())

    def test_payload_hash_and_unlisted_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_bundle(directory)
            verified, _ = worker.load_manifest(directory)
            self.assertEqual(verified["selector_sha256"], manifest["selector_sha256"])
            (directory / "unexpected.bin").write_bytes(b"unlisted")
            with self.assertRaises(worker.InvalidInput):
                worker.load_manifest(directory)
            (directory / "unexpected.bin").unlink()
            (directory / "encoder.onnx").write_bytes(b"changed")
            with self.assertRaises(worker.InvalidInput):
                worker.load_manifest(directory)

    def test_threshold_provenance_and_digest_are_enforced(self):
        for field, value in (("fit_split", "test"), ("validation_sha256", "sha256:" + "2" * 64),
                             ("lower_logit", "NaN"), ("upper_logit", "-2")):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest = self.make_bundle(directory)
                manifest["selection"][field] = value
                self.save(directory, manifest)
                with self.assertRaises(worker.InvalidInput):
                    worker.load_manifest(directory)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.make_bundle(directory)
            manifest["selection"]["upper_logit"] = "2"
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(worker.InvalidInput):
                worker.load_manifest(directory)

    def test_canonical_hash_uses_literal_unicode_and_safe_integers(self):
        payload = '{"a":"é","z":1}'.encode()
        self.assertEqual(worker.canonical_hash({"z": 1, "a": "é"}),
                         "sha256:" + hashlib.sha256(payload).hexdigest())
        for invalid in (1.0, 9007199254740992, "\ud800"):
            with self.assertRaises(worker.InvalidInput):
                worker.canonical_hash(invalid)


if __name__ == "__main__":
    unittest.main()
