"""Pure metric behavior with nonprivate synthetic records, not audit/outcome evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_router", ROOT / "python/evaluate-router.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def fixture():
    record = {"example_id": "synthetic-one", "split": "challenge",
              "label": {"decision": "none", "selected_ids": []}}
    gold = {
        "decision": "selected", "selected_ids": ["a", "b"], "verified_negative_ids": ["negative"],
        "unknown_ids": ["alternative", "policy", "unknown"], "acceptable_alternative_sets": [["alternative"]],
        "activate_now": ["a", "b"]}
    request = {"requiredIds": ["policy"], "candidates": [{"id": x} for x in ("a", "negative", "alternative", "policy", "unknown")]}
    case = {"record": record, "requests": {name: copy.deepcopy(request) for name in evaluation.BASELINES},
            "strata": ["challenge"], "timing": None, "contextual_gold": gold}
    observation = {"example_id": "synthetic-one", "baseline": "finetuned", "request": request,
        "response": {"status": "selected", "reasonCode": "matched", "selectedIds": ["a", "policy", "unknown"]},
        "loaded_ids": None, "trace": None}
    return case, observation


def trace(run_id, completed=None, tokens=None):
    return {"run_id": run_id, "fixture_sha256": "sha256:" + "a" * 64,
            "environment_sha256": "sha256:" + "b" * 64, "seed": 42,
            "evidence_sha256": "sha256:" + "c" * 64,
            "measurements": {name: completed if name == "completed_task" else tokens if name == "input_tokens" else None
                             for name in evaluation.MEASURES}, "activations": None}


class EvaluationTests(unittest.TestCase):
    def test_unknown_and_policy_do_not_inflate_pair_precision(self):
        case, observation = fixture()
        report = evaluation.summarize([(case, observation)], 1)
        self.assertEqual(report["precision"], 1)
        self.assertEqual(report["recall"], .5)
        self.assertEqual(report["known_pair_coverage"], .5)
        self.assertEqual(report["counts"]["selected_unknown"], 2)
        self.assertEqual(report["shortlist_recall"], .5)
        self.assertEqual(report["exact_set_accuracy"], 0)

    def test_alternative_whole_set_not_union(self):
        case, observation = fixture()
        observation["response"]["selectedIds"] = ["alternative", "policy"]
        self.assertEqual(evaluation.summarize([(case, observation)], 1)["exact_set_accuracy"], 1)
        observation["response"]["selectedIds"] = ["alternative", "a", "b", "policy"]
        self.assertEqual(evaluation.summarize([(case, observation)], 1)["exact_set_accuracy"], 0)

    def test_false_activation_uses_verified_negatives_only(self):
        case, observation = fixture()
        observation["response"]["selectedIds"] = ["a", "negative", "unknown", "policy"]
        report = evaluation.summarize([(case, observation)], 1)
        self.assertEqual(report["precision"], .5)
        self.assertEqual(report["verified_negative_activation_rate"], 1)

    def test_policy_only_requires_full_negative_proof_for_conditional_none(self):
        case, observation = fixture()
        request = observation["request"]
        request.update(candidateCoverage="full", excludedIds=[], intent="Review", action=None)
        observation["response"].update(reasonCode="policy_only", selectedIds=["policy"],
            scores={"values": [{"id": c["id"], "score": -2} for c in request["candidates"]]})
        case["contextual_gold"].update(decision="none", selected_ids=[], acceptable_alternative_sets=[], activate_now=[])
        manifests = {"finetuned": {"selection": {"lower_logit": "-1"}}}
        report = evaluation.summarize([(case, observation)], 1, manifests)
        self.assertEqual(report["confusion"], {"none:none": 1})
        self.assertEqual(report["exact_set_accuracy"], 1)
        for missing_proof in ("shortlist", "gray", "missing_scores", "missing_manifest"):
            changed = copy.deepcopy(observation)
            selected_manifests = manifests
            if missing_proof == "shortlist":
                changed["request"]["candidateCoverage"] = "shortlist"
            elif missing_proof == "gray":
                changed["response"]["scores"]["values"][0]["score"] = 0
            elif missing_proof == "missing_scores":
                changed["response"]["scores"] = None
            else:
                selected_manifests = {}
            report = evaluation.summarize([(case, changed)], 1, selected_manifests)
            self.assertEqual(report["confusion"], {"none:uncertain": 1})
            self.assertEqual(report["exact_set_accuracy"], 0)

    def test_unknown_extras_do_not_turn_verified_exact_selection_into_error(self):
        case, observation = fixture()
        observation["response"]["selectedIds"] = ["a", "b", "policy", "unknown"]
        report = evaluation.summarize([(case, observation)], 1)
        self.assertEqual(report["exact_set_accuracy"], 1)
        self.assertEqual(report["unknown_pair_coverage"], .5)
        self.assertEqual(report["selected_unknown_rate"], .5)

    def test_uncertain_is_not_a_none_success(self):
        case, observation = fixture()
        case["contextual_gold"].update(decision="none", selected_ids=[], acceptable_alternative_sets=[], activate_now=[])
        observation["response"].update(status="uncertain", reasonCode="insufficient_evidence", selectedIds=[])
        report = evaluation.summarize([(case, observation)], 1)
        self.assertEqual(report["confusion"], {"none:uncertain": 1})
        self.assertEqual(report["exact_set_accuracy"], 0)
        self.assertEqual(report["abstention_rate"], 1)

    def test_absent_task_measurements_stay_missing(self):
        case, observation = fixture()
        report = evaluation.summarize([(case, observation)], 2)
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertEqual(report["measurements"]["completed_task"], {"measured": 0, "missing": 2, "mean": None, "p95": None})
        self.assertEqual(report["activation_timing"], {"unobserved": 1})

    def test_timing_requires_evidence_and_actual_activation(self):
        case, observation = fixture()
        case["timing"] = {"support_sequence": 4, "deadline_sequence": 8}
        observation["trace"] = trace("timing")
        self.assertEqual(evaluation.timing_counts(case, observation), {"unobserved": 1})
        observation["trace"]["activations"] = [{"runtime_id": "a", "event_sequence": 8}, {"runtime_id": "b", "event_sequence": 9}]
        self.assertEqual(evaluation.timing_counts(case, observation), {"on_time": 1, "late": 1})
        observation["trace"]["activations"] = [{"runtime_id": "a", "event_sequence": 3}]
        self.assertEqual(evaluation.timing_counts(case, observation), {"early": 1, "missing_activation": 1})

    def test_paired_difference_does_not_impute_missing(self):
        case, observation = fixture()
        observation["trace"] = trace("tuned", completed=True, tokens=30)
        control = copy.deepcopy(observation)
        control.update(baseline="no-router", response=None, loaded_ids=[])
        control["trace"] = trace("control", completed=False, tokens=50)
        plan = {"cases": [case]}
        report = evaluation.score(plan, [observation, control])
        paired = report["paired"]["challenge/all"]["finetuned"]["no-router"]
        self.assertEqual(paired["completed_task"], {"pairs": 1, "missing_pairs": 0, "mean_difference": 1})
        self.assertEqual(paired["input_tokens"]["mean_difference"], -20)
        self.assertIsNone(paired["compliance"]["mean_difference"])
        control["trace"]["seed"] += 1
        with self.assertRaises(ValueError):
            evaluation.score(plan, [observation, control])

    def test_duplicates_and_foreign_examples_are_rejected(self):
        case, observation = fixture()
        with self.assertRaises(ValueError):
            evaluation.score({"cases": [case]}, [observation, observation])
        observation["example_id"] = "foreign"
        with self.assertRaises(ValueError):
            evaluation.score({"cases": [case]}, [observation])

    def test_missing_no_router_loads_do_not_claim_zero_false_activations(self):
        case, observation = fixture()
        observation.update(baseline="no-router", response=None)
        report = evaluation.summarize([(case, observation)], 1)
        self.assertIsNone(report["verified_negative_activation_rate"])
        self.assertIsNone(report["recall"])

    def test_saved_response_identity_scores_and_thresholds_are_bound(self):
        worker = evaluation.module(ROOT / "python/router-worker.py")
        validators = evaluation.validators()
        digest = "sha256:" + "a" * 64
        request = {
            "version": 1, "requestId": "request", "sessionKey": "session", "generation": 0,
            "catalogDigest": digest, "contextDigest": digest, "policyDigest": digest,
            "selectorDigest": digest, "event": "context", "intent": "Review this change",
            "action": None, "candidateCoverage": "full", "requiredIds": [],
            "excludedIds": [], "maxSelections": 1,
            "candidates": [{"id": "review", "name": "review", "description": "Review code", "contentDigest": digest}],
        }
        manifest = {"selector_sha256": digest, "scorer_sha256": digest,
                    "selection": {"lower_logit": "-1", "upper_logit": "1", "max_optional": 1}}
        reply = worker.decide(request, request["candidates"], [2.0], -1, 1, digest, 1)
        case = {"record": {"example_id": "synthetic"}, "catalog": {"entries": [
            {"runtime_id": "review", "content_sha256": digest}]}, "requests": {"finetuned": request}}
        observation = {"version": 1, "example_id": "synthetic", "baseline": "finetuned",
                       "request": request, "response": reply, "loaded_ids": None, "trace": None}
        evaluation.validate_observation(observation, case, {"finetuned": manifest}, validators, worker)
        stale = copy.deepcopy(observation)
        stale["response"]["generation"] += 1
        with self.assertRaises(ValueError):
            evaluation.validate_observation(stale, case, {"finetuned": manifest}, validators, worker)
        changed = copy.deepcopy(observation)
        changed["response"]["scores"]["values"][0]["score"] = -2
        with self.assertRaises(ValueError):
            evaluation.validate_observation(changed, case, {"finetuned": manifest}, validators, worker)
        foreign = copy.deepcopy(observation)
        foreign["response"]["scores"]["scorerDigest"] = "sha256:" + "b" * 64
        with self.assertRaises(ValueError):
            evaluation.validate_observation(foreign, case, {"finetuned": manifest}, validators, worker)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{"a":"\\ud800"}'):
            with self.assertRaises((ValueError, UnicodeError)):
                evaluation.decode(payload)


class BundleAdmissionTests(unittest.TestCase):
    def test_below_quality_control_is_admitted_but_never_as_tuned(self):
        trainer = evaluation.module(ROOT / "python/train-router.py")
        worker = evaluation.module(ROOT / "python/router-worker.py")
        targets = dict(target_precision=1, target_negative_precision=1,
                       min_calibration_support=1, min_coverage=1)
        scores, labels = [-3, -2, 2, 3], [1, 0, 1, 0]
        thresholds, method = trainer.fit_baseline_control(scores, labels, SimpleNamespace(**targets))
        metrics = trainer.metrics(scores, labels, thresholds)
        identity = trainer.digest("authored-validation-fixture")
        provenance = {"dataset_sha256": identity, "split_sha256": trainer.digest("frozen-split")}
        from test.test_router_worker import source_binding, REVISION
        provenance["contextual_binding"] = source_binding()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            design = Path(temporary) / "evaluation-design.json"
            trainer.write_json(design, {"version": 1, "status": "authorized_contextual_evaluation",
                "target_sha256": provenance["contextual_binding"]["target_sha256"], "contextual_binding": provenance["contextual_binding"],
                "plan_sha256": evaluation.hash_value(provenance)})
            authority = {"design": design, "design_sha256": evaluation.hash_file(design)}
            # Payload-integrity fixture only, not a claim of model inference.
            (root / "control.onnx").write_bytes(b"authored-integrity-fixture")
            (root / "training.lock").write_text("authored-lock-fixture")
            (root / "tokenizer.json").write_text("{}")
            (root / "config.json").write_text("{}")
            bundle_provenance = {**provenance,
                "contextual_binding": worker.compact_source_set_binding(provenance["contextual_binding"])}
            trainer.write_json(root / "recipe.json", {"role": "evaluation_only", "weights": "unchanged_pretrained",
                "runtime_revision": REVISION, "baseline_control_method": method, "targets": targets, "provenance": bundle_provenance})
            validation = {"role": "evaluation_only", "fit_split": "validation", **bundle_provenance,
                "threshold_source": "pretrained_torch_validation", "method": method,
                "thresholds": [trainer.decimal_string(x) for x in thresholds], "targets": targets,
                "targets_met": False, "fit_metrics": metrics, "artifact_validation_metrics": metrics}
            trainer.write_json(root / "validation.json", validation)
            trainer.write_json(root / "evaluation.json", {"role": "evaluation_only", "acceptance_passed": True,
                "quality_targets_met": False, "metrics": {"validation": metrics}})
            trainer.write_json(root / "parity.json", {"comparison": {"passed": True}})
            trainer.make_control_manifest(root, thresholds, provenance, REVISION, "cross-encoder/ms-marco-TinyBERT-L2-v2")
            admitted = evaluation.admit_bundle(root, "pretrained", provenance, worker, **authority)
            self.assertEqual(float(admitted["selection"]["lower_logit"]), -2)
            with self.assertRaises(ValueError):
                evaluation.admit_bundle(root, "finetuned", provenance, worker, **authority)
            # The same fixture cannot lie about quality, even after valid rehashing.
            (root / "manifest.json").unlink()
            validation["targets_met"] = True
            trainer.write_json(root / "validation.json", validation)
            trainer.write_json(root / "evaluation.json", {"role": "evaluation_only", "acceptance_passed": True,
                "quality_targets_met": True, "metrics": {"validation": metrics}})
            trainer.make_control_manifest(root, thresholds, provenance, REVISION, "cross-encoder/ms-marco-TinyBERT-L2-v2")
            with self.assertRaises(ValueError):
                evaluation.admit_bundle(root, "pretrained", provenance, worker, **authority)


if __name__ == "__main__":
    unittest.main()
