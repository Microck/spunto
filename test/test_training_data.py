"""Pure data/calibration contract tests; no model download or dependency mocks."""
import importlib.util
import hashlib
import json
import io
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import shutil
import unittest
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "train_router", Path(__file__).resolve().parents[1] / "python" / "train-router.py"
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "training_source_fixtures", Path(__file__).resolve().parents[1] / "test/source-set-fixtures.py")
source_fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(source_fixtures)


def rows():
    authored = [
        {"action": None, "intent": "Inspect a browser form.", "description": "Review form accessibility.",
         "label": 1, "group_id": "train-form", "split": "train"},
        {"action": None, "intent": "Inspect a browser form.", "description": "Upload an iOS release.",
         "label": 0, "group_id": "train-form", "split": "train"},
        {"action": None, "intent": "Review keyboard navigation.", "description": "Inspect keyboard access.",
         "label": 1, "group_id": "validation-keyboard", "split": "validation"},
        {"action": None, "intent": "Review keyboard navigation.", "description": "Deploy a server.",
         "label": 0, "group_id": "validation-keyboard", "split": "validation"},
        {"action": None, "intent": "Explain a font choice.", "description": "Explain typography decisions.",
         "label": 1, "group_id": "test-font", "split": "temporal_test"},
        {"action": None, "intent": "Explain a font choice.", "description": "Manage database backups.",
         "label": 0, "group_id": "test-font", "split": "temporal_test"},
    ]
    # Inner parser fixtures only. Canonical admission is exercised below with
    # real pinned evidence; this mapping never authorizes an export.
    for row in authored:
        row["contextual"] = {"binding_sha256": trainer.digest("parser fixture"), "contributors": [{
            "pair_id": trainer.digest([row["group_id"], row["description"]]),
            "annotation_sha256": trainer.digest("parser annotation"),
            "identities": {"example_id": row["group_id"]}, "runtime_id": "parser-skill",
            **{key: trainer.digest(key) for key in ("body_sha256", "input_sha256", "request_sha256",
                                                   "action_sha256", "source_record_sha256")}}]}
    return authored


class TrainingDataTests(unittest.TestCase):
    def load(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            approved, encoded = {}, []
            for row in records:
                # Invalid expanded fixture rows exercise the public row validator;
                # the disk parser only receives the canonical factored encoding.
                trainer.validate_pair(trainer.parse_json(json.dumps(row)))
                approved[trainer.digest(row)] = row
                encoded.append(trainer.private_inputs.compact_training_pair(row))
            path.write_text("".join(json.dumps(row) + "\n" for row in encoded), encoding="utf-8")
            return trainer.load_rows(path, approved_pairs=approved)

    def assert_rejected(self, records, code):
        with self.assertRaisesRegex(trainer.ContractError, "^" + code + "$"):
            self.load(records)

    def test_factored_disk_label_cannot_coerce_an_admitted_integer(self):
        row = rows()[0]
        approved = {trainer.digest(row): row}
        for label in (True, 1.0):
            with self.subTest(label_type=type(label).__name__):
                encoded = trainer.private_inputs.compact_training_pair(row)
                encoded["label"] = label
                with self.assertRaisesRegex(trainer.ContractError, "^contextual_input_source_mapping$"):
                    trainer.load_rows((json.dumps(encoded) + "\n").encode(), approved_pairs=approved)

    def test_streamed_partition_digest_matches_original_canonical_bytes(self):
        special = dict(rows()[0], intent='Résumé "quoted"\\path\n\t\x00',
                       description='日本語 and café', label=0)
        reversed_keys = dict(reversed(list(special.items())))
        for records in ([], [special], [reversed_keys, *rows()], list(reversed(rows()))):
            with self.subTest(records=len(records)):
                original = json.dumps(records, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")
                self.assertEqual(trainer.canonical_bytes(records), original)
                self.assertEqual(trainer._partition_digest(records),
                                 "sha256:" + hashlib.sha256(original).hexdigest())
        self.assertNotEqual(trainer._partition_digest(rows()), trainer._partition_digest(list(reversed(rows()))))

    @unittest.skipUnless(sys.platform == "linux", "requires Linux address-space limits")
    def test_private_bytes_load_at_scale_matches_file_without_whole_text_copy(self):
        # One non-BMP character widens a whole-file Unicode copy to four bytes
        # per character. Keep the actual bytes API within a disposable 320 MiB
        # process while verifying the same rows and canonical partition hashes.
        script = r'''
import gc
import json
from pathlib import Path
import resource
import tempfile
from test.test_training_data import rows, trainer

resource.setrlimit(resource.RLIMIT_AS, (320 * 1024**2, 320 * 1024**2))
approved = {}
with tempfile.TemporaryDirectory(prefix="training-bytes-scale-") as temporary:
    path = Path(temporary) / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        supplied = rows()
        for index in range(3072):
            supplied.append(dict(supplied[0], intent="Scale request " + str(index),
                description=("\U00010000" if index == 0 else "") + "x" * 16384))
        for row in supplied:
            approved[trainer.digest(row)] = row
            stream.write(json.dumps(trainer.private_inputs.compact_training_pair(row), ensure_ascii=False) + "\n")
    raw = path.read_bytes()
    partitions = trainer.load_rows(raw, approved_pairs=approved)
    counts = {split: len(records) for split, records in partitions.items()}
    byte_hashes = {split: trainer._partition_digest(records) for split, records in partitions.items()}
    del partitions
    gc.collect()
    from_file = trainer.load_rows(path, approved_pairs=approved)
    assert byte_hashes == {split: trainer._partition_digest(records) for split, records in from_file.items()}
    assert from_file["train"][2]["description"].startswith("\U00010000")
    print(json.dumps(counts, sort_keys=True))
'''
        completed = subprocess.run([sys.executable, "-B", "-c", script],
                                   cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), {"train": 3074, "validation": 2, "temporal_test": 2})

    def test_keeps_only_supplied_pairs_and_separate_holdouts(self):
        supplied = rows()
        supplied[-1]["split"] = "unseen_family_test"
        supplied[-1]["group_id"] = "unseen-family"
        partitions = self.load(supplied)
        self.assertEqual(partitions["train"], supplied[:2])
        self.assertEqual(partitions["validation"], supplied[2:4])
        self.assertEqual(partitions["temporal_test"], [supplied[4]])
        self.assertEqual(partitions["unseen_family_test"], [supplied[5]])
        self.assertNotIn("test", partitions)

    def test_cross_split_group_rejected_without_printing_identifier(self):
        supplied = rows()
        supplied[2]["group_id"] = supplied[0]["group_id"]
        self.assert_rejected(supplied, "cross_split_group")

    def test_duplicate_pair_across_splits_ignores_group_and_label(self):
        supplied = rows()
        supplied[4].update(intent=supplied[0]["intent"],
                           description=supplied[0]["description"], label=0)
        self.assert_rejected(supplied, "cross_split_duplicate")

    def test_normalized_duplicates_cannot_cross_splits(self):
        supplied = rows()
        supplied[0]["intent"] = "Review Café fonts."
        supplied[4].update(intent=" Review  Cafe\u0301\tfonts. ",
                           description=supplied[0]["description"])
        self.assert_rejected(supplied, "cross_split_duplicate")

    def test_duplicate_and_contradictory_labels_rejected(self):
        supplied = rows()
        supplied.append(dict(supplied[0]))
        self.assert_rejected(supplied, "duplicate_pair")
        supplied[-1]["label"] = 0
        self.assert_rejected(supplied, "contradictory_label")

    def test_binary_labels_exclude_bool_string_null_and_nonfinite(self):
        for invalid in (True, False, "1", None, -1, 2, 0.5, 10**500):
            with self.subTest(label_type=type(invalid).__name__):
                supplied = rows()
                supplied[0]["label"] = invalid
                self.assert_rejected(supplied, "invalid_binary_label")
        for invalid in (float("nan"), float("inf"), float("-inf")):
            supplied = rows()
            supplied[0]["label"] = invalid
            self.assert_rejected(supplied, "nonfinite_json")

    def test_missing_fit_and_test_splits_rejected(self):
        for split in ("train", "validation"):
            self.assert_rejected([row for row in rows() if row["split"] != split], "missing_fit_split")
        self.assert_rejected(rows()[:4], "missing_test_split")
        self.assert_rejected([], "missing_fit_split")

    def test_calibration_and_training_need_both_classes(self):
        for split in ("train", "validation"):
            supplied = rows()
            for row in supplied:
                if row["split"] == split:
                    row["label"] = 1
            self.assert_rejected(supplied, "fit_split_needs_both_labels")

    def test_unknown_fields_and_unknown_labels_do_not_become_features(self):
        supplied = rows()
        supplied[0]["skill_id"] = "opaque-key"
        self.assert_rejected(supplied, "invalid_row_fields")
        supplied = rows()
        supplied[0]["label"] = "unknown"
        self.assert_rejected(supplied, "invalid_binary_label")
        supplied = rows()
        supplied[0]["split"] = "quarantine"
        self.assert_rejected(supplied, "invalid_split")

    def test_empty_text_and_invalid_unicode_rejected(self):
        for field in ("group_id",):
            supplied = rows()
            supplied[0][field] = " \t"
            self.assert_rejected(supplied, "invalid_row_text")
        supplied = rows()
        supplied[0]["intent"] = "\ud800"
        self.assert_rejected(supplied, "invalid_unicode")

    def test_json_duplicates_and_nonfinite_numbers_rejected(self):
        for text, code in (("{\"label\":0,\"label\":1}", "duplicate_json_key"),
                           ("{\"label\":NaN}", "nonfinite_json"),
                           ("{", "invalid_json")):
            with self.assertRaisesRegex(trainer.ContractError, "^" + code + "$"):
                trainer.parse_json(text)
        supplied = rows()
        supplied[0]["label"] = 1.0
        self.assert_rejected(supplied, "invalid_binary_label")



class PairEncodingTests(unittest.TestCase):
    def test_exact_inputs_aliases_and_contributor_agreement(self):
        d = source_fixtures.d
        import copy
        supplied = copy.deepcopy(rows()[:2])
        for row in supplied:
            row.update(intent="Café\nAction: literal text 日本語 \U00010000",
                       action={"toolName": "read", "summary": "Inspect\nquoted \"text\"."})
        first = supplied[0]["contextual"]["contributors"][0]
        supplied[0]["contextual"]["contributors"].append(dict(first, runtime_id="alias", pair_id=d.digest("alias")))
        supplied[1]["contextual"]["contributors"][0]["runtime_id"] = "other"
        catalog = {"entries": [{"runtime_id": sid, "description": supplied[index]["description"],
                               "content_sha256": first["body_sha256"]}
                              for sid, index in (("parser-skill", 0), ("alias", 0), ("other", 1))]}
        contexts = {}
        encoded = [trainer.private_inputs.compact_training_pair(row, contexts) for row in supplied]
        table = {"format": "contextual-pair-contexts-v2", "binding_sha256": supplied[0]["contextual"]["binding_sha256"], "cases": contexts}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "audited-pairs.jsonl"
            path.write_bytes(b"".join(d.canonical_bytes(row) + b"\n" for row in encoded))
            trainer.write_json(root / "pair-contexts.json", table)
            self.assertEqual([d.canonical_bytes(row) for row in d.read_training_pairs(path, catalog=catalog)],
                             [d.canonical_bytes(row) for row in supplied])
            for field in ("intent", "action", "group_id", "split", "description"):
                with self.subTest(field=field):
                    changed = copy.deepcopy(table)
                    changed_catalog = copy.deepcopy(catalog)
                    if field == "description":
                        changed_catalog["entries"][1][field] = "Other alias text"
                    else:
                        # A second contributor must not smuggle different case
                        # inputs past an unchanged expanded-row commitment.
                        other = copy.deepcopy(contexts[first["identities"]["example_id"]])
                        other["identities"]["example_id"] = "other-case"
                        other[field] = None if field == "action" else "changed"
                        changed["cases"]["other-case"] = other
                        encoded[0]["contextual"]["contributors"][1]["case_id"] = "other-case"
                    trainer.write_json(root / "pair-contexts.json", changed)
                    path.write_bytes(b"".join(d.canonical_bytes(row) + b"\n" for row in encoded))
                    with self.assertRaisesRegex(d.ContractError, "^contextual_input_source_mapping$"):
                        list(d.read_training_pairs(path, catalog=changed_catalog))
                    encoded[0]["contextual"]["contributors"][1]["case_id"] = first["identities"]["example_id"]
        conflicting = dict(supplied[0], intent="Changed input for the same case")
        with self.assertRaisesRegex(ValueError, "^private_pair_context_conflict$"):
            trainer.private_inputs.compact_training_pair(conflicting, contexts)


class ExportDataTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.umask, os.umask(0o077))

    def make_export(self, root):
        f = source_fixtures
        temporary = tempfile.TemporaryDirectory(prefix="training-export-fixture-")
        self.addCleanup(temporary.cleanup)
        fixture = f.Fixture(Path(temporary.name))
        source = fixture.export()
        self.PIN = fixture.pin
        self.frozen = fixture.root / "frozen"
        dataset = f.d
        shutil.copytree(source, root, dirs_exist_ok=True)
        return dataset, dataset.read_json(root / "manifest.json"), dataset.read_json(root / "privacy-scan.json")

    def rebind(self, root, dataset, manifest, scan):
        trainer.write_json(root / "privacy-scan.json", scan)
        rebound = dataset.artifact_manifest(root, manifest["kind"], manifest["catalog_sha256"],
            sorted(dataset.CONTEXTUAL_EXPORT_FILES), manifest["dataset_sha256"],
            manifest["split_sha256"], manifest["recipe_sha256"], manifest["privacy_class"],
            manifest["parents"], contextual_binding=manifest["contextual_binding"])
        trainer.write_json(root / "manifest.json", rebound)

    def test_canonical_split_and_train_only_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, manifest, scan = self.make_export(root)
            verified, _, _, pairs = dataset.verify_contextual_export(root, self.PIN)
            frozen = self.frozen
            self.assertEqual(verified["split_sha256"], dataset.file_digest(frozen / "splits.jsonl"))
            self.assertNotEqual(verified["split_sha256"], trainer._partition_digest(list(pairs.values())))
            with self.assertRaisesRegex(trainer.ContractError, "missing_fit_split"):
                trainer.load_export(root / "audited-pairs.jsonl", root / "manifest.json", approval_sha256=self.PIN)

    def test_corrupt_payload_and_rebound_privacy_claim_are_rejected(self):
        for field in ("bytes", "hash", "pairs_sha256", "catalog_sha256", "frozen_artifact_sha256",
                      "split_sha256", "recipe_sha256", "pair_projection", "export_approved"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset, manifest, scan = self.make_export(root)
                if field == "bytes":
                    with (root / "audited-pairs.jsonl").open("a") as stream:
                        stream.write("\n")
                elif field == "hash":
                    path = root / "audited-pairs.jsonl"
                    path.write_bytes(path.read_bytes().replace(b'"label":0', b'"label":1', 1))
                else:
                    scan[field] = True if field == "export_approved" else dataset.digest("wrong")
                    self.rebind(root, dataset, manifest, scan)
                with self.assertRaises(trainer.ContractError):
                    trainer.load_export(root / "audited-pairs.jsonl", root / "manifest.json", approval_sha256=self.PIN)

    def test_rehashed_group_action_and_pair_identity_cannot_change_source_mapping(self):
        for field in ("group_id", "action", "pair_id", "label"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset, manifest, scan = self.make_export(root)
                rows = list(dataset.rows(root / "audited-pairs.jsonl"))
                if field == "pair_id":
                    rows[0]["contextual"]["contributors"][0]["pair_id"] = dataset.digest("forged")
                elif field == "label":
                    rows[0][field] = 1 - rows[0][field]
                else:
                    path = root / "pair-contexts.json"
                    table = dataset.read_json(path)
                    case_id = rows[0]["contextual"]["contributors"][0]["case_id"]
                    table["cases"][case_id][field] = "different-group" if field == "group_id" else {"toolName": "inspect", "summary": "changed"}
                    trainer.write_json(path, table)
                (root / "audited-pairs.jsonl").write_bytes(b"".join(dataset.canonical_bytes(row) + b"\n" for row in rows))
                manifest["dataset_sha256"] = scan["pairs_sha256"] = dataset.file_digest(root / "audited-pairs.jsonl")
                self.rebind(root, dataset, manifest, scan)
                with self.assertRaisesRegex(trainer.ContractError, "contextual_input_source_mapping"):
                    trainer.load_export(root / "audited-pairs.jsonl", root / "manifest.json", approval_sha256=self.PIN)

    def test_context_identity_tampering_and_unused_contexts_are_rejected(self):
        for mutation in ("identity", "unused", "missing", "intent", "action", "group_id", "split", "old_format"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset, manifest, scan = self.make_export(root)
                path = root / "pair-contexts.json"
                table = dataset.read_json(path)
                case_id = next(iter(table["cases"]))
                if mutation == "identity":
                    table["cases"][case_id]["identities"]["event_id"] = "forged-event"
                elif mutation == "unused":
                    table["cases"]["unused-case"] = table["cases"][case_id]
                elif mutation == "missing":
                    del table["cases"][case_id]
                elif mutation == "old_format":
                    table["format"] = "contextual-pair-contexts-v1"
                else:
                    table["cases"][case_id][mutation] = {"toolName": "inspect", "summary": "changed"} if mutation == "action" else "changed"
                trainer.write_json(path, table)
                self.rebind(root, dataset, manifest, scan)
                with self.assertRaisesRegex(trainer.ContractError, "contextual_input_source_mapping"):
                    trainer.load_export(root / "audited-pairs.jsonl", root / "manifest.json", approval_sha256=self.PIN)

    def test_manifest_kind_privacy_and_declared_split_are_enforced(self):
        for field, value in (("kind", "split"), ("privacy_class", "approved_remote_private"),
                             ("publication_allowed", True), ("split_sha256", "invalid-digest")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset, manifest, _ = self.make_export(root)
                manifest[field] = value
                manifest["artifact_sha256"] = dataset.digest({
                    key: item for key, item in manifest.items() if key not in ("created_at", "artifact_sha256")})
                trainer.write_json(root / "manifest.json", manifest)
                with self.assertRaises(trainer.ContractError):
                    trainer.load_export(root / "audited-pairs.jsonl", root / "manifest.json", approval_sha256=self.PIN)

class ThresholdTests(unittest.TestCase):
    def test_fit_excludes_tied_ambiguous_scores(self):
        scores = [-3.0, -2.0, 0.0, 0.0, 2.0, 3.0]
        labels = [0, 0, 0, 1, 1, 1]
        policy = trainer.fit_thresholds(scores, labels, 1.0, 1.0, 2, 0.6)
        self.assertEqual([trainer.decision(score, policy) for score in scores], [0, 0, None, None, 1, 1])
        report = trainer.metrics(scores, labels, policy)
        self.assertEqual(report["positive_precision"], 1.0)
        self.assertEqual(report["negative_precision"], 1.0)
        self.assertAlmostEqual(report["coverage"], 2 / 3)
        self.assertAlmostEqual(report["recall"], 2 / 3)

    def test_fitted_decisions_survive_small_export_drift(self):
        for scores, labels, expected, coverage in (
            ([-3., -2., 0., 0., 2., 3.], [0, 0, 0, 1, 1, 1],
             [0, 0, None, None, 1, 1], 2 / 3),
            ([-3., -2., 2., 3.], [0, 0, 1, 1], [0, 0, 1, 1], 1.),
        ):
            with self.subTest(scores=scores):
                policy = trainer.fit_thresholds(scores, labels, 1., 1., 2, coverage)
                self.assertLess(*policy)
                self.assertEqual([trainer.decision(s, policy) for s in scores], expected)
                for cutoff in policy:
                    self.assertLess(max(s for s in scores if s < cutoff), cutoff)
                    self.assertLess(cutoff, min(s for s in scores if s > cutoff))
                    self.assertNotIn(cutoff, scores)
                # Both drift directions must preserve accepted and abstained pairs.
                for delta in (-1e-6, 1e-6):
                    exported = [s + delta for s in scores]
                    self.assertEqual([trainer.decision(s, policy) for s in exported], expected)
                    report = trainer.metrics(exported, labels, policy)
                    self.assertEqual(report["positive_precision"], 1.)
                    self.assertEqual(report["negative_precision"], 1.)
                    self.assertEqual(report["coverage"], coverage)

    def test_adjacent_boundaries_keep_distinct_inclusive_decisions(self):
        # Even adjacent representable scores must keep distinct decisions.
        import math
        adjacent = [1., math.nextafter(1., math.inf)]
        policy = trainer.fit_thresholds(adjacent, [0, 1], 1., 1., 1, 1.)
        self.assertLess(*policy)
        self.assertEqual([trainer.decision(s, policy) for s in adjacent], [0, 1])

    def test_impossible_targets_have_no_fallback_thresholds(self):
        self.assertIsNone(trainer.fit_thresholds([0.0, 0.0], [0, 1], 0.9, 0.9, 1, 0.5))
        self.assertIsNone(trainer.fit_thresholds([-1.0, 1.0], [0, 1], 1.0, 1.0, 2, 0.5))
        self.assertIsNone(trainer.fit_thresholds([-2.0, 0.0, 0.0, 2.0], [0, 0, 1, 1], 1.0, 1.0, 1, 0.75))

    def test_poor_baseline_control_reports_unmet_quality_with_finite_thresholds(self):
        args = SimpleNamespace(target_precision=1, target_negative_precision=1,
                               min_calibration_support=1, min_coverage=1)
        scores, labels = [-3, -2, 2, 3], [1, 0, 1, 0]
        policy, method = trainer.fit_baseline_control(scores, labels, args)
        self.assertEqual(policy, (-2, 2))
        self.assertEqual(method, "validation-balanced-accuracy-full-coverage-v1")
        report = trainer.metrics(scores, labels, policy)
        self.assertFalse(trainer.meets_targets(report, args))
        self.assertEqual(report["coverage"], 1)
        self.assertEqual((report["positive_precision"], report["negative_precision"]), (.5, .5))
        self.assertIsNone(trainer.fit_thresholds(scores, labels, 1, 1, 1, 1))
        for constant in (-1.0, 0.0, 1.0):
            policy, _ = trainer.fit_baseline_control([constant, constant], [0, 1], args)
            self.assertLess(*policy)
            self.assertEqual([trainer.decision(constant, policy)] * 2, [0, 0])
        with self.assertRaisesRegex(trainer.ContractError, "calibration_needs_both_labels"):
            trainer.fit_baseline_control([0, 1], [1, 1], args)

    def test_boundaries_are_inclusive_and_not_probabilities(self):
        thresholds = (-2.0, 3.0)
        self.assertEqual(trainer.decision(-2.0, thresholds), 0)
        self.assertEqual(trainer.decision(3.0, thresholds), 1)
        self.assertIsNone(trainer.decision(0.9, thresholds))
        report = trainer.metrics([-1000.0, 1000.0], [0, 1], thresholds)
        self.assertEqual(report["bce"], 0.0)

    def test_rejects_nonfinite_and_missing_model_scores(self):
        for scores in ([float("nan"), 1.0], [float("inf"), 1.0], [1.0]):
            with self.assertRaisesRegex(trainer.ContractError, "invalid_model_scores"):
                trainer.fit_thresholds(scores, [0, 1], 0.9, 0.9, 1, 0.5)

    def test_drift_detects_boundary_and_ranking_changes_below_numeric_tolerance(self):
        pairs = [("one event", "description one"), ("one event", "description two")]
        report = trainer.drift([1.0, 1.01], [1.02, 0.99], (-1.0, 1.0), pairs, 0.1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["decision_changes"], 1)
        self.assertEqual(report["ranking_pair_changes"], 1)
        self.assertLess(report["max_absolute_logit_error"], report["tolerance"])

    def test_split_drift_separates_zero_margin_flips_from_real_flips(self):
        # 1.0 sits exactly on a fitted threshold; 2.0 is far from both.
        report = trainer.split_drift_report([1.0, 2.0, 0.0], [0.99995, 2.0, 0.0], (-1.0, 1.0), 1e-4)
        self.assertEqual((report["decision_changes"], report["boundary_decision_changes"]), (0, 1))
        report = trainer.split_drift_report([1.0, 2.0], [1.0, 0.5], (-1.0, 1.0), 1e-4)
        self.assertEqual((report["decision_changes"], report["boundary_decision_changes"]), (1, 0))

    def test_digest_serialization_matches_cross_language_contract(self):
        self.assertEqual(trainer.canonical_bytes({"z": [2, True, None], "a": {"text": "café\n"}}),
                         b'{"a":{"text":"caf\xc3\xa9\\n"},"z":[2,true,null]}')
        for invalid in (0.5, 2**53, {"threshold": float("inf")}, "\udfff"):
            with self.assertRaises(trainer.ContractError):
                trainer.canonical_bytes(invalid)
        self.assertEqual(trainer.decimal_string(1e-7), "0.0000001")


class ExportTests(unittest.TestCase):
    def test_required_candidates_and_optional_int8_have_distinct_outcomes(self):
        for baseline, tuned, quantized, expected in (
            (True, True, True, 0), (True, True, False, 0),
            (False, True, True, 3), (True, False, True, 3), (False, False, False, 3),
        ):
            exports = {name: {"passed": passed} for name, passed in zip(
                ("pretrained-fp32", "fp32", "int8-arm64"), (baseline, tuned, quantized))}
            result, code = trainer.export_outcome(exports, 4)
            self.assertEqual(code, expected)
            self.assertEqual(result["comparison_ready"], baseline and tuned)
            self.assertEqual(result["usable_candidates"],
                             sorted(name for name, report in exports.items() if report["passed"]))
            self.assertEqual(result["exports"], exports)

    def test_failed_calibration_retains_each_candidate_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "retry-export").mkdir()
            args = SimpleNamespace(output=root, export_name="retry-export")
            for pretrained, names in ((True, ("pretrained-fp32",)), (False, ("fp32", "int8-arm64"))):
                reports = trainer.export_candidates(
                    None, {}, {}, None, {}, {}, {}, args, None, None, pretrained=pretrained)
                self.assertEqual(set(reports), set(names))
                for name in names:
                    directory = root / "retry-export" / name
                    self.assertFalse((directory / "manifest.json").exists())
                    failure = trainer.parse_json((directory / "failure.json").read_text())
                    self.assertEqual(failure["code"], "validation_targets_unmet")
                    self.assertEqual(trainer.parse_json((root / "retry-export" / (name + ".json")).read_text()),
                                     failure)
            before = (root / "retry-export" / "pretrained-fp32" / "failure.json").read_bytes()
            with self.assertRaises(FileExistsError):
                trainer.export_candidates(None, {}, {}, None, {}, {}, {}, args, None, None, pretrained=True)
            self.assertEqual((root / "retry-export" / "pretrained-fp32" / "failure.json").read_bytes(), before)

    def test_baseline_and_tuned_thresholds_are_independently_fitted(self):
        labels = [0, 0, 1, 1]
        baseline = trainer.fit_thresholds([-9, -8, -3, -2], labels, 1, 1, 2, 1)
        tuned = trainer.fit_thresholds([-2, -1, 2, 3], labels, 1, 1, 2, 1)
        self.assertEqual([trainer.decision(s, baseline) for s in [-9, -8, -3, -2]], labels)
        self.assertEqual([trainer.decision(s, tuned) for s in [-2, -1, 2, 3]], labels)
        self.assertEqual(trainer.metrics([-9, -8, -3, -2], labels, tuned)["recall"], 0)

    def test_exported_baseline_stays_unchanged_after_model_update(self):
        # A real tiny local BERT exercises serialization and ONNX without Hub access.
        # Parent runs this in the resolved training environment; no dependency mocks.
        if any(importlib.util.find_spec(name) is None for name in
               ("torch", "numpy", "transformers", "tokenizers", "onnx", "onnxruntime", "jsonschema", "referencing")):
            self.skipTest("requires the parent's resolved training environment")
        import numpy as np
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

        torch.manual_seed(7)
        backend = Tokenizer(models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "review": 4, "deploy": 5}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                           cls_token="[CLS]", sep_token="[SEP]")
        config = BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=16, max_position_embeddings=512, num_labels=1,
                            hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
        config._attn_implementation = "eager"
        model = BertForSequenceClassification(config).eval()
        encoder = SimpleNamespace(model=model, tokenizer=tokenizer)
        encoded = {"validation": trainer.checked_tokens(tokenizer, [{"intent": "review", "action": None, "description": "review"}, {"intent": "deploy", "action": None, "description": "deploy"}])}
        scores = {"validation": trainer.score_pretrained(model, tokenizer, encoded["validation"], torch)}
        self.assertNotEqual(*scores["validation"])
        labels = {"validation": [int(v == max(scores["validation"])) for v in scores["validation"]]}
        thresholds = trainer.fit_thresholds(scores["validation"], labels["validation"], 1, 1, 1, 1)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "exports").mkdir()
            lock = root / "training.lock"
            lock.write_text("# local fixture\n")
            from test.test_router_worker import source_binding, REVISION
            args = SimpleNamespace(output=root, export_name="exports", dependency_lock=lock,
                runtime_revision=REVISION, schema_directory=Path(__file__).resolve().parents[1] / "schemas",
                fp32_atol=0.0001, int8_atol=0.25, target_precision=1, target_negative_precision=1,
                min_calibration_support=1, min_coverage=1)
            provenance = {"dataset_sha256": trainer.digest("dataset"), "split_sha256": trainer.digest("splits"),
                          "partitions": {"validation": trainer.digest("validation")}}
            provenance["contextual_binding"] = source_binding()

            recipe = {"targets": {}, "baseline_control_method": "inclusive-observed-boundaries-max-coverage-v1",
                      "provenance": provenance}
            reports = trainer.export_candidates(encoder, encoded, labels, thresholds, scores, recipe,
                                                provenance, args, torch, np, pretrained=True)
            self.assertTrue(all(not child.training for child in model.modules()))
            # Boundary roundoff may reject admission, but never changes the exported weights.
            bundle = root / "exports" / "pretrained-fp32"
            self.assertIn("metrics", reports["pretrained-fp32"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
            graph_before = (bundle / "control.onnx").read_bytes()
            import onnxruntime as ort
            session = trainer.session_for(bundle / "control.onnx", ort)
            exported = trainer.score_control_onnx(session, tokenizer.pad_token_id, encoded["validation"], np)
            np.testing.assert_allclose(exported, scores["validation"], atol=args.fp32_atol, rtol=0)
            with torch.no_grad():
                model.classifier.bias.add_(2)
            changed = trainer.score_pretrained(model, tokenizer, encoded["validation"], torch)
            np.testing.assert_allclose(np.array(changed) - scores["validation"], [2, 2], atol=0.0001, rtol=0)
            self.assertEqual((bundle / "control.onnx").read_bytes(), graph_before)
            np.testing.assert_allclose(trainer.score_control_onnx(session, tokenizer.pad_token_id, encoded["validation"], np),
                                       exported, atol=0, rtol=0)
            validation = trainer.parse_json((bundle / "validation.json").read_text())
            self.assertEqual(validation["threshold_source"], "pretrained_torch_validation")
            self.assertEqual(validation["thresholds"], [trainer.decimal_string(v) for v in thresholds])
            self.assertEqual(validation["dataset_sha256"], provenance["dataset_sha256"])
            self.assertEqual((bundle / "manifest.json").exists(), reports["pretrained-fp32"]["passed"])

    def test_failure_diagnostics_omit_private_exception_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                trainer.parse_json("{")
            except trainer.ContractError as error:
                error.args = ("PRIVATE /home/person/input secret",)
                trainer.record_failure(root, "dataset_validation", error)
            paths = list(root.glob("failure-diagnostic-*.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].stat().st_mode & 0o777, 0o600)
            payload = paths[0].read_text()
            report = json.loads(payload)
            self.assertEqual(report["stage"], "dataset_validation")
            self.assertEqual(report["exception_type"], "ContractError")
            self.assertEqual(report["frames"][-1]["function"], "parse_json")
            self.assertNotIn("PRIVATE", payload)
            self.assertNotIn("/home", payload)


class RuntimeBudgetTests(unittest.TestCase):
    def test_stage_deadline_rejects_and_restores_its_owned_timer(self):
        import signal
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), _phase_directory=Path(temporary))
            previous = signal.getsignal(signal.SIGALRM)
            with self.assertRaisesRegex(trainer.ContractError, '^trainer_stage_deadline$'):
                with trainer.trainer_budget(args):
                    self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
                    trainer.stage_budget(args, 'finalization')
                    signal.raise_signal(signal.SIGALRM)
            self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
            timings = json.loads((Path(temporary) / 'stage-timings.json').read_text())
            self.assertEqual(set(timings['elapsed_seconds']), {'preparation', 'finalization'})
            self.assertEqual(timings['limits_seconds'],
                             {'preparation': 8400, 'optimization': 5520, 'finalization': 480})
            self.assertEqual(sum(timings['limits_seconds'].values()), 14400)


class PhaseDiagnosticTests(unittest.TestCase):
    def test_contract_error_codes_are_retained_without_private_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = io.StringIO()
            args = SimpleNamespace(_phase_stream=stream, _phase_directory=root)
            for value, expected in (("pair_token_overflow", "pair_token_overflow"),
                                    ("unsupported_tokenizer_signature", "unsupported_tokenizer_signature"),
                                    ("private_budget_stale_rate", "private_budget_stale_rate"),
                                    ("private_/home/person/PRIVATE", "unknown_contract_error"),
                                    ("PRIVATE-sentinel /private/path", "unknown_contract_error")):
                error = trainer.ContractError(value)
                trainer.emit_phase(args, "tokenization", "ERROR", error=error)
                trainer.record_failure(root, "tokenization", error)
                phase = json.loads((root / "trainer-phase.json").read_text())
                self.assertEqual(phase["error_code"], expected)
                self.assertEqual(phase["exception_type"], "ContractError")
            for path in root.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertNotIn("PRIVATE", path.read_text())
                self.assertNotIn("/private/path", path.read_text())
            self.assertNotIn("PRIVATE", stream.getvalue())

    def test_missing_dataset_cli_keeps_diagnostics_private_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            completed = subprocess.run(
                [sys.executable, "-B", str(Path(trainer.__file__)),
                 "--data", str(root / "PRIVATE-secret" / "audited-pairs.jsonl"),
                 "--dataset-manifest", str(root / "PRIVATE-secret" / "manifest.json"),
                 "--dependency-lock", str(root / "PRIVATE-lock"),
                 "--output", str(output), "--runtime-revision", trainer.file_digest(Path(trainer.bundle_runtime.__file__)),
                 "--approval-sha256", "sha256:" + "a" * 64,
                 "--base", "cross-encoder/ms-marco-MiniLM-L6-v2",
                 "--target-precision", "0.95", "--target-negative-precision", "0.95",
                 "--min-calibration-support", "20", "--min-coverage", "0.5"],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stderr),
                             {"status": "error", "code": "invalid_dataset_export"})
            self.assertNotIn("PRIVATE", completed.stdout + completed.stderr)
            self.assertNotIn(str(root), completed.stdout + completed.stderr)
            self.assertFalse(output.exists())

    def test_phase_artifact_is_private_atomic_and_keeps_checkpoint_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = io.StringIO()
            args = SimpleNamespace(_phase_stream=stream, _phase_directory=root)
            trainer.emit_phase(args, "checkpoint_write", step=50)
            first = (root / "trainer-phase.json").open()
            try:
                trainer.emit_phase(args, "checkpoint_saved", "END", step=50, checkpoint_bytes=1234)
                self.assertEqual(json.load(first)["phase"], "checkpoint_write")
            finally:
                first.close()
            trainer.emit_phase(args, "training", "PROGRESS")
            report = json.loads((root / "trainer-phase.json").read_text())
            self.assertEqual((report["step"], report["checkpoint_bytes"], report["sequence"]), (50, 1234, 3))
            self.assertEqual((root / "trainer-phase.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(root.iterdir()), [root / "trainer-phase.json"])
            self.assertEqual(json.loads(stream.getvalue().splitlines()[-1]),
                             {"event": "PHASE", "phase": "training"})

    def test_unknown_exception_class_and_invalid_phase_cannot_leak(self):
        with tempfile.TemporaryDirectory() as temporary:
            stream = io.StringIO()
            args = SimpleNamespace(_phase_stream=stream, _phase_directory=Path(temporary))
            private_error = type("PRIVATE_secret", (Exception,), {})("PRIVATE arguments")
            trainer.emit_phase(args, "dependency_imports", "ERROR", error=private_error)
            payload = (Path(temporary) / "trainer-phase.json").read_text()
            self.assertEqual(json.loads(payload)["exception_type"], "Other")
            self.assertNotIn("PRIVATE", stream.getvalue() + payload)
            with self.assertRaises(trainer.ContractError):
                trainer.emit_phase(args, "PRIVATE-path")
            with self.assertRaises(trainer.ContractError):
                trainer.emit_phase(args, "training", step="PRIVATE-input")
            self.assertEqual(len(stream.getvalue().splitlines()), 1)

    def test_phase_stream_survives_abrupt_exit_without_creating_output(self):
        # A disposable stdlib child exits without cleanup, like a lost interpreter.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            code = (
                "import importlib.util, os, sys; from pathlib import Path; "
                "from types import SimpleNamespace; "
                "s=importlib.util.spec_from_file_location('trainer', sys.argv[1]); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "a=SimpleNamespace(output=Path(sys.argv[2]), _phase_stream=sys.stdout); "
                "m.emit_phase(a, 'dataset_validation'); os._exit(9)"
            )
            completed = subprocess.run([sys.executable, "-B", "-c", code, trainer.__file__, str(output)],
                                       capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 9)
            self.assertEqual(json.loads(completed.stdout), {"event": "PHASE", "phase": "dataset_validation"})
            self.assertEqual(completed.stderr, "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
