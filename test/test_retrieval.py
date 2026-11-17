"""Retrieval trust-boundary tests, independent of model weights."""
import importlib.util
from pathlib import Path
import runpy
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("retrieval", Path(__file__).resolve().parents[1] / "python/retrieval.py")
retrieval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retrieval)


class RetrievalTests(unittest.TestCase):
    def test_pair_admission_pins_minilm_decider_base_and_revision(self):
        from test.test_router_worker import ArtifactTests
        root = Path(__file__).resolve().parents[1]
        fixture = runpy.run_path(str(root / 'test/retrieval-fixtures.py'))
        spec = importlib.util.spec_from_file_location('retrieval_installer', root / 'python/install-router.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory(prefix='pair-admission-') as temporary:
            directory = Path(temporary)
            decision = directory / 'decision'
            decision.mkdir()
            artifacts = ArtifactTests()
            manifest = artifacts.make_bundle(decision)
            retrieval_manifest = fixture['make_bundle'](directory / 'retrieval', manifest['provenance']['contextual_binding'])
            self.assertEqual(installer.preflight(decision, directory / 'retrieval'), [manifest, retrieval_manifest])
            for base, revision in (
                ('cross-encoder/ms-marco-TinyBERT-L2-v2', '81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc'),
                ('cross-encoder/ms-marco-MiniLM-L6-v2', '0' * 40),
                ('other-decision-model', '0' * 40),
            ):
                manifest['model'] = {'type': 'window_bag', 'base': base, 'revision': revision}
                artifacts.save(decision, manifest)
                with self.subTest(base=base, revision=revision):
                    with self.assertRaises(ValueError):
                        retrieval.load_decider_manifest(decision)
                    with self.assertRaises(ValueError):
                        installer.preflight(decision, directory / 'retrieval')

    def test_compact_pairing_rejects_changed_source_and_full_serving_binding(self):
        from test.test_router_worker import ArtifactTests, source_binding
        root = Path(__file__).resolve().parents[1]
        fixture = runpy.run_path(str(root / 'test/retrieval-fixtures.py'))
        spec = importlib.util.spec_from_file_location('compact_installer', root / 'python/install-router.py')
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        with tempfile.TemporaryDirectory(prefix='compact-pair-') as temporary:
            directory = Path(temporary)
            decision = directory / 'decision'
            decision.mkdir()
            candidate = ArtifactTests().make_bundle(decision)
            binding = candidate['provenance']['contextual_binding']
            for field in ('source_set_sha256', 'approval_sha256', 'target_sha256', 'runtime_revision', 'fixture_only'):
                changed = {**binding, field: False if field == 'fixture_only' else 'sha256:' + '2' * 64}
                shortlist = directory / field
                fixture['make_bundle'](shortlist, changed)
                retrieval.load_manifest(shortlist)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Bundles belong to different training releases'):
                    installer.preflight(decision, shortlist)
            shortlist = directory / 'full-shape'
            manifest = fixture['make_bundle'](shortlist, binding)
            manifest['provenance']['contextual_binding'] = source_binding()
            manifest['scorer_sha256'] = retrieval.scorer_digest(manifest)
            manifest['selector_sha256'] = retrieval.selector_digest(manifest)
            fixture['save'](shortlist / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                retrieval.load_manifest(shortlist)

    def test_failed_or_over_tolerance_parity_is_not_admitted(self):
        comparison = {"passed": True, "tolerance": 1e-4, "max_absolute_embedding_error": 1e-5,
                      "max_absolute_pooled_error": 1e-5, "ranking_pair_changes": 0}
        report = {"role": "retrieval", "nonprivate": True, "cases": [{}, {}, {}, {}], "comparison": comparison}
        retrieval.validate_parity(report)
        for change in ({"max_absolute_embedding_error": 1.01e-4}, {"max_absolute_pooled_error": 1.01e-4},
                       {"tolerance": .001}, {"passed": False}, {"ranking_pair_changes": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                retrieval.validate_parity({**report, "comparison": {**comparison, **change}})


    def test_changed_context_policy_cannot_reuse_bundle_admission(self):
        from test.test_router_worker import source_binding
        root = Path(__file__).resolve().parents[1]
        fixture = runpy.run_path(str(root / 'test/retrieval-fixtures.py'))
        with tempfile.TemporaryDirectory(prefix='retrieval-context-policy-') as temporary:
            directory = Path(temporary) / 'retrieval'
            manifest = fixture['make_bundle'](directory, source_binding())
            manifest['pooling'] = {**manifest['pooling'], 'context_selection': 'head'}
            manifest['scorer_sha256'] = retrieval.scorer_digest(manifest)
            manifest['selector_sha256'] = retrieval.selector_digest(manifest)
            fixture['save'](directory / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                retrieval.load_manifest(directory)


class RetrievalProvenanceTests(unittest.TestCase):
    def test_pretrained_and_trained_bundles_share_admission_without_false_fit_claims(self):
        from test.test_router_worker import source_binding
        fixture = runpy.run_path(str(Path(__file__).with_name('retrieval-fixtures.py')))
        for trained in (False, True):
            with self.subTest(trained=trained), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / 'retrieval'
                fixture['make_bundle'](directory, source_binding(), trained=trained)
                manifest, files = retrieval.load_manifest(directory)
                self.assertIs(manifest['provenance']['training_data_used'], trained)
                self.assertEqual(manifest['provenance']['fit_split'], 'validation' if trained else None)
                self.assertEqual('training.json' in files, trained)

    def test_weight_provenance_rejects_rehashed_cross_dataset_and_false_fit_claims(self):
        from test.test_router_worker import source_binding
        fixture = runpy.run_path(str(Path(__file__).with_name('retrieval-fixtures.py')))
        for trained in (False, True):
            changes = [('dataset_sha256', 'sha256:' + '9' * 64), ('fit_split', 'test'),
                       ('training_data_used', not trained),
                       ('weights', 'unchanged_pretrained' if trained else 'fine_tuned')]
            for field, value in changes:
                with self.subTest(trained=trained, field=field), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary) / 'retrieval'
                    manifest = fixture['make_bundle'](directory, source_binding(), trained=trained)
                    retrieval.load_manifest(directory)
                    recipe = retrieval.boundary.decode_json((directory / 'recipe.json').read_bytes())
                    recipe[field] = value
                    fixture['save'](directory / 'recipe.json', recipe)
                    manifest['provenance']['recipe_sha256'] = retrieval.file_digest(directory / 'recipe.json')
                    for entry in manifest['files']:
                        if entry['path'] == 'recipe.json':
                            entry.update(sha256=manifest['provenance']['recipe_sha256'], bytes=(directory / 'recipe.json').stat().st_size)
                    manifest['scorer_sha256'] = retrieval.scorer_digest(manifest)
                    manifest['selector_sha256'] = retrieval.selector_digest(manifest)
                    fixture['save'](directory / 'manifest.json', manifest)
                    with self.assertRaises(ValueError):
                        retrieval.load_manifest(directory)


if __name__ == "__main__":
    unittest.main()
