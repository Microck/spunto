"""Disposable authored fixtures only. These reviewer records are not real dataset audits."""
import copy
import os
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

SPEC = importlib.util.spec_from_file_location('dataset', Path(__file__).resolve().parents[1] / 'python' / 'build-dataset.py')
dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset)
KEY = b'x' * 32


def opaque(value):
    return dataset.private_id(KEY, 'test', value)


def catalog_fixture():
    entries = []
    for name, description, family in [('a', 'Review keyboard navigation in a browser interface.', 'accessibility'), ('b', 'Prepare store listing screenshots.', 'store'), ('c', 'Write clear interface labels.', 'writing')]:
        entry = {'runtime_id': name, 'declared_name': name, 'description': description,
                 'content_sha256': dataset.digest(['body', name]), 'resource_tree_sha256': None,
                 'source_kind': 'fixture', 'source_ref': opaque(name), 'hidden': False,
                 'automatic_eligible': True, 'explicit_eligible': True, 'family_id': family}
        entry['routing_sha256'] = dataset.digest({k: entry[k] for k in ('runtime_id', 'description', 'hidden', 'automatic_eligible', 'explicit_eligible', 'content_sha256')})
        entries.append(entry)
    catalog = {'schema_version': 1, 'captured_at': '2026-10-01T00:00:00Z', 'loader_contract': 'authored-test-fixture', 'eligibility_sha256': dataset.digest([]), 'entries': entries}
    catalog['catalog_sha256'] = dataset.digest({k: catalog[k] for k in ('schema_version', 'loader_contract', 'eligibility_sha256', 'entries')})
    return catalog


def put_blob(directory, value):
    fingerprint = dataset.digest(value)
    path = directory / (fingerprint[7:] + '.json')
    if not path.exists():
        dataset.write_json(path, value)
    return fingerprint


def row_fixture(catalog, suffix='one', cutoff='2026-10-10T12:00:00Z'):
    event = opaque('event-' + suffix)
    group = opaque('group-' + suffix)
    return {'schema_version': 1, 'example_id': opaque('example-' + suffix),
            'source': {'adapter': 'synthetic', 'conversation_group': group, 'event_id': event, 'event_sequence': 1,
                       'cutoff_utc': cutoff, 'source_manifest_sha256': dataset.digest([]), 'lineage_sha256': dataset.digest([]),
                       'origin': 'synthetic', 'label_basis': 'synthetic_contract', 'template_group': group},
            'split': dataset.synthetic_split(group), 'event_kind': 'user_request', 'catalog_sha256': catalog['catalog_sha256'], 'policy_sha256': dataset.digest([]),
            'input': {'request': 'Review keyboard navigation without changing the interface.',
                      'evidence': [{'id': 'e1', 'source_event_id': event, 'event_sequence': 1, 'observed_at': cutoff, 'kind': 'user_intent', 'text': 'Review keyboard navigation without changing the interface.'}],
                      'action': None, 'already_loaded': [], 'candidate_ids': ['a', 'b', 'c'], 'candidate_coverage': 'full', 'required_ids': ['c'], 'excluded_ids': [], 'explicit_ids': []},
            'label': {'decision': 'selected', 'selected_ids': ['a'], 'activate_now': ['a'], 'verified_negative_ids': ['b'], 'unknown_ids': ['c'], 'acceptable_alternative_sets': [],
                      'support': [{'runtime_id': name, 'judgment': judgment, 'evidence_ids': ['e1'], 'contract_sha256': catalog['entries'][i]['content_sha256'], 'clause_ref': clause} for name, i, judgment, clause in [('a', 0, 'positive', 'trigger:keyboard'), ('b', 1, 'negative', 'exclusion:no-store')]],
                      'earliest_supported_event': event, 'activation_deadline_event': None},
            'audit': {'status': 'machine_audited', 'human_reviewed': False, 'full_catalog_reviewed': True, 'reviewers': [], 'disagreement': False, 'adjudication_sha256': None, 'reconciliation_sha256': None},
            'privacy': {'classification': 'approved_remote_private', 'sanitizer_sha256': dataset.digest(dataset.SANITIZER), 'scan_passed': True, 'export_approved': True, 'publication_allowed': False}}


def attach_audits(row, catalog, directory):
    rules = put_blob(directory, {'fixture': 'required writing instructions'})
    row['policy_sha256'] = put_blob(directory, {'schema_version': 1, 'rules_sha256': rules, **{name: sorted(row['input'][name]) for name in ('required_ids', 'excluded_ids', 'explicit_ids')}})
    if row['source']['origin'] == 'synthetic':
        source = row['source']
        events = [{'event_id': source['event_id'], 'event_sequence': source['event_sequence'], 'observed_at': source['cutoff_utc'], 'parent_event_id': None}]
        source['lineage_sha256'] = dataset.digest(events)
        source['source_manifest_sha256'] = put_blob(directory, {'schema_version': 1, 'origin': 'synthetic', 'template_group': source['template_group'], 'events': events})
    row['audit']['reviewers'] = []
    for role, family, order in [('teacher_a', 'fixture-alpha', ['a', 'b', 'c']), ('teacher_b', 'fixture-beta', ['c', 'a', 'b'])]:
        settings = {'model': family, 'model_revision': 'test-only', 'model_family': family, 'parameters': {}, 'pass_id': role, 'blinded': True, 'tools_disabled': True}
        prompt = dataset.audit_input(row, catalog, order)
        prompt_hash = put_blob(directory, prompt)
        output = {'schema_version': 1, 'prompt_sha256': prompt_hash, 'label': dict(row['label'], activate_now=sorted(row['label']['selected_ids'])), 'privacy_approved': True, 'abstraction_supported': True, 'full_catalog_reviewed': True}
        row['audit']['reviewers'].append({'role': role, 'model': family, 'model_revision': 'test-only', 'settings_sha256': put_blob(directory, settings), 'prompt_sha256': prompt_hash, 'output_sha256': put_blob(directory, output)})
    return row


def alias_catalog_fixture(source='a', description=None):
    catalog = catalog_fixture()
    target = catalog['entries'][2]
    original = next(e for e in catalog['entries'] if e['runtime_id'] == source)
    target['description'] = original['description'] if description is None else description
    target['family_id'] = original['family_id']
    target['routing_sha256'] = dataset.digest({k: target[k] for k in ('runtime_id', 'description', 'hidden', 'automatic_eligible', 'explicit_eligible', 'content_sha256')})
    catalog['catalog_sha256'] = dataset.digest({k: catalog[k] for k in ('schema_version', 'loader_contract', 'eligibility_sha256', 'entries')})
    return catalog


def add_alias_label(row, catalog, source='a'):
    field = 'selected_ids' if source == 'a' else 'verified_negative_ids'
    row['label'][field].append('c')
    row['label']['unknown_ids'] = []
    row['label']['activate_now'] = sorted(row['label']['selected_ids'])
    support = copy.deepcopy(next(s for s in row['label']['support'] if s['runtime_id'] == source))
    support.update(runtime_id='c', contract_sha256=catalog['entries'][2]['content_sha256'])
    row['label']['support'].append(support)
    return row


def attach_reconciliation(row, catalog, directory, *, unsupported=False, members=None):
    """Authored evidence with agreeing teachers and one independent re-audit difference."""
    a, b = row['audit']['reviewers'][:2]
    base = dataset.blob(directory, a['output_sha256'])
    def reviewer(role, output):
        settings = {'model': 'fixture', 'model_revision': None, 'model_family': 'fixture',
                    'parameters': {}, 'pass_id': role, 'blinded': role in dataset.BLINDED_ROLES, 'tools_disabled': False}
        return {'role': role, 'model': 'fixture', 'model_revision': None,
                'settings_sha256': put_blob(directory, settings), 'prompt_sha256': a['prompt_sha256'],
                'output_sha256': put_blob(directory, output)}
    adjudicator = reviewer('adjudicator', base)
    reaudit_output = copy.deepcopy(base)
    reaudit_output['label']['verified_negative_ids'] = []
    reaudit_output['label']['unknown_ids'] = sorted(set(base['label']['unknown_ids']) | {'b'})
    reaudit_output['label']['support'] = [s for s in reaudit_output['label']['support'] if s['runtime_id'] != 'b']
    reaudit_output['abstraction_supported'] = not unsupported
    reaudit = reviewer('reaudit', reaudit_output)
    final = dataset.conservative_label(base['label'], {'b'}, 'full', {})
    coordinator = reviewer('coordinator', dict(base, label=final, semantic_review=False, abstraction_supported=not unsupported))
    row['label'] = copy.deepcopy(final)
    row['audit']['reviewers'] += [adjudicator, reaudit, coordinator]
    row['audit']['adjudication_sha256'] = adjudicator['output_sha256']
    association = {'example_id': row['example_id'], 'output_sha256': adjudicator['output_sha256'],
                   'prompt_sha256': a['prompt_sha256'], 'teacher_a_output_sha256': a['output_sha256'],
                   'teacher_b_output_sha256': b['output_sha256'], 'original_teacher_a_prompt_sha256': a['prompt_sha256'],
                   'original_teacher_b_prompt_sha256': b['prompt_sha256']}
    group = {'schema_version': 1, 'group_id': row['source']['conversation_group'],
             'example_ids': sorted(members or [row['example_id']]), 'unsupported_reviews': [reaudit] if unsupported else []}
    proof = {'schema_version': 1, 'kind': 'conservative-reconciliation', 'example_id': row['example_id'],
             'input_sha256': a['prompt_sha256'], 'teacher_a_output_sha256': a['output_sha256'],
             'teacher_b_output_sha256': b['output_sha256'], 'base_output_sha256': adjudicator['output_sha256'],
             'base_association_sha256': put_blob(directory, association), 'reaudit_output_sha256': reaudit['output_sha256'],
             'conflicting_ids': ['b'], 'group_evidence_sha256': put_blob(directory, group),
             'clause_translation_sha256': put_blob(directory, {}), 'coordinator_output_sha256': coordinator['output_sha256'],
             'coordinator_settings_sha256': coordinator['settings_sha256']}
    row['audit']['reconciliation_sha256'] = put_blob(directory, proof)
    if unsupported:
        row['audit']['status'] = 'quarantined'
        row['split'] = 'quarantine'
        row['privacy'].update(classification='local_only', scan_passed=False, export_approved=False)
    return row


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.umask, os.umask(0o077))
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        self.catalog = catalog_fixture()
        self.row = attach_audits(row_fixture(self.catalog), self.catalog, self.audit)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, row=None):
        # These fixtures exercise retained broad provenance, not contextual admission.
        return dataset.validate_record(row or self.row, self.catalog, audit_dir=self.audit, accepted=False)

    def test_broad_labels_cannot_validate_freeze_or_project_as_contextual(self):
        for source in ('a', 'b'):
            catalog = alias_catalog_fixture(source)
            row = add_alias_label(row_fixture(catalog, 'alias-' + source), catalog, source)
            attach_audits(row, catalog, self.audit)
            path = self.write_rows('alias-' + source + '.jsonl', [row])
            with self.subTest(source=source):
                with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
                    dataset.validate_record(row, catalog, audit_dir=self.audit)
                with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
                    dataset.freeze(path, self.root / ('frozen-' + source), catalog, index=None, audit_dir=self.audit)
                with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
                    list(dataset.dataset_to_pairs(row, catalog))


    def test_role_aware_blindness(self):
        attach_reconciliation(self.row, self.catalog, self.audit)
        self.validate()
        for role in dataset.AUDIT_ROLES:
            row = copy.deepcopy(self.row)
            reviewer = next(r for r in row['audit']['reviewers'] if r['role'] == role)
            settings = dataset.blob(self.audit, reviewer['settings_sha256'])
            settings['blinded'] = not settings['blinded']
            reviewer['settings_sha256'] = put_blob(self.audit, settings)
            with self.subTest(role=role), self.assertRaisesRegex(dataset.ContractError, 'reviewer_settings'):
                self.validate(row)

    def test_agreement_reconciliation_retains_unknown_and_known_support(self):
        base = copy.deepcopy(self.row['label'])
        attach_reconciliation(self.row, self.catalog, self.audit)
        self.validate()
        self.assertFalse(self.row['audit']['disagreement'])
        self.assertEqual(self.row['label']['unknown_ids'], ['b', 'c'])
        self.assertEqual(self.row['label']['support'], base['support'][:1])
        with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
            list(dataset.dataset_to_pairs(self.row, self.catalog))

    def test_reconciliation_rejects_forged_proof(self):
        attach_reconciliation(self.row, self.catalog, self.audit)
        original = dataset.blob(self.audit, self.row['audit']['reconciliation_sha256'])
        for field, value in [('conflicting_ids', []), ('base_output_sha256', dataset.digest('forged')),
                             ('input_sha256', dataset.digest('forged')), ('reaudit_output_sha256', None)]:
            row = copy.deepcopy(self.row)
            row['audit']['reconciliation_sha256'] = put_blob(self.audit, dict(original, **{field: value}))
            with self.subTest(field=field), self.assertRaises(dataset.ContractError):
                self.validate(row)
        association = dataset.blob(self.audit, original['base_association_sha256'])
        association['output_sha256'] = dataset.digest('forged')
        self.row['audit']['reconciliation_sha256'] = put_blob(self.audit, dict(original, base_association_sha256=put_blob(self.audit, association)))
        with self.assertRaisesRegex(dataset.ContractError, 'reconciliation_base_association'):
            self.validate()

    def test_reconciliation_rejects_promotions_polarity_and_support_changes(self):
        attach_reconciliation(self.row, self.catalog, self.audit)
        for mutation in ('unknown_to_known', 'positive_to_negative', 'support'):
            row = copy.deepcopy(self.row)
            coordinator = next(r for r in row['audit']['reviewers'] if r['role'] == 'coordinator')
            output = dataset.blob(self.audit, coordinator['output_sha256'])
            label = output['label']
            if mutation == 'unknown_to_known':
                label['unknown_ids'].remove('b')
                label['verified_negative_ids'] = ['b']
            elif mutation == 'positive_to_negative':
                label['selected_ids'] = []
                label['verified_negative_ids'] = ['a']
            else:
                label['support'][0]['clause_ref'] = 'trigger:invented'
            coordinator['output_sha256'] = put_blob(self.audit, output)
            proof = dataset.blob(self.audit, row['audit']['reconciliation_sha256'])
            proof['coordinator_output_sha256'] = coordinator['output_sha256']
            row['audit']['reconciliation_sha256'] = put_blob(self.audit, proof)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(dataset.ContractError, 'reconciliation_label'):
                self.validate(row)

    def test_reconciliation_preserves_privacy_gate(self):
        attach_reconciliation(self.row, self.catalog, self.audit)
        reviewer = self.row['audit']['reviewers'][1]
        output = dataset.blob(self.audit, reviewer['output_sha256'])
        output['privacy_approved'] = False
        reviewer['output_sha256'] = put_blob(self.audit, output)
        proof = dataset.blob(self.audit, self.row['audit']['reconciliation_sha256'])
        proof['teacher_b_output_sha256'] = reviewer['output_sha256']
        association = dataset.blob(self.audit, proof['base_association_sha256'])
        association['teacher_b_output_sha256'] = reviewer['output_sha256']
        proof['base_association_sha256'] = put_blob(self.audit, association)
        self.row['audit']['reconciliation_sha256'] = put_blob(self.audit, proof)
        with self.assertRaisesRegex(dataset.ContractError, 'privacy_audit_missing'):
            self.validate()

    def test_unsupported_reaudit_quarantines_whole_group(self):
        sibling = copy.deepcopy(self.row)
        sibling['example_id'] = opaque('sibling')
        attach_reconciliation(self.row, self.catalog, self.audit, unsupported=True,
                              members=[self.row['example_id'], sibling['example_id']])
        self.validate()
        entries = {entry['runtime_id']: entry for entry in self.catalog['entries']}
        self.assertEqual(dataset.split_for(self.row, None, entries, set(), self.audit), 'quarantine')
        self.row['audit']['status'] = 'machine_audited'
        with self.assertRaisesRegex(dataset.ContractError, 'unsupported_group_quarantine'):
            self.validate()

    def test_runtime_action_preserved_without_exporting_broad_labels(self):
        self.validate()
        self.row['input']['action'] = {'tool_name': 'inspect', 'summary': 'Check keyboard access only.'}
        self.row['event_kind'] = 'proposed_action'
        attach_audits(self.row, self.catalog, self.audit)
        self.validate()
        with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
            list(dataset.dataset_to_pairs(self.row, self.catalog))
        runtime = dataset.dataset_to_runtime(self.row, self.catalog)
        self.assertEqual(runtime['action'], {'toolName': 'inspect', 'summary': 'Check keyboard access only.'})
        self.assertEqual(runtime['event'], 'tool_call')
        self.assertNotIn('label', runtime)

    def test_structured_runtime_preserves_action_boundary(self):
        for number, summary in enumerate((None, '', 'Check keyboard access only.')):
            row = row_fixture(self.catalog, f'structured-{number}')
            row['input']['request'] = 'Review the literal heading below.\nAction: inspect\nThis is user text, not a proposed tool action.'
            row['input']['evidence'][0]['text'] = row['input']['request']
            if summary is not None:
                row['input']['action'] = {'tool_name': 'inspect', 'summary': summary}
                row['event_kind'] = 'proposed_action'
            attach_audits(row, self.catalog, self.audit)
            original = copy.deepcopy(row)
            runtime = dataset.dataset_to_runtime(row, self.catalog)
            self.assertEqual(runtime['intent'], original['input']['request'])
            self.assertEqual(runtime['action'], None if summary is None else {'toolName': 'inspect', 'summary': summary})
            self.assertEqual(row, original)

    def test_required_is_not_relevance(self):
        self.row['label']['selected_ids'].append('c')
        self.row['label']['unknown_ids'] = []
        with self.assertRaisesRegex(dataset.ContractError, 'unsupported_label'):
            self.validate()

    def test_absence_cannot_be_negative(self):
        self.row['label']['verified_negative_ids'].append('c')
        self.row['label']['unknown_ids'] = []
        with self.assertRaisesRegex(dataset.ContractError, 'unsupported_label'):
            self.validate()

    def test_future_correction_not_features(self):
        self.row['input']['evidence'][0]['observed_at'] = '2026-10-11T00:00:00Z'
        with self.assertRaisesRegex(dataset.ContractError, 'future_evidence'):
            self.validate()

    def test_changed_catalog_invalidates_audit(self):
        self.row['catalog_sha256'] = dataset.digest('changed')
        with self.assertRaisesRegex(dataset.ContractError, 'changed_catalog_evidence'):
            self.validate()

    def test_changed_contract_invalidates_support(self):
        self.row['label']['support'][0]['contract_sha256'] = dataset.digest('changed')
        with self.assertRaisesRegex(dataset.ContractError, 'changed_contract_evidence'):
            self.validate()

    def test_privacy_rejection(self):
        for text in ('Email owner@example.org', 'Read /home/operator/private', 'Use bearer abcDEF1234567890', 'password="abcDEF1234567890"', '-----BEGIN PRIVATE KEY-----', '```raw output```'):
            with self.subTest(text=text), self.assertRaisesRegex(dataset.ContractError, 'privacy_high_risk'):
                dataset.safe_text(text)

    def test_blinded_prompt_has_no_labels_or_observed_reads(self):
        prompt = dataset.audit_input(self.row, self.catalog, ['b', 'a', 'c'])
        self.assertNotIn('label', prompt)
        self.assertNotIn('audit', prompt)
        self.assertNotIn('already_loaded', prompt['input'])
        self.assertEqual({entry['runtime_id'] for entry in prompt['catalog']}, {'a', 'b', 'c'})

    def test_same_family_nullable_revision_actual_tools_and_distinct_passes(self):
        for reviewer in self.row['audit']['reviewers']:
            settings = dataset.blob(self.audit, reviewer['settings_sha256'])
            settings.update(model_family='fixture-alpha', model_revision=None, tools_disabled=False)
            reviewer['model_revision'] = None
            reviewer['settings_sha256'] = put_blob(self.audit, settings)
        self.validate()
        provenance = dataset.audit_check(self.row, self.catalog, self.audit)
        self.assertTrue(provenance['same_family'])
        summary = {'settings': {}, 'reviewed_rows': 0, 'same_family_rows': 0}
        dataset.audit_summary_add(summary, self.row, self.audit)
        self.assertEqual(summary['same_family_rows'], 1)
        self.assertTrue(all(settings['model_revision'] is None and settings['tools_disabled'] is False
                            for settings in summary['settings'].values()))
        reviewer = self.row['audit']['reviewers'][1]
        settings = dataset.blob(self.audit, reviewer['settings_sha256'])
        settings['pass_id'] = 'teacher_a'
        reviewer['settings_sha256'] = put_blob(self.audit, settings)
        with self.assertRaisesRegex(dataset.ContractError, 'audit_not_independent'):
            self.validate()

    def test_copied_audit_output_rejected(self):
        self.row['audit']['reviewers'][1]['output_sha256'] = self.row['audit']['reviewers'][0]['output_sha256']
        with self.assertRaisesRegex(dataset.ContractError, 'audit_not_independent'):
            self.validate()

    def test_public_technical_findings_require_real_privacy_disposition(self):
        self.row['input']['request'] = 'Review password and token guidance at https://example.org.'
        attach_audits(self.row, self.catalog, self.audit)
        reviewer = self.row['audit']['reviewers'][1]
        output = dataset.blob(self.audit, reviewer['output_sha256'])
        output['privacy_approved'] = False
        reviewer['output_sha256'] = put_blob(self.audit, output)
        self.row['split'] = 'quarantine'
        self.row['privacy'].update(classification='local_only', scan_passed=False, export_approved=False)
        self.validate()
        findings = dataset.scan_record(self.row, {entry['runtime_id']: entry for entry in self.catalog['entries']})
        self.assertEqual({item['category'] for item in findings}, {'public_url_or_domain', 'credential_vocabulary'})
        self.row['privacy']['scan_passed'] = True
        with self.assertRaisesRegex(dataset.ContractError, 'privacy_audit_missing'):
            self.validate()
        output['privacy_approved'] = True
        reviewer['output_sha256'] = put_blob(self.audit, output)
        self.row['privacy'].update(classification='approved_remote_private', export_approved=True)
        self.validate()
        provenance = dataset.audit_check(self.row, self.catalog, self.audit)
        self.assertEqual([dataset.blob(self.audit, review['output_sha256'])['privacy_approved']
                          for review in provenance['reviews']], [True, True])
        with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
            list(dataset.dataset_to_pairs(self.row, self.catalog))

    def test_pending_current_date_closed_input_with_findings(self):
        row = row_fixture(self.catalog, 'pending-now', '2026-11-05T22:00:00Z')
        row['input']['request'] = 'Review the public token API at https://example.org.'
        attach_audits(row, self.catalog, self.audit)
        row['split'] = 'quarantine'
        row['audit'].update(status='pending', full_catalog_reviewed=False, reviewers=[])
        row['label'].update(decision='uncertain', selected_ids=[], activate_now=[], verified_negative_ids=[],
                            unknown_ids=['a', 'b', 'c'], support=[], earliest_supported_event=None)
        row['privacy'].update(classification='local_only', scan_passed=False, export_approved=False)
        self.assertIs(dataset.validate_record(row, self.catalog, audit_dir=self.audit, accepted=False), row)
        self.assertFalse(row['privacy']['scan_passed'])
        with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
            list(dataset.dataset_to_pairs(row, self.catalog))

    def test_confirmed_selection_survives_unknown_but_none_needs_full_truth(self):
        self.validate()
        self.assertEqual(self.row['label']['selected_ids'], ['a'])
        self.assertEqual(self.row['label']['unknown_ids'], ['c'])
        row = copy.deepcopy(self.row)
        row['label'].update(decision='none', selected_ids=[], activate_now=[])
        with self.assertRaisesRegex(dataset.ContractError, 'schema_mismatch'):
            self.validate(row)
        row['label'].update(decision='uncertain', verified_negative_ids=['a', 'b', 'c'], unknown_ids=[])
        with self.assertRaisesRegex(dataset.ContractError, 'uncertain_without_unresolved'):
            self.validate(row)

    def test_disagreement_requires_new_adjudication_not_rewritten_outputs(self):
        reviewer = self.row['audit']['reviewers'][1]
        output = dataset.blob(self.audit, reviewer['output_sha256'])
        output['label'].update(decision='uncertain', selected_ids=[], activate_now=[],
                               verified_negative_ids=[], unknown_ids=['a', 'b', 'c'], support=[])
        reviewer['output_sha256'] = put_blob(self.audit, output)
        original_outputs = [item['output_sha256'] for item in self.row['audit']['reviewers']]
        self.row['audit']['disagreement'] = True
        with self.assertRaisesRegex(dataset.ContractError, 'adjudication_required'):
            self.validate()
        prompt_hash = put_blob(self.audit, dataset.audit_input(self.row, self.catalog, ['b', 'c', 'a']))
        adjudication = {'schema_version': 1, 'prompt_sha256': prompt_hash,
                        'label': dict(self.row['label'], activate_now=['a']),
                        'privacy_approved': True, 'abstraction_supported': True, 'full_catalog_reviewed': True}
        settings = {'model': 'fixture-adjudicator', 'model_revision': None, 'model_family': 'fixture-alpha',
                    'parameters': {}, 'pass_id': 'adjudicator-context', 'blinded': False, 'tools_disabled': False}
        output_hash = put_blob(self.audit, adjudication)
        self.row['audit']['reviewers'].append({'role': 'adjudicator', 'model': settings['model'], 'model_revision': None,
                                              'settings_sha256': put_blob(self.audit, settings),
                                              'prompt_sha256': prompt_hash, 'output_sha256': output_hash})
        self.row['audit']['adjudication_sha256'] = output_hash
        self.validate()
        self.assertEqual([item['output_sha256'] for item in self.row['audit']['reviewers'][:2]], original_outputs)
        self.assertEqual(dataset.blob(self.audit, original_outputs[1])['label']['decision'], 'uncertain')

    def test_pending_rows_never_project(self):
        self.row['audit']['status'] = 'pending'
        self.row['split'] = 'quarantine'
        with self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
            list(dataset.dataset_to_pairs(self.row, self.catalog))

    def write_rows(self, filename, values):
        path = self.root / filename
        with dataset.new_file(path) as stream:
            for value in values:
                stream.write(dataset.canonical_bytes(value) + b'\n')
        return path

    def test_broad_holdout_rows_require_contextual_review_in_every_split(self):
        for split in ('train', 'validation'):
            suffix = next(str(i) for i in range(1000) if dataset.synthetic_split(opaque('group-' + str(i))) == split)
            row = attach_audits(row_fixture(self.catalog, suffix), self.catalog, self.audit)
            source = self.write_rows(split + '.jsonl', [row])
            with self.subTest(split=split), self.assertRaisesRegex(dataset.ContractError, 'missing_contextual_target'):
                dataset.freeze(source, self.root / (split + '-frozen'), self.catalog,
                               index=None, audit_dir=self.audit, holdout=['store'])

    def test_positive_holdout_and_reserved_intent_keep_whole_group(self):
        for holdout, reserve in ((['accessibility'], []), (['store'], [self.row['source']['conversation_group']])):
            with self.subTest(holdout=holdout):
                suffix = holdout[0]
                other = copy.deepcopy(self.row)
                other['example_id'] = opaque('descendant')
                other['input']['request'] = 'Check a second keyboard navigation constraint.'
                other['label'].update(decision='uncertain', selected_ids=[], activate_now=[],
                                      unknown_ids=['a', 'c'], support=[other['label']['support'][1]])
                attach_audits(other, self.catalog, self.audit)
                entries = {entry['runtime_id']: entry for entry in self.catalog['entries']}
                self.assertEqual(dataset.split_for(self.row, None, entries, holdout, self.audit, reserve),
                                 'unseen_family_test')
                self.assertEqual(dataset.split_for(other, None, entries, holdout, self.audit,
                                                   [self.row['source']['conversation_group']]),
                                 'unseen_family_test')

    def test_recorded_unseen_membership_keeps_non_authored_whole_group(self):
        entries = {entry['runtime_id']: entry for entry in self.catalog['entries']}
        other = copy.deepcopy(self.row)
        other['example_id'] = opaque('recorded-descendant')
        other['label']['selected_ids'] = []
        group = self.row['source']['conversation_group']
        for row in (self.row, other):
            self.assertEqual(dataset.split_for(row, None, entries, (), self.audit,
                             recorded_unseen_groups=(group,)), 'unseen_family_test')
        other['audit']['status'] = 'pending'
        self.assertEqual(dataset.split_for(other, None, entries, (), self.audit,
                         recorded_unseen_groups=(group,)), 'quarantine')

    def test_current_synthetic_dates_and_descendants_do_not_use_history_boundaries(self):
        row = row_fixture(self.catalog, cutoff='2026-11-05T23:00:00Z')
        attach_audits(row, self.catalog, self.audit)
        self.validate(row)
        entries = {entry['runtime_id']: entry for entry in self.catalog['entries']}
        self.assertEqual(dataset.split_for(row, None, entries, set(), self.audit), self.row['split'])
        self.assertEqual(dataset.temporal_split(row['source']['cutoff_utc']), 'quarantine')
        for stamp, split in [('2026-10-14T23:59:59Z', 'train'), ('2026-10-15T00:00:00Z', 'validation'),
                             ('2026-10-25T00:00:00Z', 'temporal_test'), ('2026-11-05T00:00:00Z', 'quarantine')]:
            self.assertEqual(dataset.temporal_split(stamp), split)


    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1.5}', '{"a":"\\ud800"}'):
            with self.assertRaises(dataset.ContractError):
                dataset.parse_json(text)

    def source_file(self, filename, sid, events, parent=None):
        header = {'type': 'session', 'version': 3, 'id': sid, 'timestamp': '2026-10-10T00:00:00Z', 'cwd': str(self.root / 'other-project')}
        if parent:
            header['parentSession'] = str(parent)
        return self.write_rows(filename, [header] + events)

    def test_omp_branches_corrections_imports_and_temporal_crossing(self):
        def event(identity, parent, when, text):
            return {'id': identity, 'parentId': parent, 'timestamp': when, 'type': 'message', 'message': {'role': 'user', 'content': [{'type': 'text', 'text': text}]}}
        source_dir = self.root / 'sources'
        source_dir.mkdir()
        first = self.source_file('sources/first.jsonl', 'first', [event('one', None, '2026-10-10T01:00:00Z', 'Inspect keyboard access'), event('two', 'one', '2026-10-10T02:00:00Z', 'Do not change files'), event('sibling', 'one', '2026-10-10T03:00:00Z', 'Other branch')])
        self.source_file('sources/child.jsonl', 'child', [event('child-one', None, '2026-10-20T00:00:00Z', 'A child task')], first)
        self.source_file('sources/import.jsonl', 'imported', [event('import-one', None, '2026-10-10T01:00:00Z', 'Inspect keyboard access')])
        key_path = self.root / 'key'
        key_path.write_bytes(KEY)
        key_path.chmod(0o600)
        index_path = self.root / 'index.sqlite'
        dataset.inventory(source_dir, index_path, key_path, self.root / 'excluded')
        index = dataset.open_index(index_path)
        try:
            groups = list(index.execute('SELECT component,split FROM lineage'))
            self.assertEqual(len({row['component'] for row in groups}), 1)
            self.assertEqual({row['split'] for row in groups}, {'quarantine'})
            sid = dataset.private_id(KEY, 'session', 'first')
            cutoff = dataset.private_id(KEY, 'event', [sid, 'two'])
            meta = index.execute('SELECT * FROM lineage WHERE session=?', (sid,)).fetchone()
            row = copy.deepcopy(self.row)
            row['split'] = 'quarantine'
            row['source'].update(adapter='omp-v3', origin='history_abstraction', label_basis='current_catalog_relabel', template_group=None, conversation_group=meta['component'], event_id=cutoff, event_sequence=3, cutoff_utc='2026-10-10T02:00:00Z', lineage_sha256=meta['lineage_sha256'], source_manifest_sha256=index.execute("SELECT value FROM metadata WHERE key='source_manifest_sha256'").fetchone()[0])
            row['input']['evidence'] = [{'id': 'e1', 'source_event_id': dataset.private_id(KEY, 'event', [sid, 'one']), 'event_sequence': 2, 'observed_at': '2026-10-10T01:00:00Z', 'kind': 'user_intent', 'text': 'Inspect keyboard access'}]
            dataset.source_check(row, index)
            row['input']['evidence'][0].update(source_event_id=dataset.private_id(KEY, 'event', [sid, 'sibling']), event_sequence=4, observed_at='2026-10-10T03:00:00Z')
            with self.assertRaisesRegex(dataset.ContractError, 'evidence_outside_prefix'):
                dataset.source_check(row, index)
        finally:
            index.close()

    def test_native_title_preamble_preserves_physical_lines_and_file_hash(self):
        source_dir = self.root / 'native'
        source_dir.mkdir()
        event = {'type': 'message', 'id': 'e', 'parentId': None, 'timestamp': '2026-10-10T01:00:00Z',
                 'message': {'role': 'user', 'content': 'Check a fixture.'}}
        source = self.source_file('native/session.jsonl', 'native', [event])
        payload = source.read_bytes()
        title = {'type': 'title', 'v': 1, 'title': 'Fixture only', 'source': 'user',
                 'updatedAt': '2026-10-10T02:00:00Z', 'pad': ''}
        title['pad'] = ' ' * (256 - len(dataset.canonical_bytes(title)) - 1)
        source.write_bytes(dataset.canonical_bytes(title) + b'\n' + payload)
        original = dataset.file_digest(source)
        key = self.root / 'key'
        key.write_bytes(KEY)
        key.chmod(0o600)
        output = self.root / 'native.sqlite'
        self.assertEqual(dataset.inventory(source_dir, output, key, self.root / 'excluded')['events'], 1)
        with dataset.open_index(output) as index:
            self.assertEqual(index.execute('SELECT sequence FROM events').fetchone()[0], 3)
            self.assertEqual(index.execute('SELECT sha256 FROM files').fetchone()[0], original)
        self.assertEqual(dataset.file_digest(source), original)
        for name, prefix in [('unknown', b'{"type":"metadata"}\n'),
                             ('wrong-width', dataset.canonical_bytes(dict(title, pad='')) + b'\n'),
                             ('wrong-version', dataset.canonical_bytes(dict(title, v=2)) + b'\n'),
                             ('double-slot', (dataset.canonical_bytes(title) + b'\n') * 2)]:
            source.write_bytes(prefix + payload)
            failed = self.root / (name + '.sqlite')
            with self.subTest(name=name), self.assertRaises(dataset.ContractError):
                dataset.inventory(source_dir, failed, key, self.root / 'excluded')
            with self.assertRaisesRegex(dataset.ContractError, 'incomplete_inventory'):
                dataset.open_index(failed)


if __name__ == '__main__':
    unittest.main()
