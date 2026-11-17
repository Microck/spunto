"""Authored local fixtures, never review receipts or approval of real candidates."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_dataset import catalog_fixture, dataset, opaque, put_blob, row_fixture


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = catalog_fixture()
        self.row = row_fixture(self.catalog)
        self.row['split'] = 'quarantine'
        self.row['audit'].update(status='pending', reviewers=[], full_catalog_reviewed=False)
        self.row['privacy'].update(classification='local_only', scan_passed=False, export_approved=False)
        self.row['label'].update(decision='uncertain', selected_ids=[], verified_negative_ids=[], unknown_ids=['a', 'b', 'c'],
                                 support=[], activate_now=[], earliest_supported_event=None)
        rules = put_blob(self.root, {'fixture': 'not a real policy review'})
        self.row['policy_sha256'] = put_blob(self.root, {'schema_version': 1, 'rules_sha256': rules,
            **{key: self.row['input'][key] for key in ('required_ids', 'excluded_ids', 'explicit_ids')}})
        source = self.row['source']
        events = []
        for index, (role, text) in enumerate((('user', self.row['input']['request']),
                ('assistant', 'A suggestion is not user authorization.'), ('toolCall', ''),
                ('toolResult', 'The inspection completed.'), ('proposed_action', ''))):
            events.append({'event_id': source['event_id'] if index == 0 else opaque(f'prefix-{index}'),
                'event_sequence': index, 'observed_at': source['cutoff_utc'],
                'parent_event_id': events[-1]['event_id'] if events else None, 'role': role, 'text': text,
                'action': {'tool_name': 'bash', 'summary': ''} if role in ('toolCall', 'proposed_action') else None,
                'tool_call_event_id': events[-1]['event_id'] if role == 'toolResult' else None})
        self.row['input']['evidence'][0]['event_sequence'] = 0
        source.update(event_id=events[-1]['event_id'], event_sequence=4)
        self.row['input']['request'] += '\n\nLast step evidence (not policy):\nThe inspection completed.'
        self.row['input']['action'] = events[-1]['action'].copy()
        self.row['event_kind'] = 'proposed_action'
        self.manifest = {'schema_version': 2, 'origin': 'synthetic', 'template_group': source['template_group'],
            'serialization': 'omp-18.16.0-compile-intent-v1', 'request_sha256': dataset.digest(self.row['input']['request']),
            'action_event_id': events[-1]['event_id'], 'events': events}
        self.bind()

    def tearDown(self):
        self.temporary.cleanup()

    def bind(self):
        self.row['source']['lineage_sha256'] = dataset.digest(self.manifest['events'])
        self.row['source']['source_manifest_sha256'] = put_blob(self.root, self.manifest)

    def validate(self):
        return dataset.validate_record(self.row, self.catalog, audit_dir=self.root, accepted=False)

    def test_native_empty_action_pending_cli_and_admission_gate(self):
        self.assertEqual(self.validate(), self.row)
        with self.assertRaisesRegex(dataset.ContractError, '^missing_contextual_target$'):
            list(dataset.dataset_to_pairs(self.row, self.catalog))
        dataset.write_json(self.root / 'catalog.json', self.catalog)
        dataset.write_json(self.root / 'rows.jsonl', self.row)
        args = [sys.executable, '-B', str(dataset.ROOT / 'python/build-dataset.py'), 'extract', '--catalog', str(self.root / 'catalog.json'),
                '--audit-dir', str(self.root), '--input', str(self.root / 'rows.jsonl'), '--output', str(self.root / 'checked.jsonl')]
        process = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual((process.returncode, process.stderr), (0, ''))
        self.assertEqual(json.loads(process.stdout), {'records': 1})
        self.assertEqual(next(dataset.rows(self.root / 'checked.jsonl')), self.row)
        with self.assertRaisesRegex(dataset.ContractError, '^missing_contextual_target$'):
            dataset.validate_record(self.row, self.catalog, audit_dir=self.root)

    def test_invalid_action_shapes_and_privacy(self):
        original = copy.deepcopy(self.row)
        for value, code in ((None, 'schema_mismatch'), (3, 'schema_mismatch'), (' ', 'empty_abstraction'),
                            ('Contact person@example.com', 'privacy_high_risk')):
            with self.subTest(value=value):
                self.row = copy.deepcopy(original)
                self.row['input']['action']['summary'] = value
                with self.assertRaisesRegex(dataset.ContractError, code):
                    self.validate()
        self.row = copy.deepcopy(original)
        del self.row['input']['action']['summary']
        with self.assertRaisesRegex(dataset.ContractError, 'schema_mismatch'):
            self.validate()
        self.row = copy.deepcopy(original)
        self.row['input']['action']['tool_name'] = ''
        with self.assertRaisesRegex(dataset.ContractError, 'schema_mismatch'):
            self.validate()
        for field in ('request', 'evidence'):
            self.row = copy.deepcopy(original)
            if field == 'request':
                self.row['input']['request'] = ''
            else:
                self.row['input']['evidence'][0]['text'] = ''
            with self.assertRaisesRegex(dataset.ContractError, 'empty_abstraction'):
                self.validate()

    def test_committed_hash_rejects_source_tampering(self):
        path = self.root / (self.row['source']['source_manifest_sha256'][7:] + '.json')
        self.manifest['events'][0]['role'] = 'assistant'
        path.write_bytes(dataset.canonical_bytes(self.manifest))
        with self.assertRaisesRegex(dataset.ContractError, '^audit_blob_digest$'):
            self.validate()

    def test_rehashed_role_order_ancestry_action_request_tampering(self):
        original_manifest, original_row = copy.deepcopy(self.manifest), copy.deepcopy(self.row)
        mutations = [
            lambda m: m['events'][0].update(role='assistant'),
            lambda m: m['events'].reverse(),
            lambda m: m['events'][1].update(parent_event_id=None),
            lambda m: m['events'][-1]['action'].update(tool_name='inspect'),
            lambda m: m.update(request_sha256=dataset.digest('different request')),
            lambda m: m['events'][3].update(tool_call_event_id=m['events'][0]['event_id']),
            lambda m: m.update(action_event_id=None),
            lambda m: m.update(action_event_id=m['events'][0]['event_id']),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.manifest, self.row = copy.deepcopy(original_manifest), copy.deepcopy(original_row)
                mutate(self.manifest)
                self.bind()
                with self.assertRaises(dataset.ContractError):
                    self.validate()
        self.manifest, self.row = original_manifest, original_row
        self.row['input']['request'] += ' Changed.'
        with self.assertRaisesRegex(dataset.ContractError, 'synthetic_prefix_request'):
            self.validate()

    def test_assistant_cannot_be_labeled_user_or_tool_evidence(self):
        for kind in ('user_intent', 'tool_fact', 'phase', 'restore_fact'):
            event = self.manifest['events'][1]
            self.row['input']['evidence'] = [{'id': 'assistant', 'source_event_id': event['event_id'],
                'event_sequence': event['event_sequence'], 'observed_at': event['observed_at'], 'kind': kind, 'text': event['text']}]
            with self.subTest(kind=kind), self.assertRaisesRegex(dataset.ContractError, 'evidence_role_mismatch'):
                self.validate()

    def test_exact_description_mapping_requires_source_and_meaning(self):
        entry = self.catalog['entries'][0]
        text = entry['description']
        trigger, binding = dataset.description_span_reference(entry, 0, len(text), text, 'trigger')
        exclusion, other = dataset.description_span_reference(entry, 0, len(text), text, 'exclusion')
        self.assertEqual(binding, other)
        self.assertEqual(trigger, 'trigger:description-span.' + dataset.digest(binding)[7:])
        self.assertEqual(exclusion, trigger.replace('trigger:', 'exclusion:'))
        for start, end, quote, kind in ((0, len(text), 'invented body', 'trigger'), (0, len(text), text, 'positive'),
                                      (-1, len(text), text, 'trigger')):
            with self.assertRaises(dataset.ContractError):
                dataset.description_span_reference(entry, start, end, quote, kind)

    def test_action_can_remain_pending_before_current_cutoff(self):
        previous = self.manifest['events'][-1]
        current = dict(previous, event_id=opaque('later-user'), event_sequence=5, parent_event_id=previous['event_id'],
                       role='user', text='Keep the pending action scoped.', action=None)
        self.manifest['events'].append(current)
        self.row['source'].update(event_id=current['event_id'], event_sequence=5)
        self.row['event_kind'] = 'user_request'
        self.row['input']['request'] = (self.manifest['events'][0]['text'] + '\n\n' + current['text']
            + '\n\nLast step evidence (not policy):\nThe inspection completed.')
        self.manifest['request_sha256'] = dataset.digest(self.row['input']['request'])
        self.bind()
        self.assertEqual(self.validate()['input']['action'], previous['action'])

    def test_actual_host_serializer_matches_prefix(self):
        # Call the real host serializer in a disposable process, not a copied implementation.
        runner = self.root / 'host-prefix.ts'
        runner.write_text('import { compileIntent } from ' + json.dumps(str(dataset.ROOT / 'src/session.ts')) + ';\n'
                          'const cases = await Bun.stdin.json();\n'
                          'console.log(JSON.stringify(cases.map(messages => compileIntent(messages))));\n')
        cases = []
        event_cases = (self.manifest['events'], [
                dict(self.manifest['events'][0], text='A user correction.'),
                dict(self.manifest['events'][1], role='compactionSummary', text='Prior constraints.'),
                dict(self.manifest['events'][1], role='branchSummary', text='Branch constraints.'),
                dict(self.manifest['events'][1], text='A suggestion.'),
                dict(self.manifest['events'][1], text='')])
        for events in event_cases:
            messages = []
            for event in events:
                role, text = event['role'], event['text']
                if role == 'user':
                    messages.append({'role': role, 'content': text})
                elif role in ('assistant', 'toolResult'):
                    messages.append({'role': role, 'content': [{'type': 'text', 'text': text}]})
                elif role in ('compactionSummary', 'branchSummary'):
                    messages.append({'role': role, 'summary': text})
            cases.append(messages)
        process = subprocess.run(["bun", str(runner)],
            input=json.dumps(cases), capture_output=True, text=True, timeout=30)
        self.assertEqual((process.returncode, process.stderr), (0, ''))
        self.assertEqual(json.loads(process.stdout), [self.row['input']['request'],
            'A user correction.\n\nPrior constraints.\n\nBranch constraints.'])
        self.validate()
        events = event_cases[1]
        for index, event in enumerate(events):
            event.update(event_id=opaque(f'summary-{index}'), event_sequence=index,
                         parent_event_id=opaque(f'summary-{index - 1}') if index else None)
        self.manifest.update(events=events, action_event_id=None,
                             request_sha256=dataset.digest(json.loads(process.stdout)[1]))
        self.row['source'].update(event_id=events[-1]['event_id'], event_sequence=len(events) - 1)
        self.row['input'].update(request=json.loads(process.stdout)[1], action=None,
            evidence=[{'id': 'correction', 'source_event_id': events[0]['event_id'], 'event_sequence': 0,
                       'observed_at': events[0]['observed_at'], 'kind': 'user_constraint', 'text': events[0]['text']}])
        self.row['event_kind'] = 'session_restore'
        self.bind()
        self.validate()


if __name__ == '__main__':
    unittest.main()
