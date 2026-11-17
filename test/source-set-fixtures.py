"""Owned disposable fixture generator. All opinions and grants here are TEST ONLY."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
spec = importlib.util.spec_from_file_location('fixture_codec', ROOT / 'python/build-dataset.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
dataset = d

KEY = b'x' * 32
PROOFS = {}


def proof(reference):
    return d.portable.SourceProof(PROOFS[reference['declaration']])


def opaque(value):
    return dataset.private_id(KEY, 'test', value)


def catalog_fixture(names=('a', 'b', 'c')):
    entries = []
    for name, (_, description, family) in zip(names, [('a', 'Review keyboard navigation in a browser interface.', 'accessibility'), ('b', 'Prepare store listing screenshots.', 'store'), ('c', 'Write clear interface labels.', 'writing')]):
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


def quote(text, start=0, end=None):
    raw = text.encode()
    end = len(raw) if end is None else end
    return {'start_byte': start, 'end_byte': end, 'quote': raw[start:end].decode(),
            'source_sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
            'quote_sha256': 'sha256:' + hashlib.sha256(raw[start:end]).hexdigest()}


class Fixture:
    def __init__(self, directory, *, mixed=False, distinct=True, raw_role='train', split='train',
                 collision=False, same_request=False, long=False, imported=False, plan_change=None,
                 multipart=False, empty_challenge=False, namespace='one', names=('a', 'b', 'c'),
                 masked=False, diagnostic=False, exclude_fit=False, fixture_source=True,
                 conversation_namespace=None, template_namespace=None, request_namespace=None,
                 unavailable=False):
        self.root = Path(directory)
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        self.commands = []
        self.catalog = catalog_fixture(names)
        self.bodies = {}
        for entry in self.catalog['entries']:
            body = 'Fixture contract for ' + entry['runtime_id'] + '.'
            entry['content_sha256'] = quote(body)['source_sha256']
            self.bodies[entry['runtime_id']] = body
            self.put('source/bodies/' + entry['content_sha256'][7:] + '.md', body, raw=True)
        if collision:
            self.catalog['entries'][1]['description'] = self.catalog['entries'][0]['description']
        for entry in self.catalog['entries']:
            entry['routing_sha256'] = d.digest({k: entry[k] for k in ('runtime_id', 'description', 'hidden', 'automatic_eligible', 'explicit_eligible', 'content_sha256')})
        self.catalog['catalog_sha256'] = d.digest({k: self.catalog[k] for k in ('schema_version', 'loader_contract', 'eligibility_sha256', 'entries')})
        self.put('catalog.json', self.catalog)
        self.target = self.put('source/target.md', 'Fixture contextual rubric.\nAdditional authority clauses.\n', raw=True)
        self.rubric = self.put('source/rubric.md', 'Fixture contextual rubric.\n', raw=True) if distinct else self.target
        self.records, assignments = [], []
        for i in range(2):
            row = row_fixture(self.catalog, 'v2-' + str(i))
            del row['label']
            row['input']['candidate_ids'] = [name for name in names if not unavailable or name != names[1]]
            row['input']['candidate_coverage'] = 'shortlist' if unavailable else 'full'
            row['example_id'] = 'synthetic-case-' + namespace + '-' + str(i)
            row['split'] = 'challenge' if empty_challenge and i == 0 else split
            active = 'Cancel the earlier task. Now review keyboard access é 雪 ' + (request_namespace or namespace) + ' ' + str(0 if same_request else i) + '.'
            old = 'Earlier canceled task: inspect a fixture.' + (' bounded context' * 900 if long else '')
            events = []
            for n, (role, text) in enumerate([('user', old), ('assistant', 'Earlier suggestion, not authorization.'),
                    ('toolCall', ''), ('toolResult', 'Inert fixture inspection evidence.'), ('user', active)]):
                events.append({'event_id': 'synthetic-event-' + str(i) + '-' + str(n), 'event_sequence': n,
                    'observed_at': row['source']['cutoff_utc'], 'parent_event_id': events[-1]['event_id'] if events else None,
                    'role': role, 'text': text, 'action': {'tool_name': 'inspect', 'summary': ''} if role == 'toolCall' else None,
                    'tool_call_event_id': events[-1]['event_id'] if role == 'toolResult' else None})
            raw_messages = copy.deepcopy(events)
            if multipart:
                for n in (0, 3, 4):
                    text = raw_messages[n].pop('text')
                    extra = 'Preserve the second text block é 雪.'
                    raw_messages[n]['content'] = [
                        {'type': 'text', 'text': text},
                        {'type': 'image', 'data': 'inert fixture', 'mimeType': 'image/png'},
                        {'type': 'text', 'text': extra}]
                    events[n]['text'] = text + '\n' + extra
                active = events[-1]['text']
            if imported:
                for message in raw_messages:
                    message['speaker'] = 'producer-' + message.pop('role')
            case = {'case_id': row['example_id'], 'raw_role': 'calibration' if empty_challenge and i == 0 else raw_role,
                    'conversation_group': 'authored conversation ' + (conversation_namespace or namespace) + (' empty' if empty_challenge and i == 0 else ''),
                    'template_group': 'authored template ' + (template_namespace or namespace) + (' empty' if empty_challenge and i == 0 else ''),
                    'synthetic': True, 'fixture_only': fixture_source, 'messages': raw_messages}
            case_ref = self.put('source/case-' + str(i) + '.json', case)
            request = self.compile(events)
            prefix = {'schema_version': 2, 'origin': 'synthetic', 'template_group': case['template_group'],
                      'serialization': 'omp-18.16.0-compile-intent-v1', 'request_sha256': d.digest(request),
                      'action_event_id': None, 'events': events}
            prefix_ref = self.put('source/prefix-' + str(i) + '.json', prefix)
            row['source'].update(provenance_kind=d.AUTHORED_CONTEXTUAL, conversation_group=case['conversation_group'],
                template_group=case['template_group'], event_id=events[-1]['event_id'], event_sequence=4,
                source_manifest_sha256=prefix_ref['sha256'], lineage_sha256=d.digest(events))
            row['authored'] = {'case_id': case['case_id'], 'messages': raw_messages, 'case_reference': case_ref,
                'prefix_reference': prefix_ref, 'role_plan_sha256': '', 'event_metadata': {'basis': 'SYNTHETIC',
                **{k: row['source'][k] for k in ('event_id', 'event_sequence', 'cutoff_utc')}}}
            row['authored']['source_fields'] = {key: '/' + key for key in
                ('synthetic', 'case_id', 'messages', 'conversation_group', 'template_group')}
            row['authored']['message_mapping'] = [{'message_index': n, 'event_id': event['event_id'],
                'raw_role': event['role'], 'role': event['role'], 'fields': {key: '/' + key
                for key in ('role', 'text', 'action', 'tool_call_event_id')}} for n, event in enumerate(events)]
            if multipart:
                for n in (0, 3, 4):
                    row['authored']['message_mapping'][n]['fields']['text'] = '/content'
            if imported:
                for mapping in row['authored']['message_mapping']:
                    mapping['raw_role'] = 'producer-' + mapping['raw_role']
                    mapping['fields']['role'] = '/speaker'
            row['prior_broad_provenance'] = {'state': 'absent_new_synthetic_source'}
            row['input'].update(request=request, required_ids=[], explicit_ids=[],
                evidence=[{'id': 'active', 'source_event_id': events[-1]['event_id'], 'event_sequence': 4,
                    'observed_at': events[-1]['observed_at'], 'kind': 'user_intent', 'text': active}])
            row['privacy'].update(classification='local_only', export_approved=False)
            row['audit'] = {'status': 'contextual_pair_audited', 'scope': 'contextual_pairs', 'human_reviewed': False,
                'full_catalog_reviewed': False, 'reviewers': [], 'disagreement': False,
                'adjudication_sha256': None, 'reconciliation_sha256': None}
            rules = put_blob(self.audit, {'fixture_only': True})
            row['policy_sha256'] = put_blob(self.audit, {'schema_version': 1, 'rules_sha256': rules,
                **{k: row['input'][k] for k in ('required_ids', 'excluded_ids', 'explicit_ids')}})
            assignments.append({'case_id': row['example_id'], 'split': row['split'], 'raw_role': case['raw_role'],
                'source_reference': case_ref, 'role_pointer': '/raw_role', 'reviewed_ids': list(names[:2]),
                **{k: case[k] for k in ('conversation_group', 'template_group')}})
            self.records.append(row)
        if mixed:
            broad = attach_audits(row_fixture(self.catalog, 'prior-broad'), self.catalog, self.audit)
            # Existing broad source bytes are committed before the mixed annotation.
            self.prior_broad = copy.deepcopy(broad)
            broad_ref = self.put('source/prior-broad.json', {k: v for k, v in broad.items() if k != 'audit'})
            assignments.append({'case_id': broad['example_id'], 'split': broad['split'], 'raw_role': broad['split'],
                'source_reference': broad_ref, 'role_pointer': '/split', 'reviewed_ids': ['a', 'b'],
                **{k: broad['source'][k] for k in ('conversation_group', 'template_group')}})
            self.records.append(broad)
        self.plan = {'format': 'contextual-role-plan-v2', 'assignments': assignments}
        if plan_change:
            plan_change(self.plan)
        plan_ref = self.put('source/role-plan.json', self.plan)
        for row in self.records:
            if d.authored_contextual(row):
                row['authored']['role_plan_sha256'] = plan_ref['sha256']
        self.pairs, self.mapping = [], {'format': 'contextual-identity-map-v2', 'pairs': []}
        snapshots = [{k: v for k, v in row.items() if k != 'audit'} for row in self.records]
        snapshot_ref = self.put('source/snapshots.json', snapshots)
        for i, row in enumerate(self.records):
            authored = d.authored_contextual(row)
            source_ref = row['authored']['case_reference'] if authored else dict(snapshot_ref, json_pointer='/' + str(i))
            raw_request = row['input']['evidence'][0]['text'] if authored else row['input']['request']
            start = row['input']['request'].encode().index(raw_request.encode())
            compiled_quote = quote(row['input']['request'], start, start + len(raw_request.encode()))
            for entry in self.catalog['entries'][:2]:
                skill = entry['runtime_id']
                if empty_challenge and i == 0:
                    status = 'unresolved'
                elif skill == names[0]:
                    status = 'contextual'
                elif i == 1 and not empty_challenge:
                    status = 'unresolved'
                else:
                    status = 'nonapplicable'
                pair_id = 'sha256:' + hashlib.sha256((row['example_id'] + '\0' + skill).encode()).hexdigest()
                binding = {'case_id': row['example_id'], 'runtime_id': skill, 'body_sha256': entry['content_sha256'],
                    'target_sha256': self.target['sha256'], 'request_sha256': quote(raw_request)['source_sha256'],
                    'source_reference': source_ref}
                opinions, refs, mapped = {}, {}, {}
                for role, foreign in [('author', 'author-opaque-' + str(i) + '-' + skill),
                                      ('reviewer', [row['example_id'], skill]),
                                      ('adjudicator', {'record': ['decision-' + str(i), skill], 'revision': 1})]:
                    value = dict(binding, producer_pair=foreign, producer_role='original-' + role, status=status)
                    if role == 'adjudicator':
                        value.update(affirmative_exclusion='Fixture exclusion.' if status == 'nonapplicable' else '',
                            missing_facts='Fixture facts absent.' if status == 'unresolved' else '', opinion_references=refs.copy(),
                            privacy_scope='local_only', privacy_reviewed=True, full_catalog_reviewed=False)
                    ref = self.put('source/' + str(i) + '-' + skill + '-' + role + '.json', value)
                    mapped[role] = {'reference': ref, 'external_id': foreign, 'raw_role': value['producer_role'],
                        'fields': {**{k: '/' + k for k in binding}, 'external_id': '/producer_pair', 'raw_role': '/producer_role'}}
                    if role != 'adjudicator':
                        opinions[role], refs[role] = value, ref
                pair = {'pair_id': pair_id, 'runtime_id': skill,
                    'identities': {'example_id': row['example_id'], 'split': row['split'], 'event_kind': row['event_kind'],
                        'request_index': 4 if authored else 0, **{k: row['source'][k] for k in ('event_id', 'conversation_group', 'template_group')}},
                    'source_identity': {'case_id': row['example_id'], 'target_version': 'contextual-training-alignment-v1', 'canonical_pair_id': pair_id},
                    'cutoff_source': row['source'], 'input_sha256': d.digest(row['input']), 'catalog_sha256': row['catalog_sha256'],
                    'source_relevance': {'state': 'absent_new_synthetic_source'} if authored else {
                        'label_snapshot_sha256': d.digest(row['label']), 'original_label_type': 'positive' if skill == 'a' else 'verified_negative'},
                    'privacy': row['privacy'], 'privacy_evidence': dict(snapshot_ref, json_pointer='/' + str(i) + '/privacy'),
                    'opinions': opinions, 'opinion_references': refs, 'adjudication_reference': mapped['adjudicator']['reference'],
                    'body_sha256': entry['content_sha256'], 'body_quote': quote(self.bodies[skill]),
                    'request_sha256': quote(row['input']['request'])['source_sha256'], 'request_quote': compiled_quote,
                    'status': status, 'affirmative_exclusion': value['affirmative_exclusion'], 'missing_facts': value['missing_facts'],
                    'family_mask': masked, 'disposition': 'masked' if masked else 'unresolved' if status == 'unresolved' else 'ineligible' if skill not in row['input']['candidate_ids'] else 'diagnostic' if diagnostic else 'eligible_resolved',
                    'conjunctive_groups': [], 'admitted': False, 'eligibility': {'catalog_route': 'automatic',
                        'automatic_eligible': True, 'explicit_eligible': True, 'active_explicit_invocation': False}}
                self.pairs.append(pair)
                self.mapping['pairs'].append({'pair_id': pair_id, 'records': mapped, 'quote_binding': {
                    'event_id': row['source']['event_id'], 'raw_quote': quote(raw_request), 'compiled_quote': compiled_quote} if authored else None})
        privacy_ref = self.put('source/privacy-findings.json', {'export_approved': False, 'publication_allowed': False})
        for pair, item in zip(self.pairs, self.mapping['pairs']):
            old_ref = pair['adjudication_reference']
            original = d.read_json(self.root / old_ref['path'])
            original['original_opinions'] = pair['opinions']
            new_ref = self.put(old_ref['path'], original)
            pair['adjudication_reference'] = new_ref
            item['records']['adjudicator']['reference'] = new_ref
            for role, record in item['records'].items():
                record['external_id_present'] = True
                record['fields'] = {key: {'origin': 'record', 'pointer': value,
                    'transform': 'reference' if key == 'source_reference' else 'value'} for key, value in record['fields'].items()}
                record['context'] = {'reference': new_ref,
                    'opinion_pointer': '' if role == 'adjudicator' else '/original_opinions/' + role,
                    'opinion_reference_pointer': None if role == 'adjudicator' else '/opinion_references/' + role}
            item['review_binding'] = {'status': '/status', 'affirmative_exclusion': '/affirmative_exclusion',
                'missing_facts': '/missing_facts', 'privacy_reference': privacy_ref}
            row = next(row for row in self.records if row['example_id'] == pair['identities']['example_id'])
            if row['split'] != 'train' or diagnostic:
                fit = (row['split'] == 'validation' and pair['status'] != 'unresolved' and
                       not masked and not diagnostic and not exclude_fit and pair['runtime_id'] in row['input']['candidate_ids'] and
                       pair['runtime_id'] not in ('unslop', 'emil-unslop-writing'))
                pair['purposes'] = {'calibration_fit': fit, 'diagnostic': diagnostic,
                    'evaluation': row['split'] in ('temporal_test', 'unseen_family_test') and not diagnostic,
                    'fit_exclusions': ['fixture-exclusion'] if exclude_fit else []}
                purpose_ref = self.put('source/purpose-' + pair['pair_id'][7:] + '.json', {
                    'case': row['example_id'], 'split': row['split'], 'fit': fit,
                    'exclusions': pair['purposes']['fit_exclusions'], 'mask': masked,
                    'status': pair['status'], 'diagnostic': diagnostic})
                item['purpose_binding'] = {'reference': purpose_ref,
                    **{key + '_pointer': '/' + value for key, value in (
                        ('fit', 'fit'), ('exclusions', 'exclusions'), ('mask', 'mask'),
                        ('status', 'status'), ('diagnostic', 'diagnostic'), ('case', 'case'), ('split', 'split'))}}
        self.schema = ROOT / 'schemas/contextual-annotation-v2.schema.json'
        self.seal()

    def compile(self, events):
        # Execute the shared host serializer. No copied first/last-message implementation.
        runner = self.root / 'host-prefix.ts'
        runner.write_text('import {compileIntent} from ' + json.dumps(str(ROOT / 'src/session.ts')) + ';\n'
                          'console.log(JSON.stringify(compileIntent(await Bun.stdin.json())));\n')
        messages = []
        for e in events:
            if e['role'] == 'user':
                messages.append({'role': 'user', 'content': e['content'] if 'content' in e else e['text']})
            elif e['role'] in ('assistant', 'toolResult'):
                messages.append({'role': e['role'], 'content': e['content'] if 'content' in e else
                                 [{'type': 'text', 'text': e['text']}]})
        process = subprocess.run(["bun", str(runner)],
            input=json.dumps(messages), capture_output=True, text=True, timeout=30)
        assert process.returncode == 0, process.stderr
        runner.unlink()
        return json.loads(process.stdout)

    def put(self, name, value, raw=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        return {'path': name, 'sha256': d.file_digest(path), 'json_pointer': ''}

    def seal(self):
        for pair in self.pairs:
            pair.pop('opinions', None)
            for name in ('request_quote', 'body_quote'):
                pair[name].pop('quote', None)
        for item in self.mapping['pairs']:
            if item['quote_binding'] is not None:
                for name in ('raw_quote', 'compiled_quote'):
                    item['quote_binding'][name].pop('quote', None)
        self.put('source/dataset.jsonl', ''.join(d.canonical_bytes(r).decode() + '\n' for r in self.records), raw=True)
        identity_ref = self.put('source/identity-map.json', self.mapping)
        annotation = {'format': 'contextual-annotation-v2', 'target_version': 'contextual-training-alignment-v1',
            'target_sha256': self.target['sha256'], 'schema_sha256': d.file_digest(self.schema), 'pairs': self.pairs}
        self.put('annotation.json', annotation)
        receipt = {'fixture_only': True, 'proposed_pair_allowlist': sorted(p['pair_id'] for p in self.pairs if p['disposition'] == 'eligible_resolved'),
            'excluded_dispositions': {x: sum(p['disposition'] == x for p in self.pairs) for x in ('unresolved', 'masked', 'ineligible', 'diagnostic')}, 'target_sha256': self.target['sha256'], 'rubric_sha256': self.rubric['sha256'],
            'target_reference': {k: self.target[k] for k in ('path', 'sha256')},
            'rubric_reference': {k: self.rubric[k] for k in ('path', 'sha256')},
            'schema_sha256': d.file_digest(self.schema), 'annotation_sha256': d.file_digest(self.root / 'annotation.json'),
            'source_hashes': {str(p.relative_to(self.root)): d.file_digest(p) for folder in ('source', 'audit')
                for p in (self.root / folder).rglob('*') if p.is_file()},
            'pairs': [{k: p[k] for k in ('pair_id', 'runtime_id', 'identities', 'request_sha256', 'body_sha256',
                'disposition', 'status', 'privacy_evidence', 'opinion_references')} for p in self.pairs],
            'role_plan_sha256': d.file_digest(self.root / 'source/role-plan.json'), 'identity_map_sha256': identity_ref['sha256'],
            'dataset_schema_sha256': d.file_digest(ROOT / 'schemas/dataset.schema.json'),
            'source_contract_sha256': d.file_digest(ROOT / 'docs/contextual-source-v2.md')}
        self.put('receipt.json', receipt)
        approval = {'status': 'approved_local_canonical_integration_only', 'fixture_only': True,
            'source_set_execution': {'purpose': 'local-canonical-integration',
                                     'runtime_revision': d.file_digest(ROOT / 'python/build-dataset.py')},
            **{k: receipt[k] for k in ('target_sha256', 'rubric_sha256', 'schema_sha256', 'annotation_sha256',
                'role_plan_sha256', 'identity_map_sha256', 'dataset_schema_sha256', 'source_contract_sha256')},
            'proposed_receipt_sha256': d.file_digest(self.root / 'receipt.json'),
            'approved_pair_ids': [p['pair_id'] for p in self.pairs if p['disposition'] == 'eligible_resolved'],
            'excluded_dispositions': {x: sum(p['disposition'] == x for p in self.pairs) for x in ('unresolved', 'masked', 'ineligible', 'diagnostic')}}
        self.put('fixture-operator-approval.json', approval)
        self.member_pin = d.file_digest(self.root / 'fixture-operator-approval.json')
        receipt = self.root / 'receipt.json'
        self.member_reference, self.member_proof = d.portable.prepare_member(
            receipt, self.root / 'annotation.json', self.schema,
            {name: self.root / name for name in d.read_json(receipt)['source_hashes']})
        self.references, self.pin = source_set(self.root / 'set', [self])
        self.put('set/references.json', self.references)

    def admission(self):
        return d.load_contextual_admission(self.references, self.pin, proof=proof(self.references))

    def cli(self, command, source, output=None):
        args = [sys.executable, '-B', str(ROOT / 'python/build-dataset.py'), command, '--catalog', str(self.root / 'catalog.json'),
            '--audit-dir', str(self.audit), '--input', str(source), '--approval-sha256', self.pin]
        if command in ('freeze', 'validate'):
            args += ['--contextual-admission', str(self.root / 'set/references.json')]
        if output is not None:
            args += ['--output', str(output)]
        process = subprocess.run(args, capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': str(self.root)})
        self.commands.append({'argv': args, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
        assert process.returncode == 0, self.commands[-1]
        return json.loads(process.stdout)

    def export(self):
        frozen, exported = self.root / 'frozen', self.root / 'export'
        self.cli('freeze', Path(self.references['declaration']), frozen)
        self.cli('validate', frozen / 'dataset.jsonl')
        self.cli('export', frozen, exported)
        return exported


def source_set(directory, fixtures):
    directory = Path(directory)
    directory.mkdir(mode=0o700)
    members = []
    objects = {}
    for fixture in fixtures:
        proposal = d.ContextualProposal(**fixture.member_reference, proof=fixture.member_proof)
        objects.update(fixture.member_proof.files)
        raw = (fixture.root / 'fixture-operator-approval.json').read_bytes()
        objects[d.file_digest(raw)] = raw
        identity = d.member_identity(proposal)
        members.append((d.digest(identity), {'identity': identity, 'reference': fixture.member_reference},
                        {'member_sha256': d.digest(identity),
                         'approval': fixture.member_pin,
                         'approval_sha256': fixture.member_pin}))
    members.sort(key=lambda item: item[0])
    declaration = {'format': 'contextual-source-set-v2', 'purpose': 'local-canonical-integration',
                   'fixture_only': True, 'members': [item[1] for item in members]}
    d.write_json(directory / 'declaration.json', declaration)
    declaration_raw = (directory / 'declaration.json').read_bytes()
    declaration_key = d.file_digest(declaration_raw)
    objects[declaration_key] = declaration_raw
    joint = {'format': 'contextual-source-set-approval-v2',
        'status': 'approved_local_canonical_integration_only', 'purpose': declaration['purpose'],
        'fixture_only': True, 'runtime_revision': d.file_digest(ROOT / 'python/build-dataset.py'),
        'declaration_sha256': d.file_digest(directory / 'declaration.json'),
        'source_set_sha256': d.digest({'purpose': declaration['purpose'], 'fixture_only': True,
                                      'members': [item[1]['identity'] for item in members]}),
        'members': [{key: item[2][key] for key in ('member_sha256', 'approval_sha256')} for item in members]}
    d.write_json(directory / 'joint-approval.json', joint)
    joint_raw = (directory / 'joint-approval.json').read_bytes()
    joint_key = d.file_digest(joint_raw)
    objects[joint_key] = joint_raw
    reference = {'declaration': declaration_key, 'approval': joint_key,
                 'member_approvals': [item[2] for item in members]}
    PROOFS[declaration_key] = objects
    d.portable.SourceProof(objects).write(directory / 'proof')
    return reference, joint_key


