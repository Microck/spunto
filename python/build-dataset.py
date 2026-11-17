#!/usr/bin/env python3
"""Local OMP lineage inventory, audited dataset validation, freezing and private export."""
from __future__ import annotations

import argparse
import contextlib
from collections.abc import Mapping
import hashlib
import hmac
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
import unicodedata
import weakref
from datetime import datetime, timezone
from functools import lru_cache

_portable_spec = importlib.util.spec_from_file_location(
    'router_private_inputs', Path(__file__).with_name('private-training-inputs.py'))
portable = importlib.util.module_from_spec(_portable_spec)
_portable_spec.loader.exec_module(portable)

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE = 8 * 1024 * 1024
# Per-record review references grow with the dataset; keep the report within
# the canonical JSON reader's bound rather than discarding audit evidence.
MAX_PRIVACY_REPORT_BYTES = MAX_LINE
BOUNDARIES = ('2026-10-15T00:00:00Z', '2026-10-25T00:00:00Z', '2026-11-05T00:00:00Z')
PAIR_FORMAT = 'intent-action-window-bag-v2'
SANITIZER = {'version': 2, 'mode': 'reject-secrets-review-public-technical-text', 'anonymity_proof': False}
SYNTHETIC_POLICY = {'version': 1, 'seed': 'curation-template-split-v1',
                    'algorithm': 'sha256-canonical-seed-group-first64-mod100',
                    'train_below': 70, 'validation_below': 90}
PAIR_PROJECTION = {'version': 7, 'masked_splits': ['train', 'validation'],
                   'mask': 'holdout-family-all-known-labels',
                   'source_bound_mask': 'retain-original-family-mask-across-role-remapping',
                   'source_bound_purposes': 'pinned-calibration-fit-or-independent-test-only;exclude-diagnostics-ineligible-neutral-unknown',
                   'deduplication': {'identity': ['intent_utf8_bytes', 'action', 'description_utf8_bytes'],
                                     'scope': 'same-canonical-example-and-label', 'order': 'after-family-mask'}}
PAIR_ACCOUNTING_STAGES = ('pre_mask', 'masked', 'projected', 'deduplicated', 'emitted')


class ContractError(ValueError):
    """Content-free diagnostic; never include a source value in an error."""


def require(condition, code):
    if not condition:
        raise ContractError(code)


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'invalid_arguments\n')


def cli_failure(error):
    if isinstance(error, ContractError):
        code = error.args[0] if len(error.args) == 1 else None
        if not isinstance(code, str) or re.fullmatch(r'[a-z][a-z0-9_]{0,95}', code) is None:
            code = 'contract_failure'
        status = 2
    elif isinstance(error, (OSError, sqlite3.Error)):
        code, status = 'local_io_failure', 3
    elif isinstance(error, ImportError):
        code, status = 'dependency_unavailable', 4
    elif isinstance(error, KeyboardInterrupt):
        code, status = 'interrupted', 130
    elif isinstance(error, (ValueError, KeyError, TypeError)):
        code, status = 'invalid_source', 2
    else:
        code, status = 'unexpected_failure', 1
    print(code, file=sys.stderr)
    return status


def private_dir(path):
    path = Path(path)
    if not path.exists():
        private_dir(path.parent)
        path.mkdir(mode=0o700)
    require(not path.is_symlink() and path.is_dir() and stat.S_IMODE(path.stat().st_mode) == 0o700,
            'nonprivate_output_directory')


def immutable_put(path, payload):
    """Same exclusive, byte-idempotent private writer as the sealed proposal producer."""
    path = Path(path)
    private_dir(path.parent)
    if path.exists() or path.is_symlink():
        require(not path.is_symlink() and path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o600,
                'nonprivate_output_file')
        require(path.read_bytes() == payload, 'immutable_output_drift')
        return
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(payload)


def canonical_bytes(value):
    def check(item):
        if item is None or isinstance(item, (str, bool)):
            return
        if type(item) is int and abs(item) <= 9007199254740991:
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            for child in item.values():
                check(child)
            return
        raise ContractError('invalid_digest_value')
    check(value)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except UnicodeError:
        raise ContractError('invalid_unicode') from None


def digest(value):
    return 'sha256:' + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    if isinstance(path, bytes):
        return 'sha256:' + hashlib.sha256(path).hexdigest()
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return 'sha256:' + checksum.hexdigest()


def parse_json(payload, *, source=False):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def constant(_):
        raise ContractError('nonfinite_json')
    try:
        value = json.loads(payload, object_pairs_hook=object_pairs, parse_constant=constant)
        if source:
            # OMP usage records contain floats. They are not canonical digest inputs
            # or exported features; still reject nonfinite values and bad Unicode.
            try:
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            except (ValueError, UnicodeError):
                raise ContractError('invalid_source_json') from None
        else:
            canonical_bytes(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ContractError('invalid_json') from None


def rows(path, *, source=False):
    with io.BytesIO(path) if isinstance(path, bytes) else Path(path).open('rb') as stream:
        while line := stream.readline(MAX_LINE + 1):
            require(len(line) <= MAX_LINE, 'line_capacity')
            yield parse_json(line, source=source)


def read_json(path, *, source=False):
    maximum = 64 * 1024 * 1024 if source else MAX_LINE
    if isinstance(path, bytes):
        raw = path
    else:
        require(Path(path).stat().st_size <= maximum, 'document_capacity')
        with Path(path).open('rb') as stream:
            raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'document_capacity')
    return parse_json(raw, source=source)


def timestamp(value):
    require(isinstance(value, str) and value.endswith('Z'), 'invalid_timestamp')
    try:
        return datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError:
        raise ContractError('invalid_timestamp') from None


def temporal_split(value):
    instant = timestamp(value)
    for boundary, split in zip(BOUNDARIES, ('train', 'validation', 'temporal_test')):
        if instant < timestamp(boundary):
            return split
    return 'quarantine'


def synthetic_split(group):
    bucket = int(digest([SYNTHETIC_POLICY['seed'], group])[7:23], 16) % 100
    if bucket < SYNTHETIC_POLICY['train_below']:
        return 'train'
    return 'validation' if bucket < SYNTHETIC_POLICY['validation_below'] else 'challenge'


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def normalized_model_input(inputs):
    """Match the scorer's rendered first sequence, including literal action text."""
    rendered = inputs['request']
    if inputs['action'] is not None:
        rendered += '\nAction: ' + inputs['action']['tool_name'] + '\n' + inputs['action']['summary']
    return normalized(rendered)


def private_id(key, domain, value):
    return 'hmac:' + hmac.new(key, canonical_bytes([domain, value]), hashlib.sha256).hexdigest()


def load_key(path):
    path = Path(path)
    if not path.exists():
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
            stream.write(os.urandom(32))
    require(not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600, 'key_permissions')
    key = path.read_bytes()
    require(len(key) == 32, 'key_length')
    return key


def new_file(path):
    parent = Path(path).parent
    require(parent.is_dir() and not parent.is_symlink() and stat.S_IMODE(parent.stat().st_mode) == 0o700,
            'nonprivate_output_directory')
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb')


def write_json(path, value):
    with new_file(path) as stream:
        stream.write(canonical_bytes(value) + b'\n')


@lru_cache(maxsize=4)
def validators(schema_dir=ROOT / 'schemas'):
    from jsonschema import Draft202012Validator, FormatChecker
    from referencing import Registry, Resource
    from referencing.exceptions import NoSuchResource
    def refuse(uri):
        raise NoSuchResource(ref=uri)
    registry = Registry(retrieve=refuse)
    schemas = {}
    for name in ('catalog', 'dataset', 'routing'):
        schema = read_json(Path(schema_dir) / (name + '.schema.json'))
        registry = registry.with_resource(schema['$id'], Resource.from_contents(schema))
        schemas[name] = schema
    return {name: Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()) for name, schema in schemas.items()}


def schema_check(validator, value):
    try:
        require(not any(validator.iter_errors(value)), 'schema_mismatch')
    except ContractError:
        raise
    except Exception:
        raise ContractError('schema_reference_failure') from None


def validate_catalog(catalog, validator):
    schema_check(validator, catalog)
    entries = catalog['entries']
    ids = [entry['runtime_id'] for entry in entries]
    require(ids == sorted(set(ids)), 'catalog_order_or_duplicate')
    require(catalog['catalog_sha256'] == digest({k: catalog[k] for k in ('schema_version', 'loader_contract', 'eligibility_sha256', 'entries')}), 'catalog_digest')
    for entry in entries:
        require(entry['routing_sha256'] == digest({k: entry[k] for k in ('runtime_id', 'description', 'hidden', 'automatic_eligible', 'explicit_eligible', 'content_sha256')}), 'routing_digest')
    return {entry['runtime_id']: entry for entry in entries}


# Hard rejection never redacts input. Ambiguous public technical text requires
# independent review of the exact closed input before a passed scan is claimed.
HIGH_RISK = re.compile(r'(?i)([\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\d{1,3}\.){3}\d{1,3}|(?:/home/|/Users/|/tmp/|[a-z]:\\)|-----BEGIN|```|<\s*(?:html|script|thinking)|\b(?:sk|ghp|gho|AKIA)[_-]?[a-z0-9]{12,}|\b[a-f0-9]{32,}\b|\bbearer\s+[a-z0-9._~+/-]{12,}|\b(?:password|passwd|api[_ -]?key|secret|token|private[_ -]?key)\s*[:=]\s*["\']?[^\s"\']+)')
REVIEW_PATTERNS = {
    'public_url_or_domain': re.compile(r'(?i)(https?://|www\.|\b[a-z0-9.-]+\.(?:com|dev|org|net|io)\b)'),
    'credential_vocabulary': re.compile(r'(?i)\b(?:bearer|password|passwd|api[_ -]?key|secret|token|private[_ -]?key)\b'),
}


def safe_text(text):
    require(isinstance(text, str) and bool(text.strip()), 'empty_abstraction')
    require(not HIGH_RISK.search(text), 'privacy_high_risk')
    require(not any(ord(c) < 32 and c not in '\n\t' for c in text), 'privacy_control')
    require(not any(unicodedata.category(c) in ('Cc', 'Cf', 'Cs') and c not in '\n\t' for c in text), 'privacy_unicode')
    return [name for name, pattern in REVIEW_PATTERNS.items() if pattern.search(text)]


def scan_record(row, entries):
    fields = [('input.request', row['input']['request'])]
    fields.extend((f'input.evidence.{i}.text', evidence['text']) for i, evidence in enumerate(row['input']['evidence']))
    if row['input']['action'] is not None:
        fields.append(('input.action.tool_name', row['input']['action']['tool_name']))
        # Native tool actions have no inferred summary. Only this exact empty field is optional.
        if row['input']['action']['summary'] != '':
            fields.append(('input.action.summary', row['input']['action']['summary']))
    # Both auditors see the full catalog, including noncandidate descriptions.
    fields.extend((f'catalog.{i}.description', entries[skill]['description']) for i, skill in enumerate(sorted(entries)))
    return [{'field': field, 'category': category} for field, text in fields for category in safe_text(text)]


def open_index(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'missing_index')
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    marker = connection.execute("SELECT value FROM metadata WHERE key='version'").fetchone()
    if marker is None or marker[0] != '2':
        connection.close()
        raise ContractError('incomplete_inventory')
    return connection


def join_groups(db, left, right):
    for node in (left, right):
        db.execute('INSERT OR IGNORE INTO components VALUES (?,?)', (node, node))
    a = db.execute('SELECT component FROM components WHERE node=?', (left,)).fetchone()[0]
    b = db.execute('SELECT component FROM components WHERE node=?', (right,)).fetchone()[0]
    if a != b:
        db.execute('UPDATE components SET component=? WHERE component=?', (min(a, b), max(a, b)))


def omp_records(path):
    """Read native framing without renumbering physical lines or retaining titles."""
    with Path(path).open('rb') as stream:
        line = stream.readline(MAX_LINE + 1)
        require(len(line) <= MAX_LINE, 'line_capacity')
        record = parse_json(line, source=True)
        sequence = 1
        if isinstance(record, dict) and record.get('type') == 'title':
            require(set(record) in ({'type', 'v', 'title', 'updatedAt', 'pad'},
                                    {'type', 'v', 'title', 'updatedAt', 'pad', 'source'}),
                    'source_preamble_fields')
            require(type(record['v']) is int and record['v'] == 1
                    and all(isinstance(record[name], str) for name in ('title', 'updatedAt', 'pad'))
                    and record.get('source', 'auto') in ('auto', 'user')
                    and not record['pad'].strip(' ') and len(line) == 256 and line.endswith(b'\n'),
                    'source_preamble_framing')
            timestamp(record['updatedAt'])
            line = stream.readline(MAX_LINE + 1)
            require(len(line) <= MAX_LINE, 'line_capacity')
            record = parse_json(line, source=True)
            sequence = 2
        require(isinstance(record, dict) and record.get('type') == 'session'
                and type(record.get('version')) is int and record['version'] == 3,
                'unsupported_source_header')
        yield sequence, record
        while line := stream.readline(MAX_LINE + 1):
            sequence += 1
            require(len(line) <= MAX_LINE, 'line_capacity')
            record = parse_json(line, source=True)
            require(isinstance(record, dict) and record.get('type') not in ('title', 'session'),
                    'source_event_framing')
            yield sequence, record


def inventory(source_root, output, key_path, excluded_project, links=None):
    """Stream source records into a task-owned SQLite index; no raw text is stored."""
    require(Path(source_root).is_dir(), 'source_root_missing')
    key = load_key(key_path)
    with new_file(output):
        pass
    db = sqlite3.connect(output)
    try:
        db.executescript('''
          CREATE TABLE components(node TEXT PRIMARY KEY, component TEXT NOT NULL);
          CREATE INDEX component_index ON components(component);
          CREATE TABLE sessions(id TEXT PRIMARY KEY, excluded INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE files(path_key TEXT PRIMARY KEY, session TEXT, sha256 TEXT, parent_ref TEXT);
          CREATE TABLE events(id TEXT PRIMARY KEY, session TEXT, parent TEXT, sequence INTEGER,
            observed_at TEXT, kind TEXT, fingerprint TEXT, raw_id_key TEXT);
          CREATE INDEX event_session ON events(session);
          CREATE INDEX event_fingerprint ON events(fingerprint);
          CREATE INDEX event_raw_id ON events(raw_id_key);
          CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
          CREATE TABLE private_map(key TEXT PRIMARY KEY, kind TEXT, value TEXT);
          CREATE TABLE lineage(session TEXT PRIMARY KEY, component TEXT, lineage_sha256 TEXT, split TEXT);
          CREATE TABLE calls(call_key TEXT, event_id TEXT, PRIMARY KEY(call_key,event_id));
          CREATE TABLE results(event_id TEXT PRIMARY KEY, call_key TEXT);
        ''')
        count = 0
        excluded = Path(excluded_project).resolve()
        for base, directories, filenames in os.walk(source_root, followlinks=False):
            directories.sort()
            for filename in sorted(filenames):
                if not filename.endswith('.jsonl'):
                    continue
                path = Path(base) / filename
                require(not path.is_symlink(), 'source_symlink')
                before = file_digest(path)
                stream = iter(omp_records(path))
                _, header = next(stream)
                require(isinstance(header.get('id'), str) and header['id'], 'source_session_id')
                sid = private_id(key, 'session', header['id'])
                db.execute('INSERT OR IGNORE INTO private_map VALUES (?,?,?)', (sid, 'session', header['id']))
                db.execute('INSERT OR IGNORE INTO private_map VALUES (?,?,?)', (private_id(key, 'path', str(path.resolve())), 'path', str(path.resolve())))
                cwd = header.get('cwd')
                reject = temporal_split(header.get('timestamp')) == 'quarantine'
                if not isinstance(cwd, str):
                    reject = True
                else:
                    source_cwd = Path(cwd).resolve()
                    reject |= source_cwd.is_relative_to(excluded) or source_cwd.is_relative_to(ROOT)
                db.execute('INSERT OR IGNORE INTO sessions VALUES (?,?)', (sid, int(reject)))
                if reject:
                    db.execute('UPDATE sessions SET excluded=1 WHERE id=?', (sid,))
                join_groups(db, sid, sid)
                parent_ref = header.get('parentSession')
                require(parent_ref is None or isinstance(parent_ref, str), 'source_parent_type')
                # OMP parentSession is a source filename. Resolve relative to this file.
                parent_key = None
                if parent_ref:
                    parent_path = Path(parent_ref)
                    if not parent_path.is_absolute():
                        parent_path = path.parent / parent_path
                    parent_key = private_id(key, 'path', str(parent_path.resolve()))
                db.execute('INSERT INTO files VALUES (?,?,?,?)', (private_id(key, 'path', str(path.resolve())), sid, before, parent_key))
                for sequence, event in stream:
                    require(isinstance(event, dict), 'source_event_type')
                    if 'id' not in event:
                        continue
                    require(isinstance(event['id'], str), 'source_event_id')
                    event_id = private_id(key, 'event', [sid, event['id']])
                    parent = event.get('parentId')
                    require(parent is None or isinstance(parent, str), 'source_parent_id')
                    parent_id = private_id(key, 'event', [sid, parent]) if parent else None
                    observed = event.get('timestamp')
                    timestamp(observed)
                    message = event.get('message', {})
                    kind = message.get('role', event.get('type')) if isinstance(message, dict) else event.get('type')
                    if isinstance(message, dict):
                        content = message.get('content', [])
                        if kind == 'assistant' and isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'toolCall':
                                    require(isinstance(block.get('id'), str), 'source_call_id')
                                    db.execute('INSERT OR IGNORE INTO calls VALUES (?,?)', (private_id(key, 'call', [sid, block['id']]), event_id))
                        if kind == 'toolResult':
                            require(isinstance(message.get('toolCallId'), str), 'source_result_call')
                            db.execute('INSERT OR IGNORE INTO results VALUES (?,?)', (event_id, private_id(key, 'call', [sid, message['toolCallId']])))
                    fingerprint = None
                    if kind == 'user':
                        content = message.get('content', '')
                        if isinstance(content, list):
                            content = '\n'.join(block['text'] for block in content if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str))
                        if isinstance(content, str) and content.strip():
                            fingerprint = private_id(key, 'duplicate-text', normalized(content))
                    values = (event_id, sid, parent_id, sequence, observed, kind, fingerprint, private_id(key, 'raw-event-id', event['id']))
                    db.execute('INSERT OR IGNORE INTO private_map VALUES (?,?,?)', (event_id, 'event', event['id']))
                    existing = db.execute('SELECT * FROM events WHERE id=?', (event_id,)).fetchone()
                    if existing:
                        require(existing == values, 'conflicting_source_copy')
                    else:
                        db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?)', values)
                    for other, in db.execute('SELECT DISTINCT session FROM events WHERE (fingerprint=? AND fingerprint IS NOT NULL) OR raw_id_key=?', (fingerprint, values[-1])):
                        join_groups(db, sid, other)
                require(file_digest(path) == before, 'source_changed_during_read')
                count += 1
                db.commit()
        for sid, parent_key in db.execute('SELECT session,parent_ref FROM files WHERE parent_ref IS NOT NULL'):
            parent = db.execute('SELECT session FROM files WHERE path_key=?', (parent_key,)).fetchone()
            if parent:
                join_groups(db, sid, parent[0])
            else:
                db.execute('UPDATE sessions SET excluded=1 WHERE id=?', (sid,))
        if links:
            for link in rows(links):
                require(set(link) == {'left', 'right', 'relation', 'evidence_sha256'}, 'lineage_link_fields')
                require(link['relation'] in ('import', 'resume', 'delegate', 'fork', 'copy'), 'lineage_relation')
                require(re.fullmatch(r'sha256:[0-9a-f]{64}', link['evidence_sha256']) is not None, 'lineage_evidence')
                for node in (link['left'], link['right']):
                    require(db.execute('SELECT 1 FROM sessions WHERE id=?', (node,)).fetchone(), 'unknown_lineage_session')
                join_groups(db, link['left'], link['right'])
        # All source events, not only chosen examples, determine the temporal component.
        for component, in db.execute('SELECT DISTINCT component FROM components ORDER BY component'):
            checksum = hashlib.sha256()
            splits = set()
            rejected = False
            for sid, excluded_flag in db.execute('SELECT s.id,s.excluded FROM sessions s JOIN components c ON c.node=s.id WHERE c.component=? ORDER BY s.id', (component,)):
                checksum.update(canonical_bytes(sid) + b'\n')
                rejected |= bool(excluded_flag)
                for event in db.execute('SELECT id,parent,sequence,observed_at FROM events WHERE session=? ORDER BY sequence,id', (sid,)):
                    checksum.update(canonical_bytes(list(event)) + b'\n')
                    splits.add(temporal_split(event[3]))
                    if event[1] and not db.execute('SELECT 1 FROM events WHERE id=? AND session=? AND sequence<?', (event[1], sid, event[2])).fetchone():
                        rejected = True
            split = next(iter(splits)) if len(splits) == 1 and not rejected else 'quarantine'
            lineage_hash = 'sha256:' + checksum.hexdigest()
            db.execute('INSERT INTO lineage SELECT node,?,?,? FROM components WHERE component=?', (component, lineage_hash, split, component))
        checksum = hashlib.sha256()
        for source in db.execute('SELECT path_key,session,sha256,parent_ref FROM files ORDER BY path_key'):
            checksum.update(canonical_bytes(list(source)) + b'\n')
        require(count > 0, 'empty_source_inventory')
        links_hash = file_digest(links) if links else digest([])
        source_hash = digest({'files_sha256': 'sha256:' + checksum.hexdigest(), 'links_sha256': links_hash})
        for name, value in {'source_manifest_sha256': source_hash, 'links_sha256': links_hash, 'version': '2'}.items():
            db.execute('INSERT INTO metadata VALUES (?,?)', (name, value))
        db.commit()
        return {'files': count, 'events': db.execute('SELECT count(*) FROM events').fetchone()[0], 'source_manifest_sha256': source_hash}
    finally:
        db.close()


def source_check(row, index):
    source = row['source']
    if source['origin'] == 'synthetic':
        require(source['adapter'] == 'synthetic' and source['template_group'] == source['conversation_group'], 'synthetic_template_group')
        timestamp(source['cutoff_utc'])
        return
    require(source['adapter'] == 'omp-v3' and source['template_group'] is None, 'unsupported_adapter')
    require(index is not None, 'source_index_required')
    event = index.execute('SELECT e.*,l.component,l.lineage_sha256,l.split FROM events e JOIN lineage l ON l.session=e.session WHERE e.id=?', (source['event_id'],)).fetchone()
    require(event is not None, 'source_event_missing')
    require(source['conversation_group'] == event['component'] and source['lineage_sha256'] == event['lineage_sha256'], 'source_lineage_mismatch')
    manifest = index.execute("SELECT value FROM metadata WHERE key='source_manifest_sha256'").fetchone()[0]
    require(source['source_manifest_sha256'] == manifest, 'source_manifest_mismatch')
    require(source['event_sequence'] == event['sequence'] and source['cutoff_utc'] == event['observed_at'], 'source_cutoff_mismatch')
    if row['split'] != 'quarantine':
        require(event['split'] != 'quarantine', 'source_lineage_quarantined')
        require(row['split'] in (event['split'], 'unseen_family_test', 'challenge'), 'source_temporal_split')
    # Walk only the active parent chain. Sibling branches and future corrections fail.
    needed = {}
    for item in row['input']['evidence']:
        needed.setdefault(item['source_event_id'], []).append(item)
    current = event
    last_sequence = source['event_sequence'] + 1
    needed_calls = set()
    while current is not None:
        require(current['sequence'] < last_sequence, 'source_parent_cycle')
        last_sequence = current['sequence']
        needed_calls.discard(current['id'])
        if current['id'] in needed:
            for evidence in needed.pop(current['id']):
                require(evidence['event_sequence'] == current['sequence'] and evidence['observed_at'] == current['observed_at'], 'evidence_source_mismatch')
                if evidence['kind'] in ('user_intent', 'user_constraint'):
                    require(current['kind'] == 'user', 'evidence_role_mismatch')
                if evidence['kind'] == 'tool_fact':
                    require(current['kind'] == 'toolResult', 'evidence_role_mismatch')
                    calls = list(index.execute('SELECT c.event_id FROM calls c JOIN results r ON r.call_key=c.call_key WHERE r.event_id=?', (current['id'],)))
                    require(len(calls) == 1, 'tool_call_join')
                    needed_calls.add(calls[0][0])
        if not current['parent']:
            break
        current = index.execute('SELECT * FROM events WHERE id=?', (current['parent'],)).fetchone()
        require(current is not None, 'source_parent_missing')
    require(not needed, 'evidence_outside_prefix')
    require(not needed_calls, 'tool_call_outside_prefix')


def audit_input(row, catalog, candidate_order):
    """Teacher-safe closed input. Never includes observed reads, labels or later history."""
    require(set(candidate_order) == {entry['runtime_id'] for entry in catalog['entries']} and len(candidate_order) == len(catalog['entries']), 'audit_catalog_order')
    entries = {entry['runtime_id']: entry for entry in catalog['entries']}
    return {'schema_version': 1, 'catalog_sha256': row['catalog_sha256'], 'policy_sha256': row['policy_sha256'],
            'source': row['source'], 'event_kind': row['event_kind'],
            'input': {k: v for k, v in row['input'].items() if k != 'already_loaded'},
            'catalog': [entries[key] for key in candidate_order]}


def blob(directory, expected):
    if isinstance(directory, (ContextualProposal, _AdmittedMember)):
        return directory.audit_blob(expected)
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', expected) is not None, 'invalid_blob_digest')
    path = Path(directory) / (expected[7:] + '.json')
    require(path.is_file() and not path.is_symlink(), 'audit_blob_missing')
    value = read_json(path)
    require(digest(value) == expected, 'audit_blob_digest')
    return value


def policy_check(row, audit_dir):
    policy = blob(audit_dir, row['policy_sha256'])
    require(set(policy) == {'schema_version', 'rules_sha256', 'required_ids', 'excluded_ids', 'explicit_ids'} and policy['schema_version'] == 1, 'policy_artifact')
    blob(audit_dir, policy['rules_sha256'])
    for name in ('required_ids', 'excluded_ids', 'explicit_ids'):
        require(policy[name] == sorted(row['input'][name]), 'policy_artifact_mismatch')


def synthetic_check(row, audit_dir):
    source = row['source']
    manifest = blob(audit_dir, source['source_manifest_sha256'])
    require(isinstance(manifest, dict), 'synthetic_manifest')
    if manifest.get('schema_version') == 2:
        synthetic_prefix_check(row, manifest)
        return
    require(set(manifest) == {'schema_version', 'origin', 'template_group', 'events'} and manifest['schema_version'] == 1 and manifest['origin'] == 'synthetic', 'synthetic_manifest')
    require(manifest['template_group'] == source['template_group'] and digest(manifest['events']) == source['lineage_sha256'], 'synthetic_lineage')
    events = {}
    for event in manifest['events']:
        require(set(event) == {'event_id', 'event_sequence', 'observed_at', 'parent_event_id'} and event['event_id'] not in events, 'synthetic_event')
        events[event['event_id']] = event
    current = events.get(source['event_id'])
    require(current is not None and current['event_sequence'] == source['event_sequence'] and current['observed_at'] == source['cutoff_utc'], 'synthetic_cutoff')
    chain = {}
    sequence = source['event_sequence'] + 1
    while current is not None:
        require(type(current['event_sequence']) is int and current['event_sequence'] < sequence, 'synthetic_cycle')
        sequence = current['event_sequence']
        chain[current['event_id']] = current
        parent = current['parent_event_id']
        require(parent is None or parent in events, 'synthetic_parent_missing')
        current = events.get(parent)
    for evidence in row['input']['evidence']:
        event = chain.get(evidence['source_event_id'])
        require(event is not None and event['event_sequence'] == evidence['event_sequence'] and event['observed_at'] == evidence['observed_at'], 'synthetic_evidence_prefix')


def synthetic_prefix_check(row, manifest):
    """Verify a committed synthetic prefix, not the truth or review of its premises."""
    source = row['source']
    ordered = manifest.get('serialization') == 'ordered-message-content-join-v1'
    require(set(manifest) == {'schema_version', 'origin', 'template_group', 'serialization', 'request_sha256', 'action_event_id', 'events'}
            and manifest['schema_version'] == 2 and manifest['origin'] == 'synthetic'
            and (manifest['serialization'] == 'omp-18.16.0-compile-intent-v1' or
                 (ordered and authored_contextual(row) and 'projection' in row['authored'])), 'synthetic_prefix_manifest')
    require(manifest['template_group'] == source['template_group']
            and digest(manifest['events']) == source['lineage_sha256'], 'synthetic_lineage')
    events = manifest['events']
    require(isinstance(events, list) and bool(events), 'synthetic_prefix_events')
    by_id, parts = {}, []
    source_offsets, serialized_bytes = {}, 0
    identity_pattern = r'(?!hmac:)[^\x00-\x1f\x7f]{1,256}' if authored_contextual(row) else r'hmac:[0-9a-f]{64}'
    parent, last_sequence, last_time, last_evidence = None, -1, None, ''
    for event in events:
        require(isinstance(event, dict) and set(event) == {'event_id', 'event_sequence', 'observed_at', 'parent_event_id',
                'role', 'text', 'action', 'tool_call_event_id'}, 'synthetic_prefix_event')
        identifier, role, text = event['event_id'], event['role'], event['text']
        require(isinstance(identifier, str) and re.fullmatch(identity_pattern, identifier) is not None
                and identifier not in by_id, 'synthetic_prefix_identity')
        instant = timestamp(event['observed_at'])
        require(event['parent_event_id'] == parent and type(event['event_sequence']) is int
                and event['event_sequence'] > last_sequence and (last_time is None or instant >= last_time), 'synthetic_prefix_order')
        require(role in ('user', 'assistant', 'toolCall', 'toolResult', 'proposed_action', 'compactionSummary', 'branchSummary')
                and isinstance(text, str), 'synthetic_prefix_role')
        if text:
            safe_text(text)
        if role in ('toolCall', 'proposed_action'):
            schema_check(validators()['dataset'].evolve(schema={'$ref': '#/properties/input/properties/action'}), event['action'])
            require(event['action'] is not None and text == '', 'synthetic_prefix_action')
            safe_text(event['action']['tool_name'])
            if event['action']['summary'] != '':
                safe_text(event['action']['summary'])
        else:
            require(event['action'] is None, 'synthetic_prefix_action')
        if role == 'toolResult':
            require(isinstance(event['tool_call_event_id'], str), 'synthetic_prefix_tool_join')
            call = by_id.get(event['tool_call_event_id'])
            require(call is not None and call['role'] == 'toolCall', 'synthetic_prefix_tool_join')
        else:
            require(event['tool_call_event_id'] is None, 'synthetic_prefix_tool_join')
        if ordered or (role == 'user' and text) or role in ('compactionSummary', 'branchSummary'):
            source_offsets[identifier] = serialized_bytes
            serialized_bytes += len(text.encode()) + 2
            parts.append(text)
        elif role in ('assistant', 'toolResult'):
            last_evidence = text
        by_id[identifier] = event
        parent, last_sequence, last_time = identifier, event['event_sequence'], instant
    require(parent == source['event_id'] and last_sequence == source['event_sequence']
            and events[-1]['observed_at'] == source['cutoff_utc'], 'synthetic_cutoff')
    if last_evidence and not ordered:
        parts.append('Last step evidence (not policy):\n' + last_evidence)
    request = '\n\n'.join(parts)
    require(len(request.encode('utf-16-le')) // 2 <= 32768, 'synthetic_prefix_capacity')
    require(request == row['input']['request'] and digest(request) == manifest['request_sha256'], 'synthetic_prefix_request')
    action_id = manifest['action_event_id']
    require(action_id is None or isinstance(action_id, str), 'synthetic_prefix_action_binding')
    action_event = by_id.get(action_id)
    if action_id is not None:
        require(action_event is not None and action_event['role'] == 'proposed_action', 'synthetic_prefix_action_binding')
    action = action_event['action'] if action_event is not None else None
    if ordered:
        # Separate source-owned actions are bound by validate_projection, not
        # invented message events. Every retained message remains unchanged.
        require(action_id is None and all(event['action'] is None for event in events),
                'synthetic_prefix_action_binding')
    else:
        require(action == row['input']['action'], 'synthetic_prefix_action_binding')
    evidence_roles = {'user_intent': ('user',), 'user_constraint': ('user',), 'tool_fact': ('toolResult',),
                      'phase': ('user',), 'restore_fact': ('compactionSummary', 'branchSummary')}
    for evidence in row['input']['evidence']:
        event = by_id.get(evidence['source_event_id'])
        require(event is not None and event['role'] in evidence_roles[evidence['kind']], 'evidence_role_mismatch')
        require(event['text'] == evidence['text'] and event['event_sequence'] == evidence['event_sequence']
                and event['observed_at'] == evidence['observed_at'], 'synthetic_evidence_prefix')
    return source_offsets


def description_span_reference(entry, start, end, quote, kind):
    """Encode an exact reviewed scope; callers supply meaning, never infer it from polarity."""
    require(kind in ('trigger', 'exclusion'), 'description_span_kind')
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(entry['description'])
            and entry['description'][start:end] == quote, 'description_span_source')
    binding = {'runtime_id': entry['runtime_id'], 'contract_sha256': entry['content_sha256'],
               'description_sha256': digest(entry['description']), 'start': start, 'end': end, 'quote': quote}
    return kind + ':description-span.' + digest(binding)[7:], binding


BLINDED_ROLES = frozenset(('teacher_a', 'teacher_b', 'reaudit'))
AUDIT_ROLES = BLINDED_ROLES | {'adjudicator', 'coordinator'}


def reconciliation_binding(proof, fingerprint, reviews):
    """Shared report boundary: validate typed proof and its retained review references."""
    if proof is None:
        require(fingerprint is None and not any(r['role'] == 'coordinator' for r in reviews), 'reconciliation_required')
        return
    check = validators()['dataset']
    schema_check(check.evolve(schema={'$ref': check.schema['$id'] + '#/$defs/reconciliation'}), proof)
    require(digest(proof) == fingerprint, 'reconciliation_digest')
    roles = {review['role']: review for review in reviews}
    require({'teacher_a', 'teacher_b', 'adjudicator', 'coordinator'} <= roles.keys(), 'reconciliation_roles')
    for role, field in (('teacher_a', 'teacher_a_output_sha256'), ('teacher_b', 'teacher_b_output_sha256'),
                        ('adjudicator', 'base_output_sha256'), ('coordinator', 'coordinator_output_sha256')):
        require(roles[role]['output_sha256'] == proof[field], 'reconciliation_output_binding')
    require(proof['reaudit_output_sha256'] == (roles['reaudit']['output_sha256'] if 'reaudit' in roles else None), 'reconciliation_reaudit_binding')
    require(roles['adjudicator']['prompt_sha256'] == roles['coordinator']['prompt_sha256'] == proof['input_sha256']
            and roles['coordinator']['settings_sha256'] == proof['coordinator_settings_sha256'], 'reconciliation_input_binding')
    require(proof['conflicting_ids'] == sorted(set(proof['conflicting_ids'])), 'reconciliation_conflict_order')


def review_artifacts(reviewer, audit_dir):
    role = reviewer['role']
    require(role in AUDIT_ROLES, 'reviewer_role')
    settings = blob(audit_dir, reviewer['settings_sha256'])
    require(set(settings) == {'model', 'model_revision', 'model_family', 'parameters', 'pass_id', 'blinded', 'tools_disabled'}, 'reviewer_settings_fields')
    require(settings['model'] == reviewer['model'] and settings['model_revision'] == reviewer['model_revision']
            and (reviewer['model_revision'] is None or bool(reviewer['model_revision']))
            and isinstance(settings['model_family'], str) and bool(settings['model_family'])
            and isinstance(settings['pass_id'], str) and bool(settings['pass_id'])
            and isinstance(settings['parameters'], dict) and settings['blinded'] is (role in BLINDED_ROLES)
            and type(settings['tools_disabled']) is bool, 'reviewer_settings')
    prompt = blob(audit_dir, reviewer['prompt_sha256'])
    require(set(prompt) == {'schema_version', 'catalog_sha256', 'policy_sha256', 'source', 'event_kind', 'input', 'catalog'}, 'audit_prompt_fields')
    output = blob(audit_dir, reviewer['output_sha256'])
    fields = {'schema_version', 'prompt_sha256', 'label', 'privacy_approved', 'abstraction_supported', 'full_catalog_reviewed'}
    if role == 'coordinator':
        fields.add('semantic_review')
        require(output.get('semantic_review') is False, 'coordinator_not_semantic_review')
    require(set(output) == fields, 'audit_output_fields')
    require(output['schema_version'] == 1 and output['prompt_sha256'] == reviewer['prompt_sha256']
            and output['full_catalog_reviewed'] is True
            and type(output['privacy_approved']) is bool and type(output['abstraction_supported']) is bool, 'audit_output_binding')
    return reviewer, settings, prompt, output


def candidate_judgments(label, candidates):
    judgments = {}
    for field in ('selected_ids', 'verified_negative_ids', 'unknown_ids'):
        for skill in label[field]:
            require(skill not in judgments and skill in candidates, 'reconciliation_base_partition')
            judgments[skill] = field
    require(judgments.keys() == candidates, 'reconciliation_base_partition')
    return judgments


def conservative_label(base, conflicts, coverage, translations):
    """Derive the only allowed revision, preserving known supports and unresolved labels."""
    require(not base['acceptable_alternative_sets'], 'reconciliation_alternatives')
    label = dict(base)
    for field in ('selected_ids', 'verified_negative_ids'):
        label[field] = sorted(set(base[field]) - conflicts)
    label['unknown_ids'] = sorted(set(base['unknown_ids']) | conflicts)
    label['activate_now'] = label['selected_ids']
    label['support'] = []
    for support in base['support']:
        if support['runtime_id'] in conflicts:
            continue
        support = dict(support)
        if support['clause_ref'].startswith('description-scope:'):
            identity = {'runtime_id': support['runtime_id'], 'contract_sha256': support['contract_sha256'],
                        'original_clause_ref': support['clause_ref']}
            reference = 'trigger:description-scope.' + digest(identity)[7:]
            require(translations.get(reference) == identity, 'reconciliation_clause_translation')
            support['clause_ref'] = reference
        label['support'].append(support)
    label['decision'] = 'selected' if label['selected_ids'] else ('uncertain' if label['unknown_ids'] or coverage != 'full' else 'none')
    if not label['selected_ids']:
        label['earliest_supported_event'] = None
        label['activation_deadline_event'] = None
    return label


def reconciliation_check(row, catalog, audit_dir, by_role, proof):
    require(proof['example_id'] == row['example_id'], 'reconciliation_example')
    association = blob(audit_dir, proof['base_association_sha256'])
    expected = {'example_id': row['example_id'], 'output_sha256': proof['base_output_sha256'],
                'prompt_sha256': proof['input_sha256'],
                'teacher_a_output_sha256': proof['teacher_a_output_sha256'],
                'teacher_b_output_sha256': proof['teacher_b_output_sha256'],
                'original_teacher_a_prompt_sha256': by_role['teacher_a'][0]['prompt_sha256'],
                'original_teacher_b_prompt_sha256': by_role['teacher_b'][0]['prompt_sha256']}
    require(all(association.get(key) == value for key, value in expected.items()), 'reconciliation_base_association')
    base = by_role['adjudicator'][3]['label']
    candidates = set(row['input']['candidate_ids'])
    judgments = candidate_judgments(base, candidates)
    conflicts = set()
    if 'reaudit' in by_role:
        revised = candidate_judgments(by_role['reaudit'][3]['label'], candidates)
        conflicts = {skill for skill in candidates if judgments[skill] != revised[skill]}
    require(proof['conflicting_ids'] == sorted(conflicts), 'reconciliation_conflicts')
    expected_label = conservative_label(base, conflicts, row['input']['candidate_coverage'],
                                       blob(audit_dir, proof['clause_translation_sha256']))
    require(by_role['coordinator'][3]['label'] == expected_label, 'reconciliation_label')
    group = blob(audit_dir, proof['group_evidence_sha256'])
    require(set(group) == {'schema_version', 'group_id', 'example_ids', 'unsupported_reviews'}
            and group['schema_version'] == 1 and group['group_id'] == row['source']['conversation_group']
            and group['example_ids'] == sorted(set(group['example_ids']))
            and row['example_id'] in group['example_ids'] and isinstance(group['unsupported_reviews'], list), 'reconciliation_group')
    unsupported = set()
    for reviewer in group['unsupported_reviews']:
        require(reviewer['role'] == 'reaudit', 'reconciliation_source_role')
        review = review_artifacts(reviewer, audit_dir)
        require(review[2]['source']['conversation_group'] == group['group_id']
                and review[2]['catalog_sha256'] == row['catalog_sha256']
                and review[3]['abstraction_supported'] is False, 'reconciliation_source_evidence')
        require(reviewer['output_sha256'] not in unsupported, 'reconciliation_source_duplicate')
        unsupported.add(reviewer['output_sha256'])
    if 'reaudit' in by_role and not by_role['reaudit'][3]['abstraction_supported']:
        require(by_role['reaudit'][0]['output_sha256'] in unsupported, 'reconciliation_source_omitted')
    require(by_role['coordinator'][3]['abstraction_supported'] is (not unsupported), 'reconciliation_source_disposition')
    if unsupported:
        require(row['audit']['status'] == 'quarantined' and row['split'] == 'quarantine'
                and not row['privacy']['scan_passed'] and not row['privacy']['export_approved'], 'unsupported_group_quarantine')
    else:
        require(row['audit']['status'] == 'machine_audited'
                and all(review[3]['abstraction_supported'] for review in by_role.values()), 'audit_output_binding')


def audit_check(row, catalog, audit_dir):
    audit = row['audit']
    require(audit['status'] in ('machine_audited', 'quarantined') and audit['full_catalog_reviewed'], 'audit_pending')
    policy_check(row, audit_dir)
    if row['source']['origin'] == 'synthetic':
        synthetic_check(row, audit_dir)
    by_role = {}
    for reviewer in audit['reviewers']:
        role = reviewer['role']
        require(role not in by_role, 'reviewer_identity')
        require(all(reviewer['output_sha256'] != prior[0]['output_sha256']
                    for prior_role, prior in by_role.items() if {role, prior_role} == {'teacher_a', 'teacher_b'}),
                'audit_not_independent')
        review = review_artifacts(reviewer, audit_dir)
        settings, prompt, output = review[1:]
        require(all(settings['pass_id'] != prior[1]['pass_id'] for prior in by_role.values()), 'audit_not_independent')
        # The stored prompt preserves each reviewer's actual full-catalog order.
        order = [entry['runtime_id'] for entry in prompt['catalog']]
        require(prompt == audit_input(row, catalog, order), 'audit_prompt_mismatch')
        by_role[role] = review
    require({'teacher_a', 'teacher_b'} <= set(by_role), 'independent_audits_required')
    a, b = by_role['teacher_a'], by_role['teacher_b']
    require(a[1]['pass_id'] != b[1]['pass_id'], 'audit_not_independent')
    if len(catalog['entries']) > 1:
        require(a[2]['catalog'] != b[2]['catalog'], 'audit_order_not_shuffled')
    disagreement = a[3]['label'] != b[3]['label']
    require(audit['disagreement'] == disagreement, 'audit_disagreement_mismatch')
    proof = blob(audit_dir, audit['reconciliation_sha256']) if audit['reconciliation_sha256'] is not None else None
    reconciliation_binding(proof, audit['reconciliation_sha256'], audit['reviewers'])
    if disagreement or proof is not None:
        require('adjudicator' in by_role, 'adjudication_required')
        require(audit['adjudication_sha256'] == by_role['adjudicator'][0]['output_sha256'], 'adjudication_digest')
        accepted = by_role['adjudicator']
    else:
        accepted = a
        require(audit['adjudication_sha256'] is None and 'adjudicator' not in by_role, 'unexpected_adjudication')
    if proof is not None:
        reconciliation_check(row, catalog, audit_dir, by_role, proof)
        accepted = by_role['coordinator']
    else:
        require(audit['status'] == 'machine_audited'
                and all(review[3]['abstraction_supported'] for review in by_role.values()), 'audit_output_binding')
    expected_label = dict(row['label'], activate_now=sorted(row['label']['selected_ids']))
    require(accepted[3]['label'] == expected_label, 'audit_label_mismatch')
    if row['privacy']['export_approved'] or (row['privacy']['scan_passed'] and scan_record(row, {entry['runtime_id']: entry for entry in catalog['entries']})):
        require(all(review[3]['privacy_approved'] is True for review in by_role.values()), 'privacy_audit_missing')
    return {'same_family': a[1]['model_family'] == b[1]['model_family'],
            'reviews': [{'role': role, 'settings': review[1],
                         **{key: review[0][key] for key in ('settings_sha256', 'prompt_sha256', 'output_sha256')}}
                        for role, review in sorted(by_role.items())]}


AUTHORED_CONTEXTUAL = 'authored_contextual_v2'


def authored_contextual(row):
    return row['source'].get('provenance_kind') == AUTHORED_CONTEXTUAL


class ContextualProposal:
    """Validate source-bound proposals without granting admission authority."""

    authorized = False

    def __init__(self, receipt, annotation, schema, evidence_map, *, proof):
        self.proof = proof
        reference = {'receipt': receipt, 'annotation': annotation, 'schema': schema,
                     'evidence_map': evidence_map}
        receipt, annotation, schema = (proof.read(key) for key in (receipt, annotation, schema))
        self.receipt = read_json(receipt, source=True)
        self.annotation = read_json(annotation, source=True)
        annotation_schema = read_json(schema)
        self.binding = {
            'target_sha256': self.receipt['target_sha256'],
            'rubric_sha256': self.receipt['rubric_sha256'],
            'schema_sha256': file_digest(schema),
            'annotation_sha256': file_digest(annotation),
            'receipt_sha256': file_digest(receipt),
            'approval_sha256': None,
        }
        self.references = reference
        for key in ('target_sha256', 'schema_sha256', 'annotation_sha256'):
            require(self.binding[key] == self.receipt[key], 'contextual_approval_binding')
        require(self.annotation['target_sha256'] == self.binding['target_sha256'] and
                self.annotation['schema_sha256'] == self.binding['schema_sha256'], 'contextual_target_binding')
        require(self.annotation.get('format') == 'contextual-annotation-v2',
                'contextual_current_representation_required')
        for key, path in (
                ('dataset_schema_sha256', ROOT / 'schemas/dataset.schema.json'),
                ('source_contract_sha256', ROOT / 'docs/contextual-source-v2.md')):
            require(self.receipt.get(key) == file_digest(path),
                    'contextual_source_contract_pin')
        require(file_digest(schema) == file_digest(ROOT / 'schemas/contextual-annotation-v2.schema.json'),
                'contextual_authored_schema')
        for key in ('role_plan_sha256', 'identity_map_sha256', 'rubric_sha256'):
            require(self.receipt.get(key) is not None, 'contextual_operator_pin')
        self.evidence = {}
        self._documents = {}
        mapping = proof.json(evidence_map)
        require(isinstance(mapping, dict) and mapping == self.receipt['source_hashes'],
                'contextual_evidence_map')
        for relative, expected in mapping.items():
            portable.safe_name(relative)
            self.evidence[relative] = proof.read(expected)
        for name in ('target', 'rubric'):
            reference = self.receipt[name + '_reference']
            require(set(reference) == {'path', 'sha256'} and
                    reference['sha256'] == self.binding[name + '_sha256'] and
                    self.receipt['source_hashes'].get(reference['path']) == reference['sha256'] and
                    reference['path'] in self.evidence, 'contextual_target_source')
        maps = [path for relative, path in self.evidence.items()
                if self.receipt['source_hashes'][relative] == self.receipt['identity_map_sha256']]
        require(len(maps) == 1, 'contextual_identity_map')
        mapping = read_json(maps[0], source=True)
        require(set(mapping) == {'format', 'pairs'} and
                mapping['format'] == 'contextual-identity-map-v2', 'contextual_identity_map')
        self.identity_map = {}
        for item in mapping['pairs']:
            require(item['pair_id'] not in self.identity_map, 'contextual_identity_duplicate')
            self.identity_map[item['pair_id']] = item
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(annotation_schema)
        self.pairs = {}
        self.records = {}
        self.by_example = {}
        for pair in self.annotation['pairs']:
            schema_check(validator, pair)
            identity = pair['pair_id']
            require(identity not in self.pairs, 'contextual_duplicate_identity')
            self.pairs[identity] = pair
            reference = pair['privacy_evidence']
            snapshot = self.resolve_reference({**reference, 'json_pointer': reference['json_pointer'].rsplit('/', 1)[0]})
            example = pair['identities']['example_id']
            require(self.records.setdefault(example, snapshot) == snapshot, 'contextual_source_conflict')
            self.by_example.setdefault(example, []).append(pair)
            self.validate_annotation(pair, snapshot)
        require(self.receipt['pairs'] == [
            {key: pair[key] for key in ('pair_id', 'runtime_id', 'identities', 'request_sha256',
                                       'body_sha256', 'disposition', 'status', 'privacy_evidence', 'opinion_references')}
            for pair in self.annotation['pairs']], 'contextual_receipt_pairs')
        require(self.receipt['proposed_pair_allowlist'] == sorted(
            key for key, pair in self.pairs.items() if pair['disposition'] == 'eligible_resolved'),
            'contextual_proposed_pair_allowlist')
        require(self.receipt['excluded_dispositions'] == {
            state: sum(pair['disposition'] == state for pair in self.pairs.values())
            for state in ('unresolved', 'masked', 'ineligible', 'diagnostic')}, 'contextual_exclusion_counts')
        datasets = [(relative, path) for relative, path in self.evidence.items() if relative.endswith('/dataset.jsonl')]
        require(len(datasets) == 1, 'contextual_source_dataset')
        source_relative, source_path = datasets[0]
        self.source_dataset_sha256 = self.receipt['source_hashes'][source_relative]
        reviewed_records = {}
        self.source_groups = {}
        group_splits = {}
        for row in rows(source_path):
            group = row['source']['conversation_group']
            self.source_groups.setdefault(group, []).append(row['example_id'])
            group_splits.setdefault(group, set()).add(row['split'])
            require(row['example_id'] in self.records, 'contextual_source_coverage')
            snapshot = self.records[row['example_id']]
            require({key: value for key, value in row.items() if key != 'audit'} == snapshot,
                    'contextual_source_snapshot')
            require(row['example_id'] not in reviewed_records, 'contextual_duplicate_source')
            reviewed_records[row['example_id']] = row
        require(reviewed_records.keys() == self.records.keys(), 'contextual_source_coverage')
        require(all(group_splits[row['source']['conversation_group']] == {row['split']}
                    for row in reviewed_records.values()), 'contextual_cross_split_group')
        self.records = reviewed_records
        require(self.identity_map.keys() == self.pairs.keys(), 'contextual_identity_coverage')
        plans = [path for relative, path in self.evidence.items()
                 if self.receipt['source_hashes'][relative] == self.receipt['role_plan_sha256']]
        require(len(plans) == 1, 'contextual_role_plan_missing')
        plan = read_json(plans[0])
        require(set(plan) == {'format', 'assignments'} and
                plan['format'] == 'contextual-role-plan-v2', 'contextual_role_plan')
        assignments, groups = {}, {}
        for item in plan['assignments']:
            require(set(item) == {'case_id', 'conversation_group', 'template_group', 'split',
                    'raw_role', 'source_reference', 'role_pointer', 'reviewed_ids'}
                    and item['case_id'] not in assignments, 'contextual_role_assignment')
            assignments[item['case_id']] = item
            for key in ('conversation_group', 'template_group'):
                if item[key] is not None:
                    require(groups.setdefault(item[key], (item['split'], item['raw_role'])) ==
                            (item['split'], item['raw_role']), 'contextual_cross_split_group')
        require(assignments.keys() == self.records.keys(), 'contextual_role_coverage')
        for row in self.records.values():
            item = assignments[row['example_id']]
            require(item['split'] == row['split'] and all(item[key] == row['source'][key]
                    for key in ('conversation_group', 'template_group')), 'contextual_role_drift')
            raw = self.resolve_reference(item['source_reference'])
            require(self.pointer(raw, item['role_pointer']) == item['raw_role'], 'contextual_raw_role')
            if authored_contextual(row):
                require(item['source_reference'] == row['authored']['case_reference'],
                        'contextual_role_source')
                allowed = {'train': {'train'}, 'TRAIN_PROPOSAL': {'train'},
                           'calibration': {'validation', 'challenge', 'temporal_test'},
                           'test': {'temporal_test', 'unseen_family_test', 'challenge'}}
                require(row['split'] in allowed.get(item['raw_role'], set()), 'contextual_role_remapping')
                self.validate_authored_source(row)
            else:
                require(raw == {k: v for k, v in row.items() if k != 'audit'} and
                        item['raw_role'] == row['split'], 'contextual_broad_role')
            reviewed = item['reviewed_ids']
            require(len(reviewed) == len(set(reviewed)) and
                    set(reviewed) == {p['runtime_id'] for p in self.by_example[row['example_id']]} and
                    (all('purposes' in p for p in self.by_example[row['example_id']]) or set(reviewed) <= set(row['input']['candidate_ids'])), 'contextual_candidate_coverage')
        projections = [row['authored']['projection'] for row in self.records.values()
                       if authored_contextual(row) and 'projection' in row['authored']]
        if projections:
            require(self.receipt.get('fixture_only') is False and
                    self.receipt.get('authority_requirements') == {
                        'purpose': 'actual-evaluation-canonical-integration', 'fixture_only': False,
                        'projection_decision_sha256': self.receipt.get('projection_decision_sha256'),
                        'codec_sha256': file_digest(__file__)}, 'contextual_source_authority_contract')
            require(all(item['decision_reference']['sha256'] == self.receipt['projection_decision_sha256']
                        for item in projections), 'contextual_projection_decision_pin')
            require(self.receipt.get('calibration_fit_pair_allowlist') == sorted(
                key for key, pair in self.pairs.items() if pair.get('purposes', {}).get('calibration_fit')),
                'contextual_fit_allowlist')
        # Original evidence remains in the immutable proof. Parsing caches are
        # unnecessary after member validation and multiply cohort memory use.
        self._documents.clear()

    def audit_blob(self, expected):
        candidates = [self.evidence[name] for name in self.evidence
                      if name.endswith('/' + expected[7:] + '.json')]
        require(candidates and all(raw == candidates[0] for raw in candidates), 'audit_blob_missing')
        value = read_json(candidates[0])
        require(digest(value) == expected, 'audit_blob_digest')
        return value

    def validate_projection(self, row, raw):
        projection = row['authored']['projection']
        require(set(projection) == {'decision_reference', 'synthetic_value_pointer', 'source_context_pointer',
                'active_request_pointer', 'active_request_hash_pointer', 'native_messages', 'compiled_intent_sha256'},
                'contextual_projection_shape')
        decision = self.resolve_reference(projection['decision_reference'])
        require(self.pointer(raw, row['authored']['source_fields']['synthetic']) ==
                self.pointer(decision, projection['synthetic_value_pointer']), 'contextual_synthetic_declaration')
        require(decision['uniform_rule'] == {'role': 'assistant', 'source_content_type': 'string',
                'native_content': '[{type:text,text:original_scalar_content}]',
                'already_native_arrays': 'preserve exactly', 'non_text_parts': 'preserve source and native rules',
                'case_specific_branches': False}, 'contextual_projection_rule')
        require(digest(row['authored']['messages']) == self.pointer(raw, projection['source_context_pointer']),
                'contextual_source_context_digest')
        native = []
        for original in row['authored']['messages']:
            message = dict(original)
            if message['role'] == 'assistant' and isinstance(message.get('content'), str):
                message['content'] = [{'type': 'text', 'text': message['content']}]
            native.append(message)
        require(native == projection['native_messages'], 'contextual_native_projection')
        active = [self.complete_message_text(m, 'user', '/content') for m in native if m['role'] == 'user']
        active = [text for text in active if text]
        require(active and active[-1] == self.pointer(raw, projection['active_request_pointer']) and
                'sha256:' + hashlib.sha256(active[-1].encode()).hexdigest() ==
                self.pointer(raw, projection['active_request_hash_pointer']), 'contextual_active_request_digest')
        require('sha256:' + hashlib.sha256(row['input']['request'].encode()).hexdigest() ==
                projection['compiled_intent_sha256'], 'contextual_compiled_intent_digest')
        manifest = self.resolve_reference(row['authored']['prefix_reference'])
        if manifest['serialization'] == 'ordered-message-content-join-v1':
            require(decision.get('intent_serialization') == {
                'serialization': 'ordered-message-content-join-v1', 'intent_pointer': '/intent',
                'action_pointer': '/action', 'action_keys': {'toolName': 'tool_name', 'summary': 'summary'}},
                'contextual_projection_rule')
            intent = '\n\n'.join(self.complete_message_text(message, message['role'], '/content') for message in native)
            action = self.pointer(raw, '/action')
            action = None if action is None else {'tool_name': action['toolName'], 'summary': action['summary']}
            require(intent == self.pointer(raw, '/intent') == row['input']['request'] and
                    action == row['input']['action'], 'contextual_source_input_binding')
        return native

    def validate_source_context_identity(self, pair, row, item):
        """Join independently stored opinions through their original sealed context identities."""
        require(authored_contextual(row) and 'projection' in row['authored'], 'contextual_source_context_identity')
        raw = self.resolve_reference(row['authored']['case_reference'])
        projection = row['authored']['projection']
        context_hash = self.pointer(raw, projection['source_context_pointer'])
        request_hash = self.pointer(raw, projection['active_request_hash_pointer'])
        expected = {'case_id': row['example_id'], 'runtime_id': pair['runtime_id'],
                    'body_sha256': pair['body_sha256'], 'target_sha256': self.binding['target_sha256'],
                    'request_sha256': request_hash, 'source_context_sha256': context_hash}
        audit = self.resolve_reference(pair['adjudication_reference'])
        for role, binding in item['records'].items():
            require(set(binding) == {'reference', 'fields', 'external_id', 'external_id_present',
                    'raw_role', 'original_reference'}, 'contextual_identity_reference')
            reference = pair['adjudication_reference'] if role == 'adjudicator' else pair['opinion_references'][role]
            require(binding['reference'] == reference, 'contextual_identity_reference')
            value = self.resolve_reference(reference)
            declaration = binding['original_reference']
            if role == 'adjudicator':
                require(declaration is None, 'contextual_adjudicator_reference')
            else:
                require(set(declaration) == {'pointer', 'path_prefix', 'path_field', 'hash_field',
                        'index_field', 'record_hash_field'}, 'contextual_original_reference')
                original = self.pointer(audit, declaration['pointer'])
                path = declaration['path_prefix'] + original[declaration['path_field']]
                index = original[declaration['index_field']]
                wanted = {'path': path, 'sha256': original[declaration['hash_field']],
                          'json_pointer': '' if path.endswith('.jsonl') else '/' + str(index)}
                if path.endswith('.jsonl'):
                    wanted['jsonl_line_1based'] = index + 1
                require(reference == wanted and digest(value) == original[declaration['record_hash_field']],
                        'contextual_original_reference')
            require(set(binding['fields']) == set(expected) | {'external_id', 'raw_role'}, 'contextual_identity_fields')
            for key, wanted in dict(expected, external_id=binding['external_id'], raw_role=binding['raw_role']).items():
                field = binding['fields'][key]
                require(set(field) == {'origin', 'pointer', 'transform'} and
                        field['origin'] in ('record', 'context'), 'contextual_identity_binding')
                require(key == 'target_sha256' or field['origin'] == 'record', 'contextual_identity_binding')
                selected = audit if field['origin'] == 'context' else value
                present = self.field_present(selected, field['pointer'])
                if field['transform'] == 'optional':
                    require(key in ('external_id', 'raw_role'), 'contextual_identity_binding')
                    actual = self.pointer(selected, field['pointer']) if present else None
                else:
                    require(field['transform'] == 'value', 'contextual_identity_binding')
                    actual = self.pointer(selected, field['pointer'])
                require(canonical_bytes(actual) == canonical_bytes(wanted), 'contextual_identity_binding')
                if key == 'external_id':
                    require(present == binding['external_id_present'], 'contextual_foreign_identity_presence')
                    if present:
                        self.check_foreign_identity(actual)

    def validate_purpose(self, pair, row):
        binding = self.identity_map[pair['pair_id']]['purpose_binding']
        require(set(binding) == {'reference', 'fit_pointer', 'exclusions_pointer', 'mask_pointer',
                'status_pointer', 'diagnostic_pointer', 'case_pointer', 'split_pointer'}, 'contextual_purpose_binding')
        plan = self.resolve_reference(binding['reference'])
        require(self.pointer(plan, binding['case_pointer']) == row['example_id'] and
                self.pointer(plan, binding['split_pointer']) == row['split'], 'contextual_purpose_role')
        require(self.pointer(plan, binding['fit_pointer']) == pair['purposes']['calibration_fit'] and
                self.pointer(plan, binding['exclusions_pointer']) == pair['purposes']['fit_exclusions'] and
                self.pointer(plan, binding['mask_pointer']) == pair['family_mask'] and
                self.pointer(plan, binding['status_pointer']) == pair['status'] and
                self.pointer(plan, binding['diagnostic_pointer']) == pair['purposes']['diagnostic'],
                'contextual_purpose_binding')
        fit = pair['purposes']['calibration_fit']
        require(not fit or (row['split'] == 'validation' and not pair['purposes']['diagnostic'] and
                not pair['family_mask'] and pair['status'] != 'unresolved' and
                pair['runtime_id'] not in ('unslop', 'emil-unslop-writing') and
                pair['eligibility']['automatic_eligible'] and pair['runtime_id'] in row['input']['candidate_ids'] and
                not pair['purposes']['fit_exclusions']), 'contextual_fit_boundary')
        evaluation = row['split'] in ('temporal_test', 'unseen_family_test') and not pair['purposes']['diagnostic']
        require(pair['purposes']['evaluation'] == evaluation, 'contextual_evaluation_purpose')

    @staticmethod
    def complete_message_text(message, role, pointer):
        """Bind the whole normalized/native payload, not a caller-selected text part."""
        slots = [key for key in ('text', 'content', 'summary') if key in message]
        require(len(slots) == 1 and pointer == '/' + slots[0], 'contextual_message_text')
        slot = slots[0]
        value = message[slot]
        if slot == 'content':
            require(role in ('user', 'assistant', 'toolResult'), 'contextual_message_content')
            if not (role == 'user' and isinstance(value, str)):
                require(isinstance(value, list), 'contextual_message_content')
                texts = []
                for part in value:
                    require(isinstance(part, dict) and isinstance(part.get('type'), str),
                            'contextual_message_content')
                    if part['type'] == 'text':
                        require(isinstance(part.get('text'), str), 'contextual_message_content')
                        texts.append(part['text'])
                value = '\n'.join(texts)
        elif slot == 'summary':
            require(role in ('compactionSummary', 'branchSummary'), 'contextual_message_content')
        require(isinstance(value, str), 'contextual_message_content')
        return value

    def validate_authored_source(self, row):
        """Authored provenance is independently sealed, never inferred from annotations."""
        require(authored_contextual(row) and 'label' not in row and
                row.get('prior_broad_provenance') == {'state': 'absent_new_synthetic_source'},
                'contextual_absent_broad_provenance')
        schema_check(validators()['dataset'], row)
        authored, source, inputs = row['authored'], row['source'], row['input']
        raw = self.resolve_reference(authored['case_reference'])
        fields = authored['source_fields']
        require(set(fields) == {'synthetic', 'case_id', 'messages', 'conversation_group', 'template_group'},
                'contextual_source_fields')
        values = {key: self.pointer(raw, pointer) for key, pointer in fields.items()}
        native = self.validate_projection(row, raw) if 'projection' in authored else authored['messages']
        require((values['synthetic'] is True or 'projection' in authored) and authored['case_id'] == row['example_id'] == values['case_id'] and
                authored['messages'] == values['messages'], 'contextual_authored_request')
        require(all(values[key] == source[key] for key in ('conversation_group', 'template_group')),
                'contextual_authored_groups')
        require(authored['role_plan_sha256'] == self.receipt['role_plan_sha256'],
                'contextual_role_plan_pin')
        require(authored['event_metadata'] == {'basis': 'SYNTHETIC', **{key: source[key]
                for key in ('event_id', 'event_sequence', 'cutoff_utc')}}, 'contextual_authored_event')
        manifest = self.resolve_reference(authored['prefix_reference'])
        require(source['source_manifest_sha256'] == authored['prefix_reference']['sha256'],
                'contextual_authored_lineage')
        mapping = authored['message_mapping']
        require(len(mapping) == len(authored['messages']) == len(manifest['events']),
                'contextual_message_coverage')
        for index, (message, event, item) in enumerate(zip(native, manifest['events'], mapping)):
            require(set(item) == {'message_index', 'event_id', 'raw_role', 'role', 'fields'} and
                    item['message_index'] == index and item['event_id'] == event['event_id'] and
                    item['role'] == event['role'] and
                    set(item['fields']) == {'role', 'text', 'action', 'tool_call_event_id'},
                    'contextual_message_mapping')
            require(self.pointer(message, item['fields']['role']) == item['raw_role'],
                    'contextual_raw_message_role')
            require(self.complete_message_text(message, event['role'], item['fields']['text']) == event['text'],
                    'contextual_message_binding')
            for key in ('action', 'tool_call_event_id'):
                pointer = item['fields'][key]
                value = None if pointer is None else self.pointer(message, pointer)
                require(value == event[key], 'contextual_message_binding')
        synthetic_prefix_check(row, manifest)
        require(inputs['required_ids'] == inputs['explicit_ids'] == inputs['already_loaded'] == [],
                'contextual_native_state')
        require(row['privacy'] == {'classification': 'local_only', 'sanitizer_sha256': digest(SANITIZER),
                'scan_passed': True, 'export_approved': False, 'publication_allowed': False},
                'contextual_authored_privacy')

    def validate_pair_review(self, pair):
        require(set(pair['opinion_references']) == {'author', 'reviewer'}
                and pair['opinion_references']['author'] != pair['opinion_references']['reviewer'],
                'contextual_pair_audit_roles')
        audit = self.resolve_reference(pair['adjudication_reference'])
        review = self.identity_map[pair['pair_id']]['review_binding']
        require(set(review) == {'status', 'affirmative_exclusion', 'missing_facts', 'privacy_reference'},
                'contextual_pair_audit_binding')
        for key in ('status', 'affirmative_exclusion', 'missing_facts'):
            value = self.pointer(audit, review[key])
            if key != 'status' and value is None:
                value = ''
            require(value == pair[key], 'contextual_pair_audit_binding')
        privacy = self.resolve_reference(review['privacy_reference'])
        require(privacy['export_approved'] is False and
                privacy.get('publication_allowed', False) is False,
                'contextual_pair_privacy_evidence')

    @staticmethod
    def pointer(value, pointer):
        require(isinstance(pointer, str) and (pointer == '' or pointer.startswith('/')),
                'contextual_pointer')
        try:
            for part in pointer.split('/')[1:]:
                part = part.replace('~1', '/').replace('~0', '~')
                value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, ValueError, TypeError):
            raise ContractError('contextual_pointer') from None
        return value

    def validate_identity(self, pair, row):
        identity = pair['source_identity']
        require(identity == {'case_id': row['example_id'], 'canonical_pair_id': pair['pair_id'],
                'target_version': self.annotation['target_version']}, 'contextual_external_identity')
        item = self.identity_map.get(pair['pair_id'])
        required_fields = {'pair_id', 'records', 'quote_binding', 'review_binding'}
        require(item is not None and required_fields <= set(item) <=
                required_fields | {'identity_basis', 'purpose_binding'} and
                set(item['records']) == {'author', 'reviewer', 'adjudicator'}, 'contextual_identity_map')
        require(('purpose_binding' in item) == ('purposes' in pair), 'contextual_purpose_binding')
        if 'purposes' in pair:
            self.validate_purpose(pair, row)
        else:
            require(row['split'] == 'train', 'contextual_purpose_required')
        require('identity_basis' not in item or item['identity_basis'] == 'sealed-message-context',
                'contextual_identity_basis')
        snapshot_ref = dict(pair['privacy_evidence'],
                            json_pointer=pair['privacy_evidence']['json_pointer'].rsplit('/', 1)[0])
        source_ref = row['authored']['case_reference'] if authored_contextual(row) else snapshot_ref
        raw_request = row['input']['request']
        if authored_contextual(row):
            quote_binding = item['quote_binding']
            require(set(quote_binding) == {'event_id', 'raw_quote', 'compiled_quote'},
                    'contextual_quote_mapping')
            manifest = self.resolve_reference(row['authored']['prefix_reference'])
            source_offsets = synthetic_prefix_check(row, manifest)
            events = manifest['events']
            active = [(index, event) for index, event in enumerate(events) if event['role'] == 'user' and event['text']]
            require(active and quote_binding['event_id'] == active[-1][1]['event_id'] and
                    pair['identities']['request_index'] == active[-1][0], 'contextual_quote_active_event')
            active_event = active[-1][1]
            raw_request = active_event['text']
            raw_quote = quote_binding['raw_quote']
            raw_bytes = raw_request.encode()
            start, end = raw_quote['start_byte'], raw_quote['end_byte']
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(raw_bytes) and
                    raw_quote['source_sha256'] == 'sha256:' + hashlib.sha256(raw_bytes).hexdigest() and
                    raw_quote['quote_sha256'] == 'sha256:' + hashlib.sha256(raw_bytes[start:end]).hexdigest() and
                    quote_binding['compiled_quote'] == pair['request_quote'] and
                    raw_quote['quote_sha256'] == pair['request_quote']['quote_sha256'] and
                    pair['request_quote']['start_byte'] == source_offsets[active_event['event_id']] + start and
                    pair['request_quote']['end_byte'] == source_offsets[active_event['event_id']] + end,
                    'contextual_quote_mapping')
        else:
            require(item['quote_binding'] is None, 'contextual_quote_mapping')
        if item.get('identity_basis') == 'sealed-message-context':
            self.validate_source_context_identity(pair, row, item)
            return
        expected = {'case_id': row['example_id'], 'runtime_id': pair['runtime_id'],
                    'body_sha256': pair['body_sha256'], 'target_sha256': self.binding['target_sha256'],
                    'request_sha256': 'sha256:' + hashlib.sha256(raw_request.encode()).hexdigest(),
                    'source_reference': source_ref}
        for role, record in item['records'].items():
            reference = pair['adjudication_reference'] if role == 'adjudicator' else pair['opinion_references'][role]
            require(set(record) == {'reference', 'fields', 'external_id', 'external_id_present', 'raw_role', 'context'}
                    and record['reference'] == reference, 'contextual_identity_reference')
            value = self.resolve_reference(reference)
            context = record['context']
            require(set(context) == {'reference', 'opinion_pointer', 'opinion_reference_pointer'}
                    and context['reference'] == pair['adjudication_reference'], 'contextual_identity_context')
            witness = self.resolve_reference(context['reference'])
            require(self.pointer(witness, context['opinion_pointer']) == value, 'contextual_original_opinion')
            if role == 'adjudicator':
                require(context['opinion_pointer'] == '' and context['opinion_reference_pointer'] is None,
                        'contextual_identity_context')
            else:
                require(self.canonical_reference(self.pointer(witness, context['opinion_reference_pointer'])) ==
                        reference, 'contextual_original_reference')
            require(set(record['fields']) == set(expected) | {'external_id', 'raw_role'},
                    'contextual_identity_fields')
            foreign = record['external_id']
            present = self.field_present(value, record['fields']['external_id']['pointer'])
            require(type(record['external_id_present']) is bool and present == record['external_id_present'],
                    'contextual_foreign_identity_presence')
            if present:
                self.check_foreign_identity(foreign)
            else:
                require(foreign is None, 'contextual_foreign_identity_presence')
            for key, wanted in dict(expected, external_id=foreign, raw_role=record['raw_role']).items():
                binding = record['fields'][key]
                require(set(binding) == {'origin', 'pointer', 'transform'} and
                        binding['origin'] in ('record', 'context'), 'contextual_identity_binding')
                if key not in ('target_sha256', 'source_reference'):
                    require(binding['origin'] == 'record', 'contextual_identity_binding')
                selected = value if binding['origin'] == 'record' else witness
                if binding['transform'] == 'optional':
                    require(key in ('external_id', 'raw_role'), 'contextual_identity_binding')
                    actual = self.pointer(selected, binding['pointer']) if self.field_present(selected, binding['pointer']) else None
                else:
                    actual = self.pointer(selected, binding['pointer'])
                    if binding['transform'] == 'reference':
                        require(key == 'source_reference', 'contextual_identity_binding')
                        actual = self.canonical_reference(actual)
                    else:
                        require(binding['transform'] == 'value', 'contextual_identity_binding')
                require(canonical_bytes(actual) == canonical_bytes(wanted), 'contextual_identity_binding')

    @classmethod
    def field_present(cls, value, pointer):
        require(isinstance(pointer, str) and pointer.startswith('/'), 'contextual_pointer')
        try:
            cls.pointer(value, pointer)
            return True
        except ContractError:
            return False

    @staticmethod
    def canonical_reference(reference):
        require(isinstance(reference, dict) and {'path', 'sha256', 'json_pointer'} <= reference.keys(),
                'contextual_reference_shape')
        return {key: reference[key] for key in ('path', 'sha256', 'json_pointer', 'jsonl_line_1based')
                if key in reference}

    @staticmethod
    def check_foreign_identity(identity):
        """Preserve JSON structure without producer-specific decoding or hash aliases."""
        require(isinstance(identity, (str, list, dict)) and bool(identity), 'contextual_foreign_identity')
        pending, count = [(identity, 0)], 0
        while pending:
            value, depth = pending.pop()
            count += 1
            require(depth <= 4 and count <= 128, 'contextual_foreign_identity')
            if isinstance(value, str):
                require(0 < len(value) <= 256 and not re.search(r'[\x00-\x1f\x7f]', value),
                        'contextual_foreign_identity')
            elif isinstance(value, (list, dict)):
                require(0 < len(value) <= 16, 'contextual_foreign_identity')
                if isinstance(value, dict):
                    pending.extend((key, depth + 1) for key in value)
                    children = value.values()
                else:
                    children = value
                pending.extend((item, depth + 1) for item in children)
            else:
                require(value is None or type(value) is bool or
                        (type(value) is int and abs(value) <= 9007199254740991), 'contextual_foreign_identity')
        require(len(canonical_bytes(identity)) <= 4096, 'contextual_foreign_identity')

    def resolve_reference(self, reference):
        relative = reference['path']
        require(relative in self.evidence and
                self.receipt['source_hashes'][relative] == reference['sha256'], 'contextual_reference_digest')
        if relative not in self._documents:
            path = self.evidence[relative]
            require(len(path) <= 64 * 1024 * 1024, 'source_document_capacity')
            if relative.endswith('.jsonl'):
                self._documents[relative] = list(rows(path, source=True))
            else:
                self._documents[relative] = parse_json(path, source=True)
        value = self._documents[relative]
        if relative.endswith('.jsonl'):
            line = reference.get('jsonl_line_1based')
            require(type(line) is int and 1 <= line <= len(value), 'contextual_jsonl_line')
            value = value[line - 1]
        else:
            require('jsonl_line_1based' not in reference, 'contextual_jsonl_line')
        return self.pointer(value, reference['json_pointer'])

    def validate_annotation(self, pair, row):
        skill, ids = pair['runtime_id'], pair['identities']
        require(ids['example_id'] == row['example_id'] and ids['split'] == row['split'] and
                ids['event_kind'] == row['event_kind'], 'contextual_source_identity')
        for key in ('event_id', 'conversation_group', 'template_group'):
            require(ids[key] == row['source'][key], 'contextual_source_identity')
        require(pair['pair_id'] == 'sha256:' + hashlib.sha256(
            (row['example_id'] + '\0' + skill).encode()).hexdigest(), 'contextual_pair_identity')
        require(pair['cutoff_source'] == row['source'] and pair['input_sha256'] == digest(row['input']) and
                pair['catalog_sha256'] == row['catalog_sha256'], 'contextual_cutoff_binding')
        self.validate_identity(pair, row)
        self.validate_pair_review(pair)
        if authored_contextual(row):
            require('label' not in row and pair['source_relevance'] ==
                    row['prior_broad_provenance'] == {'state': 'absent_new_synthetic_source'},
                    'contextual_absent_broad_provenance')
        else:
            require(pair['source_relevance']['label_snapshot_sha256'] == digest(row['label']),
                    'contextual_broad_provenance')
            original = pair['source_relevance']['original_label_type']
            require(original in ('positive', 'verified_negative') and
                    skill in row['label']['selected_ids' if original == 'positive' else 'verified_negative_ids'] and
                    skill not in row['label']['unknown_ids'], 'contextual_unknown_source')
        require(pair['privacy'] == row['privacy'] == self.resolve_reference(pair['privacy_evidence']),
                'contextual_privacy_binding')
        body_paths = [path for relative, path in self.evidence.items()
                      if relative.endswith('/bodies/' + pair['body_sha256'][7:] + '.md')]
        require(len(body_paths) == 1, 'contextual_body_missing')
        body = body_paths[0]
        require('sha256:' + hashlib.sha256(body).hexdigest() == pair['body_sha256'],
                'contextual_body_digest')
        request = row['input']['request'].encode()
        for quote, raw in ((pair['request_quote'], request), (pair['body_quote'], body)):
            start, end = quote['start_byte'], quote['end_byte']
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(raw) and
                    quote['source_sha256'] == 'sha256:' + hashlib.sha256(raw).hexdigest() and
                    quote['quote_sha256'] == 'sha256:' + hashlib.sha256(raw[start:end]).hexdigest(),
                    'contextual_quote_binding')
        require(pair['request_sha256'] == 'sha256:' + hashlib.sha256(request).hexdigest(),
                'contextual_request_digest')
        status = pair['status']
        require(bool(pair['affirmative_exclusion']) == (status in ('response_only', 'nonapplicable')) and
                bool(pair['missing_facts']) == (status == 'unresolved'), 'contextual_status_evidence')
        unavailable = pair['runtime_id'] not in row['input']['candidate_ids']
        diagnostic = pair.get('purposes', {}).get('diagnostic', False)
        require(pair['disposition'] == ('masked' if pair['family_mask'] else
                'unresolved' if status == 'unresolved' else 'ineligible' if unavailable else
                'diagnostic' if diagnostic else 'eligible_resolved'), 'contextual_disposition')
        require(pair['conjunctive_groups'] == [] and pair['admitted'] is False, 'contextual_author_not_authority')
        route = pair['eligibility']['catalog_route']
        require(route in ('automatic', 'explicit', 'body-explicit-only-unmet'), 'contextual_eligibility')
        if route == 'automatic':
            require(pair['eligibility']['automatic_eligible'], 'contextual_eligibility')
        if route == 'explicit' and not diagnostic:
            require(skill in row['input']['explicit_ids'] and pair['eligibility']['active_explicit_invocation'],
                    'contextual_explicit_required')

    def validate_record(self, row, catalog):
        require(self.authorized, 'contextual_admission_authority_required')
        source = {key: value for key, value in row.items() if key != 'contextual'}
        require(row['example_id'] in self.records and source == self.records[row['example_id']],
                'contextual_record_not_reviewed')
        require(catalog['catalog_sha256'] == row['catalog_sha256'], 'contextual_catalog_binding')
        entries = {entry['runtime_id']: entry for entry in catalog['entries']}
        for pair in self.by_example[row['example_id']]:
            skill = pair['runtime_id']
            require(skill in entries and (pair['disposition'] != 'eligible_resolved' or
                    (skill in row['input']['candidate_ids'] and skill not in row['input']['excluded_ids'])),
                    'contextual_candidate_ineligible')
            require(entries[skill]['content_sha256'] == pair['body_sha256'], 'contextual_catalog_body')
            for key in ('automatic_eligible', 'explicit_eligible'):
                require(entries[skill][key] == pair['eligibility'][key], 'contextual_catalog_eligibility')



def member_approval_requirements(receipt, receipt_sha256):
    """Required decision payload, not an issued grant or an inferred trusted pin."""
    payload = {
        'status': 'approved_local_canonical_integration_only',
        'fixture_only': receipt['fixture_only'],
        'source_set_execution': {'purpose': 'local-canonical-integration', 'runtime_revision': file_digest(__file__)},
        **{key: receipt[key] for key in ('target_sha256', 'rubric_sha256', 'schema_sha256',
            'annotation_sha256', 'role_plan_sha256', 'identity_map_sha256', 'dataset_schema_sha256',
            'source_contract_sha256')},
        'proposed_receipt_sha256': receipt_sha256,
        'approved_pair_ids': receipt['proposed_pair_allowlist'],
        'excluded_dispositions': receipt['excluded_dispositions'],
    }
    if 'authority_requirements' in receipt:
        payload['authority_requirements'] = receipt['authority_requirements']
    return payload


class _MemberAdmission(ContextualProposal):
    """Internal member verification; only a joint source-set grant permits consumption."""

    def __init__(self, approval, receipt, annotation, schema, evidence_map, *, approval_sha256, proof):
        require(isinstance(approval_sha256, str) and
                re.fullmatch(r'sha256:[0-9a-f]{64}', approval_sha256), 'contextual_approval_pin_required')
        require(approval == approval_sha256, 'contextual_approval_digest')
        self.approval = read_json(proof.read(approval), source=True)
        require(self.approval.get('status') == 'approved_local_canonical_integration_only',
                'contextual_approval_scope')
        require(self.approval.get('source_set_execution') == {
            'purpose': 'local-canonical-integration', 'runtime_revision': file_digest(__file__)},
            'contextual_authority_implementation')
        proposed = read_json(proof.read(receipt), source=True)
        fixture = proposed.get('fixture_only') is True and self.approval.get('fixture_only') is True
        if 'authority_requirements' in proposed:
            require(self.approval.get('authority_requirements') == proposed['authority_requirements'],
                    'contextual_authority_purpose')
            require(proposed['authority_requirements'].get('codec_sha256') == file_digest(__file__),
                    'contextual_authority_implementation')
            require(self.approval.get('fixture_only') is False, 'contextual_fixture_authority_rejected')
        require(self.approval['proposed_receipt_sha256'] == receipt, 'contextual_receipt_binding')
        for key in ('target_sha256', 'schema_sha256', 'annotation_sha256'):
            require(self.approval[key] == proposed[key], 'contextual_approval_binding')
        for key in ('dataset_schema_sha256', 'source_contract_sha256', 'role_plan_sha256',
                    'identity_map_sha256', 'rubric_sha256'):
            require(self.approval.get(key) == proposed.get(key) and proposed.get(key) is not None,
                    'contextual_operator_pin')
        super().__init__(receipt, annotation, schema, evidence_map, proof=proof)
        if fixture:
            require(all(authored_contextual(row) and
                        self.resolve_reference(row['authored']['case_reference']).get('fixture_only') is True
                        for row in self.records.values()), 'contextual_fixture_source_required')
        approved = self.approval['approved_pair_ids']
        require(len(approved) == len(set(approved)) and set(approved) ==
                {key for key, pair in self.pairs.items() if pair['disposition'] == 'eligible_resolved'},
                'contextual_approved_pair_allowlist')
        require(self.approval['excluded_dispositions'] == {
            disposition: sum(pair['disposition'] == disposition for pair in self.pairs.values())
            for disposition in ('unresolved', 'masked', 'ineligible', 'diagnostic')}, 'contextual_exclusion_counts')
        require(self.approval.keys() == member_approval_requirements(proposed, receipt).keys(),
                'contextual_approval_fields')
        self.approved = frozenset(approved)
        self.binding['approval_sha256'] = approval_sha256
        self.references['approval'] = approval
        self.authorized = True


class _EvidenceFiles(Mapping):
    """Keep sealed evidence references, not every member's original review bytes."""

    def __init__(self, proof, hashes):
        self.proof, self.hashes = proof, hashes

    def __iter__(self):
        return iter(self.hashes)

    def __len__(self):
        return len(self.hashes)

    def __contains__(self, key):
        return key in self.hashes

    def __getitem__(self, key):
        return self.proof.read(self.hashes[key])


class _AdmissionStore:
    """Disposable derived pair state; original evidence remains solely in the CAS."""

    def __init__(self):
        directory = tempfile.TemporaryDirectory(prefix='contextual-admission-')
        path = Path(directory.name) / 'pairs.sqlite'
        path.touch(mode=0o600)
        self.connection = sqlite3.connect(path)
        # This cache is rebuilt after any failure; it needs no durable journal.
        self.connection.execute('PRAGMA journal_mode=OFF')
        self.connection.executescript('CREATE TABLE pairs(id TEXT PRIMARY KEY, member TEXT, example TEXT, payload BLOB); CREATE INDEX pair_member ON pairs(member); CREATE INDEX pair_example ON pairs(example);')
        self.cache = {}
        weakref.finalize(self, self.close, self.connection, directory)

    @staticmethod
    def close(connection, directory):
        connection.close()
        directory.cleanup()

    def retain(self, member, identity):
        try:
            self.connection.executemany('INSERT INTO pairs VALUES (?,?,?,?)',
                ((key, identity, pair['identities']['example_id'], canonical_bytes(pair))
                 for key, pair in member.pairs.items()))
            self.connection.commit()
        except sqlite3.IntegrityError:
            raise ContractError('contextual_duplicate_identity') from None

    def case_pairs(self, example):
        if example not in self.cache:
            if len(self.cache) == 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[example] = [parse_json(raw, source=True) for raw, in self.connection.execute(
                'SELECT payload FROM pairs WHERE example=? ORDER BY rowid', (example,))]
        return self.cache[example]


class _PairIndex(Mapping):
    def __init__(self, store, member=None):
        self.store, self.member = store, member

    def __iter__(self):
        query, args = ('SELECT id FROM pairs ORDER BY rowid', ()) if self.member is None else (
            'SELECT id FROM pairs WHERE member=? ORDER BY rowid', (self.member,))
        return (key for key, in self.store.connection.execute(query, args))

    def __len__(self):
        query, args = ('SELECT count(*) FROM pairs', ()) if self.member is None else (
            'SELECT count(*) FROM pairs WHERE member=?', (self.member,))
        return self.store.connection.execute(query, args).fetchone()[0]

    def __getitem__(self, key):
        query, args = ('SELECT payload FROM pairs WHERE id=?', (key,)) if self.member is None else (
            'SELECT payload FROM pairs WHERE id=? AND member=?', (key, self.member))
        row = self.store.connection.execute(query, args).fetchone()
        if row is None:
            raise KeyError(key)
        return parse_json(row[0], source=True)


class _CasePairs(Mapping):
    def __init__(self, store, records):
        self.store, self.records = store, records

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        if key not in self.records:
            raise KeyError(key)
        return self.store.case_pairs(key)


class _AdmittedMember:
    """Consumer state after complete member validation; no retained review parser."""

    authorized = True
    validate_record = ContextualProposal.validate_record
    audit_blob = ContextualProposal.audit_blob
    resolve_reference = ContextualProposal.resolve_reference
    pointer = staticmethod(ContextualProposal.pointer)

    def __init__(self, member, store, identity):
        self.binding, self.references = member.binding, member.references
        self.receipt = {key: value for key, value in member.receipt.items()
                        if key not in ('pairs', 'proposed_pair_allowlist', 'calibration_fit_pair_allowlist')}
        self.records, self.source_groups = member.records, member.source_groups
        self.source_dataset_sha256, self.approved = member.source_dataset_sha256, member.approved
        self.evidence = _EvidenceFiles(member.proof, self.receipt['source_hashes'])
        self._documents = {}
        self.pairs = _PairIndex(store, identity)
        self.by_example = _CasePairs(store, self.records)


class ContextualAdmission:
    """Verify the whole declared source set before exposing any admitted records."""

    def __init__(self, declaration, approval, member_approvals, *, approval_sha256, proof, progress=None):
        require(isinstance(approval_sha256, str) and
                re.fullmatch(r'sha256:[0-9a-f]{64}', approval_sha256), 'contextual_approval_pin_required')
        require(approval == approval_sha256, 'contextual_approval_digest')
        # The independently pinned joint bytes authenticate every member pin before
        # a member is reconstructed. Object placement conveys no authority.
        joint = read_json(proof.read(approval), source=True)
        declared = read_json(proof.read(declaration), source=True)
        require(set(declared) == {'format', 'purpose', 'fixture_only', 'members'} and
                declared['format'] == 'contextual-source-set-v2' and
                declared['purpose'] == 'local-canonical-integration' and
                type(declared['fixture_only']) is bool, 'contextual_source_set_contract')
        members = declared['members']
        require(isinstance(members, list) and members, 'contextual_source_set_empty')
        require(isinstance(member_approvals, list) and len(member_approvals) == len(members),
                'contextual_member_authority_coverage')
        require(joint.get('members') == [
            {key: item[key] for key in ('member_sha256', 'approval_sha256')}
            for item in member_approvals], 'contextual_member_authority_coverage')
        identities, admitted = [], []
        store = _AdmissionStore()
        stored_member_ids = set()
        for item, authority in zip(members, member_approvals):
            require(set(item) == {'identity', 'reference'} and
                    set(authority) == {'member_sha256', 'approval', 'approval_sha256'},
                    'contextual_source_set_member')
            reference = item['reference']
            require(isinstance(reference, dict) and set(reference) ==
                    {'receipt', 'annotation', 'schema', 'evidence_map'},
                    'contextual_admission_reference_fields')
            member = _MemberAdmission(**reference, approval=authority['approval'],
                                      approval_sha256=authority['approval_sha256'], proof=proof)
            identity = member_identity(member)
            require(identity == item['identity'] and authority['member_sha256'] == digest(identity),
                    'contextual_member_identity')
            require(member.receipt.get('fixture_only') is declared['fixture_only'] and
                    member.approval.get('fixture_only') is declared['fixture_only'],
                    'contextual_fixture_authority_rejected')
            identity_hash = digest(identity)
            require(identity_hash not in stored_member_ids, 'contextual_source_set_order')
            stored_member_ids.add(identity_hash)
            identities.append(identity)
            store.retain(member, identity_hash)
            admitted.append(_AdmittedMember(member, store, identity_hash))
            del member
            if progress is not None:
                progress('source_members_verified', len(admitted), len(members))
        member_ids = [digest(identity) for identity in identities]
        require(member_ids == sorted(set(member_ids)), 'contextual_source_set_order')
        self.source_set_sha256 = digest({'purpose': declared['purpose'],
            'fixture_only': declared['fixture_only'], 'members': identities})
        expected_joint = {'format': 'contextual-source-set-approval-v2',
            'status': 'approved_local_canonical_integration_only',
            'purpose': declared['purpose'], 'fixture_only': declared['fixture_only'],
            'source_set_sha256': self.source_set_sha256, 'runtime_revision': file_digest(__file__),
            'declaration_sha256': declaration,
            'members': [{'member_sha256': key, 'approval_sha256': authority['approval_sha256']}
                        for key, authority in zip(member_ids, member_approvals)]}
        require(joint == expected_joint, 'contextual_joint_authority_binding')
        self.approval = joint
        check_source_set(admitted)
        self.records, self.source_groups, self.owners = {}, {}, {}
        self.pairs = _PairIndex(store)
        self.by_example = _CasePairs(store, self.records)
        for member in admitted:
            for example, row in member.records.items():
                self.records[example] = row
                self.owners[example] = member
            self.source_groups.update(member.source_groups)
        self.approved = frozenset(pair for member in admitted for pair in member.approved)
        self.authorized = True
        self.binding = {'version': 1, 'target_sha256': identities[0]['target_sha256'],
            'catalog_sha256': identities[0]['catalog_sha256'], 'source_set_sha256': self.source_set_sha256,
            'runtime_revision': file_digest(__file__), 'approval_sha256': approval_sha256,
            'purpose': declared['purpose'], 'fixture_only': declared['fixture_only'],
            'members': [{'member_sha256': key, 'identity': identity,
                         'approval_sha256': authority['approval_sha256']}
                        for key, identity, authority in zip(member_ids, identities, member_approvals)]}
        self.references = {'declaration': declaration, 'approval': approval,
                           'member_approvals': member_approvals}
        self.declaration = declared
        self.proof = proof
        proof.complete()

    def record_binding(self, example):
        require(example in self.owners, 'contextual_record_not_reviewed')
        return {'binding_sha256': digest(self.binding),
                'pair_ids': sorted(pair['pair_id'] for pair in self.by_example[example])}

    def validate_record(self, row, catalog):
        example = row['example_id']
        require(example in self.owners, 'contextual_record_not_reviewed')
        require(row.get('contextual') == self.record_binding(example), 'contextual_record_binding')
        member = self.owners[example]
        member.validate_record(row, catalog)


def check_source_set(members):
    """Global checks shared by proposal reconstruction and trusted admission."""
    require(members, 'contextual_source_set_empty')
    identities = [member_identity(member) for member in members]
    require(len({identity['target_sha256'] for identity in identities}) == 1 and
            len({identity['catalog_sha256'] for identity in identities}) == 1,
            'contextual_source_set_common_contract')
    examples, pairs, inputs, groups = set(), set(), set(), {}
    for member, identity in zip(members, identities):
        key = digest(identity)
        require(not pairs & member.pairs.keys(), 'contextual_duplicate_identity')
        pairs.update(member.pairs)
        for example, row in member.records.items():
            require(example not in examples, 'contextual_duplicate_source')
            examples.add(example)
            for field in ('conversation_group', 'template_group'):
                group = row['source'][field]
                if group is not None:
                    require(groups.setdefault((field, group), (key, row['split'])) == (key, row['split']),
                            'contextual_cross_source_group')
            input_key = digest(normalized_model_input(row['input']))
            require(input_key not in inputs, 'contextual_duplicate_model_input')
            inputs.add(input_key)


def member_identity(member):
    """Content identity deliberately excludes approval hashes and filesystem paths."""
    catalog_ids = {row['catalog_sha256'] for row in member.records.values()}
    require(len(catalog_ids) == 1, 'contextual_catalog_binding')
    return {**{key: value for key, value in member.binding.items() if key != 'approval_sha256'},
        'catalog_sha256': next(iter(catalog_ids)), 'source_dataset_sha256': member.source_dataset_sha256,
        **{key: member.receipt[key] for key in
           ('dataset_schema_sha256', 'source_contract_sha256', 'role_plan_sha256', 'identity_map_sha256')},
        'projection_sha256': digest(PAIR_PROJECTION)}


def load_contextual_admission(reference, approval_sha256, *, proof, progress=None):
    require(isinstance(reference, dict) and
            set(reference) == {'declaration', 'approval', 'member_approvals'},
            'contextual_admission_reference_fields')
    return ContextualAdmission(**reference, approval_sha256=approval_sha256, proof=proof, progress=progress)


def validate_record(row, catalog, *, index=None, audit_dir=None, schema_dir=ROOT / 'schemas', accepted=True, admission=None):
    """Validate one canonical dataset row. Returns the unchanged row; raises fixed codes."""
    if accepted:
        require(admission is not None and 'contextual' in row, 'missing_contextual_target')
        admission.validate_record(row, catalog)
    checks = validators(schema_dir)
    entries = validate_catalog(catalog, checks['catalog'])
    schema_check(checks['dataset'], row)
    require(row['catalog_sha256'] == catalog['catalog_sha256'], 'changed_catalog_evidence')
    source, inputs = row['source'], row['input']
    if not authored_contextual(row):
        source_check(row, index)
    if source['label_basis'] == 'historical_snapshot':
        require(timestamp(catalog['captured_at']) <= timestamp(source['cutoff_utc']), 'future_historical_catalog')
    candidates = set(inputs['candidate_ids'])
    require(candidates <= entries.keys(), 'candidate_membership')
    for field in ('required_ids', 'excluded_ids', 'explicit_ids'):
        require(set(inputs[field]) <= entries.keys(), 'policy_membership')
    require(set(inputs['required_ids']) <= candidates and not set(inputs['required_ids']) & set(inputs['excluded_ids']), 'policy_conflict')
    require(all(entries[key]['explicit_eligible'] for key in inputs['required_ids']), 'required_ineligible')
    eligible = {key for key, entry in entries.items() if entry['automatic_eligible'] and key not in inputs['excluded_ids']} | set(inputs['required_ids'])
    require(candidates <= eligible, 'candidate_eligibility')
    if inputs['candidate_coverage'] == 'full':
        require(candidates == eligible, 'incomplete_full_catalog')
    if authored_contextual(row):
        require(admission is not None, 'contextual_source_kind_authority')
        if not accepted:
            admission.validate_record(row, catalog)
        require(audit_dir is not None, 'audit_directory_required')
        policy_check(row, audit_dir)
        scan_record(row, entries)
        return row
    label = row['label']
    positive, negative, unknown = (set(label[k]) for k in ('selected_ids', 'verified_negative_ids', 'unknown_ids'))
    require(not (positive & negative or positive & unknown or negative & unknown) and positive | negative | unknown == candidates, 'label_partition')
    if label['decision'] == 'uncertain':
        require(bool(unknown) or inputs['candidate_coverage'] != 'full', 'uncertain_without_unresolved')
    require(not positive & set(inputs['excluded_ids']), 'excluded_positive')
    evidence = {item['id']: item for item in inputs['evidence']}
    require(len(evidence) == len(inputs['evidence']), 'duplicate_evidence_id')
    for item in evidence.values():
        require(item['event_sequence'] <= source['event_sequence'] and timestamp(item['observed_at']) <= timestamp(source['cutoff_utc']), 'future_evidence')
    supports = {}
    for support in label['support']:
        skill = support['runtime_id']
        require(skill not in supports and skill in positive | negative, 'support_membership')
        require(support['judgment'] == ('positive' if skill in positive else 'negative'), 'support_judgment')
        require(set(support['evidence_ids']) <= evidence.keys(), 'support_evidence_missing')
        require(support['contract_sha256'] == entries[skill]['content_sha256'], 'changed_contract_evidence')
        require(re.fullmatch(r'(?:trigger|exclusion):[A-Za-z0-9_.-]+', support['clause_ref']) is not None, 'support_clause_ref')
        supports[skill] = support
    require(supports.keys() == positive | negative, 'unsupported_label')
    for alternative in label['acceptable_alternative_sets']:
        require(set(alternative) <= candidates and not set(alternative) & (negative | set(inputs['excluded_ids'])), 'alternative_conflict')
    loaded = {}
    for item in inputs['already_loaded']:
        require(item['runtime_id'] in entries and item['runtime_id'] not in loaded, 'loaded_membership')
        loaded[item['runtime_id']] = item['content_sha256']
    active = {skill for skill in positive if loaded.get(skill) != entries[skill]['content_sha256']}
    require(set(label['activate_now']) == active, 'activation_mismatch')
    if label['earliest_supported_event'] is not None:
        require(label['earliest_supported_event'] in {item['source_event_id'] for item in evidence.values()}, 'earliest_evidence_missing')
    require(row['privacy']['sanitizer_sha256'] == digest(SANITIZER), 'sanitizer_version')
    findings = scan_record(row, entries)
    if accepted or row['audit']['reconciliation_sha256'] is not None or row['privacy']['export_approved'] or (findings and row['privacy']['scan_passed']):
        require(audit_dir is not None, 'audit_directory_required')
        audit_check(row, catalog, audit_dir)
    elif audit_dir is not None:
        policy_check(row, audit_dir)
        if source['origin'] == 'synthetic':
            synthetic_check(row, audit_dir)
    return row


def dataset_to_runtime(row, catalog):
    """Return only canonical runtime content fields; host supplies fresh identity/digests."""
    entries = {entry['runtime_id']: entry for entry in catalog['entries']}
    inputs = row['input']
    action = inputs['action']
    return {'version': row['schema_version'], 'catalogDigest': row['catalog_sha256'], 'policyDigest': row['policy_sha256'],
            'event': 'tool_call' if row['event_kind'] == 'proposed_action' else 'context', 'intent': inputs['request'],
            'action': None if action is None else {'toolName': action['tool_name'], 'summary': action['summary']},
            'candidates': [{'id': key, 'name': key, 'description': entries[key]['description'], 'contentDigest': entries[key]['content_sha256']} for key in sorted(inputs['candidate_ids'])],
            'candidateCoverage': inputs['candidate_coverage'], 'requiredIds': sorted(inputs['required_ids']), 'excludedIds': sorted(inputs['excluded_ids'])}


def empty_pair_accounting():
    return {stage: {'0': 0, '1': 0} for stage in PAIR_ACCOUNTING_STAGES}


def add_pair_accounting(summary, split, accounting):
    totals = summary.setdefault(split, empty_pair_accounting())
    for stage in PAIR_ACCOUNTING_STAGES:
        for label in ('0', '1'):
            totals[stage][label] += accounting[stage][label]


def dataset_to_pairs(row, catalog, *, holdout=(), accounting=None, admission=None):
    """Project only independently approved contextual judgments, never broad labels."""
    require(admission is not None, 'missing_contextual_target')
    admission.validate_record(row, catalog)
    if row['split'] == 'quarantine' or row['audit']['status'] not in ('machine_audited', 'contextual_pair_audited'):
        return
    if accounting is None:
        accounting = empty_pair_accounting()
    inputs = row['input']
    source_action = inputs['action']
    action = None if source_action is None else {'toolName': source_action['tool_name'], 'summary': source_action['summary']}
    entries = {entry['runtime_id']: entry for entry in catalog['entries']}
    seen = {}
    binding_sha256 = digest(admission.binding)
    for annotation in sorted(admission.by_example[row['example_id']], key=lambda pair: pair['pair_id']):
        skill = annotation['runtime_id']
        if (skill in ('unslop', 'emil-unslop-writing') or annotation['status'] == 'unresolved'
                or annotation['disposition'] in ('ineligible', 'diagnostic')):
            continue
        masked = annotation['family_mask']
        if 'purposes' in annotation and not masked:
            purpose = annotation['purposes']
            if not (purpose['calibration_fit'] or purpose['evaluation']):
                continue
        label = int(annotation['status'] == 'contextual')
        class_key = str(label)
        accounting['pre_mask'][class_key] += 1
        if masked:
            accounting['masked'][class_key] += 1
            continue
        require(annotation['pair_id'] in admission.approved, 'contextual_pair_not_approved')
        accounting['projected'][class_key] += 1
        description = entries[skill]['description']
        key = normalized(description)
        contributor = {'pair_id': annotation['pair_id'], 'annotation_sha256': digest(annotation),
                       'identities': annotation['identities'], 'runtime_id': skill,
                       'body_sha256': annotation['body_sha256'], 'input_sha256': annotation['input_sha256'],
                       'request_sha256': annotation['request_sha256'], 'action_sha256': digest(action),
                       'source_record_sha256': digest(admission.records[row['example_id']])}
        if key in seen:
            # Check all contributors before yielding. An opposite label must not
            # leave a partially emitted model input behind.
            require(seen[key]['description'] == description and seen[key]['label'] == label,
                    'duplicate_training_pair')
            seen[key]['contextual']['contributors'].append(contributor)
            accounting['deduplicated'][class_key] += 1
            continue
        accounting['emitted'][class_key] += 1
        seen[key] = {'intent': inputs['request'], 'action': action, 'description': description, 'label': label,
                     'group_id': row['source']['conversation_group'], 'split': row['split'],
                     'contextual': {'binding_sha256': binding_sha256, 'contributors': [contributor]}}
    yield from seen.values()


def split_for(row, index, entries, holdout, audit_dir=None, reserved_groups=(), *, recorded_unseen_groups=()):
    """Keep operator reservations separate from previously recorded unseen membership."""
    if authored_contextual(row):
        require(row['source']['conversation_group'] not in reserved_groups, 'contextual_authored_reserved_group')
        return row['split']
    if row['audit']['status'] != 'machine_audited' or not row['privacy']['scan_passed']:
        return 'quarantine'
    source = row['source']
    if source['origin'] == 'history_abstraction':
        found = index.execute('SELECT split FROM lineage WHERE component=? LIMIT 1', (source['conversation_group'],)).fetchone()
        require(found is not None, 'source_group_missing')
        split = found[0]
    else:
        split = synthetic_split(source['template_group'])
    if split == 'quarantine':
        return split
    if (source['conversation_group'] in reserved_groups
            or source['conversation_group'] in recorded_unseen_groups
            or any(entries[skill]['family_id'] in holdout for skill in row['label']['selected_ids'])):
        return 'unseen_family_test'
    return split


def audit_summary_add(summary, row, audit_dir):
    """Retain actual settings once; count correlation without claiming family independence."""
    if authored_contextual(row):
        summary['pair_scoped_rows'] = summary.get('pair_scoped_rows', 0) + 1
        return
    families = {}
    for reviewer in row['audit']['reviewers']:
        key = reviewer['settings_sha256']
        settings = blob(audit_dir, key)
        summary['settings'][key] = settings
        families[reviewer['role']] = settings['model_family']
    summary['reviewed_rows'] += 1
    summary['same_family_rows'] += int(families['teacher_a'] == families['teacher_b'])




def freeze(dataset, output, catalog, *, index, audit_dir, holdout=(), reserved_groups=(), admission=None, progress=None):
    """Freeze whole groups and duplicate inputs using disk indexes, never fit labels."""
    require(admission is not None, 'missing_contextual_target')
    require(str(dataset) == admission.references['declaration'],
            'contextual_source_set_binding')
    entries = validate_catalog(catalog, validators()['catalog'])
    require(set(holdout) <= {entry['family_id'] for entry in entries.values()} and None not in holdout, 'unknown_holdout_family')
    if holdout:
        require(all(entry['family_id'] for entry in entries.values()), 'family_review_incomplete')
        equivalent = {}
        for entry in entries.values():
            for key in (entry['content_sha256'], normalized(entry['description'])):
                require(key not in equivalent or equivalent[key] == entry['family_id'], 'equivalent_family_split')
                equivalent[key] = entry['family_id']
    holdout, reserved_groups = set(holdout), set(reserved_groups)
    require(all(isinstance(group, str) and re.fullmatch(r'hmac:[0-9a-f]{64}', group) for group in reserved_groups), 'reserved_group_id')
    audit_summary = {'reviewed_rows': 0, 'same_family_rows': 0, 'settings': {}}
    pair_counts, masked_counts, pair_accounting = {}, {}, {}
    output = Path(output)
    with tempfile.TemporaryDirectory(prefix='dataset-freeze-', dir=output.parent) as temporary:
        db = sqlite3.connect(Path(temporary) / 'rows.sqlite')
        db.executescript('CREATE TABLE records(id TEXT PRIMARY KEY, group_id TEXT, split TEXT, payload BLOB); CREATE INDEX record_group ON records(group_id);')
        for completed, source in enumerate(sorted(admission.records.values(), key=lambda row: row['example_id']), 1):
            row = dict(source)
            row['contextual'] = admission.record_binding(row['example_id'])
            audit_dir = admission.owners[source['example_id']]
            validate_record(row, catalog, index=index, audit_dir=audit_dir, admission=admission)
            split = split_for(row, index, entries, holdout, audit_dir, reserved_groups)
            require(split == row['split'], 'contextual_reviewed_split_changed')
            try:
                db.execute('INSERT INTO records VALUES (?,?,?,?)', (row['example_id'], row['source']['conversation_group'], split, canonical_bytes(row)))
            except sqlite3.IntegrityError:
                raise ContractError('duplicate_example') from None
            if progress is not None:
                progress('freeze_records_validated', completed, len(admission.records))
        require({key for key, in db.execute('SELECT id FROM records')} == set(admission.records),
                'contextual_source_coverage')
        require(reserved_groups <= {group for group, in db.execute('SELECT DISTINCT group_id FROM records')}, 'reserved_group_missing')
        group_proofs = {}
        for payload, in db.execute('SELECT payload FROM records'):
            row = parse_json(payload)
            audit_dir = admission.owners[row['example_id']]
            if row['audit']['reconciliation_sha256'] is not None:
                proof = blob(audit_dir, row['audit']['reconciliation_sha256'])
                group = blob(audit_dir, proof['group_evidence_sha256'])
                prior = group_proofs.setdefault(group['group_id'], proof['group_evidence_sha256'])
                require(prior == proof['group_evidence_sha256'], 'reconciliation_group_disagreement')
                # The pilot is a partial annotation, not a replacement conversation.
                # Verify the entire sealed source group, not just sampled members.
                members = sorted(admission.source_groups[group['group_id']])
                require(members == group['example_ids'], 'reconciliation_group_membership')
        unseen_groups = {group for group, in db.execute("SELECT DISTINCT group_id FROM records WHERE split='unseen_family_test'")}
        output.mkdir(mode=0o700)
        with new_file(output / 'dataset.jsonl') as destination, new_file(output / 'splits.jsonl') as assignments:
            for completed, (example_id, group, split, payload) in enumerate(db.execute('SELECT * FROM records ORDER BY id'), 1):
                row = parse_json(payload)
                require(split == row['split'], 'contextual_reviewed_split_changed')
                audit_dir = admission.owners[row['example_id']]
                row['split'] = split
                if split != 'quarantine':
                    accounting = empty_pair_accounting()
                    emitted = sum(1 for _ in dataset_to_pairs(row, catalog, holdout=holdout, accounting=accounting, admission=admission))
                    add_pair_accounting(pair_accounting, split, accounting)
                    masked_counts[split] = masked_counts.get(split, 0) + sum(accounting['masked'].values())
                    if emitted:
                        pair_counts[split] = pair_counts.get(split, 0) + emitted
                    audit_summary_add(audit_summary, row, audit_dir)
                destination.write(canonical_bytes(row) + b'\n')
                assignments.write(canonical_bytes({'example_id': example_id, 'group_id': group, 'split': split}) + b'\n')
                if progress is not None:
                    progress('freeze_records_projected', completed, len(admission.records))
        counts = dict(db.execute('SELECT split,count(*) FROM records GROUP BY split'))
        db.close()
    recipe = {'schema_version': 7, 'boundaries': list(BOUNDARIES), 'holdout_families': sorted(holdout),
              'contextual_binding': admission.binding,
              'synthetic_policy': SYNTHETIC_POLICY, 'reserved_groups': sorted(reserved_groups),
              'unseen_groups': sorted(unseen_groups), 'pair_projection': PAIR_PROJECTION,
              'pair_counts': pair_counts, 'masked_pair_counts': masked_counts, 'audit_summary': audit_summary,
              'pair_accounting': pair_accounting,
              'pair_format': PAIR_FORMAT, 'sanitizer_sha256': digest(SANITIZER), 'source_sha256': admission.source_set_sha256}
    write_json(output / 'recipe.json', recipe)
    write_json(output / 'contextual-admission.json', admission.references)
    admission.proof.write(output / 'proof')
    manifest = artifact_manifest(output, 'split', catalog['catalog_sha256'], ['dataset.jsonl', 'splits.jsonl', 'recipe.json', 'contextual-admission.json'], file_digest(output / 'dataset.jsonl'), file_digest(output / 'splits.jsonl'), digest(recipe), 'private_local', contextual_binding=admission.binding)
    write_json(output / 'manifest.json', manifest)
    return {'counts': counts, 'artifact_sha256': manifest['artifact_sha256']}


def artifact_manifest(directory, kind, catalog_hash, filenames, dataset_hash, split_hash, recipe_hash, privacy, parents=(), *, contextual_binding=None):
    manifest = {'schema_version': 1, 'kind': kind, 'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'files': [{'path': name, 'sha256': file_digest(Path(directory) / name), 'bytes': (Path(directory) / name).stat().st_size} for name in sorted(filenames)],
                'parents': sorted(set(parents)), 'catalog_sha256': catalog_hash, 'dataset_sha256': dataset_hash, 'split_sha256': split_hash,
                'recipe_sha256': recipe_hash, 'base_model_revision': None, 'runtime_revision': file_digest(__file__),
                'privacy_class': privacy, 'publication_allowed': False}
    if kind in ('dataset', 'split'):
        require(contextual_binding is not None, 'missing_contextual_target')
        manifest['contextual_binding'] = contextual_binding
    manifest['artifact_sha256'] = digest({k: v for k, v in manifest.items() if k != 'created_at'})
    return manifest


def verify_manifest(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    checks = validators()
    schema = {'$ref': checks['catalog'].schema['$id'] + '#/$defs/artifactManifest'}
    schema_check(checks['catalog'].evolve(schema=schema), manifest)
    require(manifest['runtime_revision'] == file_digest(__file__), 'contextual_runtime_revision')
    require(manifest['artifact_sha256'] == digest({k: v for k, v in manifest.items() if k not in ('created_at', 'artifact_sha256')}), 'artifact_digest')
    names = [item['path'] for item in manifest['files']]
    require(names == sorted(set(names)), 'artifact_file_order')
    for item in manifest['files']:
        path = directory / item['path']
        require(not Path(item['path']).is_absolute() and '..' not in Path(item['path']).parts and path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink(), 'artifact_path')
        require(path.is_file() and path.stat().st_size == item['bytes'] and file_digest(path) == item['sha256'], 'artifact_file_digest')
    return manifest


CONTEXTUAL_EXPORT_FILES = frozenset(('audited-pairs.jsonl', 'pair-contexts.json', 'privacy-scan.json', 'contextual-admission.json', 'catalog.json', 'recipe.json', 'training-projection.json'))


def validate_contextual_recipe(recipe, admission):
    require(isinstance(recipe, dict) and recipe.get('contextual_binding') == admission.binding and
            recipe.get('source_sha256') == admission.source_set_sha256, 'contextual_recipe_binding')
    require(set(recipe) == {'schema_version', 'boundaries', 'holdout_families', 'pair_format', 'sanitizer_sha256', 'source_sha256',
                            'synthetic_policy', 'reserved_groups', 'unseen_groups', 'pair_projection', 'pair_counts',
                            'masked_pair_counts', 'audit_summary', 'pair_accounting', 'contextual_binding'}
            and recipe['schema_version'] == 7 and recipe['boundaries'] == list(BOUNDARIES)
            and recipe['pair_format'] == PAIR_FORMAT and recipe['sanitizer_sha256'] == digest(SANITIZER)
            and recipe['synthetic_policy'] == SYNTHETIC_POLICY and recipe['pair_projection'] == PAIR_PROJECTION,
            'frozen_recipe_contract')


def training_projection(admission, pair_hashes):
    """Locally verified consumption state. Original review evidence stays in proof/."""
    return {'format': 'admitted-training-projection-v1', 'binding': admission.binding,
            'references': admission.references, 'pair_sha256s': sorted(pair_hashes),
            'local_proof_commitment': digest(sorted(admission.proof.files)),
            'records': {key: {'split': row['split'], 'input': {name: row['input'][name] for name in
                ('request', 'action', 'required_ids', 'excluded_ids')}} for key, row in admission.records.items()},
            'by_example': {key: [{name: pair[name] for name in
                ('pair_id', 'runtime_id', 'family_mask', 'disposition', 'status')} |
                ({'purposes': pair['purposes']} if 'purposes' in pair else {}) for pair in pairs]
                for key, pairs in admission.by_example.items()}}


class AdmittedTrainingProjection:
    """Consumption view authenticated by a separately pinned whole-capsule permit.

    This is not source admission and cannot freeze or export a dataset. The local
    prepare_payload path replays full admission before these bytes can be reviewed.
    """
    def __init__(self, value):
        self.binding = value['binding']
        self.source_set_sha256 = self.binding['source_set_sha256']
        self.records = value['records']
        self.by_example = value['by_example']



def read_training_pairs(path, *, catalog=None):
    """Expand the sole disk encoding; authorization remains the verifier's job."""
    path = Path(path)
    table = read_json(path.parent / 'pair-contexts.json', source=True)
    code = 'contextual_input_source_mapping'
    require(isinstance(table, dict) and set(table) == {'format', 'binding_sha256', 'cases'} and
            table['format'] == 'contextual-pair-contexts-v2' and isinstance(table['cases'], dict), code)
    if catalog is None:
        catalog = read_json(path.parent / 'catalog.json')
    entries = {entry['runtime_id']: entry for entry in catalog['entries']}
    used = set()
    for row in rows(path):
        require(isinstance(row, dict) and set(row) == {'label', 'contextual'}, code)
        contextual = row['contextual']
        require(isinstance(contextual, dict) and set(contextual) == {'binding_sha256', 'row_sha256', 'contributors'} and
                contextual['binding_sha256'] == table['binding_sha256'] and
                isinstance(contextual['contributors'], list) and contextual['contributors'], code)
        contributors = []
        inputs, description = None, None
        for compact in contextual['contributors']:
            require(isinstance(compact, dict) and set(compact) == {'pair_id', 'annotation_sha256', 'runtime_id', 'case_id'} and
                    isinstance(compact['case_id'], str) and compact['case_id'] in table['cases'] and
                    isinstance(compact['runtime_id'], str) and compact['runtime_id'] in entries, code)
            case_id = compact['case_id']
            common = table['cases'][case_id]
            if case_id not in used:
                require(isinstance(common, dict) and set(common) == set(portable.PAIR_CONTEXT_FIELDS + portable.PAIR_INPUT_FIELDS) and
                        isinstance(common['identities'], dict) and common['identities'].get('example_id') == case_id, code)
                used.add(case_id)
            case_inputs = {key: common[key] for key in portable.PAIR_INPUT_FIELDS}
            entry = entries[compact['runtime_id']]
            if inputs is None:
                inputs, description = case_inputs, entry['description']
            require(case_inputs == inputs and entry['description'] == description, code)
            contributors.append({**{key: common[key] for key in portable.PAIR_CONTEXT_FIELDS},
                                 **{key: compact[key] for key in ('pair_id', 'annotation_sha256', 'runtime_id')},
                                 'body_sha256': entry['content_sha256']})
        expanded = {**inputs, 'description': description, 'label': row['label'],
                    'contextual': {'binding_sha256': contextual['binding_sha256'], 'contributors': contributors}}
        require(digest(expanded) == contextual['row_sha256'], code)
        yield expanded
    require(used == set(table['cases']), code)

def verify_export_metadata(directory):
    """Common export integrity and privacy checks for local and remote consumers."""
    directory = Path(directory)
    manifest = verify_manifest(directory)
    require('contextual_binding' in manifest, 'missing_contextual_target')
    require({item['path'] for item in manifest['files']} == CONTEXTUAL_EXPORT_FILES, 'contextual_export_files')
    scan = read_json(directory / 'privacy-scan.json')
    require(isinstance(scan, dict) and set(scan) == {
        'schema_version', 'scan_passed', 'export_approved', 'publication_allowed', 'anonymity_proof',
        'pair_format', 'sanitizer_sha256', 'pairs_sha256', 'catalog_sha256', 'frozen_artifact_sha256',
        'split_sha256', 'recipe_sha256', 'counts', 'masked_pair_counts', 'pair_projection',
        'pair_accounting', 'audit_summary', 'privacy_reviews', 'contextual_binding', 'holdout_families'},
        'contextual_privacy_report_fields')
    recipe = read_json(directory / 'recipe.json')
    require(isinstance(recipe, dict) and digest(recipe) == manifest['recipe_sha256'], 'contextual_recipe_binding')
    require(all(key in recipe and recipe[key] == scan[other] for key, other in
                (('holdout_families', 'holdout_families'), ('pair_counts', 'counts'),
                 ('masked_pair_counts', 'masked_pair_counts'), ('pair_accounting', 'pair_accounting'),
                 ('audit_summary', 'audit_summary'))), 'contextual_recipe_report_binding')
    catalog = read_json(directory / 'catalog.json')
    validate_catalog(catalog, validators()['catalog'])
    require(manifest['kind'] == 'dataset' and manifest['privacy_class'] == 'private_local' and
            scan['schema_version'] == 3 and scan['export_approved'] is False and
            scan['scan_passed'] is True and scan['publication_allowed'] is False and
            scan['anonymity_proof'] is False and
            manifest['catalog_sha256'] == scan['catalog_sha256'] == catalog['catalog_sha256'] and
            manifest['dataset_sha256'] == scan['pairs_sha256'] == file_digest(directory / 'audited-pairs.jsonl') and
            manifest['split_sha256'] == scan['split_sha256'] and
            manifest['recipe_sha256'] == scan['recipe_sha256'] and
            manifest['parents'] == [scan['frozen_artifact_sha256']] and
            scan['sanitizer_sha256'] == digest(SANITIZER) and
            scan['pair_format'] == PAIR_FORMAT and scan['pair_projection'] == PAIR_PROJECTION,
            'contextual_export_contract')
    return manifest, scan, recipe, catalog


def verify_training_projection(directory, approval_sha256, manifest_file_sha256):
    """Verify permitted consumption bytes without shipping local review evidence."""
    directory = Path(directory)
    require(file_digest(directory / 'manifest.json') == manifest_file_sha256, 'contextual_manifest_binding')
    manifest, scan, recipe, catalog = verify_export_metadata(directory)
    value = read_json(directory / 'training-projection.json', source=True)
    require(set(value) == {'format', 'binding', 'references', 'pair_sha256s', 'local_proof_commitment', 'records', 'by_example'}
            and value['format'] == 'admitted-training-projection-v1', 'contextual_projection_contract')
    require(value['binding'] == manifest['contextual_binding'] and
            value['binding']['approval_sha256'] == approval_sha256 and
            value['references'] == read_json(directory / 'contextual-admission.json'), 'contextual_export_binding')
    admission = AdmittedTrainingProjection(value)
    validate_contextual_recipe(recipe, admission)
    pairs = {}
    binding_sha256 = digest(admission.binding)
    for pair in read_training_pairs(directory / 'audited-pairs.jsonl', catalog=catalog):
        key = digest(pair)
        require(key not in pairs, 'contextual_duplicate_projection')
        require(pair['contextual']['binding_sha256'] == binding_sha256, 'contextual_record_binding')
        pairs[key] = pair
    require(sorted(pairs) == value['pair_sha256s'], 'contextual_input_source_mapping')
    return manifest, scan, admission, pairs


def verify_contextual_export(directory, approval_sha256, *, proof=None, progress=None):
    """Verify the exact model-input mapping before any consumer loads training rows."""
    directory = Path(directory)
    manifest, scan, recipe, catalog = verify_export_metadata(directory)
    admission = load_contextual_admission(read_json(directory / 'contextual-admission.json'),
        approval_sha256, proof=proof if proof is not None else portable.SourceProof.load(directory / 'proof'), progress=progress)
    validate_contextual_recipe(recipe, admission)
    require(manifest['contextual_binding'] == scan['contextual_binding'] == admission.binding,
            'contextual_export_binding')
    expected, accounting, counts = {}, {}, {}
    reviews = []
    summary = {'reviewed_rows': 0, 'same_family_rows': 0, 'settings': {}}
    entries = {entry['runtime_id']: entry for entry in catalog['entries']}
    assignments = hashlib.sha256()
    for completed, source in enumerate(sorted(admission.records.values(), key=lambda row: row['example_id']), 1):
        assignments.update(canonical_bytes({'example_id': source['example_id'],
            'group_id': source['source']['conversation_group'], 'split': source['split']}) + b'\n')
        row = {**source, 'contextual': admission.record_binding(source['example_id'])}
        owner = admission.owners[source['example_id']]
        validate_record(row, catalog, audit_dir=owner, admission=admission)
        if row['split'] != 'quarantine':
            audit_summary_add(summary, row, owner)
            reviews.append({'record_sha256': digest(row), 'findings': scan_record(row, entries),
                'disposition': 'pair_scoped_local_only' if authored_contextual(row) else 'independently_approved_private',
                'reconciliation_sha256': row['audit']['reconciliation_sha256'],
                'reconciliation': blob(owner, row['audit']['reconciliation_sha256']) if row['audit']['reconciliation_sha256'] is not None else None,
                'reviews': [{key: reviewer[key] for key in ('role', 'settings_sha256', 'prompt_sha256', 'output_sha256')}
                            for reviewer in row['audit']['reviewers']]})
        record_accounting = empty_pair_accounting()
        for pair in dataset_to_pairs(row, catalog, holdout=scan['holdout_families'],
                                     accounting=record_accounting, admission=admission):
            key = digest(pair)
            require(key not in expected, 'contextual_duplicate_projection')
            expected[key] = pair
            counts[pair['split']] = counts.get(pair['split'], 0) + 1
        add_pair_accounting(accounting, row['split'], record_accounting)
        if progress is not None:
            progress('export_records_verified', completed, len(admission.records))
    observed = [digest(pair) for pair in read_training_pairs(directory / 'audited-pairs.jsonl', catalog=catalog)]
    require(len(observed) == len(set(observed)) and set(observed) == set(expected),
            'contextual_input_source_mapping')
    require(counts == scan['counts'] and accounting == scan['pair_accounting'], 'contextual_projection_counts')
    require(scan['privacy_reviews'] == reviews and scan['audit_summary'] == summary,
            'contextual_privacy_evidence_binding')
    require('sha256:' + assignments.hexdigest() == manifest['split_sha256'], 'contextual_split_binding')
    require(read_json(directory / 'training-projection.json', source=True) == training_projection(admission, expected),
            'contextual_projection_binding')
    return manifest, scan, admission, expected


def export_pairs(frozen, output, catalog, *, index, audit_dir, approval_sha256=None, progress=None):
    frozen, output = Path(frozen), Path(output)
    source_manifest = verify_manifest(frozen)
    require('contextual_binding' in source_manifest, 'missing_contextual_target')
    admission = load_contextual_admission(read_json(frozen / 'contextual-admission.json'),
        approval_sha256, proof=portable.SourceProof.load(frozen / 'proof'), progress=progress)
    require(source_manifest['contextual_binding'] == admission.binding, 'contextual_manifest_binding')
    require(source_manifest['kind'] == 'split' and source_manifest['catalog_sha256'] == catalog['catalog_sha256'], 'frozen_catalog_mismatch')
    require(source_manifest['dataset_sha256'] == file_digest(frozen / 'dataset.jsonl') and source_manifest['split_sha256'] == file_digest(frozen / 'splits.jsonl'), 'frozen_projection_digest')
    recipe = read_json(frozen / 'recipe.json')
    require(digest(recipe) == source_manifest['recipe_sha256'], 'frozen_recipe_digest')
    validate_contextual_recipe(recipe, admission)
    entries = validate_catalog(catalog, validators()['catalog'])
    output.mkdir(mode=0o700)
    counts = {}
    masked_counts, privacy_reviews = {}, []
    pair_accounting = {}
    pair_contexts = {}
    audit_summary = {'reviewed_rows': 0, 'same_family_rows': 0, 'settings': {}}
    holdout = set(recipe['holdout_families'])
    with tempfile.TemporaryDirectory(prefix='dataset-export-', dir=output) as temporary:
        db = sqlite3.connect(Path(temporary) / 'export.sqlite')
        db.executescript('CREATE TABLE groups(id TEXT PRIMARY KEY, split TEXT); CREATE TABLE pairs(key TEXT PRIMARY KEY, split TEXT, label INTEGER); CREATE TABLE assignments(id TEXT PRIMARY KEY, group_id TEXT, split TEXT);')
        for assignment in rows(frozen / 'splits.jsonl'):
            require(set(assignment) == {'example_id', 'group_id', 'split'}, 'split_assignment_fields')
            db.execute('INSERT INTO assignments VALUES (?,?,?)', (assignment['example_id'], assignment['group_id'], assignment['split']))
        require({identity for identity, in db.execute('SELECT id FROM assignments')} ==
                set(admission.records), 'contextual_source_coverage')
        # Publish model input only after every record and cross-record collision
        # has passed. A failed export must not leave a partial pairs file.
        staged_pairs = Path(temporary) / 'audited-pairs.jsonl'
        with new_file(staged_pairs) as stream:
            for completed, row in enumerate(rows(frozen / 'dataset.jsonl'), 1):
                assignment = db.execute('SELECT group_id,split FROM assignments WHERE id=?', (row['example_id'],)).fetchone()
                audit_dir = admission.owners[row['example_id']]
                require(assignment == (row['source']['conversation_group'], row['split']), 'split_assignment_mismatch')
                db.execute('DELETE FROM assignments WHERE id=?', (row['example_id'],))
                if row['split'] == 'quarantine':
                    continue
                validate_record(row, catalog, index=index, audit_dir=audit_dir, admission=admission)
                require(row['split'] == split_for(
                    row, index, entries, holdout, audit_dir, recipe['reserved_groups'],
                    recorded_unseen_groups=recipe['unseen_groups']), 'frozen_split_policy')
                if not authored_contextual(row):
                    require(row['privacy']['classification'] == 'approved_remote_private' and row['privacy']['export_approved'] and row['privacy']['scan_passed'], 'private_export_not_approved')
                audit_summary_add(audit_summary, row, audit_dir)
                accounting = empty_pair_accounting()
                privacy_reviews.append({'record_sha256': digest(row), 'findings': scan_record(row, entries),
                                        'disposition': 'pair_scoped_local_only' if authored_contextual(row) else 'independently_approved_private',
                                        'reconciliation_sha256': row['audit']['reconciliation_sha256'],
                                        'reconciliation': blob(audit_dir, row['audit']['reconciliation_sha256']) if row['audit']['reconciliation_sha256'] is not None else None,
                                        'reviews': [{key: reviewer[key] for key in ('role', 'settings_sha256', 'prompt_sha256', 'output_sha256')}
                                                    for reviewer in row['audit']['reviewers']]})
                group = row['source']['conversation_group']
                old = db.execute('SELECT split FROM groups WHERE id=?', (group,)).fetchone()
                require(old is None or old[0] == row['split'], 'cross_split_group')
                db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)', (group, row['split']))
                rendered_key = normalized_model_input(row['input'])
                for pair in dataset_to_pairs(row, catalog, holdout=holdout, accounting=accounting, admission=admission):
                    key = digest([rendered_key, normalized(pair['description'])])
                    require(db.execute('SELECT 1 FROM pairs WHERE key=?', (key,)).fetchone() is None, 'duplicate_training_pair')
                    db.execute('INSERT INTO pairs VALUES (?,?,?)', (key, pair['split'], pair['label']))
                    stream.write(canonical_bytes(portable.compact_training_pair(pair, pair_contexts)) + b'\n')
                    counts[pair['split']] = counts.get(pair['split'], 0) + 1
                add_pair_accounting(pair_accounting, row['split'], accounting)
                masked_counts[row['split']] = masked_counts.get(row['split'], 0) + sum(accounting['masked'].values())
                if progress is not None:
                    progress('export_records_projected', completed, len(admission.records))
        require(db.execute('SELECT count(*) FROM assignments').fetchone()[0] == 0, 'unused_split_assignment')
        require(sum(counts.values()) > 0, 'empty_pair_export')
        require(counts == recipe['pair_counts'] and masked_counts == recipe['masked_pair_counts'], 'frozen_pair_counts')
        require(pair_accounting == recipe['pair_accounting'], 'frozen_pair_accounting')
        require(audit_summary == recipe['audit_summary'], 'frozen_audit_summary')
        db.close()
        os.replace(staged_pairs, output / 'audited-pairs.jsonl')
    report = {'schema_version': 3, 'scan_passed': True, 'export_approved': False, 'publication_allowed': False, 'anonymity_proof': False,
              'contextual_binding': admission.binding, 'holdout_families': sorted(holdout),
              'pair_format': PAIR_FORMAT, 'sanitizer_sha256': digest(SANITIZER), 'pairs_sha256': file_digest(output / 'audited-pairs.jsonl'),
              'catalog_sha256': catalog['catalog_sha256'], 'frozen_artifact_sha256': source_manifest['artifact_sha256'],
              'split_sha256': source_manifest['split_sha256'], 'recipe_sha256': source_manifest['recipe_sha256'],
              'counts': counts, 'masked_pair_counts': masked_counts, 'pair_projection': PAIR_PROJECTION,
              'pair_accounting': pair_accounting,
              'audit_summary': audit_summary, 'privacy_reviews': privacy_reviews}
    report_bytes = canonical_bytes(report) + b'\n'
    require(len(report_bytes) <= MAX_PRIVACY_REPORT_BYTES, 'privacy_report_capacity')
    with new_file(output / 'privacy-scan.json') as stream:
        stream.write(report_bytes)
    write_json(output / 'contextual-admission.json', admission.references)
    admission.proof.write(output / 'proof')
    write_json(output / 'catalog.json', catalog)
    write_json(output / 'recipe.json', recipe)
    write_json(output / 'pair-contexts.json', {'format': 'contextual-pair-contexts-v2',
        'binding_sha256': digest(admission.binding), 'cases': pair_contexts})
    pair_hashes = [digest(pair) for pair in read_training_pairs(output / 'audited-pairs.jsonl', catalog=catalog)]
    write_json(output / 'training-projection.json', training_projection(admission, pair_hashes))
    manifest = artifact_manifest(output, 'dataset', catalog['catalog_sha256'], sorted(CONTEXTUAL_EXPORT_FILES), report['pairs_sha256'], source_manifest['split_sha256'], source_manifest['recipe_sha256'], 'private_local', [source_manifest['artifact_sha256']], contextual_binding=admission.binding)
    write_json(output / 'manifest.json', manifest)
    # The trainer binds the manifest's exact bytes; the launcher checks its privacy gate.
    return {'pairs': sum(counts.values()), 'artifact_sha256': manifest['artifact_sha256']}


def parser():
    cli = PrivateArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument('--version', action='version', version='build-dataset 1')
    commands = cli.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory', help='Index OMP v3 files locally without exporting message text.')
    inv.add_argument('--source-root', required=True, type=Path)
    inv.add_argument('--output', required=True, type=Path)
    inv.add_argument('--key-file', required=True, type=Path)
    inv.add_argument('--excluded-project', required=True, type=Path)
    inv.add_argument('--links', type=Path, help='Reviewed local ancestry links; see docs/retraining.md.')
    for name in ('extract', 'validate', 'freeze', 'export'):
        command = commands.add_parser(name, help={'extract': 'Check separately authored canonical pending abstractions; keep quarantine.', 'validate': 'Validate schema, provenance and real audit artifacts.', 'freeze': 'Freeze temporal, lineage and family split assignments.', 'export': 'Export approved private structured intent-action-window-bag-v2 pairs only.'}[name])
        command.add_argument('--catalog', required=True, type=Path)
        command.add_argument('--index', type=Path)
        command.add_argument('--audit-dir', type=Path, required=name != 'extract')
        command.add_argument('--input', required=True, type=Path)
        if name != 'extract':
            command.add_argument('--approval-sha256', required=True, help='Operator-pinned external approval digest; never read from the input payload.')
        if name in ('freeze', 'validate'):
            command.add_argument('--contextual-admission', required=True, type=Path, help='Local sealed evidence references, without approval authority.')
        if name != 'validate':
            command.add_argument('--output', required=True, type=Path)
        if name == 'freeze':
            command.add_argument('--holdout-family', action='append', default=[])
            command.add_argument('--reserved-group', action='append', default=[], help='Pre-label reserved holdout-intent group; repeat for each group.')
    return cli


def run(args):
    if args.command == 'inventory':
        return inventory(args.source_root, args.output, args.key_file, args.excluded_project, args.links)
    catalog = read_json(args.catalog)
    validate_catalog(catalog, validators()['catalog'])
    admission = load_contextual_admission(read_json(args.contextual_admission), args.approval_sha256,
        proof=portable.SourceProof.load(args.contextual_admission.parent / 'proof')) if args.command in ('freeze', 'validate') else None
    with contextlib.closing(open_index(args.index)) if args.index else contextlib.nullcontext(None) as index:
        if args.command == 'freeze':
            return freeze(args.input, args.output, catalog, index=index, audit_dir=args.audit_dir,
                          holdout=args.holdout_family, reserved_groups=args.reserved_group, admission=admission)
        if args.command == 'export':
            return export_pairs(args.input, args.output, catalog, index=index, audit_dir=args.audit_dir, approval_sha256=args.approval_sha256)
        count = 0
        with new_file(args.output) if args.command == 'extract' else contextlib.nullcontext(None) as destination:
            for row in rows(args.input):
                if args.command == 'extract':
                    require(row['split'] == 'quarantine' and row['audit']['status'] == 'pending' and not row['audit']['reviewers'] and not row['privacy']['export_approved'], 'extraction_must_be_pending')
                    require(not row['label']['selected_ids'] and not row['label']['verified_negative_ids'], 'extraction_cannot_label')
                audit = admission.owners[row['example_id']] if admission is not None else args.audit_dir
                validate_record(row, catalog, index=index, audit_dir=audit, accepted=args.command == 'validate', admission=admission)
                if destination:
                    destination.write(canonical_bytes(row) + b'\n')
                count += 1
        return {'records': count}


def main():
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        return cli_failure(error)


if __name__ == '__main__':
    sys.exit(main())
