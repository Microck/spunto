#!/usr/bin/env python3
"""Closed private input snapshots. This module has no provider or ML dependencies."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
MAX_ENTRIES = 8192
MAX_TOTAL = 1024 ** 3
# Bounded local source evidence, separate from model weights.
MAX_SOURCE_TOTAL = 8 * 1024 ** 3
MAX_SOURCE_ENTRIES = 16384
MAX_FILE = MAX_TOTAL
MAX_JSON = 64 * 1024 ** 2
MAX_DEPTH = 128
DIGEST = re.compile(r'sha256:[0-9a-f]{64}')
TEACHER_ARGUMENTS = ('teacher_model', 'teacher_revision', 'teacher_instruction_sha256',
                     'teacher_candidate_rule', 'distillation_alpha', 'distillation_temperature')



class InputError(ValueError):
    """Fixed diagnostics only. Never include source text or paths."""


def require(condition, code):
    if not condition:
        raise InputError(code)


def exact(value, fields, code):
    require(isinstance(value, dict) and set(value) == set(fields.split()), code)


def sha(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    require(isinstance(value, str) and DIGEST.fullmatch(value) is not None, 'private_digest')
    return value


def canonical(value):
    # Semantic identities use integers and decimal strings, never floating point.
    def check(item, depth=0):
        require(depth <= MAX_DEPTH, 'private_json_depth')
        if item is None or type(item) in (str, bool):
            return
        if type(item) is int:
            require(abs(item) <= 9007199254740991, 'private_json_number')
            return
        require(type(item) in (dict, list), 'private_json_type')
        if isinstance(item, dict):
            require(all(type(key) is str for key in item), 'private_json_key')
            item = item.values()
        for child in item:
            check(child, depth + 1)
    check(value)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (UnicodeError, ValueError, RecursionError):
        raise InputError('private_json_encoding') from None


def digest(value):
    return sha(canonical(value))


PAIR_CONTEXT_FIELDS = ('identities', 'input_sha256', 'request_sha256', 'action_sha256', 'source_record_sha256')
PAIR_INPUT_FIELDS = ('intent', 'action', 'group_id', 'split')


def compact_training_pair(row, contexts=None):
    """One disk encoding; consumers authenticate the fully expanded semantic hash."""
    contextual = row['contextual']
    exact(contextual, 'binding_sha256 contributors', 'private_pair_contextual_fields')
    contributors = []
    for contributor in contextual['contributors']:
        exact(contributor, 'pair_id annotation_sha256 identities runtime_id body_sha256 input_sha256 request_sha256 action_sha256 source_record_sha256',
              'private_pair_contributor_fields')
        case_id = contributor['identities']['example_id']
        if contexts is not None:
            common = {key: contributor[key] for key in PAIR_CONTEXT_FIELDS} | {key: row[key] for key in PAIR_INPUT_FIELDS}
            require(contexts.setdefault(case_id, common) == common, 'private_pair_context_conflict')
        contributors.append({key: contributor[key] for key in ('pair_id', 'annotation_sha256', 'runtime_id')} |
                            {'case_id': case_id})
    return {'label': row['label'], 'contextual': {'binding_sha256': contextual['binding_sha256'],
                                               'row_sha256': digest(row), 'contributors': contributors}}


def parse(raw):
    require(type(raw) is bytes and len(raw) <= MAX_JSON, 'private_json_capacity')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'private_json_duplicate_key')
            result[key] = value
        return result
    def invalid(_):
        raise InputError('private_json_number')
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        canonical(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise InputError('private_json_encoding') from None


def safe_name(name):
    require(type(name) is str and len(name) <= 1024 and
            re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name) is not None and
            all(part not in ('.', '..') for part in name.split('/')), 'private_path')
    require(len(name.split('/')) <= 32, 'private_path_depth')
    return name


class SafeTree:
    """Read through held directory descriptors, never resolved ambient leaf paths."""

    def __init__(self, root, *, private_root=True):
        absolute = Path(root).absolute()
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in absolute.parts[1:]:
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = following
            if private_root:
                require(stat.S_IMODE(os.fstat(fd).st_mode) == 0o700, 'private_directory_mode')
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def close(self):
        os.close(self.fd)

    def read(self, name, maximum=MAX_FILE):
        parts = safe_name(name).split('/')
        fd = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = following
                require(stat.S_IMODE(os.fstat(fd).st_mode) == 0o700, 'private_directory_mode')
            leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                before = os.fstat(leaf)
                require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'private_regular_file')
                require(stat.S_IMODE(before.st_mode) == 0o600, 'private_file_mode')
                require(0 <= before.st_size <= maximum, 'private_file_capacity')
                with os.fdopen(os.dup(leaf), 'rb') as stream:
                    raw = stream.read(maximum + 1)
                after = os.fstat(leaf)
                require(len(raw) == before.st_size and
                        (before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                        (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'private_file_changed')
                return raw
            finally:
                os.close(leaf)
        finally:
            os.close(fd)

    def names(self, *, maximum_entries=MAX_ENTRIES):
        files, directories = set(), set()
        def visit(fd, prefix, depth):
            require(depth <= 32, 'private_path_depth')
            with os.scandir(fd) as entries:
                for entry in entries:
                    name = safe_name(prefix + entry.name)
                    metadata = entry.stat(follow_symlinks=False)
                    require(not stat.S_ISLNK(metadata.st_mode), 'private_regular_file')
                    require(len(files) + len(directories) < maximum_entries * 2, 'private_entry_capacity')
                    if stat.S_ISDIR(metadata.st_mode):
                        require(stat.S_IMODE(metadata.st_mode) == 0o700, 'private_directory_mode')
                        directories.add(name)
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            visit(child, name + '/', depth + 1)
                        finally:
                            os.close(child)
                    else:
                        require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, 'private_regular_file')
                        files.add(name)
        visit(self.fd, '', 0)
        return files, directories


@contextmanager
def tree(root, *, private_root=True):
    opened = None
    try:
        opened = SafeTree(root, private_root=private_root)
        yield opened
    except OSError:
        raise InputError('private_file_access') from None
    finally:
        if opened is not None:
            opened.close()


def inventory_entries(entries, *, roles, maximum=MAX_TOTAL, maximum_entries=MAX_ENTRIES):
    require(isinstance(entries, list) and 0 < len(entries) <= maximum_entries, 'private_entry_capacity')
    result, total = {}, 0
    for entry in entries:
        exact(entry, 'path bytes sha256 role', 'private_inventory_fields')
        name = safe_name(entry['path'])
        fingerprint(entry['sha256'])
        require(entry['role'] in roles, 'private_inventory_role')
        require(type(entry['bytes']) is int and 0 <= entry['bytes'] <= MAX_FILE, 'private_file_capacity')
        total += entry['bytes']
        require(total <= maximum, 'private_total_capacity')
        require(name not in result, 'private_inventory_duplicate')
        result[name] = entry
    require(list(result) == sorted(result), 'private_inventory_order')
    return result, total


def expected_directories(names):
    return {'/'.join(name.split('/')[:i]) for name in names for i in range(1, len(name.split('/')))}


class Snapshot:
    """Verified bytes are immutable; transports must not reopen the authoring tree."""

    def __init__(self, files):
        require(all(type(name) is str and type(raw) is bytes for name, raw in files.items()), 'private_snapshot_type')
        self.files = MappingProxyType(dict(files))

    def read(self, name):
        require(name in self.files, 'private_unlisted_file')
        return self.files[name]

    def json(self, name):
        return parse(self.read(name))

    def write(self, root):
        root = Path(root)
        names = {safe_name(name) for name in self.files}
        directories = expected_directories(names)
        require(not names & directories, 'private_inventory_collision')
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        root.chmod(0o700, follow_symlinks=False)
        # Create every level explicitly. pathlib's recursive mkdir applies its mode
        # only to the final level, and the caller's umask must not expose parents.
        for name in sorted(directories, key=lambda name: (name.count('/'), name)):
            destination = root / name
            destination.mkdir(mode=0o700)
            destination.chmod(0o700, follow_symlinks=False)
        # Completion markers are deliberately last. No transport may launch from a partial tree.
        for name in sorted(names, key=lambda name: (name == 'manifest.json', name)):
            destination = root / name
            with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(self.files[name])


def read_inventory(root, *, format_name, roles, extra=(), maximum=MAX_TOTAL):
    with tree(root) as source:
        manifest_raw = source.read('manifest.json', MAX_JSON)
        manifest = parse(manifest_raw)
        exact(manifest, 'format entries total_bytes', 'private_manifest_fields')
        require(manifest['format'] == format_name, 'private_manifest_format')
        entries, total = inventory_entries(manifest['entries'], roles=roles, maximum=maximum)
        require(type(manifest['total_bytes']) is int and manifest['total_bytes'] == total,
                'private_inventory_total')
        require('manifest.json' not in entries and not set(extra) & entries.keys(), 'private_inventory_self_reference')
        require(total + len(manifest_raw) <= maximum, 'private_total_capacity')
        names = set(entries) | {'manifest.json'} | set(extra)
        observed, directories = source.names()
        require(observed == names and directories == expected_directories(names), 'private_inventory_closure')
        files = {'manifest.json': manifest_raw}
        for name, entry in entries.items():
            raw = source.read(name, entry['bytes'])
            require(len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], 'private_inventory_digest')
            files[name] = raw
        consumed = total + len(manifest_raw)
        for name in extra:
            raw = source.read(name, min(MAX_JSON, maximum - consumed))
            consumed += len(raw)
            files[name] = raw
        return Snapshot(files), manifest


class _ProofFiles(Mapping):
    """Resolve a closed object index without retaining the whole review corpus in RAM."""

    def __init__(self, objects):
        self.objects = MappingProxyType(dict(objects))

    def __iter__(self):
        return iter(self.objects)

    def __len__(self):
        return len(self.objects)

    def __contains__(self, key):
        return key in self.objects

    def __getitem__(self, key):
        value = self.objects[key]
        if isinstance(value, bytes):
            return value
        root, _, size = value
        with tree(root) as source:
            raw = source.read('objects/' + key[7:], size)
        require(len(raw) == size and sha(raw) == key, 'private_object_digest')
        return raw


class SourceProof:
    """One finite CAS for both local and relocated source admission."""

    def __init__(self, files, *, sources=()):
        objects = {}
        for source in sources:
            objects.update(source.files.objects)
        for key, raw in files.items():
            fingerprint(key)
            require(type(raw) is bytes and len(raw) <= MAX_FILE and sha(raw) == key, 'private_object_digest')
            objects.setdefault(key, raw)
        self.files = _ProofFiles(objects)
        self.used = set()
        self._check_capacity()

    def _check_capacity(self):
        require(len(self.files) <= MAX_SOURCE_ENTRIES, 'private_entry_capacity')
        require(sum(len(v) if isinstance(v, bytes) else v[2]
                    for v in self.files.objects.values()) <= MAX_SOURCE_TOTAL, 'private_total_capacity')

    def subset(self, keys):
        keys = set(keys)
        require(keys <= self.files.keys(), 'private_object_missing')
        result = SourceProof({})
        result.files = _ProofFiles({key: self.files.objects[key] for key in keys})
        result._check_capacity()
        return result

    @classmethod
    def load(cls, root):
        root = Path(root).absolute()
        with tree(root) as source:
            manifest_raw = source.read('manifest.json', MAX_JSON)
            manifest_pin = sha(manifest_raw)
            manifest = parse(manifest_raw)
            exact(manifest, 'format entries total_bytes', 'private_manifest_fields')
            require(manifest['format'] == 'contextual-proof-v1', 'private_manifest_format')
            entries = [{k: v for k, v in entry.items() if k != 'source'} for entry in manifest['entries']]
            indexed, total = inventory_entries(entries, roles={'evidence'}, maximum=MAX_SOURCE_TOTAL,
                                               maximum_entries=MAX_SOURCE_ENTRIES)
            require(type(manifest['total_bytes']) is int and manifest['total_bytes'] == total and
                    total + len(manifest_raw) <= MAX_SOURCE_TOTAL, 'private_inventory_total')
            local_names = {'manifest.json'} | {entry['path'] for entry in manifest['entries'] if 'source' not in entry}
            names, directories = source.names(maximum_entries=MAX_SOURCE_ENTRIES)
            require(names == local_names and directories == expected_directories(names), 'private_inventory_closure')
            objects, origins = {}, {}
            for entry in manifest['entries']:
                key = entry['sha256']
                require(entry['path'] == 'objects/' + key[7:], 'private_object_name')
                if 'source' in entry:
                    exact(entry, 'path bytes sha256 role source', 'private_inventory_fields')
                    origin = entry['source']
                    exact(origin, 'root manifest_file_sha256', 'private_proof_source_fields')
                    origin_root = origin['root']
                    require(type(origin_root) is str and Path(origin_root).is_absolute() and
                            Path(origin_root) != root, 'private_proof_source_root')
                    fingerprint(origin['manifest_file_sha256'])
                    pin = (origin_root, origin['manifest_file_sha256'])
                    if pin not in origins:
                        with tree(origin_root) as original:
                            raw = original.read('manifest.json', MAX_JSON)
                            require(sha(raw) == pin[1], 'private_proof_source_digest')
                            document = parse(raw)
                            exact(document, 'format entries total_bytes', 'private_manifest_fields')
                            require(document['format'] == 'contextual-proof-v1', 'private_manifest_format')
                            original_entries, original_total = inventory_entries(
                                [{k: v for k, v in item.items() if k != 'source'}
                                 for item in document['entries']], roles={'evidence'}, maximum=MAX_SOURCE_TOTAL,
                                maximum_entries=MAX_SOURCE_ENTRIES)
                            require(document['total_bytes'] == original_total and
                                    original_total + len(raw) <= MAX_SOURCE_TOTAL, 'private_inventory_total')
                            local = {item['path'] for item in document['entries'] if 'source' not in item}
                            names, directories = original.names(maximum_entries=MAX_SOURCE_ENTRIES)
                            require(names == local | {'manifest.json'} and
                                    directories == expected_directories(names), 'private_inventory_closure')
                            origins[pin] = {name: original_entries[name] for name in local}
                    require(origins[pin].get(entry['path']) == indexed[entry['path']], 'private_proof_source_entry')
                    location = (origin_root, pin[1], entry['bytes'])
                    with tree(origin_root) as original:
                        raw = original.read(entry['path'], entry['bytes'])
                else:
                    location = (str(root), manifest_pin, entry['bytes'])
                    raw = source.read(entry['path'], entry['bytes'])
                require(len(raw) == entry['bytes'] and sha(raw) == key, 'private_inventory_digest')
                objects[key] = location
        result = cls({})
        result.files = _ProofFiles(objects)
        result._check_capacity()
        return result

    def read(self, key):
        fingerprint(key)
        require(key in self.files, 'private_object_missing')
        self.used.add(key)
        return self.files[key]

    def json(self, key):
        return parse(self.read(key))

    def complete(self):
        require(self.used == self.files.keys(), 'private_unreachable_object')

    def write(self, root):
        entries, files = [], {}
        for key, value in sorted(self.files.objects.items()):
            entry = {'path': 'objects/' + key[7:], 'sha256': key, 'role': 'evidence'}
            if isinstance(value, bytes):
                entry['bytes'] = len(value)
                files[entry['path']] = value
            else:
                origin, pin, size = value
                entry.update(bytes=size, source={'root': origin, 'manifest_file_sha256': pin})
            entries.append(entry)
        files['manifest.json'] = canonical({'format': 'contextual-proof-v1', 'entries': entries,
                                            'total_bytes': sum(entry['bytes'] for entry in entries)}) + b'\n'
        Snapshot(files).write(root)


def prepare_member(receipt, annotation, schema, evidence_paths):
    """Authoring-only input. Exact explicit paths, no admission and no authority issuance."""
    files = {}
    def add(path):
        path = Path(path).absolute()
        # Original reviewed inputs may predate the private capsule mode contract.
        # Use bounded no-follow reads without modifying those source permissions.
        require(not any(part.is_symlink() for part in (path, *path.parents)), 'private_regular_file')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'private_regular_file')
            require(info.st_size <= MAX_FILE, 'private_file_capacity')
            with os.fdopen(os.dup(fd), 'rb') as stream:
                raw = stream.read(MAX_FILE + 1)
            require(len(raw) == info.st_size, 'private_file_changed')
        finally:
            os.close(fd)
        key = sha(raw)
        files[key] = raw
        return key
    reference = {name: add(path) for name, path in
                 (('receipt', receipt), ('annotation', annotation), ('schema', schema))}
    proposed = parse(files[reference['receipt']])
    require(isinstance(evidence_paths, dict) and evidence_paths.keys() == proposed['source_hashes'].keys(),
            'private_member_namespace')
    mapping = {}
    for name, path in sorted(evidence_paths.items()):
        safe_name(name)
        key = add(path)
        require(key == proposed['source_hashes'][name], 'private_member_object_binding')
        mapping[name] = key
    raw = canonical(mapping) + b'\n'
    reference['evidence_map'] = sha(raw)
    files[sha(raw)] = raw
    return reference, SourceProof(files)

