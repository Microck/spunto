"""Hash-coherent inert bundles and an owned JSONL child. No ML, network or installation."""
import importlib.util
import json
from pathlib import Path
import shutil
import venv

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('fixture_retrieval', ROOT / 'python/retrieval.py')
retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval)


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def make_bundle(directory, binding, runtime_root=ROOT / 'python', *, trained=False):
    """A real retrieval-role manifest over inert bytes, for admission, never inference."""
    binding = retrieval.boundary.compact_source_set_binding(binding)
    directory.mkdir(mode=0o700)
    for name in ('encoder.onnx', 'tokenizer.json', 'config.json'):
        (directory / name).write_bytes(b'{}')
    save(directory / 'embeddings.json', {'format': 'content-addressed-skill-embeddings-v1', 'entries': []})
    comparison = {'passed': True, 'tolerance': 1e-4, 'max_absolute_embedding_error': 0,
                  'max_absolute_pooled_error': 0, 'ranking_pair_changes': 0}
    save(directory / 'parity.json', {'role': 'retrieval', 'nonprivate': True, 'cases': [{}, {}, {}, {}],
                                    'comparison': comparison})
    training = {'role': 'retrieval', 'weights': 'fine_tuned' if trained else 'unchanged_pretrained',
        'fit_split': 'validation' if trained else None, 'training_data_used': trained,
        'contextual_binding': binding, 'dataset_sha256': 'sha256:' + '6' * 64,
        'split_sha256': 'sha256:' + '7' * 64, 'training_recipe_sha256': 'sha256:' + '8' * 64 if trained else None}
    if trained:
        save(directory / 'training.json', {**training, 'chosen_epoch': 1})
    save(directory / 'recipe.json', {**training, 'export': {'variant': 'fp32'}})
    for name in ('evaluation.json', 'validation.json'):
        save(directory / name, {**training, 'comparison': comparison})
    manifest = {'schema_version': 1, 'role': 'retrieval', 'model': retrieval.MODEL, 'graphs': retrieval.GRAPHS,
        'tokenizer_file': 'tokenizer.json', 'model_config_file': 'config.json', 'pooling': retrieval.POOLING,
        'embedding_table': 'embeddings.json', 'selection': {'top_k': 8, 'tie_break': 'id'}, 'limits': retrieval.LIMITS,
        'provenance': {'contextual_binding': binding, 'catalog_sha256': binding['catalog_sha256'],
            **{name: training[name] for name in ('dataset_sha256', 'split_sha256', 'weights', 'training_recipe_sha256',
                                               'fit_split', 'training_data_used')},
            'training_report_sha256': retrieval.file_digest(directory / 'training.json') if trained else None,
            'recipe_sha256': retrieval.file_digest(directory / 'recipe.json'),
            'runtime_revision': retrieval.file_digest(runtime_root / 'retrieval-worker.py'),
            'runtime_files': {name: retrieval.file_digest(runtime_root / name)
                              for name in ('retrieval.py', 'router-worker.py', 'pair-windows.py')}},
        'files': [{'path': p.name, 'sha256': retrieval.file_digest(p), 'bytes': p.stat().st_size}
                  for p in sorted(directory.iterdir())], 'publication_allowed': True}
    manifest['scorer_sha256'] = retrieval.scorer_digest(manifest)
    manifest['selector_sha256'] = retrieval.selector_digest(manifest)
    save(directory / 'manifest.json', manifest)
    return manifest


TRANSPORT_WORKER = '''"""Disposable transport fixture, never a model runtime."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--artifact', type=Path, required=True)
parser.add_argument('--decision-artifact', type=Path, required=True)
args = parser.parse_args()
with (args.artifact.parent / 'owned-pids.jsonl').open('a') as log:
    log.write(json.dumps(os.getpid()) + '\\n')
if (args.artifact.parent / 'exit-startup').exists():
    print('backend_unavailable', file=sys.stderr, flush=True)
    raise SystemExit(7)
retrieval = json.loads((args.artifact / 'manifest.json').read_text())
decision = json.loads((args.decision_artifact / 'manifest.json').read_text())
spec = importlib.util.spec_from_file_location('transport_boundary', args.artifact.parent / 'python/router-worker.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
time.sleep(.9)
identity = ('version', 'requestId', 'sessionKey', 'generation', 'catalogDigest', 'contextDigest', 'policyDigest', 'selectorDigest')
for line in sys.stdin:
    request = json.loads(line)
    boundary.validate_request(request)
    with (args.artifact.parent / 'requests.jsonl').open('a') as log:
        log.write(json.dumps(request) + '\\n')
    if request['requestId'] == 'startup-decider' and (args.artifact.parent / 'stall-startup-decider').exists():
        time.sleep(9.5)
    if request['intent'] == 'cancel':
        time.sleep(2)
    if request['intent'] == 'crash':
        raise SystemExit(1)
    is_retrieval = request['selectorDigest'] == retrieval['selector_sha256']
    manifest = retrieval if is_retrieval else decision
    result = {key: request[key] for key in identity}
    selected = [] if is_retrieval else request['requiredIds']
    result.update(status='uncertain' if is_retrieval else 'selected' if selected else 'none', selectedIds=selected,
        reasonCode='inadequate_candidates' if is_retrieval else 'policy_only' if selected else 'no_match',
        scores={'kind': 'ranking', 'scorerDigest': manifest['scorer_sha256'], 'calibrationDigest': None,
                'values': [{'id': c['id'], 'score': -2.0} for c in request['candidates']]})
    if request['requestId'] == 'startup-decider' and (args.artifact.parent / 'reject-startup-decider').exists():
        result.update(status='uncertain', selectedIds=[], reasonCode='backend_unavailable', scores=None)
    print(json.dumps(result), flush=True)
'''


def make_pair(directory):
    from test.test_router_worker import ArtifactTests
    runtime = directory / 'python'
    runtime.mkdir(mode=0o700)
    for name in ('router-worker.py', 'pair-windows.py', 'retrieval.py'):
        shutil.copyfile(ROOT / 'python' / name, runtime / name)
    (runtime / 'retrieval-worker.py').write_text(TRANSPORT_WORKER)
    venv.create(directory / 'venv', with_pip=False, symlinks=True)
    decision = directory / 'decision'
    decision.mkdir(mode=0o700)
    manifest = ArtifactTests().make_bundle(decision)
    make_bundle(directory / 'retrieval', manifest['provenance']['contextual_binding'], runtime)
