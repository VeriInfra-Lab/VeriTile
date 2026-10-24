import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
BASE = Path('/tmp/veritile-comparator-restore-20261015')
ARCHIVE = BASE / 'public-entrypoints-93d38e39-evidence.tar.gz'
ROOT = BASE / 'public-entrypoints-93d38e39-evidence'
COMMIT = '93d38e399ac513eb98aa3624d9aeb1c21858757a'
sha = lambda data: hashlib.sha256(data).hexdigest()

def unique(pairs):
    result = {}
    for k, v in pairs:
        assert k not in result, ('duplicate JSON key', k)
        result[k] = v
    return result

def load(name):
    return json.loads(payload[name], object_pairs_hook=unique)

def safe(name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts
    assert '\\' not in name and str(p) == name and name != '.'

assert sha(ARCHIVE.read_bytes()) == '1ba741709746f29d0499d1560820f1bece91b99e70bde5934419c2db6bc67150'
payload, members, dirs = {}, set(), set()
with tarfile.open(ARCHIVE, 'r:gz') as archive:
    for m in archive.getmembers():
        safe(m.name)
        assert m.name not in members, ('duplicate archive member', m.name)
        members.add(m.name)
        p = PurePosixPath(m.name)
        assert p.parts[0] == ROOT.name
        assert not m.linkname and (m.isdir() or m.isfile()), (m.name, m.type)
        if m.isdir():
            dirs.add(m.name)
            continue
        relative = p.relative_to(ROOT.name).as_posix()
        data = archive.extractfile(m).read()
        assert len(data) == m.size
        assert (ROOT / relative).is_file() and not (ROOT / relative).is_symlink()
        assert (ROOT / relative).resolve().is_relative_to(ROOT.resolve())
        assert (ROOT / relative).read_bytes() == data
        payload[relative] = data
assert not any(PurePosixPath(p).parent.as_posix() in payload for p in payload)
hashes = {name: sha(data) for name, data in payload.items()}
assert hashes['artifact-receipt.json'] == '6857574900d6434983acc4b11385e5af3b17cf67a7f3ba060f6a6c9ea08e0ecf'
assert hashes['verification.json'] == '8185fe0a8ecfa9bcb08b592537530e6b08e0bf842e73fd57395f3e08189356e4'
assert hashes['raw/collection-receipt.json'] == '99a0a066ef7200894b43bc48a5c8dd03bf2ad9c8b1c5e4b77c7db313c7f828cf'

sums = {}
for line in payload['SHA256SUMS'].decode().splitlines():
    digest, name = line.split('  ', 1)
    safe(name)
    assert name not in sums
    assert hashes[name] == digest
    sums[name] = digest
assert set(sums) == set(payload) - {'SHA256SUMS'}
receipt = load('artifact-receipt.json')
assert receipt['commit'] == COMMIT
assert set(receipt['files']) == set(payload) - set(receipt['excludes'])
assert set(receipt['excludes']) == {'artifact-receipt.json', 'SHA256SUMS'}
for name, row in receipt['files'].items():
    safe(name)
    assert row['sha256'] == hashes[name] and row['bytes'] == len(payload[name])
collection = load('raw/collection-receipt.json')
assert {name[4:] for name in payload if name.startswith('raw/')} == set(collection['files']) | {'collection-receipt.json'}
for name, digest in collection['files'].items():
    safe(name)
    assert hashes['raw/' + name] == digest

frozen = load('provenance/current-source-inputs.json')
manifest = frozen['files']
assert len(manifest) == 7177
assert load('raw/run/source-after.json') == manifest
official_inputs = load('raw/official-inputs.json')
assert len(official_inputs) == 4303
assert load('raw/run/official-snapshot-source-after.json') == official_inputs
assert all(manifest[name] == digest for name, digest in official_inputs.items())
assert load('provenance/current-source-receipt.json')['files'] == manifest
binding = load('provenance/commit-binding.json')
assert binding['commit'] == COMMIT and binding['matches'] and not binding['errors'] and not binding['extra_commit_paths']
assert binding['source_manifest_sha256'] == hashes['provenance/current-source-inputs.json'] == collection['source_manifest_sha256']

verification = load('verification.json')
runs = load('raw/run/results.json')
assert verification['accepted'] and runs['accepted'] and runs['source_unchanged'] and collection['accepted']
assert len(runs['runs']) == len(verification['runs']) == 6
files = [r['source'] for r in runs['runs']]
assert len(set(files)) == 6 and runs['files'] == files
catalog_path = 'bench/review-checkpoints/2026-10-15/original-scope-extensions/catalog-supplement.json'
assert hashes['raw/catalog-supplement.json'] == manifest[catalog_path]

# Read every historical Git blob, independently of the producer's receipt.
tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', COMMIT], cwd=REPO)
entries = []
for entry in tree.split(b'\0'):
    if not entry:
        continue
    metadata, name = entry.split(b'\t', 1)
    mode, kind, oid = metadata.split()
    assert kind == b'blob'
    entries.append((name.decode(), oid.decode()))
assert {name for name, _ in entries} == set(manifest) and len(entries) == 7177
needed = set(files) | {'scripts/check_comparator.py', 'scripts/comparator_common.py',
    'scripts/setup-comparator.sh', 'scripts/kernel-manifest.tsv', 'bench/audit_trust_prep.py',
    'bench/current_catalog_vendor/spec_sheet.py', catalog_path}
blobs, total_bytes = {}, 0
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for name, oid in entries:
    proc.stdin.write((oid + '\n').encode())
    proc.stdin.flush()
    result_oid, kind, size = proc.stdout.readline().split()
    assert result_oid.decode() == oid and kind == b'blob'
    data = proc.stdout.read(int(size))
    assert len(data) == int(size) and proc.stdout.read(1) == b'\n'
    assert sha(data) == manifest[name], name
    total_bytes += len(data)
    if name in needed:
        blobs[name] = data
proc.stdin.close()
assert proc.wait() == 0
assert set(blobs) == needed
for leaf in ['check_comparator.py', 'comparator_common.py', 'setup-comparator.sh']:
    assert payload['provenance/official-runner/' + leaf] == blobs['scripts/' + leaf]

# Reconstruct the unchanged official runner's exact wrappers without replaying Lean.
runner_tree = ast.parse(blobs['scripts/check_comparator.py'].decode())
inventory = next(ast.literal_eval(node.value) for node in runner_tree.body
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INVENTORY' for t in node.targets))
manifest_rows = [line.split('\t') for line in blobs['scripts/kernel-manifest.tsv'].decode().splitlines()]
def audited_source(text, relative):
    lines = text.splitlines()
    names = list(dict.fromkeys(row[2] for row in manifest_rows
        if len(row) >= 5 and row[1] == relative and row[4] == 'proven'))
    new_lines = list(lines)
    if 'import VeriTile.Meta.StatementAudit' not in new_lines:
        last = max((i for i, line in enumerate(lines) if line.startswith('import ')), default=-1)
        new_lines.insert(last + 1, 'import VeriTile.Meta.StatementAudit')
    footer = ['', '-- ==== external trust audit (appended by bench/audit_trust.sh) ====', '#auditModuleAxioms']
    footer += [f'#axiomsClean {name}' for name in names]
    source_module = '.'.join((*Path(relative).parts[:3], 'Kernels'))
    footer += [f'#auditModuleSpecs from [{source_module}]' if f'import {source_module}' in lines else '#auditModuleSpecs']
    return '\n'.join(new_lines + footer) + '\n'

# Use only reviewed pure parsing helpers from the pinned vendored catalog parser.
parser_tree = ast.parse(blobs['bench/current_catalog_vendor/spec_sheet.py'].decode())
definitions = {'declaration_scan', 'split_decls', 'trim_decl', 'split_statement'}
constants = {'DECL_KW', 'DECL_RE'}
selected = [node for node in parser_tree.body if
    (isinstance(node, ast.FunctionDef) and node.name in definitions) or
    (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
parser = {'re': re}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-pure-catalog-parser>', 'exec'), parser)

catalog = load('raw/catalog-supplement.json')
declarations = [d for task in catalog['tasks'] for d in task['declarations'] if d['file'] in files]
assert len(declarations) == 11 and len({(d['file'], d['declaration']) for d in declarations}) == 11
mapping = load('raw/target-mapping.json')
assert len(mapping) == 26
expected_mapping, checked_runs, checked_declarations = [], [], []
tools = load('provenance/tool-provenance.json')
for index, (run, verified) in enumerate(zip(runs['runs'], verification['runs'])):
    prefix = f'raw/official-jobs/{index:02d}/'
    result = load(prefix + 'result.json')
    targets = load(prefix + 'targets.json')
    config = load(prefix + 'config.json')
    source = run['source']
    assert run['exit'] == verified['exit'] == result['comparator_exit'] == 0
    assert run['accepted'] and result['accepted'] and len(run['official_results']) == 1
    report = run['official_results'][0]
    assert report['result'] == result
    assert report['sha256'] == hashes[prefix + 'result.json'] == verified['official_result_sha256']
    assert result['source'] == verified['source'] == source and result['mode'] == 'source-replay'
    assert result['comparator_sha256'] == tools['tools']['comparator']
    assert run['argv'] == verified['argv'] == ['python3', 'scripts/check_comparator.py', '--file', source,
        '--trust', '--workspace', '/home/judge/current-library-workspace']
    assert len(targets['theorems']) == len(targets['aliases']) == result['theorem_count'] == verified['theorem_count']
    assert len(set(targets['theorems'])) == len(targets['theorems'])
    assert targets['aliases'] == [f'veritile_comparator_target_{i}' for i in range(result['theorem_count'])]
    assert config['theorem_names'] == targets['aliases']
    assert config['permitted_axioms'] == ['propext', 'Quot.sound', 'Classical.choice']
    assert config['enable_nanoda'] is False
    tag = 'File_' + sha((source + 'True').encode())[:24]
    assert config['challenge_module'] == config['solution_module'] == 'VeriTileComparator.' + tag
    metadata = f'/home/judge/current-library-workspace/project/.lake/comparator/{tag}/targets.json'
    footer = inventory.replace('__SELECTED__', '#[]').replace('__REPORT__', json.dumps(metadata))
    reconstructed = 'import Lean\n' + audited_source(blobs[source].decode(), source) + '\n' + footer
    assert payload[prefix + 'Source.lean'].decode() == reconstructed, source
    original_decls = {d['name']: d for d in parser['split_decls'](blobs[source].decode())}
    declarations_here = [d for d in declarations if d['file'] == source]
    for declaration in declarations_here:
        d = original_decls[declaration['declaration']]
        assert d['kind'] == declaration['kind'] == 'specification'
        assert sha(parser['split_statement'](d['text'])[0].encode()) == declaration['statement_sha256']
        hits = [n for n in targets['theorems'] if n == declaration['declaration'] or n.endswith('.' + declaration['declaration'])]
        assert len(hits) == 1
        checked_declarations.append({'file': source, 'declaration': declaration['declaration'],
            'theorem': hits[0], 'statement_sha256': declaration['statement_sha256']})
    for theorem, alias in zip(targets['theorems'], targets['aliases']):
        matched = [d for d in declarations_here if theorem == d['declaration'] or theorem.endswith('.' + d['declaration'])]
        expected_mapping.append({'file': source, 'theorem': theorem, 'alias': alias,
            'original_job': f'official-jobs/{index:02d}', 'catalog_declarations': matched})
    build = payload[prefix + 'build.log'].decode()
    log = payload[prefix + 'comparator.log'].decode()
    assert build.startswith(('ℹ', '✔', '⚠', 'info:', 'warning:'))
    assert re.search(r'Build completed successfully \(\d+ jobs\)\.\s*$', build)
    assert log.startswith('Running as unit:')
    for marker in ['Running Lean default kernel on solution.', 'Lean default kernel accepts the solution',
                   'Your solution is okay!', 'Finished with result: success', 'status=0/SUCCESS']:
        assert marker in log
    assert re.search(r'Memory peak: .+\n?$', log)
    assert not re.search(r'^(?:error:|uncaught exception:|\[FAIL\])', log, re.M)
    assert 'Axiom audit:' in build and 'Spec audit:' in build
    export_lines = [line for line in log.splitlines() if line.startswith('Exporting #[')]
    assert len(export_lines) == 2
    for line in export_lines:
        for alias in targets['aliases']:
            assert alias in line
    checked_runs.append({'file': source, 'targets': len(targets['theorems']),
        'registered_declarations': len(declarations_here), 'official_exit': 0, 'comparator_exit': 0,
        'result_sha256': hashes[prefix + 'result.json'], 'source_wrapper_sha256': hashes[prefix + 'Source.lean'],
        'source_wrapper_matches_pinned_runner': True,
        'build_log_sha256': hashes[prefix + 'build.log'], 'build_log_lines': len(build.splitlines()),
        'comparator_log_sha256': hashes[prefix + 'comparator.log'], 'comparator_log_lines': len(log.splitlines())})
assert mapping == expected_mapping
assert len(checked_declarations) == 11

controls = load('provenance/controls/comparator-controls/results.json')
assert len(controls) == 4
for control in controls:
    case = control['case']
    expected = 0 if case == 'positive' else 1
    assert control['exit'] == expected and control['expected_behavior_verified']
    challenge = f'provenance/controls/comparator-controls/{case}/Challenge.lean'
    assert control['challenge_before_sha256'] == control['challenge_after_sha256'] == hashes[challenge]
    log = payload[f'provenance/controls/comparator-controls/{case}/comparator.log'].decode()
    marker = {'positive': 'Your solution is okay!', 'changed-statement': 'theorem statement do not match',
        'sorry': 'sorryAx', 'source-write': 'permission denied'}[case]
    assert marker in log
isolation = load('provenance/controls/isolation-probe.json')
assert isolation['accepted'] and isolation['uid'] == 2000 and isolation['landlock_abi'] == 6
assert len(isolation['checks']) == 5 and all(row['exit'] == row['expected'] for row in isolation['checks'])
for relative in ['scripts/check_comparator.py', 'scripts/comparator_common.py', 'bench/audit_trust_prep.py']:
    assert isolation['media_hashes'][relative] == manifest[relative]
for tool, digest in tools['tools'].items():
    assert isolation['media_hashes']['bin/' + tool] == digest

transport = BASE / 'current-targeted-evidence.tar.gz'
assert sha(transport.read_bytes()) == verification['raw_archive_sha256'] == 'ce27a6015b63197f9ca14ffd6b8895146e90afb2d7c72245fa6179b58531a4d4'
transport_files = set()
with tarfile.open(transport, 'r:gz') as archive:
    names = set()
    for member in archive.getmembers():
        safe(member.name)
        assert member.name not in names and not member.linkname
        names.add(member.name)
        assert member.isdir() or member.isfile()
        relative = PurePosixPath(member.name).relative_to('current-targeted-bundle').as_posix()
        if member.isfile():
            assert archive.extractfile(member).read() == payload['raw/' + relative]
            transport_files.add(relative)
assert transport_files == set(collection['files']) | {'collection-receipt.json'}

summary = {
    'archive_sha256': sha(ARCHIVE.read_bytes()), 'archive_bytes': ARCHIVE.stat().st_size,
    'archive_members': len(members), 'archive_files': len(payload), 'archive_directories': len(dirs),
    'sha256sums_entries': len(sums), 'artifact_receipt_payload_files': len(receipt['files']),
    'guest_collection_payload_files': len(collection['files']),
    'archive_traversal_links_duplicates_or_special_files': False,
    'transport_archive_payload_files': len(transport_files),
    'historical_commit': COMMIT, 'git_blobs_rehashed': len(entries), 'git_blob_bytes_rehashed': total_bytes,
    'frozen_source_guard_matches': len(manifest), 'official_source_guard_matches': len(official_inputs),
    'registered_statement_hashes_verified': len(checked_declarations),
    'source_manifest_sha256': hashes['provenance/current-source-inputs.json'],
    'frozen_source_base_head': frozen['head'],
    'archive_file_sha256': hashes, 'runs': checked_runs, 'declarations': checked_declarations,
    'controls': [{'case': r['case'], 'exit': r['exit']} for r in controls],
    'isolation_checks': [{'label': r['label'], 'exit': r['exit']} for r in isolation['checks']],
    'official_comparator_sha256': tools['tools']['comparator'],
    'official_comparator_revision': tools['repositories']['comparator']['head'],
    'historical_replay_reexecuted': False,
}
out = Path('/tmp/veritile-standardization-20261015/comparator-six-public-static-verification.json')
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ['archive_file_sha256', 'declarations', 'runs']}, indent=2))
print('OUTPUT', out, sha(out.read_bytes()))
