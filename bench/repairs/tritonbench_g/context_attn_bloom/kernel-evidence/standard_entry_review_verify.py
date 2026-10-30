"""Read-only verification of standard entry aliases over the accepted package."""
from pathlib import Path
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'bench/repairs/tritonbench_g/context_attn_bloom'
EVIDENCE = TASK / 'kernel-evidence'
EXPECTED = '0f1e157352512057afc0e8a3e52494a3f42e9cdeb38ac2b6c660247221cb586f'


def resolve(name):
    p = Path(name)
    return p if p.is_absolute() else ROOT / p


def sha(name):
    return hashlib.sha256(resolve(name).read_bytes()).hexdigest()


def read(name):
    return json.loads(resolve(name).read_text())


def pins(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            assert sha(value['path']) == value['sha256'], value['path']
        for child in value.values():
            pins(child)
    elif isinstance(value, list):
        for child in value:
            pins(child)


assert sha(EVIDENCE / 'standard-entry-delivery.json') == EXPECTED
p = read(EVIDENCE / 'standard-entry-delivery.json')
pins(p)
base = read(EVIDENCE / 'delivery.json')
assert p['accepted_base']['delivery.json']['sha256'] == '8c9321c5ce96560d773b801b68f415075ac30e9438c742f4c68efd912d56c932'
assert p['inherited_boundaries'] == base['boundaries']
assert not p['whole_admitted_domain_SM80_execution_established']
base_files = read(EVIDENCE / 'frozen-files.json')
base_map = {x['path']: x['sha256'] for x in base_files['files']}
base_review = read(EVIDENCE / 'independent-verification.json')
old_objects = {x['source']['path']: x for x in base_review['effective_objects']}
delta = read(p['delta_manifest']['path'])
pins(delta['base_frozen_files'])
pins(delta['files'])
assert len(delta['files']) == delta['file_count'] == 13
changes = delta['base_manifest_replacements']
assert len(changes) == 4
changed_paths = {x['after']['path'] for x in changes}
assert changed_paths == {
    str((TASK / 'Correct.lean').relative_to(ROOT)),
    str((TASK / 'FPEquiv.lean').relative_to(ROOT)),
    '.veritile-cache/kernel-build/lib/lean/bench/repairs/tritonbench_g/context_attn_bloom/Correct.workspace-build.json',
    '.veritile-cache/kernel-build/lib/lean/bench/repairs/tritonbench_g/context_attn_bloom/FPEquiv.workspace-build.json'}
with tarfile.open(resolve(p['accepted_base']['context-attn-bloom-kernel-package.tar.gz']['path'])) as archive:
    old_sources = {}
    for change in changes:
        before, after = change['before'], change['after']
        assert before['path'] == after['path']
        assert before['sha256'] == base_map[before['path']]
        old = archive.extractfile(before['path']).read()
        assert hashlib.sha256(old).hexdigest() == before['sha256']
        if before['path'].endswith('.lean'):
            old_sources[before['path']] = old.decode()
        pins(after)
for source in base_files['files']:
    if source['path'] not in changed_paths:
        pins(source)
assert len(base_files['files']) - len(changed_paths) == delta['unchanged_base_file_pins_verified'] == 530
with tarfile.open(resolve(p['delta_archive']['path'])) as archive:
    expected = {x['path']: x['sha256'] for x in delta['files']}
    expected[p['delta_manifest']['path']] = p['delta_manifest']['sha256']
    members = archive.getmembers()
    assert len(members) == len(expected) == 14
    assert {x.name for x in members} == set(expected)
    for member in members:
        assert member.isfile()
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected[member.name]

expected_links = {
    'correct_real': 'mathematical_real',
    'shared_source': 'shared_source',
    'correct_fp32': 'native_tensor',
    'input_tensor': 'input_tensor'}
entry_config = {
    'Correct.lean': ('Standard Real entry for the shared kernel and its independent target.',
                     ['correct_real', 'shared_source']),
    'FPEquiv.lean': ('Standard native entry for original Tensor values, output bytes, and frames.',
                     ['correct_fp32', 'input_tensor'])}
namespace = 'VeriTile.Bench.Repairs.ContextAttnBloomWhole001'
for name, (comment, aliases) in entry_config.items():
    path = TASK / name
    old = old_sources[str(path.relative_to(ROOT))]
    assert len(old.splitlines()) == 1 and old.startswith('import ')
    expected = old + 'import bench.repairs.tritonbench_g.context_attn_bloom.KernelOnly\n\n'
    expected += f'namespace {namespace}.PublicKernel\n\n/-- {comment} -/\n'
    expected += ''.join(f'alias {a} := KernelOnly.{expected_links[a]}\n' for a in aliases)
    expected += '\n' + ''.join(f'#axiomsClean {a}\n' for a in aliases)
    expected += f'\nend {namespace}.PublicKernel\n'
    assert path.read_text() == expected, name
links = read(p['alias_links']['path'])
pins(links)
assert not links['semantic_or_domain_change']
assert {(x['public'], x['accepted_endpoint']) for x in links['alias_links']} == {
    (namespace + '.PublicKernel.' + a, namespace + '.KernelOnly.' + b)
    for a, b in expected_links.items()}


def imports(source):
    return re.findall(r'^import (\S+)\s*$', resolve(source).read_text(), re.M)


effective_dependencies = {}
current_objects = {}
receipts = p['build']['current_receipts']
assert len(receipts) == p['build']['modules_rebuilt'] == 2
for item in receipts:
    r = read(item['receipt']['path'])
    source = item['source']['path']
    assert r['source'] == source and r['source_sha256'] == sha(source) and r['exit_code'] == 0
    assert len(r['objects']) == 3
    assert {str(resolve(x['path'])): x['sha256'] for x in item['objects']} == r['objects']
    for object_path, object_sha in r['objects'].items():
        assert sha(object_path) == object_sha and not resolve(object_path).is_symlink()
        current_objects[str(resolve(object_path).relative_to(ROOT))] = object_sha
    rel = Path(source).with_suffix('.olean')
    effective = next(Path(d) / rel for d in r['lean_path'] if (Path(d) / rel).is_file())
    assert str(effective) in r['objects']
    sidecar = read(effective.with_suffix('.ilean'))
    assert sidecar['module'] == '.'.join(Path(source).with_suffix('').parts)
    assert [x[0] for x in sidecar['directImports']] == imports(source)
    for module in imports(source):
        name = module.replace('.', '/') + '.lean'
        rel = Path(name).with_suffix('.olean')
        dependency = next(Path(d) / rel for d in r['lean_path'] if (Path(d) / rel).is_file())
        accepted = old_objects[name]
        assert dependency.resolve() == resolve(accepted['effective_object']['path']).resolve()
        assert sha(dependency) == accepted['effective_object']['sha256']
        assert sha(name) == accepted['source']['sha256']
        effective_dependencies[name] = accepted['effective_object']
assert len(effective_dependencies) == 3
replacements = delta['base_review_object_replacements']
assert len(replacements) == 4
for change in replacements:
    before, after = change['before'], change['after']
    assert before['path'] == after['path']
    assert after['sha256'] == current_objects[after['path']]
    name = Path(after['path']).name
    source = str((TASK / name).with_suffix('.lean').relative_to(ROOT))
    key = 'ilean' if name.endswith('.ilean') else 'effective_object'
    assert before == old_objects[source][key]

log = resolve(p['build']['log']['path']).read_text()
assert ': error' not in log and 'sorryAx' not in log and 'DISALLOWED' not in log
assert log.count('axiom footprint ⊆ standard base ✓') == p['build']['axiom_checks'] == 4
for alias in expected_links:
    assert namespace + '.PublicKernel.' + alias + ': axiom footprint ⊆ standard base ✓' in log
build_rows = [json.loads(line) for line in log.splitlines() if line.startswith('{"module":')]
assert len(build_rows) == 2 and all(x['exit_code'] == 0 for x in build_rows)
assert {x['module'] for x in build_rows} == {x['source']['path'] for x in receipts}
pins(p['accepted_base'])
assert sha(EVIDENCE / 'standard-entry-delivery.json') == EXPECTED
print(json.dumps({
    'all_checks_passed': True,
    'standard_entry_delivery_sha256': EXPECTED,
    'direct_aliases_verified': expected_links,
    'unchanged_base_file_pins_verified': 530,
    'explicit_source_and_receipt_pin_replacements': 4,
    'explicit_olean_and_ilean_replacements': 4,
    'delta_pinned_files': 13,
    'delta_archive_members_byte_verified': 14,
    'current_build_receipts': 2,
    'current_ilean_direct_import_lists_checked': 2,
    'unchanged_effective_direct_dependencies_verified': effective_dependencies,
    'standard_base_axiom_audits': 4,
    'accepted_delivery_archive_review_and_verification_unchanged': True,
    'kernel_only_and_selected_source_unchanged': True,
    'semantic_or_domain_change': False,
    'recompiled_by_reviewer': False,
    'CPU_compiler_numerics_or_GPU_experiments_run': False,
    'dependency_and_resource_boundaries': 'Inherits the accepted base review, including reused dependency caches and the B128/D256 SM80 shared-memory exclusion.'
}, indent=2))
