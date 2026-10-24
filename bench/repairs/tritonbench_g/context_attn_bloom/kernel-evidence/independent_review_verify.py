"""Read-only frozen-package review. No builds, compiler runs, or numerical runs.

Prints independent verification evidence; never imports the author's probes.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'bench/repairs/tritonbench_g/context_attn_bloom'
EVIDENCE = TASK / 'kernel-evidence'
EXPECTED = '8c9321c5ce96560d773b801b68f415075ac30e9438c742f4c68efd912d56c932'


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(resolve(path).read_bytes())


def read(path):
    return json.loads(resolve(path).read_text())


def pin(path):
    path = resolve(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'sha256': sha(path)}


checked_pins = []


def verify_pins(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            assert sha(value['path']) == value['sha256'], value['path']
            checked_pins.append(value['path'])
        for child in value.values():
            verify_pins(child)
    elif isinstance(value, list):
        for child in value:
            verify_pins(child)


delivery_path = EVIDENCE / 'delivery.json'
assert sha(delivery_path) == EXPECTED
delivery = read(delivery_path)
verify_pins(delivery)
frozen = read(EVIDENCE / 'frozen-files.json')
assert len(frozen['files']) == frozen['file_count'] == 534
verify_pins(frozen)
frozen_by_path = {p['path']: p['sha256'] for p in frozen['files']}
assert len(frozen_by_path) == 534
with tarfile.open(resolve(delivery['archive']['path'])) as archive:
    expected_members = dict(frozen_by_path)
    index_path = str((EVIDENCE / 'frozen-files.json').relative_to(ROOT))
    expected_members[index_path] = sha(EVIDENCE / 'frozen-files.json')
    members = archive.getmembers()
    assert len(members) == len(expected_members) == 535
    assert {m.name for m in members} == set(expected_members)
    for member in members:
        assert member.isfile(), member.name
        assert digest(archive.extractfile(member).read()) == expected_members[member.name], member.name

source_audit = read(EVIDENCE / 'source-and-profile-audit.json')
entry = read('.veritile-cache/task-deliveries/context_attn_bloom/entry.json')
compatibility = source_audit['compatibility_import'][0]
changed_name = compatibility['after']['path']
for previous in entry['entry_files']:
    if previous['path'] == changed_name:
        added_line = (compatibility['only_change'] + '\n').encode()
        current = resolve(changed_name).read_bytes()
        assert current.count(added_line) == 1
        assert digest(current.replace(added_line, b'')) == previous['sha256'] == compatibility['before']['sha256']
    else:
        verify_pins(previous)
verify_pins({k: v for k, v in source_audit.items() if k != 'compatibility_import'})
verify_pins(compatibility['after'])


class EraseIndexWidening(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Attribute) and node.args[0].attr == 'int64'
                and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == 'tl'):
            return node.func.value
        return node


def kernel(path):
    return next(n for n in ast.parse(resolve(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')


assert ast.dump(kernel(source_audit['original_source']['path'])) == ast.dump(
    EraseIndexWidening().visit(kernel(source_audit['selected_source']['path'])))


def imports(path):
    names = []
    for line in resolve(path).read_text().splitlines():
        m = re.match(r'^(?:public\s+)?import\s+(.+)$', line)
        if m:
            names.extend(n for n in m[1].split() if re.fullmatch(r'[A-Za-z0-9_.]+', n))
    return names


def closure(roots):
    seen, queue = set(), list(roots)
    while queue:
        source = queue.pop()
        if source in seen:
            continue
        seen.add(source)
        for module in imports(source):
            child = module.replace('.', '/') + '.lean'
            if resolve(child).is_file():
                queue.append(child)
    return seen


build = read(EVIDENCE / 'build-evidence.json')
verify_pins(build)
receipts = {}
paths = None
for record in build['receipts']:
    r = read(record['receipt']['path'])
    source = record['module']
    assert r['exit_code'] == 0 and r['source'] == source
    assert r['source_sha256'] == record['source_sha256'] == sha(source)
    assert len(r['objects']) == record['objects_verified'] == 3
    for obj, value in r['objects'].items():
        assert sha(obj) == value, obj
    if paths is None:
        paths = r['lean_path']
    else:
        assert paths == r['lean_path']
    receipts[source] = r
assert len(receipts) == build['task_modules_freshly_rebuilt'] == 37
source_closure = read(EVIDENCE / 'import-source-pins.json')
verify_pins(source_closure)
public = closure([source_closure['root_entry']])
all_sources = closure(receipts)
assert public == {x['path'] for x in source_closure['repository_Lean_sources']}
assert len(public) == source_closure['repository_Lean_source_count'] == 428
assert len(all_sources) == 432
objects = []
missing_ilean = []
ilean_checked = 0
for source in sorted(all_sources):
    assert sha(source) == frozen_by_path[source], source
    rel = Path(source).with_suffix('.olean')
    candidates = [Path(directory) / rel for directory in paths]
    effective = next(path for path in candidates if path.is_file())
    row = {'source': pin(source), 'effective_object': pin(effective),
           'resolved_path': str(effective.resolve()), 'fresh_task_receipt': source in receipts}
    sidecar = effective.with_suffix('.ilean')
    if sidecar.is_file():
        data = read(sidecar)
        assert data['module'] == '.'.join(Path(source).with_suffix('').parts), source
        assert [x[0] for x in data['directImports']] == imports(source), source
        row['ilean'] = pin(sidecar)
        row['source_direct_imports_equal_ilean'] = True
        ilean_checked += 1
    else:
        assert source not in receipts, source
        row['source_direct_imports_equal_ilean'] = 'not available: reused dependency cache has no .ilean sidecar'
        missing_ilean.append(source)
    if source in receipts:
        assert str(effective) in receipts[source]['objects'], source
        assert row['effective_object']['sha256'] == receipts[source]['objects'][str(effective)]
        assert not effective.is_symlink(), source
    else:
        row['boundary'] = 'Existing dependency object inventoried and pinned; not rebuilt in this review or the task build. Source equality is not reconstructed from object bytes.'
    objects.append(row)
assert ilean_checked == 372 and len(missing_ilean) == 60

build_log = resolve(build['log']['path']).read_text()
statement_log = resolve(build['statement_audit']['path']).read_text()
fix_log = resolve(build['review_fix_log']['path']).read_text()
for log in (build_log, statement_log, fix_log):
    assert ': error:' not in log and 'sorryAx' not in log and 'DISALLOWED' not in log
assert build_log.count('axiom footprint ⊆ standard base ✓') == 78
assert build_log.count('non-circular ✓') == 7
assert statement_log.count('axiom footprint ⊆ standard base ✓') == 4
assert fix_log.count('axiom footprint ⊆ standard base ✓') == 6
assert fix_log.count('non-circular ✓') == 1
success = [json.loads(line) for line in build_log.splitlines() if line.startswith('{"module":')]
assert {x['module'] for x in success} == set(receipts) and len(success) == 37
assert all(x['exit_code'] == 0 for x in success)
fix_success = [json.loads(line) for line in fix_log.splitlines() if line.startswith('{"module":')]
assert len(fix_success) == build['review_fix_rebuilt_modules'] == 2
assert {x['module'] for x in fix_success} == {
    'bench/repairs/tritonbench_g/context_attn_bloom/Proofs/KernelProfile.lean',
    'bench/repairs/tritonbench_g/context_attn_bloom/KernelOnly.lean'}
assert all(x['exit_code'] == 0 for x in fix_success)
assert 'KernelNative.pv_literal_factor_order: axiom footprint ⊆ standard base ✓' in fix_log
assert 'KernelNative.pv_literal_factor_order: axiom footprint ⊆ standard base ✓' in statement_log
profile_source = (TASK / 'Proofs/KernelProfile.lean').read_text()
assert '.binary .mul (.binary .add .literal (.dot ..)) .literal => Bool.true' in profile_source
assert '.binary .mul (.binary .add _ (.dot ..)) .literal => Bool.true' not in profile_source
assert 'theorem pv_literal_factor_order' in profile_source
with tarfile.open(resolve(delivery['prior_revision']['archive']['path'])) as archive:
    old_profile = archive.extractfile('bench/repairs/tritonbench_g/context_attn_bloom/Proofs/KernelProfile.lean').read().decode()
    remove_start = profile_source.index('/-- A previous PV output')
    remove_end = profile_source.index('/-- Arbitrary payload substitution', remove_start)
    reconstructed = (profile_source[:remove_start] + profile_source[remove_end:]).replace(
        '#axiomsClean pv_literal_factor_order\n', '').replace(
        '.binary .mul (.binary .add .literal (.dot ..)) .literal => Bool.true',
        '.binary .mul (.binary .add _ (.dot ..)) .literal => Bool.true')
    assert reconstructed == old_profile

# Parse concrete Lean trees, rather than checking only their leaf permutations.
tree_text = (TASK / 'Proofs/KernelNativeTrees.lean').read_text()
trees = {}
for m in re.finditer(r'^def ((?:local|tree|cross)(?:64|128)(?:_\d+)?) : (?:Tree|OrderedTree) (64|128) :=\s*(.*?)(?=\n(?:def |theorem )|\n\n|\Z)', tree_text, re.M | re.S):
    name, width, body = m.groups()
    if body.startswith('⟨'):
        assert body.endswith(', by decide⟩')
        body = body[1:-len(', by decide⟩')]
    body = re.sub(r'\.input ⟨(\d+), by decide⟩', r'\1', body)
    tokens = re.findall(r'\.node|[()]|[A-Za-z_][A-Za-z_0-9]*|\d+', body)
    assert ''.join(tokens) == re.sub(r'\s+', '', body), name
    position = 0

    def parse_tree():
        global position
        token = tokens[position]
        position += 1
        if token == '(':
            result = parse_tree()
            assert tokens[position] == ')'
            position += 1
            return result
        if token == '.node':
            return (parse_tree(), parse_tree())
        if token.isdecimal():
            return int(token)
        return trees[token]

    trees[name] = parse_tree()
    assert position == len(tokens), name


def fold_left(values):
    result = values[0]
    for value in values[1:]:
        result = (result, value)
    return result


def xor_stages(values, masks):
    for mask in masks:
        values = [(value, values[i ^ mask]) for i, value in enumerate(values)]
    return values


checked_trees = []
for width in (64, 128):
    local = [fold_left([8 * k + 2 * lane + j for k in range(width // 8) for j in (0, 1)]) for lane in range(4)]
    warp = xor_stages(local, (2, 1))
    for lane in range(4):
        assert trees[f'local{width}_{lane}'] == local[lane]
        assert trees[f'tree{width}_{lane}'] == warp[lane]
        checked_trees.append(f'tree{width}_{lane}')
    partials = []
    for group in range(8):
        local = [fold_left([64 * k + 8 * group + 2 * lane + j for k in range(width // 64) for j in (0, 1)]) for lane in range(4)]
        partial = xor_stages(local, (2, 1))[0]
        assert trees[f'cross{width}_{group}'] == partial
        partials.append(partial)
    assert trees[f'cross{width}'] == xor_stages(partials, (4, 2, 1))[0]
    checked_trees.append(f'cross{width}')
assert len(trees) == 34 and len(checked_trees) == 10

probe = read(source_audit['native_compilation']['path'])
topology = read(source_audit['native_topology_audit']['path'])
resource = read(EVIDENCE / 'resource-audit.json')
verify_pins(probe)
verify_pins(topology)
verify_pins(resource)
assert len(probe['profiles']) == len(topology['profiles']) == 10
assert not probe['gpu_execution'] and not probe['full_launcher']
native_checks = []
for profile, top, res in zip(probe['profiles'], topology['profiles'], resource['profiles']):
    assert profile['label'] == top['label'] == res['label']
    assert profile['status'] == 'offline_compile_pass'
    assert top['status'] == 'ordered_tree_and_K8_coverage_pass'
    width, block = profile['width'], profile['block']
    assert profile['grid'] == [1, 2, 2]
    assert profile['events'] == ['device-enter', 'launch', 'device-exit']
    assert profile['output_storage_offset'] > 0
    assert profile['signature']['sm_scale'] == 'fp32'
    for name in ('Q', 'K', 'V', 'Out'):
        assert profile['signature'][name] == '*fp32'
    for name in ('B_Start_Loc', 'B_Seqlen', 'Req_to_tokens', 'B_req_idx', 'b_prompt_cache_len'):
        assert profile['signature'][name] == '*i32'
    assert profile['options'] == {'num_warps': 4 if width <= 64 else 8, 'num_stages': 1}
    assert profile['metadata']['default_dot_input_precision'] == 'tf32'
    assert profile['metadata']['enable_fp_fusion']
    assert profile['metadata']['shared'] == res['shared_bytes']
    assert res['fits_sm80_opt_in_architectural_bound'] == (res['shared_bytes'] <= 166912)
    padded = 1 << (width - 1).bit_length()
    assert profile['constants']['BLOCK_DMODEL'] == padded
    assert profile['constants']['BLOCK_M'] == profile['constants']['BLOCK_N'] == block
    wm, wn = top['warps_M_N']
    assert (wn == 8) == (width > 128 or (width > 64 and block == 64))
    assert top['qk_k8_instructions'] == block * block // (16 * 8 * wm * wn) * (padded // 8)
    assert top['pv_k8_instructions'] == block * padded // (16 * 8 * wm * wn) * (block // 8)
    assert top['max_replicas_checked'] == block * block
    assert top['sum_replicas_checked'] == block * (32 if wn == 8 else 4)
    assert all(top['scalar_orientation_sites'].values()) and top['pv_live_seed_groups'] > 0
    ptx = resolve(top['ptx']['path']).read_text()
    assert ptx.count('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32') == top['qk_k8_instructions'] + top['pv_k8_instructions']
    native_checks.append(profile['label'])
assert resource['excluded_from_runnable_SM80_claim'] == ['b128-d256']
assert not delivery['whole_admitted_domain_SM80_execution_established']
gate = resource['runtime_resource_gate']
lines = resolve(gate['path']).read_text().splitlines()
assert '\n'.join(lines[n - 1] for n in gate['one_based_lines']) == gate['excerpt']

result = {
    'all_checks_passed': True,
    'reviewed_delivery_sha256': EXPECTED,
    'frozen_file_pins': 534,
    'archive_members_byte_verified': 535,
    'historical_files_byte_preserved': len(entry['entry_files']) - 1,
    'compatibility_import_is_only_historical_source_delta': True,
    'selected_source_floating_AST_preserved_after_int64_index_erasure': True,
    'fresh_task_receipts_verified': len(receipts),
    'public_repository_source_closure': len(public),
    'all_task_module_repository_source_closure': len(all_sources),
    'effective_repository_objects_inventoried': len(objects),
    'source_ilean_direct_imports_checked': ilean_checked,
    'reused_dependency_sources_without_ilean': missing_ilean,
    'build_log_axiom_checks': 78,
    'statement_log_axiom_checks': 4,
    'review_fix_log_axiom_checks': 6,
    'build_log_noncircular_checks': 7,
    'review_fix_log_noncircular_checks': 1,
    'review_fix_current_module_receipts': 2,
    'resolved_finding': 'BLOOM-NATIVE-ORIENTATION-001: QK seed restricted by node form; actual PV lower operand order proved. Initial complete archive verified and exact profile delta reconstructed.',
    'exact_concrete_reduction_trees_checked': checked_trees,
    'native_profiles_checked': native_checks,
    'native_artifact_pins_checked': sum(len(p['artifacts']) for p in probe['profiles']),
    'saved_topology_max_coordinates': sum(p['max_replicas_checked'] for p in topology['profiles']),
    'saved_topology_sum_replicas': sum(p['sum_replicas_checked'] for p in topology['profiles']),
    'known_nonlaunchable_SM80_profile': {'label': 'b128-d256', 'shared_bytes': 327680, 'architectural_limit': 166912},
    'builds_rerun': False,
    'compiler_or_CPU_or_primitive_experiments_rerun': False,
    'GPU_execution': False,
    'dependency_boundary': 'Only the 37 task modules have fresh build receipts. Remaining effective objects are inventoried reused cache; available .ilean imports checked, missing sidecars disclosed. No claim of a fresh transitive rebuild.',
    'effective_objects': objects,
}
print(json.dumps(result, indent=2))
