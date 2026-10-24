"""Independent read-only source, receipt, closure, archive and profile checks."""
from pathlib import Path
import ast
import copy
import functools
import hashlib
import json
import re
import struct
import tarfile

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
E = Path(__file__).resolve().parent
TASK = E.parent
EXPECTED = '893537f4216372e620c140c3daf7937b7b008776ef17ce0da5ce814161d6cb03'
CACHE = {}
PINS = {}
REFERENCES = 0


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def sha(value):
    value = path(value)
    stat = value.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in CACHE:
        with value.open('rb') as stream:
            CACHE[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return CACHE[key]


def pin(value):
    value = path(value)
    return {'path': str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else str(value), 'sha256': sha(value)}


def load(value):
    return json.loads(path(value).read_text())


def check(record):
    global REFERENCES
    name = str(path(record['path']))
    assert name not in PINS or PINS[name] == record['sha256'], ('conflicting pins', name)
    assert sha(name) == record['sha256'], record
    PINS[name] = record['sha256']
    REFERENCES += 1


def nested(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            check(value)
        for child in value.values():
            nested(child)
    elif isinstance(value, list):
        for child in value:
            nested(child)


def imports(source):
    return re.findall(r'^import (\S+)\s*$', path(source).read_text(), re.M)


def kernel(source):
    return next(n for n in ast.parse(path(source).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel_int8kv')


class EraseInt64(ast.NodeTransformer):
    count = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id == 'tl' and node.args[0].attr == 'int64'):
            self.count += 1
            return node.func.value
        return node


def check_trees():
    text = (TASK / 'Proofs/NativeSM90.lean').read_text()
    definitions = re.findall(r'^def ((?:local|tree)(?:64|128)_\d) : Tree \d+ := (.*?)(?=\n(?:def|theorem)|\Z)', text, re.M | re.S)
    values = {}
    for name, expression in definitions:
        expression = re.sub(r'⟨(\d+), by decide⟩', r'\1', expression)
        tokens = re.findall(r'\.node|\.input|(?:local|tree)(?:64|128)_\d|\d+|[()]', expression)
        position = 0

        def parse():
            nonlocal position
            token = tokens[position]
            position += 1
            if token == '(':
                value = parse()
                assert tokens[position] == ')'
                position += 1
                return value
            if token == '.node':
                return (parse(), parse())
            if token == '.input':
                value = int(tokens[position])
                position += 1
                return value
            return values[token]

        values[name] = parse()
        assert position == len(tokens), name
    assert len(values) == 16
    for size in (64, 128):
        local = {}
        for owner in range(4):
            leaves = [column for block in range(0, size, 8)
                      for column in (block + 2 * owner, block + 2 * owner + 1)]
            local[owner] = functools.reduce(lambda x, y: (x, y), leaves)
            assert values[f'local{size}_{owner}'] == local[owner]
        for owner in range(4):
            expected = ((local[owner], local[owner ^ 2]), (local[owner ^ 1], local[owner ^ 3]))
            assert values[f'tree{size}_{owner}'] == expected
    return sorted(values)


def main():
    check({'path': str(E / 'delivery.json'), 'sha256': EXPECTED})
    delivery = load(E / 'delivery.json')
    nested(delivery)
    freeze = load(E / 'freeze-result.json')
    nested(freeze)
    builds = load(E / 'task-builds.json')
    closure = load(E / 'import-closure.json')
    assert builds['count'] == len(builds['receipts']) == 48
    order = builds['effective_search_order']
    for row in builds['receipts']:
        nested(row)
        receipt = load(row['receipt']['path'])
        assert path(row['receipt']['path']).read_bytes() == path(row['original_receipt']['path']).read_bytes()
        assert receipt['source'] == str(path(row['source']['path']).relative_to(ROOT))
        assert receipt['source_sha256'] == row['source']['sha256']
        assert receipt['exit_code'] == 0 and receipt['lean_path'] == order
        assert receipt['objects'] == {str(path(x['path'])): x['sha256'] for x in row['objects']}
    binary = {row['path']: row['imports'] for row in
              (json.loads(line) for line in (E / 'independent-binary-imports.jsonl').read_text().splitlines())}
    reader = load(E / 'independent-binary-read-receipt.json')
    assert reader['exit_code'] == 0 and reader['existing_objects_unchanged']
    assert reader['reader_sha256'] == sha(E / 'independent_binary_imports.lean')
    assert reader['stdout_sha256'] == sha(E / 'independent-binary-imports.jsonl')
    assert reader['stderr_sha256'] == sha(E / 'independent-binary-imports.stderr')
    for name, digest in reader['objects'].items():
        check({'path': name, 'sha256': digest})
    assert len(binary) == 5
    graph = {}
    missing = []
    for row in closure['records']:
        nested(row)
        module = row['module']
        assert path(row['source']['path']) == ROOT / (module.replace('.', '/') + '.lean')
        direct = imports(row['source']['path'])
        assert row['source_imports'] == direct
        graph[module] = direct
        relative = module.replace('.', '/') + '.olean'
        effective = next(Path(lib) / relative for lib in order if (Path(lib) / relative).is_file())
        assert effective == path(row['effective_olean']['path'])
        assert str(effective.resolve()) == row['effective_olean_resolved_path']
        if row['effective_ilean']:
            interface = load(row['effective_ilean']['path'])
            assert interface['module'] == module
            assert [x[0] for x in interface['directImports']] == direct
            assert interface['directImports'] == row['compiled_direct_imports']
        else:
            missing.append(module)
            assert binary[str(effective)] == ['Init', *direct]
    seen = set()
    frontier = set()

    def visit(module):
        if module in seen:
            return
        source = ROOT / (module.replace('.', '/') + '.lean')
        if not source.is_file():
            frontier.add(module)
            return
        assert module in graph, ('unlisted repository import', module)
        seen.add(module)
        for dep in graph[module]:
            visit(dep)

    for name in delivery['public_entries']:
        visit(str((TASK / name).relative_to(ROOT)).removesuffix('.lean').replace('/', '.'))
    assert seen == set(graph) == set(closure['public_source_modules'])
    assert frontier == set(closure['external_direct_import_boundaries'])
    assert sorted(missing) == sorted(closure['missing_reused_ilean_modules'])
    expected = {str(path(row['path']).relative_to(ROOT)): row['sha256'] for row in delivery['frozen_files']}
    expected[str((E / 'delivery.json').relative_to(ROOT))] = EXPECTED
    with tarfile.open(E / 'packet.tar.gz') as archive:
        assert len(archive.getmembers()) == len(expected)
        assert set(archive.getnames()) == set(expected)
        for member in archive.getmembers():
            assert member.isfile()
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected[member.name]
    entry = load(E / 'entry.json')
    old_map = {row['path']: row['sha256'] for row in entry['files']}
    changed = []
    with tarfile.open(E / 'pre-revision-sources.tar.gz') as archive:
        assert len(archive.getmembers()) == len(old_map) == 48
        assert set(archive.getnames()) == set(old_map)
        for member in archive.getmembers():
            old = archive.extractfile(member).read()
            assert hashlib.sha256(old).hexdigest() == old_map[member.name]
            current = path(member.name).read_bytes()
            if current != old:
                changed.append(member.name)
                assert current.startswith(old)
                suffix = current[len(old):].decode()
                assert 'import bench.repairs.tritonbench_g.context_attn_fwd.KernelOnly' in suffix
                assert 'alias ' in suffix
    assert sorted(Path(x).name for x in changed) == ['Correct.lean', 'FPEquiv.lean']
    eraser = EraseInt64()
    repaired = eraser.visit(copy.deepcopy(kernel(delivery['selected_source']['path'])))
    assert eraser.count == 8
    assert ast.dump(repaired, include_attributes=False) == ast.dump(kernel(delivery['original_source']['path']), include_attributes=False)
    cpu = load(E / 'cpu-check.json')
    constants = {}
    for row in cpu['launch_constants']:
        width = row['width']
        incoming = struct.unpack('<f', struct.pack('<f', 1.0 / (width ** 0.5) * 1.4426950408889634))[0]
        bits = struct.unpack('<I', struct.pack('<f', incoming))[0]
        assert incoming.as_integer_ratio() == (row['numerator'], row['denominator'])
        assert bits == int(row['bits'], 16)
        constants[width] = row
    assert set(constants) == {16, 32, 64, 128, 256}
    profile = load(E / 'native-probe.json')
    static = load(E / 'native-static.json')
    nested(profile)
    resource = load(E / 'resource-audit.json')
    nested(resource)
    assert {(p['block'], p['width']) for p in profile['profiles']} == {(b, d) for b in (64, 128) for d in constants}
    assert len(profile['profiles']) == len(static['cases']) == 10
    saved = {x['label']: x for x in static['cases']}
    artifacts = []
    for row in profile['profiles']:
        block, width = row['block'], row['width']
        assert row['status'] == 'offline_compile_pass'
        assert row['events'] == ['device-enter', 'launch', 'device-exit']
        assert row['signature']['sm_scale'] == 'fp32'
        assert all(row['signature'][key] == '*fp32' for key in ['Q', 'K', 'V', 'Out'])
        assert all(row['signature'][key] == '*i32' for key in ['B_Start_Loc', 'B_Seqlen', 'b_prompt_cache_len'])
        assert row['options'] == {'num_warps': 4 if width <= 64 else 8, 'num_stages': 1}
        assert row['constants']['BLOCK_M'] == row['constants']['BLOCK_N'] == block
        assert row['constants']['BLOCK_DMODEL'] == width
        artifact_map = {Path(x['path']).suffix: x for x in row['artifacts']}
        assert set(artifact_map) == {'.ttir', '.ttgir', '.llir', '.ptx'}
        artifacts.extend(row['artifacts'])
        status = saved[row['label']]
        assert status['status'] == 'pass' and status['ptx_sha256'] == artifact_map['.ptx']['sha256']
        assert status['qk_n'] == (block // 2 if block == 64 and width > 64 else block)
        assert status['pv_n'] == (width // 2 if block == 64 and width > 64 else width)
        assert status['qk_k8_per_group'] == width // 8 and status['pv_k8_per_group'] == block // 8
        assert status['max_tree_thread_checks'] and status['sum_tree_thread_checks']
        assert status['pv_route_word_checks'] and status['output_division_owner_checks']
        assert status['shared_bytes'] == row['metadata']['shared']
    trees = check_trees()
    audit = (E / 'statement-audit.log').read_text()
    assert audit.count('axiom footprint ⊆ standard base ✓') == 8
    assert audit.count('non-circular ✓') == 2
    assert 'sorryAx' not in audit and 'DISALLOWED' not in audit and ': error' not in audit
    assert sha(E / 'delivery.json') == EXPECTED
    print(json.dumps({'task': 'context_attn_fwd', 'status': 'PASS',
        'delivery': pin(E / 'delivery.json'), 'verifier': pin(__file__),
        'pin_occurrences_checked': REFERENCES, 'unique_pinned_paths_checked': len(PINS),
        'frozen_files': len(delivery['frozen_files']), 'packet_archive_members': len(expected),
        'historical_archive_members': len(old_map), 'historical_files_unchanged': len(old_map) - len(changed),
        'historical_files_with_only_additive_prefix_preserving_public_entries': changed,
        'fresh_current_task_receipts': 48, 'effective_repository_closure': len(graph),
        'repository_import_edges': sum(sum(d in graph for d in deps) for deps in graph.values()),
        'external_frontier_count': len(frontier), 'source_ilean_imports_compared': len(graph) - len(missing),
        'existing_binary_headers_compared': missing,
        'binary_reader': pin(E / 'independent_binary_imports.lean'),
        'binary_read_receipt': pin(E / 'independent-binary-read-receipt.json'),
        'binary_header_log': pin(E / 'independent-binary-imports.jsonl'),
        'source_AST_equal_after_exact_int64_widenings_removed': eraser.count,
        'incoming_FP32_scale_constants': constants, 'native_offline_profiles': 10,
        'native_artifacts': len(artifacts), 'independently_parsed_Lean_reduction_trees': trees,
        'saved_static_max_tree_coordinate_checks': sum(x['max_tree_thread_checks'] for x in static['cases']),
        'saved_static_sum_tree_coordinate_checks': sum(x['sum_tree_thread_checks'] for x in static['cases']),
        'saved_static_PV_operand_coordinate_checks': sum(x['pv_route_word_checks'] for x in static['cases']),
        'saved_static_output_division_owner_checks': sum(x['output_division_owner_checks'] for x in static['cases']),
        'saved_CPU_nonempty_interpreter_launches': 7, 'saved_CPU_empty_returns': 2,
        'standard_public_axiom_checks': 8, 'noncircular_target_audits': 2,
        'reviewer_proof_rebuilds': 0, 'reviewer_CPU_compiler_GPU_experiments': 0,
        'scope': 'Hash/archive/source/import and saved structural evidence audit. No cold transitive build or hardware realization claim.'}, indent=2))


if __name__ == '__main__':
    main()
