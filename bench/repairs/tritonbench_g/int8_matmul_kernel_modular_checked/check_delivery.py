"""Freeze or verify this task's exact local proof/compiler evidence packet.

Default is read-only identity verification. --freeze creates a new delivery
once, after validating current task build receipts; it never builds Lean,
compiles Triton, runs kernels, modifies historical evidence, or edits ledgers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / '.veritile-cache/kernel-evidence/int8_matmul_kernel_modular_checked'
BUILD = ROOT / '.veritile-cache/kernel-build/lib/lean'
DELIVERY = OUT / 'delivery.json'
PREFIX = 'bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).absolute()
    name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {'path': name, 'sha256': digest(path)}


def verify_pin(ref):
    assert digest(ROOT / ref['path']) == ref['sha256'], ref['path']


def verify_tree(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            verify_pin(value)
        for item in value.values():
            verify_tree(item)
    elif isinstance(value, list):
        for item in value:
            verify_tree(item)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def imports(path):
    result = []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'\s*(?:(?:public|private)\s+)?import\s+(.+)', line)
        if match:
            for name in match.group(1).split('--', 1)[0].split():
                assert re.fullmatch(r'[\w.]+', name), (path, name)
                result.append(name)
    return result


def closure(lean_path):
    pending = [PREFIX + '.Audit']
    seen = set()
    local = []
    frontier = []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        rel = Path(name.replace('.', '/'))
        source = ROOT / rel.with_suffix('.lean')
        candidates = [Path(base) / rel.with_suffix('.olean') for base in lean_path]
        selected = next((p for p in candidates if p.is_file()), None)
        assert selected is not None, ('unresolved module', name)
        row = {'module': name, 'selected_object': pin(selected),
               'resolved_object_path': str(selected.resolve())}
        if source.is_file():
            direct = imports(source)
            row.update(source=pin(source), imports=direct)
            receipt = selected.with_suffix('.workspace-build.json')
            if receipt.is_file():
                record = json.loads(receipt.read_text())
                assert record['source_sha256'] == digest(source), ('stale imported source', name)
                for path, expected in record['objects'].items():
                    assert digest(path) == expected, ('stale imported object', path)
                row['build_receipt'] = pin(receipt)
            else:
                row['build_scope'] = 'Inherited cached dependency; source and selected object pinned, not freshly rebuilt by this task.'
            local.append(row)
            pending.extend(direct)
        else:
            frontier.append(row)
    result = {'entry': PREFIX + '.Audit', 'lean_path': lean_path,
              'scope': 'Transitive repository source/import closure and the resolved objects at its external package boundary; no claim to rebuild external packages.',
              'repository_modules': sorted(local, key=lambda r: r['module']),
              'external_boundary_objects': sorted(frontier, key=lambda r: r['module'])}
    json_write(OUT / 'import-closure.json', result)
    return result


def freeze():
    assert not DELIVERY.exists(), 'Preserve a previously frozen delivery; use a separate reviewed revision.'
    source = HERE / 'int8_matmul_kernel_modular_checked.py'
    assert digest(source) == '5fb5ed4e23f10e9e65aba367755153d8562ef7812ca8c64ee753ca9569af5059'
    assert digest(HERE / 'Kernels.lean') == 'a919d210ce8058278652281233d7ffb9374a2df69766111176715a793b3879f9'
    preserved = [
        ('bench/repairs/tritonbench_g/int8_matmul_kernel/int8_matmul_kernel_repaired.py',
         'e95569108e080703b0bc8df8a61c862eba863cefe6fa4c6321c9ced9ca496cee'),
        ('bench/tritonbench_g/int8_matmul_kernel/int8_matmul_kernel.py',
         'b63ca99ac3c6f1842e66031e898d87c62295ea86bb36711360576db0268bc3f9'),
        ('.veritile-cache/kernel-evidence/bgmv_expand_slice_physical_tile_checked/delivery.json',
         '99fcdfed195e0661fba63fa4163f25c779752fc33019e177ba4f8da6069bae00'),
        ('.veritile-cache/kernel-evidence/rotary_transform_tile_checked/independent-review-rmsnorm.json',
         '5a5e66269260b43c190b8da7d1ca7e0aba5c79660ebfe3474cd67bc5cddb18e0')]
    preserved_refs = [{'path': p, 'sha256': h} for p, h in preserved]
    verify_tree(preserved_refs)
    inspected = []
    for filename, args in [('generate_kernel_source.py', ['--check']), ('check_compiler_barrier.py', [])]:
        command = [sys.executable, str(HERE / filename), *args]
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        inspected.append({'command': command, 'exit_code': run.returncode,
                          'stdout': run.stdout, 'stderr': run.stderr, 'checker': pin(HERE / filename)})
    json_write(OUT / 'inspection-checks.json', inspected)
    module_builds = []
    lean_path = None
    for path in sorted(HERE.rglob('*.lean')):
        rel = path.relative_to(ROOT)
        receipt = (BUILD / rel).with_suffix('.workspace-build.json')
        record = json.loads(receipt.read_text())
        assert record['exit_code'] == 0 and record['source_sha256'] == digest(path), path
        objects = []
        for obj, expected in record['objects'].items():
            assert digest(obj) == expected, obj
            objects.append(pin(obj))
        module_builds.append({'source': pin(path), 'receipt': pin(receipt), 'objects': objects})
        if path.name == 'Audit.lean':
            lean_path = record['lean_path']
    assert lean_path is not None
    imported = closure(lean_path)
    historical = ROOT / '.veritile-cache/kernel-evidence/int8_matmul_kernel/canonical-launch/counterexample.json'
    defect = json.loads(historical.read_text())
    verify_tree(defect)
    old_evidence = [pin(p) for p in sorted(historical.parent.iterdir()) if p.is_file()]
    compiler_dir = OUT / 'directed-compiler'
    compiler_refs = [pin(p) for p in sorted(compiler_dir.iterdir()) if p.is_file()]
    for name in ['binder-profiles.json', 'compiler-checks.json']:
        record = json.loads((compiler_dir / name).read_text())
        # The binder's old checker path denotes the executed snapshot bytes.
        if name == 'binder-profiles.json':
            old_driver = record.pop('compiler_driver')
            assert old_driver['sha256'] == digest(compiler_dir / 'compiler-driver-at-compilation.py')
        verify_tree(record)
    package = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in {'.lean', '.py', '.md'}]
    payload = {
        'task': 'int8_matmul_kernel_modular_checked',
        'stage': 'frozen_ready_for_independent_review',
        'status': 'separate_modular_candidate_source_grid_tensor_proved',
        'candidate_source': pin(source), 'quotation': pin(HERE / 'Kernels.lean'),
        'original_input_domain_and_15_configs_preserved': True,
        'public_entries': [pin(HERE / p) for p in ['Contract.lean', 'Correct.lean', 'FPEquiv.lean', 'Audit.lean']],
        'public_declarations': ['kernel_tensor_low32', 'kernel_torch_low32', 'kernel_tensor_representable',
                                'native_tensor_kernel', 'native_tensor_lane'],
        'targets': {'native': 'low32 of the independent dense sum after defined FP32-to-signed8 conversion and packed weight extraction',
                    'representable': 'Separate signed INT32 exact-result corollary; not a guard',
                    'ideal_real': 'Distinct arbitrary-Real target; no native/truncation equivalence asserted'},
        'proof_scope': {
            'source': 'Complete unique generated quotation under explicit finite leading-static-assert projection; actual structural evaluator statements and both dynamic loops.',
            'grid': 'Actual complete source execution per program on evolving shared memory, all 15 legal configs, empty M/N and K=0, coverage and disjoint stores.',
            'tensor': 'Common typed backing with storage identity/offset/strides, explicit fresh-output separation, scalar store commutation, final encoding equality, INT32 Tensor readback and exact exterior frame.',
            'fp32_api': 'Existing shared Torch.FP32.Tensor laws and logicalValue linked to the same backing and actual data-pointer/strided physical addresses.',
            'bounds': 'Original guard envelope and existing pointer/allocation contracts reused; no new wrapper orchestration project.',
            'native': 'Primitive FP32 cast, fixed K32 MMA, and two-word add.u32 leaves only; no whole-kernel lowering premise.',
            'memory_limit': 'Typed common backing and ordinary fresh allocation ABI; no new untyped byte-heap alias theorem.',
            'compiler_limit': 'Exactly two saved offline SM90 profiles, configs 14 and 4; no all-config compiler/resource guarantee or GPU execution.'},
        'validation': {'fresh_task_modules': len(module_builds),
                       'repository_import_closure_modules': len(imported['repository_modules']),
                       'external_boundary_objects': len(imported['external_boundary_objects']),
                       'axiom_and_noncircular_audit': 'passed',
                       'source_and_compiler_inspections': 'passed',
                       'offline_compiles': 2, 'GPU_execution': False, 'new_numerical_experiments': False,
                       'historical_controls_rerun': False},
        'historical_defect': {'report': pin(historical), 'witness': defect['witness'], 'evidence': old_evidence},
        'preserved_files': preserved_refs,
        'compiler_evidence': compiler_refs,
        'compiler_driver_identity': 'binder-profiles.json original compiler_driver hash resolves to compiler-driver-at-compilation.py; current checker is separately pinned by compiler-checks.json.',
        'build_driver': pin(ROOT / '.veritile-cache/build-tools/workspace_lean_local_first_v2.py'),
        'toolchain': [pin(ROOT / 'lean-toolchain'), pin('/tmp/veritile-lean-toolchain/bin/lean')],
        'package_files': package, 'fresh_module_builds': module_builds,
        'evidence': [pin(OUT / name) for name in ['inspection-checks.json', 'import-closure.json', 'final-build.log', 'audit-build.log', 'store-final-build.log', 'bounds-build.log']],
        'shared_DSL_core_edits': False, 'ledger_or_Lake_edits': False,
        'canonical_selection_owned_by_root': True,
    }
    verify_tree(payload)
    json_write(DELIVERY, payload)
    print(json.dumps({'delivery': pin(DELIVERY), 'fresh_task_modules': len(module_builds),
                      'repository_import_closure_modules': len(imported['repository_modules'])}))


def verify():
    payload = json.loads(DELIVERY.read_text())
    verify_tree(payload)
    imports_record = json.loads((OUT / 'import-closure.json').read_text())
    verify_tree(imports_record)
    for row in imports_record['repository_modules'] + imports_record['external_boundary_objects']:
        relative = Path(row['module'].replace('.', '/') + '.olean')
        resolved = next((Path(base) / relative for base in imports_record['lean_path']
                         if (Path(base) / relative).is_file()), None)
        assert resolved is not None and pin(resolved) == row['selected_object'], row['module']
        assert str(resolved.resolve()) == row['resolved_object_path'], row['module']
    assert payload['candidate_source']['sha256'] == '5fb5ed4e23f10e9e65aba367755153d8562ef7812ca8c64ee753ca9569af5059'
    print(json.dumps({'status': 'PASS', 'delivery': pin(DELIVERY),
                      'package_files': len(payload['package_files']),
                      'fresh_task_modules': len(payload['fresh_module_builds']),
                      'repository_import_closure_modules': len(imports_record['repository_modules'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    options = parser.parse_args()
    freeze() if options.freeze else verify()
