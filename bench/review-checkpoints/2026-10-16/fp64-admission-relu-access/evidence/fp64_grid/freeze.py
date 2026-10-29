"""Validate preserved evidence and seal this isolated, partial fp64 grid package.

This does not compile Lean, run a frontend/backend/GPU, or validate numerical
callbacks. Actual producer Lean attempts are checked, not replaced or hidden.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
base = root.parent
core = base / 'vector-addition-fp64-core-wip'
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def write(relative, value):
    path = root / relative
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


assert not (root / 'evidence/receipt.json').exists(), 'Frozen package: do not mutate'
modules = sorted(root.glob('bench/**/*.lean'))
assert len(modules) == 4
dependencies = read(root / 'evidence/final-dependencies-before.json')
assert dependencies == read(root / 'evidence/final-dependencies-after.json')
assert dependencies == read(root / 'evidence/seal-dependencies.json')
assert len(dependencies['files']) == 15
for relative, row in dependencies['files'].items():
    assert sha(Path(row['source_path'])) == row['sha256'], relative
    if 'olean_sha256' in row:
        assert sha(Path(row['olean_path'])) == row['olean_sha256'], relative

core_receipt = read(core / 'evidence/receipt.json')
assert sha(core / 'evidence/receipt.json') == dependencies['frozen_core_receipt_sha256']
assert sha(core / 'evidence/scope-review.json') == dependencies['frozen_core_scope_sha256']
core_review = base / 'vector-addition-fp64-core-independent-review.json'
assert sha(core_review) == dependencies['frozen_core_independent_review_sha256']
for relative, expected in core_receipt['artifacts'].items():
    assert sha(core / relative) == expected, relative
for relative, expected in core_receipt['compiled_olean_sha256'].items():
    assert sha(core / relative) == expected, relative
    assert (root / relative).is_symlink()
    assert (root / relative).resolve() == (core / relative).resolve()
    assert sha(root / relative) == expected

attempts = read(root / 'evidence/verification-attempts.json')
results = read(root / 'evidence/verification-results.json')
assert len(attempts) == 12 and sum(row['exit_code'] != 0 for row in attempts) == 4
assert len(results) == 4 and results == attempts[-4:]
for index, row in enumerate(attempts, 1):
    assert Path(row['log']).name.startswith(f'{index:04d}-')
    assert sha(root / row['log']) == row['log_sha256']
    assert sha(root / row['source_snapshot']) == row['snapshot_sha256']
    assert row['source_sha256_before'] == row['source_sha256_after'] == row['snapshot_sha256']
    assert row['command'][2] == str(root / row['module'])
assert {row['module'] for row in results} == {str(p.relative_to(root)) for p in modules}
for row in results:
    assert row['exit_code'] == 0
    assert sha(root / row['module']) == row['source_sha256_after']
    assert sha(Path(row['command'][0])) == row['lean_executable_sha256']
    assert Path(row['command'][-1]) == (root / row['module']).with_suffix('.olean')
logs = '\n'.join((root / row['log']).read_text() for row in results)
assert not re.search(r'\b(?:warning|error):', logs)
assert logs.count('axiom footprint ⊆ standard base ✓') == 33
assert logs.count('non-circular ✓') == 5
assert sum(p.read_text().count('#auditModuleAxioms') for p in modules) == 4
assert sum(p.read_text().count('#axiomsClean ') for p in modules) == 33
assert sum(p.read_text().count('#specNonCircular ') for p in modules) == 5

# Reconcile the actual recorded search path with every non-library project
# import. Lean chooses the first prefix containing the module's top directory;
# the three immutable core olean links are consequently required in this root.
lean_paths = {row['lean_path'] for row in results}
assert len(lean_paths) == 1
search = [Path(p) for p in next(iter(lean_paths)).split(':') if p]
resolved = {}
for relative, row in dependencies['files'].items():
    if 'olean_sha256' not in row:
        continue
    compiled = Path(relative).with_suffix('.olean')
    prefix = next(prefix for prefix in search if (prefix / compiled.parts[0]).exists())
    loaded = prefix / compiled
    assert loaded.is_file(), str(loaded)
    assert loaded.resolve() == Path(row['olean_path']).resolve()
    assert sha(loaded) == row['olean_sha256']
    resolved[relative] = {
        'recorded_search_prefix': str(prefix),
        'loaded_path': str(loaded),
        'resolved_path': str(loaded.resolve()),
        'symlink': loaded.is_symlink(),
        'olean_sha256': sha(loaded),
    }
assert len(resolved) == 7
resolution_file = write('evidence/import-resolution.json', {
    'kind': 'recorded_final_lean_path_reconciliation; no Lean rerun',
    'lean_path': next(iter(lean_paths)),
    'imported_project_oleans': resolved,
    'external_library_rebuild_or_independent_validation_claimed': False,
})

routes_path = core / 'evidence/source-routes/receipt.json'
routes = read(routes_path)
assert sha(routes_path) == dependencies['source_route_receipt_sha256']
assert len(routes['profiles']) == 29
for row in routes['profiles']:
    assert sha(core / 'evidence/source-routes' / row['ttir_path']) == row['ttir_sha256']
source_sha = sha(repo / 'bench/tritonbench_g/vector_addition/vector_addition.py')
assert source_sha == routes['source_sha256']
floating_types = {'float16', 'bfloat16', 'float32', 'float64', 'float8_e4m3fn', 'float8_e5m2'}
assert sum(row['x'] in floating_types for row in routes['profiles']) == 20
assert sum(row['x'] not in floating_types for row in routes['profiles']) == 9

numerics_path = root / 'evidence/numerical-bindings.json'
numerics = read(numerics_path)
assert sha(numerics_path) == 'e073687689a27750f072f93cfc2848e8985a1850d62838dd5df48fe4e28024af'
assert numerics['families'] == read(core / 'evidence/numerical-correspondence.json')['families']
assert numerics['inherited_registry_sha256'] == sha(core / 'evidence/numerical-correspondence.json')
assert numerics['family_count'] == len(numerics['families']) == 15
assert numerics['port_instance_count'] == len(numerics['instances']) == 73
assert numerics['declaration_port_bindings'] == sum(len(i['declaration_bindings']) for i in numerics['instances']) == 120
assert numerics['new_numerical_families'] == 0 and numerics['numerical_callbacks_remain_unvalidated']
configuration_path = root / 'evidence/frontend-configuration.json'
assert numerics['configuration_sha256'] == sha(configuration_path)
configuration = read(configuration_path)
source_audit_path = base / 'vector-addition-dtype-extension-wip/evidence/extension-audit/receipt.json'
source_audit = read(source_audit_path)
assert configuration['source_audit_receipt_sha256'] == sha(source_audit_path) == routes['prior_audit_sha256']
assert configuration['source_sha256'] == source_sha
assert configuration['environment'] == source_audit['environment']
for key in ['offline_target', 'constexprs', 'frontend_options']:
    assert configuration[key] == source_audit['scope'][key]
sys.path.insert(0, str(repo / 'bench'))
from adapter_common import declarations
contracts = numerics['public_contracts']
assert len(contracts) == 2
for contract in contracts:
    parsed = declarations(root / contract['file'])[contract['parser_name']]
    assert parsed['kind'] == contract['kind'] == 'specification'
    assert parsed['statement_sha256'] == contract['statement_sha256']
keys = set()
profiles = {(r['x'], r['y']): r for r in routes['profiles']}
family_by_callback = {f['callback']: f for f in numerics['families']}
for instance in numerics['instances']:
    unsigned = {k: v for k, v in instance.items() if k != 'instance_key'}
    assert hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == instance['instance_key']
    assert instance['instance_key'] not in keys
    keys.add(instance['instance_key'])
    profile = instance['dtype_profile']
    route = profiles[(profile['x'], profile['y'])]
    assert profile['compute'] == 'float64' and profile['output'] == profile['x']
    assert instance['configuration_sha256'] == sha(configuration_path)
    assert instance['original_source_sha256'] == source_sha
    assert sha(Path(instance['raw_ttir_path'])) == instance['raw_ttir_sha256'] == route['ttir_sha256']
    assert instance['callback_source_sha256'] == sha(core / instance['callback_source'])
    assert instance['rule_id'] == family_by_callback[instance['callback']]['id']
    assert instance['numerical_correspondence'] == family_by_callback[instance['callback']]['required_correspondence']
    assert instance['declaration_bindings'] == (contracts if profile['x'] in floating_types else contracts[:1])
    assert instance['validation_evidence'] is None and instance['status'] == 'explicitly_deferred_unvalidated'
assert {i['callback'] for i in numerics['instances']} == set(family_by_callback)
assert sum(i['callback'] == 'add64' for i in numerics['instances']) == 29

gaps = [
    {'id': 'VA-F64-S1', 'gap': 'Source, extraction and compiler correspondence',
     'detail': 'The typed AST and memory/grid interpreter are task-local, separate from shared FP.Structural. Inherited raw TTIR route observations do not formally relate the original source/compiler to this interpreter.',
     'may_be_deferred_as_numerical': False},
    {'id': 'VA-F64-S2-remaining', 'gap': 'Native instruction trace and scheduling',
     'detail': 'Logical memory, masks, stores, frames and complete sequential program permutations are proved. runLanes computes/writes each lane in order with reads from a fixed program-entry snapshot; it does not execute or refine load-all/compute-all/store-all phases, native instruction interleavings or memory races.',
     'may_be_deferred_as_numerical': False},
    {'id': 'VA-F64-S3', 'gap': 'Native storage, finite addresses and memory admission',
     'detail': 'String/Nat logical regions and indexed words do not establish physical byte layout/endianness, provenance, allocation extent, native pointer admission, signed i32 pid/offset/mask arithmetic or overflow behavior.',
     'may_be_deferred_as_numerical': False},
    {'id': 'VA-F64-S4', 'gap': 'Wrapper, inactive-lane, errors and host composition',
     'detail': 'Output separation is an explicit selected-domain premise, not wrapper freshness or physical nonaliasing. Logical missing-input/undefined-conversion errors return no final memory and prove no native poison, traps, partial-write visibility or rollback. Native masked-load behavior, allocation/device dispatch and Python/Torch module/harness composition remain open.',
     'may_be_deferred_as_numerical': False},
    {'id': 'VA-F64-S5', 'gap': 'Remaining original dtype and error branches',
     'detail': 'The 28 semantically promoted fnuz frontend rejection pairs have no successful logical grid; original-program rejection/error contracts are not established by these successful fp64 profiles.',
     'may_be_deferred_as_numerical': False},
]
coverage = {
    'shared_interpreter_with_concrete_software_operations_profiles': 160,
    'new_task_local_fp64_memory_grid_profiles_with_unvalidated_callbacks': 29,
    'combined_inventory_across_two_distinct_models': 189,
    'combined_inventory_is_one_interpreter': False,
    'promoted_dtype_pairs': 217,
    'remaining_fnuz_frontend_rejection_pairs_without_successful_logical_grid': 28,
    'floating_output_profiles_with_automatic_definedness': 20,
    'integer_output_profiles_with_active_conversion_domain': 9,
    'whole_original_programs_accepted': 0,
    'whole_original_program_total': 183,
}
verification = {
    'actual_lean_exit_codes': [row['exit_code'] for row in results],
    'final_attempt_numbers': [9, 10, 11, 12],
    'preserved_actual_compile_attempts': len(attempts),
    'preserved_failed_compile_attempts': 4,
    'failure_summary': [
        '0001: initial missing imported core olean search path',
        '0002: reference-cons proof tactic failure',
        '0004: recursive program simplification proof failure',
        '0006: Except error-bind simplification proof failure',
    ],
    'named_axiom_checks': 33,
    'noncircularity_checks': 5,
    'whole_module_axiom_audits': 4,
    'warnings': 0,
    'bound_source_configuration_files': 15,
    'imported_core_and_shared_project_oleans': 7,
    'source_and_imported_project_olean_hashes_unchanged': True,
    'recorded_actual_lean_path_reconciled': True,
    'external_library_rebuild_claimed': False,
    'new_frontend_compilations': 0,
    'native_backend_compilations': 0,
    'gpu_executions': 0,
}
scope = write('evidence/scope-review.json', {
    'schema': 1,
    'reviewer': 'producer /root/full_scope_audit; independent review pending',
    'status': 'partial_task_local_typed_logical_memory_grid_ready_for_independent_review',
    'whole_original_program_accepted': False,
    'original_program_completion_increment': 0,
    'new_task_local_logical_grid_profile_increment': 29,
    'shared_interpreter_profile_increment': 0,
    'public_specification_count': 2,
    'modules': {str(p.relative_to(root)): sha(p) for p in modules},
    'public_contracts': contracts,
    'proved_scope': [
        'Real BitVec 64 payloads inherited from immutable typed binary64 core; no real-valued stand-in for fp64.',
        'Optional fully tagged logical cells; absent or wrong-tag active reads return unavailable, never a default value.',
        'Equal input region names imply complete dtype/tag/payload compatibility at every active index.',
        'runLanes computes/writes each lane in order; all reads use its fixed program-entry snapshot and stores update working memory.',
        'Active defined lanes perform actual logical stores equal to an independent result/reference; masked lanes preserve memory without input/domain constraints.',
        'Complete sequential permutations of range(ceil(N/1024)) succeed and equal the same full reference, with every non-output or output-tail cell preserved exactly.',
        'Zero-length grids, wholly masked programs, missing-input/conversion errors and grid error propagation have explicit lemmas.',
        'All 20 floating-output profiles are total under their chosen callbacks, with no finiteness premise. The 9 integer-output profiles retain defined active conversions.',
    ],
    'selected_domain_premises': {
        'input_view': 'Every active input cell has the complete declared tag and word.',
        'output_separation': 'Both logical input region names differ from the output name; wrapper freshness and physical separation are not proved.',
        'shared_input': 'Equal input region names require full tagged-cell equality, not only numeric equality.',
        'integer_conversion': 'OutputDefined requires a successful independent callback result only at active integer-output indices.',
        'program_order': 'Every permutation of the complete mathematical program list; no native instruction trace/interleaving theorem.',
    },
    'error_model': 'An Except error contains only a fault and no final memory. This claims no native poison/trap/partial-write/rollback behavior.',
    'numerical_deferrals': {
        'path': 'evidence/numerical-bindings.json', 'sha256': sha(numerics_path),
        'family_count': 15, 'new_family_count': 0, 'dtype_port_instances': 73,
        'applicable_public_declaration_port_bindings': 120,
        'precise_source_configuration_and_statement_bindings': True,
        'status': 'All unvalidated; concrete queue candidates for parent registration, no accepted numerical experiment.',
    },
    'structural_gaps': gaps,
    'coverage': coverage,
    'verification': verification,
    'frozen_core_receipt_sha256': sha(core / 'evidence/receipt.json'),
    'frozen_core_independent_review_sha256': sha(core_review),
})

# Copy and actually stage deliverable text in a disposable Git repository.
# The generated whitespace receipt and final receipt are excluded to avoid
# circular fingerprints. Compiled oleans/symlinks are bound separately.
files = sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
               and p.suffix != '.olean' and '__pycache__' not in p.parts
               and str(p.relative_to(root)) not in {'evidence/receipt.json', 'evidence/whitespace-check.json'})
with tempfile.TemporaryDirectory(prefix='veritile-fp64-grid-staged-') as temporary:
    stage = Path(temporary)
    for path in files:
        target = stage / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == sha(path)
    init_command = ['git', 'init', '-q', str(stage)]
    add_command = ['git', '-C', str(stage), 'add', '--', '.']
    subprocess.run(init_command, check=True)
    subprocess.run(add_command, check=True)
    names_command = ['git', '-C', str(stage), 'diff', '--cached', '--name-only', '-z']
    staged = subprocess.run(names_command, check=True, stdout=subprocess.PIPE).stdout.decode().split('\0')[:-1]
    assert staged == sorted(str(p.relative_to(root)) for p in files)
    command = ['git', '-C', str(stage), 'diff', '--cached', '--check']
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.returncode == 0, process.stdout
    whitespace = write('evidence/whitespace-check.json', {
        'init_command': init_command, 'init_actual_exit_code': 0,
        'add_command': add_command, 'add_actual_exit_code': 0,
        'staged_inventory_command': names_command,
        'command': command, 'actual_exit_code': process.returncode, 'output': process.stdout,
        'staged_files_sha256': {str(p.relative_to(root)): sha(p) for p in files},
        'temporary_git_directory_removed': True,
        'excluded_generated_receipts': ['evidence/whitespace-check.json', 'evidence/receipt.json'],
        'repository_git_index_modified': False,
    })

receipt = {
    'schema': 1,
    'task': 'vector_addition task-local genuine fp64 typed logical memory and complete program grid',
    'package_root': str(root),
    'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'frozen_ready_for_independent_review',
    'whole_original_program_accepted': False,
    'original_program_completion_increment': 0,
    'new_task_local_logical_grid_profile_increment': 29,
    'shared_interpreter_profile_increment': 0,
    'public_specification_count': 2,
    'public_contracts': contracts,
    'modules': {str(p.relative_to(root)): sha(p) for p in modules},
    'compiled_olean_sha256': {str(p.with_suffix('.olean').relative_to(root)): sha(p.with_suffix('.olean')) for p in modules},
    'imported_project_oleans': resolved,
    'import_resolution_sha256': sha(resolution_file),
    'verification': dict(verification, staged_whitespace_exit=process.returncode),
    'dependency_snapshot_sha256': sha(root / 'evidence/final-dependencies-before.json'),
    'frozen_core': {
        'receipt_sha256': sha(core / 'evidence/receipt.json'),
        'scope_review_sha256': sha(core / 'evidence/scope-review.json'),
        'independent_review_sha256': sha(core_review),
        'all_receipt_artifacts_rechecked': len(core_receipt['artifacts']),
        'source_and_olean_fingerprints_unchanged': True,
    },
    'inherited_source_routes': {'profiles': 29, 'receipt_sha256': sha(routes_path), 'source_sha256': source_sha},
    'numerical_bindings_sha256': sha(numerics_path),
    'frontend_configuration_sha256': sha(configuration_path),
    'statement_extractor_source_sha256': sha(repo / 'bench/adapter_common.py'),
    'numerical_callback_families': 15,
    'numerical_dtype_port_instances': 73,
    'numerical_declaration_port_bindings': 120,
    'scope_review_sha256': sha(scope),
    'whitespace_check_sha256': sha(whitespace),
    'coverage': coverage,
    'structural_gaps': gaps,
    'artifacts': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                  if p.is_file() and not p.is_symlink() and p.suffix != '.olean'
                  and '__pycache__' not in p.parts and str(p.relative_to(root)) != 'evidence/receipt.json'},
}
receipt_path = write('evidence/receipt.json', receipt)
print(json.dumps({
    'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path),
    'scope_sha256': sha(scope), 'modules': receipt['modules'],
    'public_contracts': contracts,
    'numerical_bindings_sha256': sha(numerics_path),
    'artifacts': len(receipt['artifacts']),
    'staged_text_files': len(files),
    'verification': receipt['verification'],
}, indent=2))
