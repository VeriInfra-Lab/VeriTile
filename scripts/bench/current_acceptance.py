#!/usr/bin/env python3
"""Portable acceptance wrapper for explicit sources and integration build receipts.

Does not run Lake, change registrations, rewrite frozen reports, or turn a
metadata classification into a successful build or native execution claim.
"""
from __future__ import annotations
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
import json
from scripts.bench.adapter_common import *

REQUIRED_EXTERNAL_GATES = {'legacy_source_audit', 'value_metadata', 'axiom_audit',
                           'comparator_preflight', 'library_replay', 'corpus_replay'}

def module_source(module):
    return Path(*module.split('.')).with_suffix('.lean').as_posix()

def object_file(module, roots):
    """Ordered Lean search roots are explicit caller capabilities, not JSON paths."""
    name = Path(*module.split('.')).with_suffix('.olean')
    for i, root in enumerate(roots):
        p = root / name
        if p.is_file():
            if not any(p.resolve().is_relative_to(r.resolve()) for r in roots):
                raise ValueError('object symlink escapes authorized roots: ' + module)
            return i, name.as_posix(), p
    raise ValueError('missing compiled object: ' + module)

def binding(module, roots):
    i, path, p = object_file(module, roots)
    return {'root_index': i, 'path': path, 'sha256': hash_file(p)}

def check_binding(module, record, roots):
    actual = binding(module, roots)
    if actual != record:
        raise ValueError('object or search-order binding changed: ' + module)

def make_build_receipt(root, catalog, records, object_roots, *, raw_bindings=None, lean_binary=None):
    """Adapt actual integration rows; retain failures and omissions verbatim.

    No retry, discovery, status label or later existing .olean changes a
    nonzero/missing compile result into success. `records` is keyed exactly once
    per module by the caller; conflicting attempts require an explicit choice.
    """
    rows = unique_rows(records, 'module')
    result = []
    wanted = set(catalog['registered_modules'])
    for module in sorted(wanted):
        row = rows.get(module)
        if row is None:
            result.append({'module': module, 'status': 'not_run', 'exit_code': None})
            continue
        out = {k: row.get(k) for k in ('module', 'source', 'source_sha256',
               'direct_import_olean_sha256', 'exit_code', 'log', 'log_sha256', 'olean_sha256')}
        out['status'] = 'passed' if (type(row.get('exit_code')) is int and row['exit_code'] == 0
            and row.get('status', 'passed') == 'passed') else 'failed'
        out['evidence_errors'] = []
        if out['status'] == 'passed':
            try:
                out['object'] = binding(module, object_roots)
                out['direct_objects'] = {d: binding(d, object_roots)
                    for d in (row.get('direct_import_olean_sha256') or {})}
            except ValueError as e:
                out['evidence_errors'].append(str(e))
                out['status'] = 'failed'
        result.append(out)
    environment = None
    if lean_binary:
        environment = {'lean_executable_sha256': hash_file(lean_binary),
            'lean_executable_at_recording': str(lean_binary.absolute()),
            'ordered_lean_path_at_recording': [str(p.absolute()) for p in object_roots],
            'implicit_init': binding('Init', object_roots),
            'external_direct_modules': sorted({d for row in result
                for d in (row.get('direct_import_olean_sha256') or {}) if d not in wanted}),
            'boundary': 'Compiler executable and direct/implicit Init object bindings are checked here. The raw producer and its root review bind the compile invocation. External transitive imports beyond these direct objects are not enumerated by this API.'}
    receipt = {'schema': 1, 'kind': 'integration_build_api',
        'catalog_sha256': sha(canonical(catalog)),
        'input_sha256': catalog['build_input_sha256'],
        'registered_modules': sorted(wanted), 'records': result,
        'object_root_count': len(object_roots),
        'environment': environment,
        'raw_receipt_bindings': raw_bindings or [],
        'unplanned_records': sorted(set(rows) - wanted),
        'status': 'not_validated', 'native_execution': False}
    audit = verify_build(root, catalog, receipt, object_roots, lean_binary=lean_binary)
    receipt['status'] = audit['status']
    receipt['validation'] = audit
    return receipt

def verify_build(root, catalog, receipt, object_roots, *, lean_binary=None, allow_root_relocation=False):
    errors = []
    if receipt.get('schema') != 1 or receipt.get('kind') != 'integration_build_api':
        raise ValueError('unknown integration build API')
    if receipt.get('catalog_sha256') != sha(canonical(catalog)):
        raise ValueError('build belongs to a different source catalog')
    if receipt.get('input_sha256') != catalog['build_input_sha256']:
        raise ValueError('integration input snapshot incomplete or changed')
    if receipt.get('registered_modules') != sorted(catalog['registered_modules']):
        raise ValueError('integration registration set incomplete')
    if receipt.get('object_root_count') != len(object_roots):
        raise ValueError('object resolver root count changed')
    if receipt.get('native_execution') is not False:
        raise ValueError('build cannot claim native execution')
    environment = receipt.get('environment')
    if not environment or lean_binary is None:
        errors.append('compiler environment not bound')
    else:
        if hash_file(lean_binary) != environment['lean_executable_sha256']:
            errors.append('Lean executable changed')
        if (not allow_root_relocation and environment['ordered_lean_path_at_recording'] !=
                [str(p.absolute()) for p in object_roots]):
            errors.append('ordered LEAN_PATH changed; relocation must be explicit')
        try:
            check_binding('Init', environment['implicit_init'], object_roots)
        except ValueError as e:
            errors.append(str(e))
    rows = unique_rows(receipt['records'], 'module')
    if set(rows) != set(catalog['registered_modules']):
        raise ValueError('integration record omitted or added')
    for path, expected in receipt['input_sha256'].items():
        try:
            checked(root, path, expected)
        except ValueError as e:
            errors.append(str(e))
    if receipt.get('unplanned_records'):
        errors.append('unplanned integration records: '+repr(receipt['unplanned_records']))
    if environment and environment.get('external_direct_modules') != sorted({d for row in rows.values()
            for d in (row.get('direct_import_olean_sha256') or {}) if d not in rows}):
        errors.append('external direct object inventory differs')
    passed = 0
    for module, row in rows.items():
        if row.get('status') != 'passed' or type(row.get('exit_code')) is not int or row['exit_code'] != 0:
            errors.append('build ' + row.get('status', 'unknown') + ': ' + module)
            continue
        try:
            source = module_source(module)
            if row['source'] != source or row['source_sha256'] != receipt['input_sha256'].get(source):
                raise ValueError('compiled source differs: ' + module)
            path = checked(root, source, row['source_sha256'])
            deps = imports(path.read_text())
            if set(deps) != set(row['direct_import_olean_sha256']) or set(deps) != set(row['direct_objects']):
                raise ValueError('direct dependency omitted or added: ' + module)
            checked(root, row['log'], row['log_sha256'])
            check_binding(module, row['object'], object_roots)
            if row['olean_sha256'] != row['object']['sha256']:
                raise ValueError('current object is not compiled output: ' + module)
            for dep in deps:
                check_binding(dep, row['direct_objects'][dep], object_roots)
                if row['direct_import_olean_sha256'][dep] != row['direct_objects'][dep]['sha256']:
                    raise ValueError('dependency changed since compilation: ' + module + ' -> ' + dep)
                # A project import cannot masquerade as an external library.
                if (root / module_source(dep)).is_file() and dep not in rows:
                    raise ValueError('project dependency has no build result: ' + dep)
                if dep in rows and (rows[dep].get('status') != 'passed' or rows[dep].get('exit_code') != 0):
                    raise ValueError('dependency build failed/not run: ' + dep)
            if row.get('evidence_errors'):
                raise ValueError('build evidence errors: ' + module)
            passed += 1
        except (ValueError, KeyError, TypeError) as e:
            errors.append(str(e))
    return {'status': 'passed' if not errors else 'failed', 'passed_modules': passed,
            'planned_modules': len(rows), 'errors': errors,
            'native_functional_correctness': 'not_established_by_build'}

def run_gate(source_root, catalog, name, command, *, gate_root, log, cwd=None):
    """Run a real command and bind its output to an unchanged input snapshot.

    The caller chooses the gate command; reviewing that command remains part
    of the integration review. A successful arbitrary command is not a proof.
    """
    import subprocess
    if name not in REQUIRED_EXTERNAL_GATES:
        raise ValueError('unknown acceptance gate')
    if not isinstance(command,list) or not command or not all(isinstance(s,str) and s for s in command):
        raise ValueError('gate command missing')
    for path,h in catalog['build_input_sha256'].items():checked(source_root,path,h)
    dest=gate_root/relative(log)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.resolve().is_relative_to(gate_root.resolve()) or dest.exists():
        raise ValueError('gate log must be new and contained')
    with dest.open('w') as stream:
        try:
            code=subprocess.run(command,cwd=cwd or source_root,stdout=stream,stderr=subprocess.STDOUT).returncode
        except OSError as e:
            stream.write(str(e)+'\n');code=127
    changed=[]
    for path,h in catalog['build_input_sha256'].items():
        try:checked(source_root,path,h)
        except ValueError:changed.append(path)
    return {'schema':1,'gate':name,'status':'passed' if code==0 and not changed else 'failed',
        'exit_code':code,'catalog_sha256':sha(canonical(catalog)),
        'input_sha256':catalog['build_input_sha256'],'command':command,
        'log':log,'log_sha256':hash_file(dest),'changed_inputs':changed}

def validate_gate(root, catalog, name, gate):
    if (gate.get('schema') != 1 or gate.get('gate') != name or
            gate.get('status') != 'passed' or type(gate.get('exit_code')) is not int or gate['exit_code'] != 0):
        raise ValueError('gate failed or not run: ' + name)
    if gate.get('catalog_sha256') != sha(canonical(catalog)) or gate.get('input_sha256') != catalog['build_input_sha256']:
        raise ValueError('gate belongs to different inputs: ' + name)
    if not gate.get('command') or not all(isinstance(s, str) for s in gate['command']):
        raise ValueError('gate command missing: ' + name)
    checked(root, gate['log'], gate['log_sha256'])

def acceptance(root, catalog, *, build=None, build_root=None, gate_root=None, object_roots=(), gates=None,
               lean_binary=None, allow_root_relocation=False):
    """All absent/failed external checks remain blockers, even with clean metadata."""
    from scripts.bench.current_source_catalog import validate_catalog
    result = {'schema': 1, 'catalog_sha256': sha(canonical(catalog)), 'gates': {},
        'native_functional_correctness': 'not_established_by_this_review',
        'historical_reports_modified': False}
    try:
        check = validate_catalog(root, catalog)
        result['gates']['source_inventory'] = {'status': 'passed', **check}
    except ValueError as e:
        result['gates']['source_inventory'] = {'status': 'failed', 'error': str(e)}
    missing_reg = catalog.get('unregistered_task_modules', [])
    result['gates']['module_registration'] = {'status': 'failed' if missing_reg else 'passed', 'missing': missing_reg}
    gaps = [f['path'] for t in catalog['tasks'] for f in t['files']
            if not f.get('review_expected_sha256') or f['review_expected_sha256'] != f['observed_sha256']]
    result['gates']['reviewed_source_bindings'] = {'status': 'failed' if gaps else 'passed', 'unbound_or_changed': gaps}
    absent = [t['task'] for t in catalog['tasks'] if not t['original_source_present']]
    result['gates']['original_source_presence'] = {'status': 'failed' if absent else 'passed', 'missing_original': absent,
        'interpretation': 'Candidate, compiled-graph and standalone-model sources do not fill original-source gaps.'}
    missing_python = catalog.get('missing_bound_python_sources', [])
    result['gates']['python_source_bindings'] = {'status': 'failed' if missing_python else 'passed', 'missing': missing_python}
    if build is None:
        result['gates']['integration_build'] = {'status': 'not_run'}
    else:
        try:
            result['gates']['integration_build'] = verify_build(build_root or root, catalog, build, object_roots,
                lean_binary=lean_binary, allow_root_relocation=allow_root_relocation)
        except ValueError as e:
            result['gates']['integration_build'] = {'status': 'failed', 'error': str(e)}
    gates = gates or {}
    if set(gates) - REQUIRED_EXTERNAL_GATES:
        raise ValueError('unknown acceptance gate')
    for name in sorted(REQUIRED_EXTERNAL_GATES):
        if name not in gates:
            result['gates'][name] = {'status': 'not_run'}
            continue
        try:
            validate_gate(gate_root or root, catalog, name, gates[name])
            result['gates'][name] = {'status': 'passed'}
        except ValueError as e:
            result['gates'][name] = {'status': 'failed', 'error': str(e)}
    result['mechanical_gates_passed'] = all(g['status'] == 'passed' for g in result['gates'].values())
    result['task_count'] = len(catalog['tasks'])
    result['whole_task_acceptance'] = 'requires substantive scope and native review'
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--build-root', type=Path)
    p.add_argument('--gate-root', type=Path)
    p.add_argument('--lean-binary', type=Path)
    p.add_argument('--allow-object-root-relocation', action='store_true')
    p.add_argument('--object-root', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    sub = p.add_subparsers(dest='mode', required=True)
    export = sub.add_parser('bind-build')
    export.add_argument('--records', type=Path, action='append', required=True)
    accept = sub.add_parser('accept')
    accept.add_argument('--build', type=Path)
    accept.add_argument('--gates', type=Path)
    a = p.parse_args()
    from scripts.bench.current_source_catalog import validate_catalog
    catalog = load(a.catalog)
    validate_catalog(a.root, catalog)
    if a.mode == 'bind-build':
        records = [row for path in a.records for row in load(path)]
        result = make_build_receipt(a.build_root or a.root, catalog, records, a.object_root,
            raw_bindings=[{'name': path.name, 'sha256': hash_file(path)} for path in a.records],
            lean_binary=a.lean_binary)
        ok = result['status'] == 'passed'
    else:
        result = acceptance(a.root, catalog, build=load(a.build) if a.build else None,
                            build_root=a.build_root, gate_root=a.gate_root,
                            lean_binary=a.lean_binary, allow_root_relocation=a.allow_object_root_relocation,
                            object_roots=a.object_root, gates=load(a.gates) if a.gates else None)
        ok = result['mechanical_gates_passed']
    write_json(a.output, result)
    print(json.dumps({'status': 'passed' if ok else 'failed', 'output': str(a.output)}))
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
