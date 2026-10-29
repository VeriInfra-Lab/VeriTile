#!/usr/bin/env python3
"""Collect reproducible full-corpus review evidence; never certify scope from labels.

Run from any directory with the repository's Lean and comparator tools on PATH:
  python3 bench/acceptance_tritonbench_g.py --output /path/to/new-bundle
  python3 bench/acceptance_tritonbench_g.py --check /path/to/bundle

Exit 0 requires every mechanical gate and a stable input set. This is a review
prerequisite, not a claim that every input domain or native behavior is proved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from comparator_common import require_tools

LIBRARIES = [
    'VeriTile', 'VeriTileMath', 'VeriTileFull', 'TritonBenchGVerified', 'TritonBenchGInProgress',
    'TritonBenchGLegacy', 'TritonBenchSpecExamples', 'FPFrontendRegressions',
]
REQUIRED_GATES = {
    'membership', 'source', 'inventory', 'contracts', 'build', 'comparator_preflight',
    'library_replay', 'corpus_replay', 'numerical_obligations', 'fp_contract_surface',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_profile_inputs(root):
    """Close explicit registry -> receipt -> artifact bindings, without globs.

    Runtime ABI, extern and pointer dtype profiles embed their evidence in the
    registry. Dtype constexpr profiles instead read a checkpoint receipt and
    every artifact it binds, including compiler binaries and cached bytecode.
    These explicit dependencies must survive the normal snapshot exclusions.
    This checks input integrity only; source/profile admission stays in its gate.
    """
    root = root.resolve()
    bound = {}

    def relative(name, label):
        if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
                or Path(name).is_absolute() or '..' in Path(name).parts
                or not Path(name).parts):
            raise ValueError('Invalid source-profile ' + label + ' path: ' + repr(name))
        return Path(name)

    def read(path, base, label, expected=None):
        if not path.resolve().is_relative_to(base.resolve()) or not path.is_file():
            raise ValueError('Missing or out-of-bounds source-profile ' + label + ': ' + str(path))
        data = path.read_bytes()
        actual = digest(data)
        if expected is not None and (not isinstance(expected, str)
                or re.fullmatch(r'[0-9a-f]{64}', expected) is None or actual != expected):
            raise ValueError('Source-profile ' + label + ' digest mismatch: ' + str(path))
        name = path.relative_to(root).as_posix()
        if name in bound and bound[name] != actual:
            raise ValueError('Conflicting source-profile input binding: ' + name)
        bound[name] = actual
        return data

    def load(data, label):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate source-profile ' + label + ' key: ' + key)
                result[key] = value
            return result
        try:
            return json.loads(data, object_pairs_hook=unique)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('Malformed source-profile ' + label) from error

    for registry in sorted((root / 'bench/tritonbench_g').glob('source_*profiles.json')):
        data = load(read(registry, root, 'registry'), 'registry')
        if (not isinstance(data, dict) or type(data.get('schema')) is not int
                or data['schema'] != 1 or not isinstance(data.get('profiles'), dict)):
            raise ValueError('Unsupported source-profile registry: ' + str(registry))
        for profile, record in data['profiles'].items():
            if not isinstance(record, dict):
                raise ValueError('Malformed source-profile record: ' + profile)
            if 'evidence' not in record:
                if registry.name == 'source_dtype_constexpr_profiles.json':
                    raise ValueError('Missing source-profile evidence binding: ' + profile)
                # Other currently supported readers consume inline receipts,
                # never a receipt path supplied by the profile.
                if isinstance(record.get('receipt'), str):
                    raise ValueError('Unsupported source-profile receipt path: ' + profile)
                continue
            evidence = relative(record['evidence'], 'evidence')
            if evidence.parts[:2] != ('bench', 'review-checkpoints'):
                raise ValueError('Source-profile evidence must be a reviewed repository checkpoint')
            base = root / evidence
            if (not base.resolve().is_relative_to(root / 'bench/review-checkpoints')
                    or not base.is_dir()):
                raise ValueError('Missing or out-of-bounds source-profile evidence: ' + str(base))
            expected = record.get('receipt_sha256')
            if expected is None:
                raise ValueError('Missing source-profile receipt digest: ' + profile)
            receipt = load(read(base / 'receipt.json', base, 'receipt', expected), 'receipt')
            artifacts = receipt.get('artifact_sha256') if isinstance(receipt, dict) else None
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError('Missing source-profile artifact bindings: ' + profile)
            for name, expected in artifacts.items():
                # Never apply the ordinary suffix or cache-directory filter to
                # a file explicitly read by the source evidence validator.
                if expected is None:
                    raise ValueError('Missing source-profile artifact digest: ' + name)
                read(base / relative(name, 'artifact'), base, 'artifact', expected)
    return bound


def inputs(root=ROOT):
    """Include new files as well as tracked files; exclude evidence/cache trees."""
    root = root.resolve()
    files = {root / n for n in ('lakefile.toml', 'lake-manifest.json', 'lean-toolchain')}
    for directory in ('VeriTile', 'bench', 'scripts', '.github', 'experiments/floating_point'):
        for parent, directories, names in os.walk(root / directory):
            directories[:] = [d for d in directories if d not in {
                'review-checkpoints', '__pycache__', '.lake', '_spec-sheets',
            }]
            for name in names:
                p = Path(parent) / name
                if p.is_file() and p.suffix in {
                    '.lean', '.py', '.sh', '.json', '.toml', '.tsv', '.csv', '.md', '.yml', '.yaml',
                }:
                    files.add(p)
    # Reviewed reports can explicitly depend on otherwise excluded files.
    # Preserve their exact dependency sets, including the generated spec index
    # fingerprinted by the inventory. Their own gates still check freshness;
    # collecting current bytes must not rewrite or bless an old fingerprint.
    reports = [
        ('numerical_obligations_review.json', 'input_sha256', 'numerical-review'),
        ('current_completion_inventory.json', 'input_fingerprints_sha256', 'inventory'),
    ]
    for filename, field, label in reports:
        report = root / 'bench/tritonbench_g' / filename
        if not report.is_file():
            continue
        dependencies = json.loads(report.read_text()).get(field)
        if not isinstance(dependencies, dict):
            raise ValueError('Invalid bound ' + label + ' input manifest')
        for name in dependencies:
            if (not isinstance(name, str) or not name or '\\' in name or '\0' in name
                    or Path(name).is_absolute() or '..' in Path(name).parts):
                raise ValueError('Invalid bound ' + label + ' input: ' + repr(name))
            p = root / name
            if not p.resolve().is_relative_to(root) or not p.is_file():
                raise ValueError('Invalid bound ' + label + ' input: ' + name)
            files.add(p)
    profile_inputs = source_profile_inputs(root)
    files.update(root / name for name in profile_inputs)
    captured = {p.relative_to(root).as_posix(): digest(p.read_bytes())
                for p in sorted(files) if p.is_file()}
    if any(captured.get(name) != expected for name, expected in profile_inputs.items()):
        raise ValueError('Source-profile inputs changed during snapshot')
    return captured


def successful(receipt):
    gates = receipt['gates']
    return (set(gates) == REQUIRED_GATES and
            all(g['status'] == 'passed' and g.get('exit_code') == 0
                for g in gates.values()) and
            not receipt['changed_inputs'] and not receipt['missing_sources'])


def changed(before, after):
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def fp_contract_surface(tasks):
    """Reject real/cast-only results as a replacement for a floating contract.

    Presence of an FP statement is only a necessary condition. Relative
    equivalence, accepted empirical premises and native value correctness
    remain distinct, and this gate does not promote any of them.
    """
    from numerical_obligations import FP_ADEQUACY_LABELS
    present = {'backend_equivalence_no_extra_law', 'backend_equivalence_existing_reports',
               'backend_equivalence_needs_validation', 'bit_interpreter_contract_only',
               'backend_value_contract_only'}
    missing = []
    for task in tasks:
        status = task.get('accepted_scope_review', {}).get('fp_adequacy_status')
        if status not in FP_ADEQUACY_LABELS:
            raise ValueError('Missing or invalid FP contract review: ' + task['task'])
        if status not in present:
            missing.append({'task': task['task'], 'fp_adequacy_status': status})
        elif task['accepted_scope_review'].get('accepted_scope_fp_coverage') == 'partial':
            missing.append({'task': task['task'], 'fp_adequacy_status': status,
                            'reason': 'previously_accepted_stages_still_lack_fp_contracts'})
    passed = bool(tasks) and not missing
    return {'status': 'passed' if passed else 'failed', 'exit_code': 0 if passed else 1,
            'tasks_with_fp_contracts': len(tasks) - len(missing),
            'tasks_without_adequate_fp_contracts': missing,
            'native_functional_correctness': 'not_established_by_this_gate',
            'interpretation': 'A necessary public FP-contract presence check. Neither a clean build, an empty numerical-law list nor an existing report supplies a missing floating model/value proof. Relative equivalence does not establish absolute native correctness.'}


def unregistered_modules(root=ROOT):
    libraries = tomllib.loads((root / 'lakefile.toml').read_text())['lean_lib']
    patterns = [pattern for lib in libraries if lib['name'] in LIBRARIES
                for pattern in [*lib.get('roots', []), *lib.get('globs', [])]]
    result = []
    for path in sorted((root / 'bench/tritonbench_g').rglob('*.lean')):
        module = '.'.join(path.relative_to(root).with_suffix('').parts)
        if not any(module.startswith(p[:-1]) if p.endswith('.+') else module == p
                   for p in patterns):
            result.append(module)
    return result


def run_gate(name, command, output):
    print(f'Running {name}; log: {output / (name + ".log")}', flush=True)
    started = datetime.now(timezone.utc).isoformat()
    stopped = False
    with (output / (name + '.log')).open('w') as log:
        try:
            # A failed compiler has already invalidated this gate. Preserve its
            # complete diagnostics, then stop remaining Lake jobs instead of
            # spending hours on unrelated elaborations. A successful gate still
            # requires Lake to finish all requested targets with exit zero.
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       start_new_session=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                if name == 'build' and line.startswith('error: Lean exited with code'):
                    stopped = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            process.stdout.close()
            code = process.wait()
            if stopped and code == 0:
                code = 1
        except OSError as error:
            log.write(str(error) + '\n')
            code = 127
    return {'command': command, 'exit_code': code,
            'status': 'passed' if code == 0 else 'failed', 'started_at_utc': started,
            'remaining_build_jobs_stopped_after_failure': stopped}


def verify(output):
    receipt = json.loads((output / 'receipt.json').read_text())
    failures = []
    for name, expected in receipt['artifact_sha256'].items():
        p = output / name
        if not p.is_file() or digest(p.read_bytes()) != expected:
            failures.append('changed artifact: ' + name)
    with zipfile.ZipFile(output / 'inputs.zip') as archive:
        if set(archive.namelist()) != set(receipt['input_sha256']):
            failures.append('archived input membership mismatch')
        else:
            for name, expected in receipt['input_sha256'].items():
                if digest(archive.read(name)) != expected:
                    failures.append('changed archived input: ' + name)
    stale = changed(receipt['input_sha256'], inputs())
    if stale:
        failures.append(f'current workspace differs in {len(stale)} inputs')
    print(json.dumps({'integrity_errors': failures,
                      'mechanical_gates_passed': successful(receipt),
                      'whole_task_acceptance': 'requires substantive scope review'}, indent=2))
    return int(bool(failures) or not successful(receipt))


def collect(output):
    output = output.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'bench/review-checkpoints'):
        raise ValueError('Output must be outside the repository or under bench/review-checkpoints.')
    output.mkdir(parents=True, exist_ok=False)
    before = inputs()
    with zipfile.ZipFile(output / 'inputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in before.items():
            data = (ROOT / name).read_bytes()
            if digest(data) != expected:
                raise ValueError('Input changed during snapshot: ' + name)
            archive.writestr(name, data)
    tasks = sorted(p for p in (ROOT / 'bench/tritonbench_g').iterdir()
                   if p.is_dir() and not p.name.startswith(('_', '.')))
    from port_layout import source_file
    missing = [p.name for p in tasks if source_file(p) is None]
    inventory_path = ROOT / 'bench/tritonbench_g/current_completion_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    # Preserve the actual public statements/scopes and all JIT links, including
    # missing-source rows. The inventory gate below rejects stale metadata.
    (output / 'tasks.json').write_text(json.dumps(inventory['tasks'], indent=2) + '\n')
    (output / 'task-scopes.md').write_bytes(
        (ROOT / 'bench/tritonbench_g/current_completion_inventory.md').read_bytes())
    unregistered = unregistered_modules()
    inventory_tasks = [t['task'] for t in inventory['tasks']]
    membership_ok = (not unregistered and len(inventory_tasks) == len(set(inventory_tasks))
                     and set(inventory_tasks) == {p.name for p in tasks})
    gates = {'membership': {'status': 'passed' if membership_ok else 'failed',
                            'exit_code': 0 if membership_ok else 1}}
    surface = fp_contract_surface(inventory['tasks'])
    gates['fp_contract_surface'] = {k: surface[k] for k in ('status', 'exit_code')}
    (output / 'fp_contract_surface.log').write_text(json.dumps(surface, indent=2) + '\n')
    (output / 'membership.log').write_text(json.dumps({
        'unregistered_modules': unregistered, 'inventory_matches_directories':
        set(inventory_tasks) == {p.name for p in tasks},
        'duplicate_inventory_rows': len(inventory_tasks) != len(set(inventory_tasks)),
    }, indent=2) + '\n')
    try:
        require_tools()
        gates['comparator_preflight'] = {'status': 'passed', 'exit_code': 0}
        message = 'Official comparator isolation probes passed.\n'
    except (OSError, ValueError) as error:
        gates['comparator_preflight'] = {'status': 'failed', 'exit_code': 1}
        message = str(error) + '\n'
    (output / 'comparator_preflight.log').write_text(message)
    commands = {
        'source': ['bash', 'bench/audit_tritonbench_g.sh', '--source-only'],
        'inventory': [sys.executable, 'bench/inventory_tritonbench_g_completion.py', '--check'],
        'contracts': [sys.executable, 'bench/coverage_review.py', '--json'],
        'numerical_obligations': [sys.executable, 'bench/numerical_obligations.py', '--check'],
        'build': ['lake', 'build', *LIBRARIES],
        'library_replay': [sys.executable, 'scripts/check_comparator.py', '--library'],
        'corpus_replay': ['bash', 'bench/audit_trust.sh'],
    }
    for name, command in commands.items():
        if changed(before, inputs()):
            gates[name] = {'status': 'not_run', 'reason': 'inputs changed during collection'}
        elif name.endswith('_replay') and gates['comparator_preflight']['status'] != 'passed':
            gates[name] = {'status': 'not_run', 'reason': 'official isolation unavailable'}
        else:
            gates[name] = run_gate(name, command, output)
    receipt = {
        'schema': 1, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': before, 'changed_inputs': changed(before, inputs()),
        'gates': gates, 'tasks': len(tasks), 'missing_sources': missing,
        'whole_task_acceptance': 'requires substantive scope review',
        'interpretation': 'A passing bundle checks current sources, all registered proof modules, '
                         'standard axiom gates and official replay. It does not enlarge any '
                         'theorem domain, prove host/native refinement or certify scope labels.',
    }
    receipt['mechanical_gates_passed'] = successful(receipt)
    receipt['artifact_sha256'] = {p.name: digest(p.read_bytes())
                                 for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('gates', 'tasks', 'missing_sources',
                                            'changed_inputs', 'mechanical_gates_passed')}, indent=2))
    return int(not successful(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--check', type=Path)
    args = parser.parse_args()
    return verify(args.check) if args.check else collect(args.output)


if __name__ == '__main__':
    sys.exit(main())
