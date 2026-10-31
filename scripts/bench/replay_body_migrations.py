#!/usr/bin/env python3
"""Validate and independently replay an explicitly staged body-migration batch.

Checking body pairs or building declarations never writes a completion receipt.
A receipt requires fresh axiom/specification gates and the official comparator
for every public entry, changed proof and shared regression listed in the batch.
Its acceptance applies only to the listed cases, never to the whole corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.bench.check_body_corpus import inspect
from scripts import check_comparator
from scripts.comparator_common import require_tools
from scripts.spec_sheet import split_decls
from scripts.bench.final_inventory import modules as imported_modules

DEFAULT = Path('validation/body-migration-batch.json')


def local(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping batch source: {name}')
    return path


def load_batch(root: Path, manifest: Path) -> tuple[list[dict], list[str], list[str]]:
    data = json.loads(manifest.read_text())
    if data.get('schema') != 1 or data.get('scope') != 'explicit_body_migration_batch':
        raise ValueError('Not an explicit body-migration batch')
    cases, builds, replay = [], set(), set()
    for entry in data['cases']:
        base = Path(entry['path'])
        if len(base.parts) != 3 or base.parts[:2] not in [
                ('bench', 'examples'), ('bench', 'tritonbench_g')]:
            raise ValueError('Case must name one examples or TritonBench task directory')
        if any(c['path'] == entry['path'] for c in cases):
            raise ValueError('Repeated batch case')
        kernel = (base / 'Kernels.lean').as_posix()
        python = entry['python']
        local(root, python)
        row = inspect(root, base.name, python, kernel, 'example' if base.parts[1] == 'examples' else 'task')
        if row['status'] != 'selected_body_copy_checked':
            raise ValueError(f'{base}: incomplete function-body correspondence: {row["status"]}')
        public = [(base / (name + '.lean')).as_posix() for name in ('Correct', 'FPEquiv')]
        required = entry.get('public_theorems', {})
        if set(required) != set(public):
            raise ValueError(f'{base}: explicit Real and FP theorem targets are required')
        for path in public:
            local(root, path)
            declarations = {d['name'] for d in split_decls((root / path).read_text())
                            if d['kind'] == 'specification'}
            if not declarations or not required[path] or any(
                    name.rsplit('.', 1)[-1] not in declarations for name in required[path]):
                raise ValueError(f'{path}: a Prop declaration or build is not a proved specification')
        files = sorted(p.relative_to(root).as_posix() for p in (root / base).rglob('*.lean'))
        builds.update(files)
        replay.update(public)
        cases.append({'path': entry['path'], 'python': python, 'public_entries': public,
                      'public_theorems': required,
                      'source_pairs': row['checked_pairs']})
    if not cases:
        raise ValueError('Empty migration batch')
    extra = data.get('extra_replay_sources', [])
    for path in extra:
        if not path.endswith('.lean') or not path.startswith(('bench/', 'tests/lean/', 'VeriTile/')):
            raise ValueError('Invalid extra proof source')
        local(root, path)
    builds.update(extra)
    replay.update(extra)
    return cases, sorted(builds), sorted(replay)


def fingerprint(root: Path, cases: list[dict], manifest: Path) -> dict[str, str]:
    # Include the complete project source so a concurrent dependency edit cannot
    # turn a replay of an older olean/source pair into a receipt for new code.
    files = {p for folder in ('VeriTile', 'bench', 'tests/lean')
             for p in (root / folder).rglob('*')
             if p.is_file() and p.suffix in {'.lean', '.py', '.json'}}
    files.update(p for p in (root / 'scripts').rglob('*')
                 if p.is_file() and p.suffix in {'.py', '.sh', '.patch'})
    files.update(root / name for name in ('lean-toolchain', 'lakefile.toml', 'lake-manifest.json'))
    files.add(manifest)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def case_sources(root: Path, case: dict) -> set[str]:
    """Bind each result to its actual source/proof import closure.

    The whole-checkout fingerprint guards a running replay. This subset lets
    later batches change unrelated cases without relabelling an older proof
    as a certificate for changed dependencies.
    """
    base = root / case['path']
    files = {p.relative_to(root).as_posix() for p in base.rglob('*')
             if p.is_file() and p.suffix in {'.lean', '.py', '.json'}}
    pending = [p for p in files if p.endswith('.lean')]
    while pending:
        path = pending.pop()
        for module in imported_modules((root / path).read_text()):
            if not module.startswith(('VeriTile.', 'bench.', 'tests.lean.')):
                continue
            source = module.replace('.', '/') + '.lean'
            local(root, source)
            if source not in files:
                files.add(source)
                pending.append(source)
    files.update(('lean-toolchain', 'lakefile.toml', 'lake-manifest.json'))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--build', action='store_true')
    action.add_argument('--replay', action='store_true')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = (ROOT / args.manifest).resolve()
    if not manifest.is_relative_to(ROOT):
        raise ValueError('Manifest must be in this checkout')
    cases, builds, replay = load_batch(ROOT, manifest)
    if not args.replay and (args.report or args.workspace):
        raise ValueError('Only official proof replay may produce a completion receipt')
    print(f'Explicit batch: {len(cases)} cases, {len(builds)} build modules, {len(replay)} proof/audit files.', flush=True)
    if args.check:
        print('Function bodies checked; proof completion has NOT been certified.')
        return 0
    modules = [p[:-5].replace('/', '.') for p in builds]
    if args.build:
        return subprocess.run(['lake', 'build', *modules], cwd=ROOT).returncode
    if args.report is None or args.workspace is None:
        raise ValueError('Replay requires --report and --workspace')
    if args.workspace.exists() and any(args.workspace.iterdir()):
        raise ValueError('Replay requires a fresh empty snapshot workspace')
    require_tools()
    before = fingerprint(ROOT, cases, manifest)
    dependencies = {case['path']: case_sources(ROOT, case) for case in cases}
    # Re-check all retained modules in the cases, including optional helpers.
    build = subprocess.run(['lake', 'build', *modules], cwd=ROOT)
    if build.returncode:
        raise ValueError('Batch build failed; no completion receipt was written')
    results = []
    required = {path: names for case in cases for path, names in case['public_theorems'].items()}
    args.workspace.mkdir(parents=True, exist_ok=True)
    for path in replay:
        try:
            proof = check_comparator.check(args.workspace.resolve(), ROOT / path, trust=True)
            if not proof or not proof.get('accepted'):
                raise ValueError('Official comparator returned no acceptance evidence')
            missing = set(required.get(path, [])) - set(proof.get('theorems', []))
            if missing:
                raise ValueError(f'Required public proofs were not replayed: {sorted(missing)}')
            results.append({'source': path, 'accepted': True,
                            'theorem_count': proof['theorem_count'],
                            'theorems': proof['theorems'], 'evidence_dir': proof.get('evidence_dir')})
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            results.append({'source': path, 'accepted': False, 'error': str(error)})
    stable = before == fingerprint(ROOT, cases, manifest)
    passed = {r['source'] for r in results if r['accepted']}
    shared = [p for p in replay if not any(p.startswith(c['path'] + '/') for c in cases)]
    for case in cases:
        required = [p for p in replay if p.startswith(case['path'] + '/')] + shared
        case['accepted'] = stable and all(p in passed for p in required)
        case['required_replay_sources'] = required
        case['source_sha256'] = {path: before[path] for path in sorted(dependencies[case['path']])}
    report = {'schema': 1, 'scope': 'explicit_body_migration_batch',
              'accepted': stable and all(c['accepted'] for c in cases),
              'source_unchanged': stable, 'source_sha256': before,
              'source_commit': os.environ.get('GITHUB_SHA'),
              'cases': cases, 'replay_results': results,
              'boundary': 'Listed cases only; requires body pairs, retained-module build, axiom/specification gates and official comparator. Not a full-corpus or native-execution certificate.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'accepted': report['accepted'], 'source_unchanged': stable,
                      'accepted_cases': sum(c['accepted'] for c in cases),
                      'cases': len(cases), 'report': str(args.report)}), flush=True)
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'Body migration replay FAILED: {error}', file=sys.stderr)
        sys.exit(1)
