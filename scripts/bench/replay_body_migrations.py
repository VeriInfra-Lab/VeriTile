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
import re
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.bench.check_body_corpus import inspect
from scripts import check_comparator
from scripts.comparator_common import require_tools
from scripts.spec_sheet import split_decls
from scripts.bench.final_inventory import modules as imported_modules
from scripts.bench.port_layout import real_entry

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
        public = [real_entry(root / base).relative_to(root).as_posix(),
                  (base / 'FPEquiv.lean').as_posix()]
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


def verify_public_targets(root: Path, cases: list[dict]) -> None:
    """Resolve full public names before spending time on independent replay.

    The source inventory checks declaration spelling. Lean checks namespaces;
    the comparator remains responsible for actually accepting every proof.
    """
    targets = {path: names for case in cases for path, names in case['public_theorems'].items()}
    names = sorted({name for values in targets.values() for name in values})
    if any(not re.fullmatch(r"[A-Za-z_][\w'.]*", name) for name in names):
        raise ValueError('Invalid public theorem name')
    source = ''.join(f'import {path[:-5].replace("/", ".")}\n' for path in sorted(targets))
    source += ''.join(f'#check {name}\n' for name in names)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', prefix='veritile-public-targets-') as probe:
        probe.write(source)
        probe.flush()
        subprocess.run(['lake', 'env', 'lean', probe.name], cwd=root, check=True)


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


def select_shard(cases: list[dict], builds: list[str], replay: list[str],
                 index: int, count: int) -> tuple[list[dict], list[str], list[str]]:
    if not 1 <= count <= len(cases) or not 0 <= index < count:
        raise ValueError('Invalid shard index/count')
    selected = cases[index::count]
    def include(path: str) -> bool:
        owners = [case for case in cases if path.startswith(case['path'] + '/')]
        return not owners or owners[0] in selected
    return selected, [p for p in builds if include(p)], [p for p in replay if include(p)]


def replay_order(cases: list[dict], replay: list[str]) -> tuple[list[str], set[str]]:
    """Finish shared obligations first, then checkpoint each complete case."""
    groups = [[p for p in replay if p.startswith(case['path'] + '/')]
              for case in cases]
    owned = {p for group in groups for p in group}
    shared = [p for p in replay if p not in owned]
    groups = [group for group in [shared, *groups] if group]
    return [p for group in groups for p in group], {group[-1] for group in groups}


def replay_report(cases: list[dict], replay: list[str], results: list[dict],
                  before: dict[str, str], dependencies: dict[str, set[str]],
                  stable: bool, source_commit: str | None, index: int, count: int) -> dict:
    """A checkpoint accepts only cases with every shared and owned proof."""
    passed = {r['source'] for r in results if r['accepted']}
    shared = [p for p in replay if not any(p.startswith(c['path'] + '/') for c in cases)]
    checked = []
    for case in cases:
        required = [p for p in replay if p.startswith(case['path'] + '/')] + shared
        item = {**case, 'accepted': stable and all(p in passed for p in required),
                'required_replay_sources': required,
                'source_sha256': {p: before[p] for p in sorted(dependencies[case['path']])}}
        checked.append(item)
    complete = len(results) == len(replay) and {r['source'] for r in results} == set(replay)
    return {'schema': 1, 'scope': 'explicit_body_migration_batch',
            'shard': {'index': index, 'count': count},
            'accepted': stable and complete and all(c['accepted'] for c in checked),
            'replay_complete': complete,
            'source_unchanged': stable, 'source_sha256': before,
            'source_commit': source_commit, 'cases': checked, 'replay_results': list(results),
            'boundary': 'Only cases with all required independent replays are accepted. '
                        'An unfinished batch never certifies its pending cases. '
                        'Not a full-corpus or native-execution certificate.'}


def merge_reports(root: Path, cases: list[dict], replay: list[str], manifest: Path,
                  directory: Path) -> dict:
    paths = sorted(directory.rglob('body-migration-replay*.json'))
    if not paths:
        raise ValueError('No official shard reports')
    reports = [json.loads(path.read_text()) for path in paths]
    if any(not report.get('replay_complete', True) for report in reports):
        raise ValueError('Unfinished official shard report')
    count = reports[0].get('shard', {}).get('count')
    if not isinstance(count, int) or len(reports) != count or {
            r.get('shard', {}).get('index') for r in reports} != set(range(count)):
        raise ValueError('Missing or repeated official shard reports')
    before = fingerprint(root, cases, manifest)
    for report in reports:
        if (report.get('schema') != 1 or report.get('scope') != 'explicit_body_migration_batch'
                or report.get('shard', {}).get('count') != count
                or report.get('source_commit') != os.environ.get('GITHUB_SHA')
                or report.get('source_sha256') != before or not report.get('source_unchanged')):
            raise ValueError('Shard source snapshot or scope differs from this checkout')
    joined = [case for report in reports for case in report['cases']]
    if len(joined) != len(cases) or {c['path'] for c in joined} != {c['path'] for c in cases}:
        raise ValueError('Shard cases do not cover the complete batch exactly once')
    results = [result for report in reports for result in report['replay_results']]
    for path in replay:
        checks = [r for r in results if r['source'] == path]
        if not checks or any(not r.get('accepted') for r in checks):
            raise ValueError(f'Missing or failed official replay: {path}')
    required = {path: names for case in cases for path, names in case['public_theorems'].items()}
    for path, names in required.items():
        if any(not set(names).issubset(r.get('theorems', []))
               for r in results if r['source'] == path):
            raise ValueError(f'Required public proofs were not replayed: {path}')
    if any(not report.get('accepted') for report in reports) or any(not c.get('accepted') for c in joined):
        raise ValueError('An official shard did not accept its cases')
    return {'schema': 1, 'scope': 'explicit_body_migration_batch', 'accepted': True,
            'source_unchanged': True, 'source_sha256': before,
            'source_commit': os.environ.get('GITHUB_SHA'), 'cases': joined,
            'replay_results': results, 'official_shards': count,
            'boundary': 'All listed cases and shared checks replayed on one unchanged source snapshot. No native-execution or full-corpus certificate.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--build', action='store_true')
    action.add_argument('--replay', action='store_true')
    action.add_argument('--merge-reports', type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = (ROOT / args.manifest).resolve()
    if not manifest.is_relative_to(ROOT):
        raise ValueError('Manifest must be in this checkout')
    cases, builds, replay = load_batch(ROOT, manifest)
    if args.merge_reports:
        if args.report is None or args.workspace or args.shard_count != 1 or args.shard_index != 0:
            raise ValueError('Merging requires --report and no workspace or shard selection')
        report = merge_reports(ROOT, cases, replay, manifest, args.merge_reports)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'Official shard receipts accepted {len(cases)} cases on the current source snapshot.')
        return 0
    cases, builds, replay = select_shard(cases, builds, replay, args.shard_index, args.shard_count)
    if not args.replay and (args.report or args.workspace):
        raise ValueError('Only official proof replay may produce a completion receipt')
    print(f'Explicit batch: {len(cases)} cases, {len(builds)} build modules, {len(replay)} proof/audit files.', flush=True)
    if args.check:
        print('Function bodies checked; proof completion has NOT been certified.')
        return 0
    modules = [p[:-5].replace('/', '.') for p in builds]
    if args.build:
        result = subprocess.run(['lake', 'build', *modules], cwd=ROOT)
        if result.returncode:
            return result.returncode
        verify_public_targets(ROOT, cases)
        return 0
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
    verify_public_targets(ROOT, cases)
    results = []
    required = {path: names for case in cases for path, names in case['public_theorems'].items()}
    args.workspace.mkdir(parents=True, exist_ok=True)
    ordered, checkpoints = replay_order(cases, replay)

    def checkpoint() -> dict:
        stable = before == fingerprint(ROOT, cases, manifest)
        report = replay_report(cases, replay, results, before, dependencies, stable,
                               os.environ.get('GITHUB_SHA'), args.shard_index, args.shard_count)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(args.report)
        print(json.dumps({'accepted': report['accepted'], 'source_unchanged': stable,
                          'replay_complete': report['replay_complete'],
                          'accepted_cases': sum(c['accepted'] for c in report['cases']),
                          'cases': len(cases), 'replayed_sources': len(results),
                          'report': str(args.report)}), flush=True)
        return report

    checkpoint()
    for path in ordered:
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
        if path in checkpoints:
            checkpoint()
    report = checkpoint()
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'Body migration replay FAILED: {error}', file=sys.stderr)
        sys.exit(1)
