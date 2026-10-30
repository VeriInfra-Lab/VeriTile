#!/usr/bin/env python3
"""Bind the final source tree and selected contracts, without archived versions.

This gate checks membership, imports, review scope and source integrity. Lean
builds, axiom/spec audits and official proof replay are separate required gates.
Neither this inventory nor a successful build establishes native execution.
"""
from __future__ import annotations

# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.bench.lean_import_scan import lean_code_mask
from scripts.spec_sheet import split_decls, split_statement

DIRECTORIES = ('VeriTile', 'bench/tritonbench_g', 'bench/examples',
               'tests/lean', 'bench/bug_detection', 'bench/pretrain_kernels')
REVIEW = 'validation/contracts.json'
LEDGER = 'validation/task_progress.json'
MANIFEST = 'validation/source.json'
CORPUS_PREFIXES = ('bench/tritonbench_g/', 'bench/examples/', 'tests/lean/')
BENCH_SECTIONS = {'examples', 'bug_detection', 'pretrain_kernels', 'tritonbench_g'}


def validate_layout(root, ledger):
    """Keep one selected repair per task and infrastructure outside bench."""
    entries = list((root / 'bench').iterdir())
    if {p.name for p in entries} != BENCH_SECTIONS or not all(p.is_dir() for p in entries):
        raise ValueError('bench must contain exactly the four benchmark collections')
    tasks = {row['task']: row['kernel_revision'] for row in ledger['tasks']}
    directory = root / 'bench/tritonbench_g'
    task_directories = list(directory.iterdir())
    if {p.name for p in task_directories} != set(tasks) or not all(p.is_dir() for p in task_directories):
        raise ValueError('tritonbench_g must contain exactly the selected task directories')
    for task, selection in tasks.items():
        base = f'bench/tritonbench_g/{task}/'
        if selection['source'] != base + task + '.py' or selection['kernel_source'] != base + 'Kernels.lean':
            raise ValueError(task + ': selected implementation must use its canonical task entry')
        expected = {base + name + '.lean' for name in ('Contract', 'Correct', 'FPEquiv')}
        if not expected <= set(selection['public_entries']):
            raise ValueError(task + ': missing canonical proof entries')
    forbidden = re.compile(r'(?:revisions?|Whole)\d*|Original|KernelOnly|KernelInterface|RealRevision', re.I)
    for path in directory.rglob('*'):
        if path.is_dir() and forbidden.fullmatch(path.name):
            raise ValueError('implementation version directory is not allowed: ' + str(path.relative_to(root)))


def ci_targets(paths, imports, rows):
    """Replay public contracts/tests; build their complete retained closure."""
    corpus = {p for p in paths if p.startswith(CORPUS_PREFIXES)}
    selected = {p for row in rows.values() for p in row['public_entries']}
    selected.update(row['kernel_source'] for row in rows.values())
    selected.update(p for p in corpus if p.startswith(('bench/examples/', 'tests/lean/')))
    if selected - corpus:
        raise ValueError('CI entry outside retained corpus: ' + repr(sorted(selected - corpus)))
    reached, pending = set(), list(selected)
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        reached.add(path)
        pending.extend(module.replace('.', '/') + '.lean' for module in imports.get(path, []))
    if corpus - reached:
        raise ValueError('retained modules outside CI build closure: ' + repr(sorted(corpus - reached)))
    return sorted(selected)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('missing or escaping source: ' + name)
    return path


def read(root, name):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(local(root, name).read_text(), object_pairs_hook=unique)


def modules(text):
    """Read real imports, including indented continuation lines."""
    lines = lean_code_mask(text).splitlines()
    result = set()
    for i, line in enumerate(lines):
        match = re.match(r'^\s*(?:public\s+)?(?:meta\s+)?import(?:\s+all)?\s+(.*)', line)
        if not match:
            continue
        body = match[1]
        j = i + 1
        module = r'(?:[A-Z]\w*|[A-Za-z_]\w*\.[\w.]+)'
        while j < len(lines) and re.fullmatch(r'\s+' + module + r'(?:\s+' + module + r')*\s*', lines[j]):
            body += ' ' + lines[j].strip()
            j += 1
        for name in body.split():
            if not re.fullmatch(r'[A-Za-z_][\w.]*', name):
                raise ValueError('unsupported import syntax: ' + name)
            result.add(name)
    return result


def validate_review(root, review, ledger, expected_tasks=183):
    if review.get('schema') != 1 or review.get('kind') != 'final_selected_contract_review':
        raise ValueError('unsupported final review schema')
    rows = review['tasks']
    byname = {r['task']: r for r in rows}
    selected = {r['task']: r['kernel_revision'] for r in ledger['tasks']}
    if len(rows) != expected_tasks or len(byname) != expected_tasks or set(byname) != set(selected):
        raise ValueError('final review must cover every selected task exactly once')
    for name, row in byname.items():
        for field in ('mathematical_target', 'required_conditions', 'scope', 'canonical_connection',
                      'fp_program_pair', 'numerical_dependencies', 'source_sha256', 'readers'):
            if not row.get(field):
                raise ValueError(name + ': missing explicit review field ' + field)
        if row.get('review_status') != 'reviewed_current_contract':
            raise ValueError(name + ': unreviewed selected contract')
        if row.get('native_execution') != 'not_established_by_this_review':
            raise ValueError(name + ': unreviewed native-execution claim')
        for field, selection in (('public_entries', 'public_entries'),
                                 ('kernel_source', 'kernel_source'), ('python_source', 'source')):
            if row.get(field) != selected[name].get(selection):
                raise ValueError(name + ': selected revision changed: ' + field)
        for path, expected in row['source_sha256'].items():
            if digest(local(root, path)) != expected:
                raise ValueError(name + ': reviewed source changed: ' + path)
        for entry in row['readers']:
            path = local(root, entry['path'])
            if digest(path) != entry['sha256']:
                raise ValueError(name + ': reviewed reader changed: ' + entry['path'])
            actual = [d['name'] for d in split_decls(path.read_text()) if d['kind'] == 'specification']
            if actual != [d['name'] for d in entry.get('specifications', [])]:
                raise ValueError(name + ': public specification inventory changed: ' + entry['path'])
        reviewed_paths = {r['path'] for r in row['readers']}
        for path in row['public_entries']:
            source = local(root, path)
            if any(d['kind'] == 'specification' for d in split_decls(source.read_text())) and path not in reviewed_paths:
                raise ValueError(name + ': selected public specifications omitted from review: ' + path)
        local(root, row['kernel_source'])
        if row.get('python_source'):
            local(root, row['python_source'])
    return byname


def collect(root=ROOT):
    root = root.resolve()
    review, ledger = read(root, REVIEW), read(root, LEDGER)
    validate_layout(root, ledger)
    rows = validate_review(root, review, ledger)
    paths = sorted({p.relative_to(root).as_posix() for folder in DIRECTORIES
                    for p in (root / folder).rglob('*.lean')})
    known = {p[:-5].replace('/', '.') for p in paths}
    imports = {}
    for name in paths:
        dependencies = sorted(modules(local(root, name).read_text()))
        missing = [m for m in dependencies if m.startswith(('VeriTile', 'bench.', 'tests.lean.')) and m not in known]
        if missing:
            raise ValueError(name + ': missing project imports: ' + repr(missing))
        imports[name] = dependencies
    inputs = set(paths) | {REVIEW, LEDGER, 'lean-toolchain', 'lakefile.toml', 'lake-manifest.json',
                           'validation/bench-layout.json',
                           'scripts/bench/final_inventory.py', 'scripts/bench/lean_import_scan.py',
                           'scripts/spec_sheet.py',
                           'scripts/bench/audit_trust.sh', 'scripts/bench/audit_trust_prep.py', 'scripts/bench/audit_tritonbench_g.sh',
                           'scripts/bench/select_audit_shard.py', 'scripts/bench/check_worker_results.py',
                           'scripts/check_comparator.py', 'scripts/comparator_common.py',
                           'scripts/setup-comparator.sh', 'scripts/comparator-string-literals.patch',
                           'scripts/kernel-manifest.tsv'}
    inputs.update(row['python_source'] for row in rows.values() if row.get('python_source'))
    inputs.update(p.relative_to(root).as_posix()
                  for p in (root / 'bench/tritonbench_g').rglob('*')
                  if p.is_file() and p.suffix in {'.py', '.cpp', '.json'})
    return {'schema': 1, 'kind': 'final_source_inventory',
            'tasks': sorted(rows), 'lean_files': paths, 'imports': imports,
            'ci_targets': ci_targets(paths, imports, rows),
            'source_sha256': {name: digest(local(root, name)) for name in sorted(inputs)},
            'validation_boundary': 'Membership and reviewed scope only; requires Lean build, axiom/spec gates and official replay. Imported benchmark proof goals and Python/native correspondence gaps remain separate.'}


def check(root=ROOT):
    actual, expected = collect(root), read(root, MANIFEST)
    if actual != expected:
        raise ValueError('final source inventory drift; review changes and regenerate after validation')
    return actual


def coverage(root=ROOT):
    review, ledger = read(root, REVIEW), read(root, LEDGER)
    rows = validate_review(root, review, ledger)
    result = []
    for task, row in sorted(rows.items()):
        for reader in row['readers']:
            path = reader['path']
            text = local(root, path).read_text()
            for decl in split_decls(text):
                if decl['kind'] != 'specification':
                    continue
                statement, _ = split_statement(decl['text'])
                result.append({'id': task + '--' + hashlib.sha256((path + ':' + decl['name']).encode()).hexdigest()[:12],
                    'port': task, 'file': path, 'line': text[:decl['start']].count('\n') + 1,
                    'declaration': decl['name'], 'python_file': row.get('python_source') or '',
                    'python_functions': [], 'coverage_level': 'selected_contract',
                    'numeric_model': 'mathematical' if 'Spec.Real' in statement else 'explicit_fp_contract',
                    'scope': row['scope'], 'statement': statement, 'references': [], 'io': [],
                    'issue': '; '.join(map(str, row.get('remaining_issues', [])))})
    return {'schema': 1, 'reviewed_on': '2026-10-29', 'basis': review['scope'], 'rows': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--targets', action='store_true')
    mode.add_argument('--ci-targets', action='store_true',
                      help='public entries, canonical sources, examples and regressions')
    mode.add_argument('--coverage', action='store_true')
    args = parser.parse_args()
    if args.coverage:
        print(json.dumps(coverage(args.root), ensure_ascii=False))
        return
    data = collect(args.root) if args.write else check(args.root)
    if args.write:
        (args.root / MANIFEST).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    if args.ci_targets:
        print('\n'.join(data['ci_targets']))
    elif args.targets:
        for path in data['lean_files']:
            if path.startswith(CORPUS_PREFIXES):
                print(path)
    else:
        print(json.dumps({'tasks': len(data['tasks']), 'lean_files': len(data['lean_files']),
                          'inventory': 'written' if args.write else 'passed'}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error))
