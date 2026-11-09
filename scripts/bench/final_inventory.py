#!/usr/bin/env python3
"""Discover the selected corpus, its current public entries and import closure.

An optional generated snapshot binds build and replay to the same inputs.
Historical reviews and independent proof receipts are kept separately.
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
from scripts.bench.port_layout import REAL_ENTRY_NAMES, real_entry
from scripts.spec_sheet import split_decls, split_statement

DIRECTORIES = ('VeriTile', 'bench/tritonbench_g', 'bench/examples',
               'tests/lean', 'bench/bug_detection', 'bench/pretrain_kernels')
LEDGER = 'validation/task_progress.json'
CORPUS_PREFIXES = ('bench/tritonbench_g/', 'bench/examples/', 'tests/lean/')
BENCH_SECTIONS = {'examples', 'bug_detection', 'pretrain_kernels', 'tritonbench_g'}


def validate_layout(root, ledger):
    """Keep one selected repair per task and infrastructure outside bench."""
    entries = list((root / 'bench').iterdir())
    if {p.name for p in entries} != BENCH_SECTIONS or not all(p.is_dir() for p in entries):
        raise ValueError('bench must contain exactly the four benchmark collections')
    tasks = {row['task']: row['kernel_revision'] for row in ledger['tasks']}
    if len(tasks) != len(ledger['tasks']) or not tasks:
        raise ValueError('expected nonempty, unique selected task names')
    directory = root / 'bench/tritonbench_g'
    # The body-correspondence gate owns this collection-level source index.
    # It is metadata for selected tasks, rather than another task directory.
    task_directories = [p for p in directory.iterdir()
                        if not (p.name == 'source-references.json' and p.is_file())]
    if {p.name for p in task_directories} != set(tasks) or not all(p.is_dir() for p in task_directories):
        raise ValueError('tritonbench_g must contain exactly the selected task directories')
    for task, selection in tasks.items():
        base = f'bench/tritonbench_g/{task}/'
        if selection['source'] != base + task + '.py' or selection['kernel_source'] != base + 'Kernels.lean':
            raise ValueError(task + ': selected implementation must use its canonical task entry')
        expected = {base + 'Contract.lean', base + 'FPEquiv.lean'}
        public = set(selection['public_entries'])
        real = public & {base + name for name in REAL_ENTRY_NAMES}
        if not expected <= public or len(real) != 1:
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
    # Audit retained modules directly when the public entries do not import them.
    return sorted(selected | (corpus - reached))


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


def selected_tasks(root, ledger):
    validate_layout(root, ledger)
    rows = {row['task']: row['kernel_revision'] for row in ledger['tasks']}
    for name, row in rows.items():
        directory = f'bench/tritonbench_g/{name}/'
        entries = row['public_entries']
        if len(entries) != len(set(entries)) or any(
                not path.startswith(directory) or not path.endswith('.lean') for path in entries):
            raise ValueError(name + ': invalid selected public entries')
        if real_entry(root / directory).relative_to(root).as_posix() not in entries:
            raise ValueError(name + ': Real entry differs from selected entry')
        for path in (row['source'], row['kernel_source'], *entries):
            actual = local(root, path)
            if not actual.is_relative_to(root / directory) or actual.relative_to(root).as_posix() != path:
                raise ValueError(name + ': noncanonical selected source path')
    return rows


def public_specifications(root, path):
    text = local(root, path).read_text()
    declarations = [d for d in split_decls(text) if d['kind'] == 'specification']
    if not declarations and Path(path).name in (*REAL_ENTRY_NAMES, 'FPEquiv.lean'):
        raise ValueError(path + ': no public specification')
    return text, declarations


def collect(root=ROOT):
    root = root.resolve()
    rows = selected_tasks(root, read(root, LEDGER))
    for row in rows.values():
        for path in row['public_entries']:
            public_specifications(root, path)
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
    inputs = set(paths) | {LEDGER, 'lean-toolchain', 'lakefile.toml', 'lake-manifest.json',
                           'validation/bench-layout.json',
                           'scripts/bench/final_inventory.py', 'scripts/bench/lean_import_scan.py',
                           'scripts/spec_sheet.py',
                           'scripts/bench/audit_trust.sh', 'scripts/bench/audit_trust_prep.py', 'scripts/bench/audit_tritonbench_g.sh',
                           'scripts/bench/select_audit_shard.py', 'scripts/bench/check_worker_results.py',
                           'scripts/check_comparator.py', 'scripts/comparator_common.py',
                           'scripts/setup-comparator.sh', 'scripts/comparator-string-literals.patch',
                           'scripts/kernel-manifest.tsv'}
    inputs.update(row['source'] for row in rows.values())
    inputs.update(p.relative_to(root).as_posix()
                  for p in (root / 'bench/tritonbench_g').rglob('*')
                  if p.is_file() and p.suffix in {'.py', '.cpp', '.json'})
    return {'schema': 1, 'kind': 'final_source_inventory',
            'tasks': sorted(rows), 'lean_files': paths, 'imports': imports,
            'ci_targets': ci_targets(paths, imports, rows),
            'source_sha256': {name: digest(local(root, name)) for name in sorted(inputs)},
            'validation_boundary': 'Current source membership, imports and input hashes. Proof acceptance is recorded in independent replay receipts.'}


def coverage(root=ROOT):
    root = root.resolve()
    rows = selected_tasks(root, read(root, LEDGER))
    result = []
    for task, row in sorted(rows.items()):
        for path in row['public_entries']:
            text, declarations = public_specifications(root, path)
            for decl in declarations:
                statement, _ = split_statement(decl['text'])
                result.append({'id': task + '--' + hashlib.sha256((path + ':' + decl['name']).encode()).hexdigest()[:12],
                    'port': task, 'file': path, 'line': text[:decl['start']].count('\n') + 1,
                    'declaration': decl['name'], 'python_file': row['source'],
                    'python_functions': [], 'coverage_level': 'selected_contract',
                    'numeric_model': 'mathematical' if 'Spec.Real' in statement else 'explicit_fp_contract',
                    'scope': row['proof_scope'], 'statement': statement, 'references': [], 'io': [],
                    'issue': ''})
    return {'schema': 1, 'basis': 'Current selected public Lean declarations', 'rows': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', type=Path, metavar='PATH', help='write a generated inventory snapshot')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--targets', action='store_true')
    mode.add_argument('--ci-targets', action='store_true',
                      help='public entries, canonical sources, examples and regressions')
    mode.add_argument('--coverage', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='require a match with a generated inventory')
    args = parser.parse_args()
    if args.coverage:
        if args.snapshot is not None:
            parser.error('--snapshot applies to inventory modes')
        print(json.dumps(coverage(args.root), ensure_ascii=False))
        return
    data = collect(args.root)
    if args.snapshot is not None and data != json.loads(args.snapshot.read_text()):
        raise ValueError('source inventory changed since the snapshot was generated')
    if args.write is not None:
        args.write.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
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
