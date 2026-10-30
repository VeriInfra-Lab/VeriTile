#!/usr/bin/env python3
"""Validate and replay the explicitly selected release scope, preserving gaps."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'bench/verified-scope'


def local_closure(roots):
    found, pending = set(), list(roots)
    while pending:
        module = pending.pop()
        name = module.replace('.', '/') + '.lean'
        path = ROOT / name
        if name in found or not path.is_file():
            continue
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Import escapes source root: ' + name)
        found.add(name)
        for line in path.read_text().splitlines():
            if re.match(r'^(?:public )?import ', line):
                pending.extend(re.sub(r'^(?:public )?import ', '', line).split('--')[0].split())
    return found


def read_scope():
    data = json.loads(gzip.decompress((SCOPE / 'manifest.json.gz').read_bytes()))
    raw = (SCOPE / 'roots.txt').read_bytes()
    roots = raw.decode().splitlines()
    if (data['schema'] != 1 or len(roots) != data['root_count']
            or len(set(roots)) != len(roots)
            or hashlib.sha256(raw).hexdigest() != data['roots_sha256']):
        raise ValueError('Selected proof roots changed without a scope review')
    if local_closure(roots) != set(data['source_sha256']):
        raise ValueError('Selected proof dependency closure changed or was omitted')
    for name, expected in data['build_input_sha256'].items():
        if name not in {'lean-toolchain', 'lakefile.toml', 'lake-manifest.json'}:
            raise ValueError('Unknown build input: ' + name)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Selected build configuration changed: ' + name)
    if set(data['build_input_sha256']) != {'lean-toolchain', 'lakefile.toml', 'lake-manifest.json'}:
        raise ValueError('Selected build configuration binding is incomplete')
    for path, expected in data['source_sha256'].items():
        source = ROOT / path
        if not source.resolve().is_relative_to(ROOT) or not source.is_file():
            raise ValueError('Missing selected proof dependency: ' + path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Changed selected proof dependency: ' + path)
    for module in roots:
        if module.replace('.', '/') + '.lean' not in data['source_sha256']:
            raise ValueError('Unbound selected root: ' + module)
    ledger = json.loads((ROOT / 'bench/tritonbench_g/task_progress.json').read_bytes())
    bodies = json.loads((ROOT / 'bench/source-correspondence.json').read_bytes())
    body_rows = {row['name']: row for row in bodies['tasks']}
    registered = {row['task']: row for row in data['tasks']}
    if set(registered) != {row['task'] for row in ledger['tasks']} or set(registered) != set(body_rows):
        raise ValueError('A task was added or omitted without a scope review')
    for row in ledger['tasks']:
        selected = registered[row['task']]
        kernel = row['kernel_revision']
        entries = kernel['public_entries']
        membership = 'selected' if all(p in data['source_sha256'] for p in entries) else 'pending'
        actual = {'task': row['task'], 'proof_scope': membership,
                  'body_status': body_rows[row['task']]['status'], 'entries': entries,
                  'python': kernel['source'], 'kernel': kernel['kernel_source']}
        if actual != selected:
            raise ValueError('Task disposition changed without review: ' + row['task'])
    if bodies['counts'] != data['body_counts'] or bodies['examples'] != data['examples']:
        raise ValueError('Body correspondence scope changed without review')
    selected_tests, pending_tests = set(data['selected_python_tests']), set(data['pending_python_tests'])
    if selected_tests & pending_tests or selected_tests | pending_tests != {
            p.name for p in (ROOT / 'bench/tests').glob('test_*.py')}:
        raise ValueError('Every benchmark Python test needs an explicit disposition')
    return data, roots


def public_summary(data):
    return {key: data[key] for key in ('schema', 'policy', 'root_count', 'tasks',
            'body_counts', 'pending_original_program_reviews', 'pending_python_tests',
            'known_unselected_build_failures')} | {
        'proof_task_counts': dict(Counter(row['proof_scope'] for row in data['tasks'])),
        'source_dependency_count': len(data['source_sha256']),
        'examples': [{'name': row['name'], 'python': row['python'], 'lean': row['lean'],
                      'status': row['status'], 'source_provenance': row['source_provenance']}
                     for row in data['examples']],
    }


def regressions(data):
    sys.path[:0] = [str(ROOT), str(ROOT / 'bench'), str(ROOT / 'scripts')]
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in data['selected_python_tests']:
        suite.addTests(loader.discover(str(ROOT / 'bench/tests'), pattern=name))
    suite.addTests(loader.discover(str(ROOT / 'scripts'), pattern='test_restore_historical_evidence.py'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print('Historical tests still pending:', json.dumps(data['pending_python_tests']))
    return result.wasSuccessful()


def proof_audit(roots):
    # Enumerate actual theorem objects (including private helpers) from every
    # selected module. Share the axiom traversal cache, checking its union after
    # each theorem. The imported environment must contain checked declarations.
    imports = ''.join(f'import {module}\n' for module in roots)
    modules = '#[' + ','.join('`' + module for module in roots) + ']'
    body = r'''
open Lean Elab Command
set_option maxHeartbeats 0 in
run_cmd do
  let env ← getEnv
  let modules : Array Name := __MODULES__
  for m in modules do
    unless env.header.modules.any (·.module == m) do
      throwError "Audit root not imported: {m}"
  let mut names : Array Name := #[]
  for (name, info) in env.constants.toList do
    let .thmInfo _ := info | continue
    let some idx := env.getModuleIdxFor? name | continue
    if modules.contains env.header.modules[idx.toNat]!.module then
      names := names.push name
  unless names.size > 0 do throwError "Empty selected theorem inventory"
  names := names.qsort (·.toString < ·.toString)
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let mut footprint : CollectAxioms.State := {}
  for name in names do
    unless (env.checked.get.find? name).isSome do
      throwError "Audit root missing from checked environment: {name}"
    let (_, next) := ((CollectAxioms.collect name).run env).run footprint
    footprint := next
    let bad := footprint.axioms.filter (fun a => ! allowed.contains a)
    unless bad.isEmpty do
      throwError "DISALLOWED axiom dependency from {name}: {bad}"
  logInfo m!"Selected modules: {modules.size}; checked theorems: {names.size}"
  logInfo m!"Checked dependency union: {footprint.axioms}"
'''
    return imports + body.replace('__MODULES__', modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--regressions', action='store_true')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    data, roots = read_scope()
    if args.json:
        print(json.dumps(public_summary(data), ensure_ascii=False))
        return 0
    if args.regressions and not regressions(data):
        return 1
    if args.build:
        subprocess.run(['lake', 'build', *roots], cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory(prefix='veritile-selected-proof-audit-') as tmp:
            path = Path(tmp) / 'Audit.lean'
            path.write_text(proof_audit(roots))
            subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT, check=True)
    print(json.dumps({'selected_roots': len(roots),
                      'source_dependencies': len(data['source_sha256']),
                      'body_counts': data['body_counts']['task_status'],
                      'pending_original_program_reviews': data['pending_original_program_reviews']}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f'Verified-scope check failed: {error}', file=sys.stderr)
        sys.exit(1)
