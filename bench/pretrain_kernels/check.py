#!/usr/bin/env python3
"""Check the catalog and prevent metadata-only entries being counted as Lean tasks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
REQUIRED = (
    'Kernels.lean', 'IO.lean', 'Contract.lean', 'Correct.lean', 'FPEquiv.lean',
    'Proofs/Backend.lean', 'Misc/Validation.lean',
)


def inspect():
    manifest = json.loads((BENCH / 'manifest.json').read_text())
    ids = [entry['id'] for entry in manifest['tasks']]
    assert len(ids) == len(set(ids)) == 269, 'catalog must contain 269 unique IDs'
    assert hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest() == (
        '685f6447e03895dc736d3240ccc1bf5f241fe2abce2f9cf42c079a5157a8cd75'
    ), 'catalog IDs differ from the fixed 269-case inventory'
    assert manifest['counts'] == {'local': 261, 'te': 8, 'total': 269}
    assert sum(ident.startswith('local_') for ident in ids) == 261
    assert sum(ident.startswith('te_') for ident in ids) == 8
    assert {p.name for p in (BENCH / 'tasks').iterdir() if p.is_dir()} == set(ids)
    assert not (BENCH / 'Shared').exists()
    pending, drafts = [], []
    for ident in ids:
        directory = BENCH / 'tasks' / ident
        metadata = json.loads((directory / 'task.json').read_text())
        assert metadata['id'] == ident and metadata['self_contained'], ident
        assert metadata['proof_status'] == 'UNSOLVED', ident
        assert metadata['numerical_admission'] == 'NONE', ident
        paths = sorted(directory.rglob('*.lean'))
        assert not any(p.is_symlink() for p in directory.rglob('*')), ident
        for path in paths:
            source = path.read_text()
            assert not re.search(r'\b(sorry|axiom|admit)\b', source), path
            for module in re.findall(r'^import\s+(\S+)', source, re.M):
                assert module.startswith(('VeriTile.', 'Mathlib.', 'bench.pretrain_kernels.tasks.')), (path, module)
                if module.startswith('bench.'):
                    assert module.startswith(f'bench.pretrain_kernels.tasks.{ident}.'), (path, module)
                    assert (ROOT / (module.replace('.', '/') + '.lean')).is_file(), (path, module)
                if path.name in ('Correct.lean', 'FPEquiv.lean') or 'Proofs' in path.parts:
                    assert '.Misc.' not in module, (path, module)
                if 'Proofs' in path.parts:
                    assert not module.endswith(('.Correct', '.FPEquiv')), (path, module)
        missing = [name for name in REQUIRED if not (directory / name).is_file()]
        if missing:
            assert metadata['conversion_status'] == 'PENDING', (ident, missing)
            pending.append(ident)
            continue
        assert metadata['conversion_status'] == 'LEAN_DRAFT', ident
        assert len(paths) == len(REQUIRED), ident
        kernel = (directory / 'Kernels.lean').read_text()
        assert kernel.count('triton {') == (3 if metadata.get('real_projection_status') == 'PENDING_FINITE_CONNECTION' else 2), ident
        assert 'def original : ComputeKernel' in kernel, ident
        assert 'def candidate : ComputeKernel' in kernel, ident
        assert not re.search(r'List\.(take|drop)|\.surfaceBody\.(map|filter)', kernel), ident
        real = (directory / 'Correct.lean').read_text()
        assert 'Spec.Real (' in real, ident
        assert '⊨ fun' in real or '.ImplementsOn Contract.Allowed' in real, ident
        fp = (directory / 'FPEquiv.lean').read_text()
        assert re.search(r'original (?:decode )?M Allowed D ≡\[R\] candidate (?:decode )?N Allowed D', fp), ident
        io = (directory / 'IO.lean').read_text()
        assert 'core.Ctx ≃ Inputs' in io, ident
        assert 'LogicalScheduledKernelIO Inputs OutputIndex' in io, ident
        assert 'ProgramIO α α' in io, ident
        drafts.append(ident)
    return {'catalog': len(ids), 'lean_drafts': len(drafts),
            'pending': len(pending), 'lean_modules': len(drafts) * len(REQUIRED)}, pending


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence():
    paths = sorted(p for p in BENCH.rglob('*') if p.is_file()
                   and p.name != 'validation.json' and '__pycache__' not in p.parts)
    paths += [ROOT / 'VeriTile/Triton/Float/GuardedProgramIO.lean',
              ROOT / 'VeriTile/Triton/Memory/LogicalScheduledKernelIO.lean']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def imports(path):
    return set(re.findall(r'^import\s+(\S+)', path.read_text(), re.M))


def compile_tasks(lean, library, output, ids, jobs, resume, joint):
    """Each task builds with its own products and only the base library visible."""
    assert library.is_dir(), library
    for collection in ('bug_detection', 'pretrain_kernels'):
        assert not (library / 'bench' / collection).exists(), library
    output.mkdir(parents=True, exist_ok=True)
    run = output / 'current' if resume else Path(tempfile.mkdtemp(prefix='isolated-', dir=output))

    def build(paths, destination, reuse=True):
        env = os.environ.copy()
        env['LEAN_PATH'] = os.pathsep.join(map(str, [destination, library]))
        env['LEAN_NUM_THREADS'] = '1'
        pending = {str(p.relative_to(ROOT).with_suffix('')).replace('/', '.'): p for p in paths}
        dirty = set()
        for module, source in pending.items():
            target = destination / (module.replace('.', '/') + '.olean')
            stamp = target.with_suffix('.sha256')
            if not (resume and reuse and target.exists() and stamp.exists() and stamp.read_text() == digest(source)):
                dirty.add(module)
        while True:
            affected = {m for m, p in pending.items() if imports(p) & dirty}
            if affected <= dirty:
                break
            dirty |= affected
        count = 0
        while pending:
            ready = [(m, p) for m, p in pending.items() if not imports(p) & pending.keys()]
            assert ready, 'cyclic task imports'
            for module, source in ready:
                target = destination / (module.replace('.', '/') + '.olean')
                target.parent.mkdir(parents=True, exist_ok=True)
                if module in dirty:
                    source_digest = digest(source)
                    result = subprocess.run([lean, '--tstack=65536', '--root=' + str(ROOT),
                                             '-o', str(target), str(source)], cwd=ROOT,
                                            env=env, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError(module + '\n' + result.stdout + result.stderr)
                    assert "declaration uses 'sorry'" not in result.stdout, module
                    assert digest(source) == source_digest, 'source changed during compilation: ' + module
                    target.with_suffix('.sha256').write_text(source_digest)
                del pending[module]
                count += 1
        return count

    products = {ident: run / 'tasks' / ident for ident in ids}
    count, errors = 0, []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(build, sorted((BENCH / 'tasks' / ident).rglob('*.lean')),
                               products[ident]): ident for ident in ids}
        for future in as_completed(futures):
            ident = futures[future]
            try:
                count += future.result()
                print('Independent task: ' + ident, flush=True)
            except Exception as error:
                errors.append(str(error))
                print('Failed task: ' + ident + '\n' + str(error), flush=True)
    if errors:
        raise RuntimeError('\n\n'.join(errors))
    if joint:
        aggregate = run / 'aggregate'
        for product in products.values():
            shutil.copytree(product, aggregate, dirs_exist_ok=True)
        count += build([BENCH / 'All.lean', BENCH / 'Validation.lean'], aggregate, reuse=False)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--lean', help='Lean 4.29 executable; enables compilation')
    parser.add_argument('--library-path', type=Path, default=ROOT / '.lake/build/lib/lean')
    parser.add_argument('--output', type=Path, default=ROOT / '.veritile-cache/pretrain-build')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--task', action='append', help='compile selected tasks independently')
    parser.add_argument('--resume', action='store_true', help='reuse unchanged development products')
    parser.add_argument('--record', action='store_true', help='record a complete independent and joint build')
    parser.add_argument('--verify-record', action='store_true')
    args = parser.parse_args()
    assert not args.record or args.lean, '--record requires a full Lean compilation'
    counts, pending = inspect()
    print(json.dumps(counts, indent=2))
    if pending and not args.allow_incomplete:
        print(f'Incomplete import: {len(pending)} entries are missing Lean modules.', file=sys.stderr)
        return 1
    before = evidence()
    if args.verify_record:
        record = json.loads((BENCH / 'validation.json').read_text())
        assert record['compilation'] == 'PASS' and record['sha256'] == before
        print('Compilation record matches current sources; compilation was not rerun.')
    if args.lean:
        assert not pending and not args.allow_incomplete, 'compile the complete task inventory'
        manifest = json.loads((BENCH / 'manifest.json').read_text())
        ids = [entry['id'] for entry in manifest['tasks']]
        if args.task:
            assert not args.record, 'partial build cannot replace the collection record'
            assert set(args.task) <= set(ids)
            ids = args.task
        compiler = subprocess.check_output([args.lean, '--version'], text=True).strip()
        assert 'version 4.29.0,' in compiler, compiler
        modules = compile_tasks(args.lean, args.library_path, args.output, ids,
                                args.jobs, args.resume, joint=not args.task)
        assert evidence() == before, 'sources changed during compilation'
        if args.record:
            report = dict(schema_version=3, tasks=len(ids), modules=modules, compiler=compiler,
                          compilation='PASS', independence='PASS', joint_import='PASS',
                          proof_status='UNSOLVED', source_translation='PROVISIONAL',
                          backend_bindings='PENDING', new_numerical_admissions=0, sha256=before)
            (BENCH / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Compiled {modules} modules.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
