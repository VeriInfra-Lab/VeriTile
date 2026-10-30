#!/usr/bin/env python3
"""Check independent Triton/Real/FP problem interfaces; optionally compile."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'bench/bug_detection'
FILES = {'Kernels.lean', 'IO.lean', 'Contract.lean', 'Correct.lean',
         'FPEquiv.lean', 'Proofs/Backend.lean', 'Misc/Validation.lean'}


def imports(path):
    return set(re.findall(r'^import\s+(\S+)', path.read_text(), re.M))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    manifest = json.loads((BENCH / 'manifest.json').read_text())
    ids = [t['id'] for t in manifest['tasks']]
    assert len(ids) == len(set(ids)) == 65
    assert manifest['schema_version'] == 3
    assert dict(Counter(t['suite'] for t in manifest['tasks'])) == manifest['counts']
    assert not (BENCH / 'Shared').exists()
    assert set(ids) == {p.name for p in (BENCH / 'tasks').iterdir() if p.is_dir()}
    paths = []
    for ident in ids:
        directory = BENCH / 'tasks' / ident
        prefix = 'bench.bug_detection.tasks.' + ident + '.'
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*.lean')}
        assert actual == FILES, (ident, actual ^ FILES)
        task = json.loads((directory / 'task.json').read_text())
        assert task['id'] == ident and task['self_contained']
        assert task['proof_status'] == 'UNSOLVED'
        assert task['backend_binding_status'] == 'PENDING'
        assert task['numerical_admission'] == 'NONE'
        for path in directory.rglob('*'):
            assert not path.is_symlink(), path
        for filename in FILES:
            path = directory / filename
            text = path.read_text()
            assert not re.search(r'\b(sorry|admit|axiom)\b', text), path
            for module in imports(path):
                assert module.startswith((prefix, 'VeriTile.', 'Mathlib.')), (path, module)
                if module.startswith(prefix):
                    assert (ROOT / (module.replace('.', '/') + '.lean')).is_file()
                if filename.startswith('Proofs/'):
                    assert not module.endswith(('.Correct', '.FPEquiv'))
                if filename in ('Correct.lean', 'FPEquiv.lean') or filename.startswith('Proofs/'):
                    assert '.Misc.' not in module
            paths.append(path)
        kernel = (directory / 'Kernels.lean').read_text()
        real = (directory / 'Correct.lean').read_text()
        fp = (directory / 'FPEquiv.lean').read_text()
        io = (directory / 'IO.lean').read_text()
        assert kernel.count('triton {') >= 2, ident
        assert 'def original : ComputeKernel' in kernel, ident
        assert 'def candidate : ComputeKernel' in kernel, ident
        assert not re.search(r'List\.(take|drop)|\.surfaceBody\.(map|filter)', kernel)
        assert 'Spec.Real (' in real and ('⊨ fun' in real or '.ImplementsOn Contract.Allowed' in real)
        assert 'original M Allowed D ≡[R] candidate N Allowed D' in fp
        assert 'structure Inputs where' in io and 'core.Ctx ≃ Inputs' in io
        assert 'LogicalScheduledKernelIO Inputs OutputIndex' in io
        assert 'ProgramIO α α' in io
        assert 'counterexampleProblem' in fp and '.InputCells vals initial' in fp
    return manifest, ids, paths


def evidence_paths():
    return sorted(p for p in BENCH.rglob('*') if p.is_file()
                  and p.name != 'validation.json' and '__pycache__' not in p.parts) + [
        ROOT / 'VeriTile/Triton/Float/GuardedProgramIO.lean',
        ROOT / 'VeriTile/Triton/Memory/LogicalScheduledKernelIO.lean']


def compile_all(lean, output, jobs, library_path, paths, ids, joint, resume):
    output.mkdir(parents=True, exist_ok=True)
    run = output / 'current' if resume else Path(tempfile.mkdtemp(prefix='isolated-', dir=output))
    run.mkdir(parents=True, exist_ok=True)
    assert library_path.is_dir(), library_path
    # No benchmark products are accepted as library dependencies.
    for collection in ('bug_detection', 'pretrain_kernels'):
        assert not (library_path / 'bench' / collection).exists(), library_path

    def build_group(group, target_root, reuse=True):
        env = os.environ.copy()
        env['LEAN_PATH'] = os.pathsep.join(map(str, [target_root, library_path]))
        env['LEAN_NUM_THREADS'] = '1'
        pending = {str(p.relative_to(ROOT).with_suffix('')).replace('/', '.'): p for p in group}
        # A changed local dependency invalidates all of its importers, even
        # when the importer itself has the same source hash.
        dirty = set()
        for module, source in pending.items():
            target = target_root / (module.replace('.', '/') + '.olean')
            stamp = target.with_suffix('.sha256')
            if not (resume and reuse and target.exists() and stamp.exists() and stamp.read_text() == sha256(source)):
                dirty.add(module)
        while True:
            importers = {m for m, p in pending.items() if imports(p) & dirty}
            if importers <= dirty:
                break
            dirty |= importers
        count = 0
        while pending:
            ready = [(m, p) for m, p in pending.items() if not imports(p) & set(pending)]
            assert ready, 'cyclic import graph'
            for module, source in ready:
                target = target_root / (module.replace('.', '/') + '.olean')
                target.parent.mkdir(parents=True, exist_ok=True)
                stamp = target.with_suffix('.sha256')
                digest = sha256(source)
                if module in dirty:
                    proc = subprocess.run([lean, '--root=' + str(ROOT), '-o', str(target), str(source)],
                                          cwd=ROOT, env=env, text=True, capture_output=True)
                    if proc.returncode:
                        raise RuntimeError(module + '\n' + proc.stdout + proc.stderr)
                    assert not re.search(r"declaration uses 'sorry'", proc.stdout), module
                    assert sha256(source) == digest, 'source changed during compilation: ' + module
                    if proc.stdout.strip():
                        print(proc.stdout, end='', flush=True)
                    stamp.write_text(digest)
                del pending[module]
                count += 1
        return count

    outputs = {ident: run / 'tasks' / ident for ident in ids}
    count = 0
    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(build_group,
                   [p for p in paths if p.is_relative_to(BENCH / 'tasks' / ident)], outputs[ident]): ident
                   for ident in ids}
        for future in as_completed(pending):
            try:
                count += future.result()
                print('Independent task: ' + pending[future], flush=True)
            except Exception as error:
                errors.append(str(error))
                print('Failed task: ' + pending[future] + '\n' + str(error), flush=True)
    if errors:
        raise RuntimeError('\n\n'.join(errors))
    if joint:
        aggregate = run / 'aggregate'
        for directory in outputs.values():
            shutil.copytree(directory, aggregate, dirs_exist_ok=True)
        count += build_group([BENCH / 'All.lean', BENCH / 'Validation.lean'], aggregate, reuse=False)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean', help='Lean 4.29 executable; enables compilation')
    parser.add_argument('--library-path', type=Path, help='compiled VeriTile and Mathlib modules')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / '.veritile-cache/bug-detection-build/check')
    parser.add_argument('--task', action='append', help='compile selected independent problems')
    parser.add_argument('--resume', action='store_true', help='reuse unchanged outputs during development')
    parser.add_argument('--record', action='store_true', help='write complete compilation evidence')
    parser.add_argument('--verify-record', action='store_true')
    args = parser.parse_args()
    manifest, ids, paths = inventory()
    if args.task:
        assert not args.record, 'partial compilation cannot replace the collection record'
        assert set(args.task) <= set(ids)
        ids = args.task
    before = {str(p.relative_to(ROOT)): sha256(p) for p in evidence_paths()}
    if args.verify_record:
        record = json.loads((BENCH / 'validation.json').read_text())
        assert record['compilation'] == 'PASS' and record['sha256'] == before
        print('Compilation record matches current sources; compilation was not rerun.')
    report = dict(schema_version=3, tasks=len(ids), layout='PASS', independence='PASS',
                  compilation='NOT_RUN', proof_status='UNSOLVED',
                  source_translation='PENDING', backend_bindings='PENDING', new_numerical_admissions=0)
    if args.lean:
        lean = str(Path(args.lean).resolve())
        report['compiler'] = subprocess.check_output([lean, '--version'], text=True).strip()
        assert 'version 4.29.0,' in report['compiler']
        library_path = args.library_path or ROOT / '.lake/build/lib/lean'
        report['compiled_modules'] = compile_all(lean, args.output.resolve(), args.jobs,
                                               library_path.resolve(), paths, ids, not args.task, args.resume)
        report['compilation'] = 'PASS'
        report['isolated_tasks'] = len(ids)
        report['joint_import'] = 'PASS' if not args.task else 'NOT_RUN'
        assert before == {str(p.relative_to(ROOT)): sha256(p) for p in evidence_paths()}, 'sources changed during build'
        report['sha256'] = before
    if args.record:
        assert args.lean, '--record requires a full Lean compilation'
        (BENCH / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
