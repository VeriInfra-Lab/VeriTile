#!/usr/bin/env python3
"""Compare native scalar operations to source-bound Lean software bit results.

Finite observations never discharge a universal Lean hypothesis. In particular,
NaN payload mismatches are retained, not normalized away to manufacture PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = [HERE / 'ScalarCases.lean', ROOT / 'VeriTile/Triton/Float/ScalarOps.lean',
         ROOT / 'VeriTile/Triton/Float/BitValue.lean']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_hashes():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in MODEL}


def validate_reference(data):
    if data.get('schema') != 1 or data.get('source_sha256') != model_hashes():
        raise ValueError('Reference is missing its exact current Lean source binding; regenerate it.')
    cases = data.get('cases', [])
    if not cases:
        raise ValueError('Reference has no cases.')
    for c in cases:
        if c.get('op') not in {'add', 'sub', 'mul', 'div', 'neg', 'abs'}:
            raise ValueError('Unknown reference operation.')
        if any(type(c.get(k)) is not int or not 0 <= c[k] < 2**32 for k in ('a', 'b', 'expected')):
            raise ValueError('Reference requires uint32 input and result words.')
    return cases


def generate(path):
    before = model_hashes()
    completed = subprocess.run(['lake', 'env', 'lean', '--run', str(HERE / 'ScalarCases.lean')],
                               cwd=ROOT, text=True, capture_output=True, check=True)
    data = json.loads(completed.stdout)
    if before != model_hashes():
        raise ValueError('Model changed while generating reference.')
    data['source_sha256'] = before
    data['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    data['native_execution_performed'] = False
    data['universal_native_equality_proved'] = False
    validate_reference(data)
    with path.open('x') as output:
        json.dump(data, output, indent=2)
        output.write('\n')
    print(json.dumps({'reference': str(path), 'cases': len(data['cases']), 'native_execution_performed': False}))


def is_nan(word):
    return word & 0x7f800000 == 0x7f800000 and word & 0x007fffff != 0


def compare(cases, actual):
    if len(actual) != len(cases):
        raise ValueError('Native output length differs from reference.')
    differences = []
    for i, (case, result) in enumerate(zip(cases, actual)):
        if result != case['expected']:
            differences.append({**case, 'case_index': i, 'actual': result,
                                'both_results_nan': is_nan(result) and is_nan(case['expected'])})
    return differences


def run(reference, output):
    data = json.loads(reference.read_text())
    cases = validate_reference(data)
    import numpy as np
    import torch
    import triton
    from native_kernels import scalar32_kernel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable; native validation was not run.')
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for op in ('add', 'sub', 'mul', 'div', 'div_rn', 'neg', 'abs'):
        selected = [c for c in cases if c['op'] == ('div' if op == 'div_rn' else op)]
        def tensor(key):
            return torch.from_numpy(np.array([c[key] for c in selected], dtype=np.uint32).view(np.int32)).cuda()
        a, b = tensor('a'), tensor('b')
        out = torch.empty_like(a)
        compiled = scalar32_kernel[(triton.cdiv(len(selected), 256),)](
            a, b, out, len(selected), op, 256, num_warps=4, enable_fp_fusion=False)
        torch.cuda.synchronize()
        actual = [int(x) for x in out.cpu().numpy().view(np.uint32)]
        differences = compare(selected, actual)
        artifacts = {}
        for name in ('ptx', 'ttir', 'ttgir', 'llir'):
            content = compiled.asm.get(name)
            if isinstance(content, str):
                path = output / f'{op}.{name}'
                path.write_text(content)
                artifacts[path.name] = digest(path)
        results.append({'operation': op, 'count': len(selected), 'output_words': actual,
                        'mismatches': differences, 'artifact_sha256': artifacts,
                        'status': 'MISMATCH' if differences else 'MATCH_ON_TESTED_INPUTS'})
    report = {'schema': 1, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'reference_sha256': digest(reference), 'source_sha256': model_hashes(),
              'runner_sha256': digest(Path(__file__)), 'kernel_sha256': digest(HERE / 'native_kernels.py'),
              'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
              'triton': triton.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'launch': {'block': 256, 'num_warps': 4, 'enable_fp_fusion': False}, 'results': results,
              'native_execution_performed': True, 'universal_native_equality_proved': False,
              'all_tested_cases_match': all(not r['mismatches'] for r in results),
              'limit': 'Primitive probes only. Finite matching inputs do not prove whole-kernel equality, reductions, casts, transcendental accuracy or compiler refinement.'}
    (output / 'reference.json').write_bytes(reference.read_bytes())
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'counts': dict(Counter(r['status'] for r in results)),
                      'universal_native_equality_proved': False}))
    return 0 if report['all_tested_cases_match'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate-reference', type=Path)
    mode.add_argument('--run', type=Path, metavar='REFERENCE')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.generate_reference:
        generate(args.generate_reference)
        return 0
    if not args.output:
        parser.error('--run requires a new --output directory')
    return run(args.run, args.output)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
