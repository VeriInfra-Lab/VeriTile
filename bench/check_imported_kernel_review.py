#!/usr/bin/env python3
"""Guard the reviewed 334-case snapshot; this is not a semantic certificate.

The ledger contains human review findings and offline graph observations.
This command checks their source snapshot and recomputes structural facts and
small counterexamples. It neither solves the Lean goals nor admits FP rules.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'bench/imported_kernel_review.json'
COLLECTIONS = {'bug_detection': 65, 'pretrain_kernels': 269}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def location(path, needle):
    source = path.read_text()
    return {'path': str(path.relative_to(ROOT)),
            'line': source[:source.index(needle)].count('\n') + 1}


def bodies(source):
    result = {}
    for match in re.finditer(r'^def (original|candidate|realOriginal) : ComputeKernel :=', source, re.M):
        start = source.index('  triton {\n', match.end()) + len('  triton {\n')
        end = re.search(r'^  }$', source[start:], re.M)
        assert end, match[1]
        result[match[1]] = source[start:start + end.start()]
    return result


def output_formula(source):
    return source.split('def output :', 1)[1].split(':=', 1)[1].split('\nend Contract', 1)[0].strip()


def task_facts(collection, ident):
    directory = ROOT / 'bench' / collection / 'tasks' / ident
    metadata = json.loads((directory / 'task.json').read_text())
    kernel_path = directory / 'Kernels.lean'
    source = kernel_path.read_text()
    programs = bodies(source)
    contract = (directory / 'Contract.lean').read_text()
    correct = (directory / 'Correct.lean').read_text()
    fp = (directory / 'FPEquiv.lean').read_text()
    io = (directory / 'IO.lean').read_text()
    validation = (directory / 'Misc/Validation.lean').read_text()
    expected = re.sub(r'\s+', '', output_formula(contract))
    displayed = re.sub(r'\s+', '', correct.split('Spec.Real (', 1)[1])
    assert expected in displayed, ident
    assert 'correctnessProblem = Spec.Real (' in validation, ident
    assert 'def correctnessProblem : Prop :=' in correct, ident
    assert 'def equivalenceProblem ' in fp, ident
    assert 'core.Ctx ≃ Inputs' in io and 'LogicalScheduledKernelIO Inputs OutputIndex' in io, ident
    allowed = contract.split(' : IO.Inputs) : Prop :=', 1)[1].split('\ndef output :', 1)[0].strip()
    view = metadata.get('interface', {})
    inputs = view.get('inputs', metadata.get('inputs'))
    shape = view.get('output_shape', metadata.get('output'))
    precision = view.get('output_precision', metadata.get('output_precision'))
    ports = {x['name'] for x in inputs} if collection == 'bug_detection' else {x[0] for x in inputs}
    scratch = view.get('scratch', metadata.get('scratch'))
    ports |= {x['name'] for x in scratch} if collection == 'bug_detection' else {x[0] for x in scratch}
    ports.add('out')
    sort_sites, nan_stores = [], []
    for program, body in programs.items():
        body_start = source.index('  triton {\n', source.index('def ' + program + ' :')) + len('  triton {\n')
        def body_location(offset):
            return {'path': str(kernel_path.relative_to(ROOT)),
                    'line': source[:body_start + offset].count('\n') + 1}
        used = set(re.findall(r'\$\((\w+)_ptr\)', body))
        assert used <= ports, (ident, used - ports)
        rows = dict(re.findall(r'^\s*(\w+)_row_values = (.*)$', body, re.M))
        for match in re.finditer(r'^[ \t]*(\w+)_rank = tl.full.*$', body, re.M):
            prefix = match[1]
            row = rows[prefix]
            if re.search(r'dtype=tl\.(bfloat16|float16|float32|float64)\b', row):
                sort_sites.append({'program': program, **body_location(match.start()),
                                   'nan_order_explicit': prefix + '_other_nan =' in body,
                                   'values_selected_by_load': prefix + '_value = tl.sum(tl.where(' not in body})
        for match in re.finditer(r'^[ \t]*(\w+)_value = tl.bitcast.*\n\s*tl.store\(\$\((\w+)_ptr\)', body, re.M):
            name = match[2]
            declared = next((x[2] for x in scratch if not isinstance(x, dict) and x[0] == name), None)
            if declared in ('bf16', 'fp16'):
                nan_stores.append({'program': program, 'buffer': name,
                                   'stored': 'fp32', 'declared': declared,
                                   **body_location(match.start())})
    canonical = lambda s: re.sub(r'\bcandidate_', '', s).strip()
    bindings = directory / 'Proofs/Backend.lean'
    return {
        'collection': collection, 'id': ident, 'suite': metadata['suite'],
        'family': metadata.get('family'), 'inputs': inputs, 'output_shape': shape,
        'output_precision': precision, 'scope': 'fixed complete declared shape',
        'real_goal': location(directory / 'Correct.lean', 'def correctnessProblem'),
        'fp_goal': location(directory / 'FPEquiv.lean', 'def equivalenceProblem'),
        'real_formula_sha256': hashlib.sha256(expected.encode()).hexdigest(),
        'real_formula_result': output_formula(contract).splitlines()[-1].strip(),
        'public_real_domain': allowed,
        'fp_parameters': ['M', 'N', 'Allowed', 'D', 'R'] + (['decode'] if 'metadataAdmissible' in fp else []),
        'native_bindings': [location(bindings, 'def ' + name + 'BackendBindingProblem')
                            for name in ('original', 'candidate')],
        'real_projection': metadata.get('real_projection_status', 'DIRECT'),
        'metadata_decoder': metadata.get('metadata_decoder_binding_status', 'NOT_REQUIRED'),
        'proof_status': metadata['proof_status'],
        'source_connection': 'PENDING', 'backend_connection': metadata['backend_binding_status'],
        'current_rule_admissions': 0, 'proved_rule_applications': 0,
        'real_contract_equality_check': 'Lean Eq.refl in Misc/Validation.lean',
        'same_arithmetic_body_after_scratch_renaming': canonical(programs['original']) == canonical(programs['candidate']),
        'float_rank_sort_sites': sort_sites, 'nan_scratch_tag_mismatches': nan_stores,
        'int64_in_lowering': 'tl.int64' in source,
    }


def snapshot():
    paths = []
    for collection in COLLECTIONS:
        bench = ROOT / 'bench' / collection
        paths.extend(bench.rglob('*.lean'))
        paths.extend((bench / 'tasks').glob('*/task.json'))
        paths.extend([bench / 'manifest.json', bench / 'check.py', bench / 'README.md'])
    paths.extend([ROOT / 'AGENTS.md', ROOT / 'bench/MAIN_THEOREM_CONVENTIONS.md',
                  ROOT / 'lakefile.toml', ROOT / 'bench/tests/FPGuardedProgramIO.lean',
                  ROOT / 'VeriTile/Triton/Float/GuardedProgramIO.lean',
                  ROOT / 'VeriTile/Triton/Memory/LogicalScheduledKernelIO.lean',
                  Path(__file__)])
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def probes():
    values = [float('nan'), 3.0, 2.0]
    ranks = [sum(y > x or (y == x and j < i) for j, y in enumerate(values))
             for i, x in enumerate(values)]
    lowered = [sum(i for i, rank in enumerate(ranks) if rank == k) for k in range(len(values))]
    assert ranks == [0, 0, 1] and lowered == [1, 2, 0]
    assert lowered != [0, 1, 2]
    # This second probe concerns a primitive value-selection lowering. It does
    # not assert an observable final-output difference for every containing task.
    assert math.copysign(1.0, sum([-0.0, 0.0])) == 1.0
    alphabet = [float('nan'), -float('nan'), -float('inf'), -1.0, -0.0, 0.0, 1.0, float('inf')]
    checked = 0
    for length in range(1, 5):
        for values in itertools.product(alphabet, repeat=length):
            for descending in (True, False):
                def beats(j, i):
                    y, x = values[j], values[i]
                    yn, xn = y != y, x != x
                    nan_first = yn and not xn if descending else not yn and xn
                    strict = y > x if descending else y < x
                    return nan_first or (not yn and not xn and strict) or ((yn and xn or y == x) and j < i)
                fixed_ranks = [sum(beats(j, i) for j in range(length)) for i in range(length)]
                assert sorted(fixed_ranks) == list(range(length))
                actual = [fixed_ranks.index(k) for k in range(length)]
                def key(i):
                    x = values[i]; nonfinite_nan = math.isnan(x)
                    return ((not nonfinite_nan if descending else nonfinite_nan),
                            0.0 if nonfinite_nan else -x if descending else x, i)
                assert actual == sorted(range(length), key=key)
                gathered = [values[i] for i in actual]
                assert all(struct.pack('!d', x) == struct.pack('!d', values[i]) for x, i in zip(gathered, actual))
                checked += 1
    return {'nan_sort': {'ranks': ranks, 'lowered_indices': lowered, 'required_indices': [0, 1, 2]},
            'selected_negative_zero_sum': {'input_sign': -1, 'sum_sign': 1},
            'fixed_comparator_and_gather': {'small_alphabet_cases': checked, 'status': 'PASS',
                                           'scope': 'scalar ordering and payload selection only'}}


def main():
    record = json.loads(RECORD.read_text())
    assert record['source_sha256'] == snapshot(), 'reviewed sources changed; a new review is required'
    for relative, expected in record.get('main_integration', {}).get('dependency_source_sha256', {}).items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), 'dependency escapes the repository'
        assert digest(path) == expected, 'validated dependency changed: ' + relative
    assert record['probes'] == probes()
    counts = Counter()
    for key, row in record['tasks'].items():
        facts = task_facts(row['facts']['collection'], row['facts']['id'])
        assert facts == row['facts'], key
        counts[facts['collection']] += 1
    assert dict(counts) == COLLECTIONS
    print(json.dumps({'reviewed_snapshot': 'PASS', 'tasks': dict(counts),
                      'semantic_certification': 'NOT_ESTABLISHED',
                      'open_findings': {k: len(v['tasks']) for k, v in record['findings'].items()}}, indent=2))


if __name__ == '__main__':
    main()
