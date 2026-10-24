"""Integrate read-only scope reviews without accepting or changing any proof."""
import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
BATCH = ROOT / 'bench/review-checkpoints/2026-10-16/original-scope-review-24'
CAT = ROOT / 'bench/review-checkpoints/2026-10-16/fp64-admission-relu-access/current-source-catalog.json'
LEDGER = ROOT / 'bench/original_program_scope.json'
BASE = '74cb67ae99cd5cf77096dc74868d8adcb243b46a'
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import canonical, declarations, sha, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    assert git('rev-parse', 'HEAD').decode().strip() == BASE
    assert git('branch', '--show-current').decode().strip() == 'fp-examples'
    assert not git('status', '--porcelain'), 'require clean integration base'
    assert not BATCH.exists(), 'refuse evidence overwrite'
    sources = {
        'batch-01-review.json': (TMP / 'remaining-original-scope-audit/batch-01-review.json', '14637a2817702322b8d71d4c0194fb14deac9d31dba91038918db781008fdc65'),
        'var_len_copy.json': (TMP / 'var-len-copy-original-scope-root-review.json', '791b755953e8e53b682d30ad695beca9e09686886304266942ed8070bab38af4'),
        'destindex_copy_kv1.json': (TMP / 'root-copy-square-scope-review/destindex_copy_kv1.json', '06dd9595f169a05ce9d3efb82f083599f5443a44a638b85ab659f586da25802e'),
        'destindex_copy_kv2.json': (TMP / 'root-copy-square-scope-review/destindex_copy_kv2.json', '59ff7867a0ef236d83f306bdb4568f046b2c81ddd21fc08ea0cfc36c1ab7fd6d'),
        'square_matrix.json': (TMP / 'root-copy-square-scope-review/square_matrix.json', '16a14ee664016eb6197b83783830c176f242941c3a99f868dfabdb66edd0d4fd'),
    }
    loaded = {}
    for name, (path, expected) in sources.items():
        assert digest(path) == expected, name
        loaded[name] = json.loads(path.read_text())
    cohort = loaded['batch-01-review.json']
    assert cohort['task_count'] == 20 and cohort['public_declarations_checked'] == 63
    assert cohort['structural_gap_count'] == 37
    reviews = [(r, 'batch-01-review.json') for r in cohort['tasks']]
    reviews += [(loaded[name], name) for name in sources if name != 'batch-01-review.json']
    assert len(reviews) == len({r['task'] for r, _ in reviews}) == 24
    catalog = json.loads(CAT.read_text())
    old = json.loads(LEDGER.read_text())
    updated = copy.deepcopy(old)
    assert old['catalog_sha256'] == sha(canonical(catalog))
    rows = {r['task']: r for r in updated['tasks']}
    old_rows = {r['task']: r for r in old['tasks']}
    assert Counter(r['initial_assessment'] for r in rows.values()) == {
        'structural_gaps_confirmed': 42, 'full_scope_not_yet_verified': 141}
    assert len(rows) == 183 and all(r['status'] == 'pending' for r in rows.values())
    catalog_rows = {r['task']: r for r in catalog['tasks']}
    parsed = {}
    current_index = []
    historical_differences = []
    for task in catalog['tasks']:
        for declaration in task['declarations']:
            rel = declaration['file']
            if rel not in parsed:
                p = ROOT / rel
                parsed[rel] = (p.read_text(), declarations(p), digest(p))
            text, actual, file_hash = parsed[rel]
            name = declaration['declaration']
            d = actual[name]
            assert d['kind'] == 'specification'
            assert d['statement_sha256'] == declaration['statement_sha256']
            line = text.count('\n', 0, d['start']) + 1
            current = {'file': rel, 'line': line, 'sha256': file_hash}
            record = {'task': task['task'], 'declaration': name,
                      'statement_sha256': d['statement_sha256'], 'current_source': current}
            historical = {k: declaration[k] for k in ('line', 'sha256', 'statement_ref') if k in declaration}
            mismatches = {k: {'retained': historical[k], 'current': current[k]}
                          for k in ('line', 'sha256') if k in historical and historical[k] != current[k]}
            if 'statement_ref' in historical:
                for k in ('file', 'line', 'sha256'):
                    if k in historical['statement_ref'] and historical['statement_ref'][k] != current[k]:
                        mismatches['statement_ref.' + k] = {'retained': historical['statement_ref'][k], 'current': current[k]}
            if mismatches:
                historical_differences.append({'task': task['task'], 'file': rel,
                                              'declaration': name, 'differences': mismatches})
            current_index.append(record)
    assert len(current_index) == 936
    checked_inputs = {}
    public_review_count = 0
    gap_count = 0
    for review, filename in reviews:
        task = review['task']
        row = rows[task]
        assert row['initial_assessment'] == 'full_scope_not_yet_verified', task
        assert review['status'] == 'structural_gaps_confirmed'
        assert review['whole_original_program_accepted'] is False and review['completion_increment'] == 0
        assert review['structural_gaps']
        for rel, h in review['source_sha256'].items():
            assert digest(ROOT / rel) == h, rel
            checked_inputs[rel] = h
        if 'all_input_sha256' in review:
            assert review['all_input_sha256'] == row['input_sha256'], task
        for files in row['input_sha256'].values():
            for rel, h in files.items():
                assert digest(ROOT / rel) == h, rel
                checked_inputs[rel] = h
        expected_public = {(d['file'], d['declaration']): d['statement_sha256']
                           for d in catalog_rows[task]['declarations']}
        observed_public = {(d['file'], d['declaration']): d['statement_sha256']
                           for d in review['public_declarations_checked']}
        assert observed_public == expected_public, task
        for gap in review['structural_gaps']:
            assert gap['can_defer_as_numeric_assumption'] is False
            for e in gap.get('evidence', []):
                assert digest(ROOT / e['path']) == e['sha256']
                lines = (ROOT / e['path']).read_text().splitlines()
                assert 1 <= e['line'] <= e.get('end_line', e['line']) <= len(lines)
                if 'excerpt' in e:
                    assert '\n'.join(lines[e['line'] - 1:e['end_line']]) == e['excerpt']
        row['initial_structural_gaps'] = copy.deepcopy(row['structural_gaps'])
        row['structural_gaps'] = copy.deepcopy(review['structural_gaps'])
        row['initial_assessment'] = 'structural_gaps_confirmed'
        row['scope_gap_review'] = {'path': (BATCH / 'evidence' / filename).relative_to(ROOT).as_posix(),
                                 'sha256': sources[filename][1], 'task': task,
                                 'status': 'reviewed_gaps_not_acceptance'}
        assert row['review'] is None and not row['deferred_numerical_assumptions']
        public_review_count += len(observed_public)
        gap_count += len(review['structural_gaps'])
    names = {r['task'] for r, _ in reviews}
    assert all(row == old_rows[name] for name, row in rows.items() if name not in names)
    assert public_review_count == 79 and gap_count == 49
    assert Counter(r['initial_assessment'] for r in rows.values()) == {
        'structural_gaps_confirmed': 66, 'full_scope_not_yet_verified': 117}
    updated['scope_audit_checkpoint'] = (BATCH / 'README.md').relative_to(ROOT).as_posix()
    report = {'schema': 1, 'kind': 'root_scope_cohort_integration_review',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'reviewer': 'root',
        'base_commit': BASE, 'status': 'scope_gaps_reviewed_no_acceptance',
        'task_count': 24, 'public_declarations_rebound': public_review_count,
        'specific_structural_gaps': gap_count, 'changed_task_names': sorted(names),
        'reviewed_input_sha256': checked_inputs, 'catalog_file_sha256': digest(CAT),
        'previous_ledger_sha256': digest(LEDGER),
        'independent_source_spot_checks': ['quantize_global', 'rowwise_quantization_triton',
            'dequantize_rowwise', 'nested_loops_processing'],
        'root_direct_reviews': ['var_len_copy', 'destindex_copy_kv1', 'destindex_copy_kv2', 'square_matrix'],
        'preservation': {'other_task_rows_unchanged': 159, 'lean_sources_changed': 0,
            'original_python_changed': 0, 'public_statements_changed': 0,
            'historical_catalog_records_changed': 0, 'new_numerical_deferrals': 0},
        'progress': {'original_programs': 183, 'accepted': 0, 'pending': 183,
            'structural_gaps_confirmed': 66, 'scope_review_pending': 117},
        'boundary': 'Manual source/spec scope review and exact binding checks only. Neither review counts nor declaration counts are completed original programs.'}
    index = {'schema': 1, 'kind': 'current_declaration_source_coordinates',
        'catalog': {'path': CAT.relative_to(ROOT).as_posix(), 'sha256': digest(CAT)},
        'declaration_count': 936, 'declarations': current_index,
        'retained_historical_coordinate_differences': historical_differences,
        'interpretation': 'Current locations and file hashes are independently parsed from current source. Existing catalog declaration records deliberately retain historical metadata; they are unchanged. This index establishes no new semantic coverage or proof acceptance.'}
    # All source, statement, population and preservation checks precede writes.
    (BATCH / 'evidence').mkdir(parents=True)
    for name, (path, h) in sources.items():
        target = BATCH / 'evidence' / name
        target.write_bytes(path.read_bytes())
        assert digest(target) == h
    write_json(BATCH / 'root-review.json', report)
    write_json(BATCH / 'current-declaration-index.json', index)
    (BATCH / 'integrate.py').write_bytes(Path(__file__).read_bytes())
    write_json(LEDGER, updated)
    command = [sys.executable, '-B', 'bench/check_original_program_scope.py', '--root', str(ROOT),
               '--catalog', str(CAT), '--ledger', str(LEDGER), '--output', str(BATCH / 'scope-gate.json')]
    with (BATCH / 'scope-gate.log').open('w') as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    observed = json.loads((BATCH / 'scope-gate.json').read_text())
    assert result.returncode == 1 and observed['counts'] == {'not_accepted': 183}, observed
    assert observed['errors'] == ['183 original-program scopes remain unaccepted']
    assert all(t['error'] == 'scope remains pending' for t in observed['tasks'])
    for rel, h in checked_inputs.items():
        assert digest(ROOT / rel) == h, 'input changed during scope check: ' + rel
    text = '''# Original-program scope review: 24 additional tasks

Whole-original-program acceptance remains **0/183**, with **183 pending**.
The ledger now distinguishes **66 tasks with confirmed structural gaps** and
**117 awaiting a full scope review**. This batch adds 24 reviews, checks 79 public
statements and records 49 specific structural gaps. Existing proofs are preserved.

The reviewed cohort is the first 20 remaining tasks plus `var_len_copy`,
`destindex_copy_kv1`, `destindex_copy_kv2` and `square_matrix`. Each evidence record
binds the original source, current proof statements and concrete omitted behavior.
Ordinary bounds/layout premises are not classified as defects. Eligible numerical
correspondences are described without registering or validating them in this batch.

Representative findings are the missing `llrint` and int8 store in the two
quantization programs, unassembled wrapper grids, and dtype routes outside the
current materialized families. Existing rowwise maximum computation, canonical
seeded RNG, complete reverse scans and all 24 nested-loop stores remain credited.

`current-declaration-index.json` separately records the actual source location and
file hash of all 936 current public declarations. It also lists differences from
retained historical coordinates, including the two rowwise quantization records.
Historical catalogs and the original metadata baseline were not rewritten.

The scope gate actually exited 1: all 183 rows remain pending, with no extra
catalog/population/binding error. No Lean, comparator, frontend or GPU execution
was rerun for this source-preserving review. The preceding 13-module batch and
its build/global-gate results remain in `../fp64-admission-relu-access/`.
'''
    (BATCH / 'README.md').write_text(text)
    validation = {'schema': 1, 'kind': 'scope_review_integration_validation',
        'base_commit': BASE, 'status': 'review_integration_passed_whole_scope_pending',
        'scope_gate': {'command': command, 'exit_code': result.returncode,
            'result_sha256': digest(BATCH / 'scope-gate.json'), 'log_sha256': digest(BATCH / 'scope-gate.log')},
        'progress': report['progress'], 'current_declarations_indexed': len(current_index),
        'retained_historical_coordinate_differences': len(historical_differences),
        'checked_source_input_count': len(checked_inputs), 'new_lean_runs': 0,
        'file_sha256': {p.relative_to(ROOT).as_posix(): digest(p)
                       for p in [LEDGER, *sorted(BATCH.rglob('*'))] if p.is_file()}}
    write_json(BATCH / 'validation.json', validation)
    print(json.dumps({'status': validation['status'], 'progress': report['progress'],
        'checked_source_inputs': len(checked_inputs), 'historical_coordinate_differences': len(historical_differences),
        'validation_sha256': digest(BATCH / 'validation.json')}, indent=2))


if __name__ == '__main__':
    main()
