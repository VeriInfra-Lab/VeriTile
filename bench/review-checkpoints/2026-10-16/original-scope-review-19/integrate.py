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
BATCH = ROOT / 'bench/review-checkpoints/2026-10-16/original-scope-review-19'
CAT = ROOT / 'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics/current-source-catalog.json'
LEDGER = ROOT / 'bench/original_program_scope.json'
BASE = 'fc183f5dae5e3f09f549b8151a9a1e4c366beb02'
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
        'batch-02-part1-review.json': (TMP / 'remaining-original-scope-audit/batch-02-part1-review.json', 'f2f43ea2c3cbf2f230fa597fcc777daaee2d84c17e3ade750c95b428ec82ceb9'),
        'batch-02-part2-review.json': (TMP / 'remaining-original-scope-audit/batch-02-part2-review.json', '307b7b690bb56bb418b6f17d3a8194607b555aac744dd24d5d0a2a62b74dfa52'),
        'batch-02-part3-review.json': (TMP / 'remaining-original-scope-audit/batch-02-part3-review.json', '7f13992196c904c5b055abeae4cc692511a36d224916280d400dadea78341934'),
        'aggregate-receipt.json': (TMP / 'remaining-original-scope-audit/batch-02-remaining-17-aggregate-receipt.json', 'a92c0d7a488266ccc085a15abfa643d05dfaf3916dfac666f28ff8842dd61888'),
        'quant_transpose_kernel.json': (TMP / 'quant-transpose-original-scope-root-review.json', '164b599df19a175b69ab1a48b0ae9bfcbb5fa4b81a34904ba5a833f302b04d5f'),
        'swiglu_backward.json': (TMP / 'swiglu-backward-original-scope-root-review.json', '01b7dd3959cbdc06a798a29e59393a236aeb7e4e4c89a98eee97fbd0cfbe7a28'),
    }
    loaded = {}
    for name, (path, expected) in sources.items():
        assert digest(path) == expected, name
        loaded[name] = json.loads(path.read_text())
    aggregate = loaded['aggregate-receipt.json']
    assert aggregate['task_count'] == 17 and aggregate['public_declarations_checked'] == 53
    assert aggregate['structural_gap_count'] == 36 and aggregate['evidence_ranges_checked'] == 97
    reviews = []
    for name in ('batch-02-part1-review.json', 'batch-02-part2-review.json', 'batch-02-part3-review.json'):
        cohort = loaded[name]
        assert cohort['task_count'] == len(cohort['tasks'])
        assert cohort['public_declarations_checked'] == sum(len(r['public_declarations_checked']) for r in cohort['tasks'])
        assert cohort['structural_gap_count'] == sum(len(r['structural_gaps']) for r in cohort['tasks'])
        reviews += [(r, name) for r in cohort['tasks']]
    assert {r['task'] for r, _ in reviews} == {r['task'] for r in aggregate['tasks']}
    for name in ('quant_transpose_kernel.json', 'swiglu_backward.json'):
        reviews.append((loaded[name], name))
    assert len(reviews) == len({r['task'] for r, _ in reviews}) == 19
    # Preserve all producer receipt dependencies without changing frozen bytes.
    for field in ('inventory', 'source_ledger_snapshot', 'validator'):
        entry = aggregate[field]
        path = Path(entry['path'])
        sources['producer/' + path.name] = (path, entry['sha256'])
    for task in aggregate['tasks']:
        for field in ('note', 'read_inputs'):
            entry = task[field]
            sources['producer/' + entry['path']] = (TMP / 'remaining-original-scope-audit' / entry['path'], entry['sha256'])
    for path, expected in sources.values():
        assert digest(path) == expected, str(path)
    assert digest(ROOT / aggregate['catalog']['path']) == aggregate['catalog']['sha256']
    inventory = json.loads(Path(aggregate['inventory']['path']).read_text())
    inventory_tasks = {row['task']: row for row in inventory['tasks']}
    catalog = json.loads(CAT.read_text())
    old = json.loads(LEDGER.read_text())
    updated = copy.deepcopy(old)
    assert old['catalog_sha256'] == sha(canonical(catalog))
    rows = {r['task']: r for r in updated['tasks']}
    old_rows = {r['task']: r for r in old['tasks']}
    assert Counter(r['initial_assessment'] for r in rows.values()) == {
        'structural_gaps_confirmed': 66, 'full_scope_not_yet_verified': 117}
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
    assert len(current_index) == 938
    checked_inputs = {}
    for rel, h in aggregate['source_sha256'].items():
        assert digest(ROOT / rel) == h, rel
        checked_inputs[rel] = h
    evidence_ranges_checked = 0
    mapped_functions_checked = 0
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
        for d in review['public_declarations_checked']:
            rel = d['file']
            name = d.get('parser_name', d['declaration'])
            current = parsed[rel][1][name]
            if 'statement' in d:
                from adapter_common import split_statement
                assert split_statement(current['text'])[0] == d['statement']
            if 'sha256' in d:
                assert digest(ROOT / rel) == d['sha256']
            if 'line' in d:
                assert parsed[rel][0].count('\n', 0, current['start']) + 1 == d['line']
        stages = review['production_stage_mapping']
        if filename.startswith('batch-02-'):
            production = inventory_tasks[task]['production']
            assert {s['function'] for s in stages} == {s['name'] for s in production}
            for stage in stages:
                original = next(p for p in production if p['name'] == stage['function'])
                assert (stage['path'], stage['line'], stage['end_line']) == (original['file'], original['line'], original['end_line'])
                assert digest(ROOT / stage['path']) == stage['sha256']
        else:
            import ast
            for stage in stages:
                nodes = [n for n in ast.walk(ast.parse((ROOT / stage['path']).read_text())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == stage['function']]
                assert len(nodes) == 1
                assert (nodes[0].lineno, nodes[0].end_lineno) == (stage['line'], stage['end_line'])
        mapped_functions_checked += len(stages)
        for gap in review['structural_gaps']:
            assert gap['can_defer_as_numeric_assumption'] is False
            for e in gap.get('evidence', []):
                evidence_ranges_checked += 1
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
    assert public_review_count == 58 and gap_count == 42
    assert Counter(r['initial_assessment'] for r in rows.values()) == {
        'structural_gaps_confirmed': 85, 'full_scope_not_yet_verified': 98}
    updated['scope_audit_checkpoint'] = (BATCH / 'README.md').relative_to(ROOT).as_posix()
    report = {'schema': 1, 'kind': 'root_scope_cohort_integration_review',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'reviewer': 'root',
        'base_commit': BASE, 'status': 'scope_gaps_reviewed_no_acceptance',
        'task_count': 19, 'public_declarations_rebound': public_review_count,
        'specific_structural_gaps': gap_count, 'changed_task_names': sorted(names),
        'reviewed_input_sha256': checked_inputs, 'catalog_file_sha256': digest(CAT),
        'previous_ledger_sha256': digest(LEDGER),
        'evidence_ranges_checked': evidence_ranges_checked,
        'production_functions_checked': mapped_functions_checked,
        'independent_source_spot_checks': ['matrix_vector_multip', 'rmsnorm_fused_llama',
            'dequantize_matmul', 'logsumexp_fwd', 'token_softmax_llama'],
        'root_direct_reviews': ['quant_transpose_kernel', 'swiglu_backward'],
        'preservation': {'other_task_rows_unchanged': 164, 'lean_sources_changed': 0,
            'original_python_changed': 0, 'public_statements_changed': 0,
            'historical_catalog_records_changed': 0, 'new_numerical_deferrals': 0},
        'progress': {'original_programs': 183, 'accepted': 0, 'pending': 183,
            'structural_gaps_confirmed': 85, 'scope_review_pending': 98},
        'boundary': 'Manual source/spec scope review and exact binding checks only. Neither review counts nor declaration counts are completed original programs.'}
    index = {'schema': 1, 'kind': 'current_declaration_source_coordinates',
        'catalog': {'path': CAT.relative_to(ROOT).as_posix(), 'sha256': digest(CAT)},
        'declaration_count': 938, 'declarations': current_index,
        'retained_historical_coordinate_differences': historical_differences,
        'interpretation': 'Current locations and file hashes are independently parsed from current source. Existing catalog declaration records deliberately retain historical metadata; they are unchanged. This index establishes no new semantic coverage or proof acceptance.'}
    # All source, statement, population and preservation checks precede writes.
    (BATCH / 'evidence').mkdir(parents=True)
    for name, (path, h) in sources.items():
        target = BATCH / 'evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True)
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
    text = '''# Original-program scope review: 19 additional tasks

Whole-original-program acceptance remains **0/183**, with **183 pending**.
The ledger now distinguishes **85 tasks with confirmed structural gaps** and
**98 awaiting a full scope review**. This batch adds 19 reviews, checks 58 public
statements and records 42 specific structural gaps. Existing proofs are preserved.

The cohort consists of 17 frozen batch-02 reviews plus root reviews of
`quant_transpose_kernel` and `swiglu_backward`. Every task's full current Lean/Python
input set, exact public declarations and quoted evidence ranges were rebound to
the clean integration base. The 164 other task rows are unchanged. Producer
receipts, notes, input records and validator are retained verbatim under evidence.

The findings distinguish complete kernel results from omitted original stages.
Examples include the final `torch.mm` in dequantize_matmul, host final logsumexp,
the general matrix-vector loop beyond the proved one-block result, independent
input/output dtype routes and complete wrapper/grid composition. Current complete
rmsnorm loops, reverse-scan tail semantics, embedding vocabulary branches, and
source-ordered arithmetic remain credited. Ordinary bounds/layout premises are
not themselves structural gaps. No new numerical deferral was registered.

`current-declaration-index.json` separately records actual locations and file hashes
for all 938 current public declarations. Historical catalogs and the original
metadata baseline were not rewritten. The current catalog was not regenerated.

The scope CLI actually exited 1 with exactly 183 pending scopes and no other
population, catalog or binding error. No Lean, comparator, frontend or GPU run
was repeated for this source-preserving review. Prior build/global-gate results
remain in `../relu-source-grid-and-exact-numerics/`.
'''
    (BATCH / 'README.md').write_text(text)
    doc = ROOT / 'bench/original_program_scope.md'
    old_text = doc.read_text()
    old_progress = 'The current full-scope review inventory is 66 programs with confirmed structural\ngaps and 117 without a completed scope review.'
    new_progress = 'The current full-scope review inventory is 85 programs with confirmed structural\ngaps and 98 without a completed scope review.'
    assert old_text.count(old_progress) == 1
    doc.write_text(old_text.replace(old_progress, new_progress))
    validation = {'schema': 1, 'kind': 'scope_review_integration_validation',
        'base_commit': BASE, 'status': 'review_integration_passed_whole_scope_pending',
        'scope_gate': {'command': command, 'exit_code': result.returncode,
            'result_sha256': digest(BATCH / 'scope-gate.json'), 'log_sha256': digest(BATCH / 'scope-gate.log')},
        'progress': report['progress'], 'current_declarations_indexed': len(current_index),
        'retained_historical_coordinate_differences': len(historical_differences),
        'checked_source_input_count': len(checked_inputs), 'new_lean_runs': 0,
        'file_sha256': {p.relative_to(ROOT).as_posix(): digest(p)
                       for p in [LEDGER, doc, *sorted(BATCH.rglob('*'))] if p.is_file()}}
    write_json(BATCH / 'validation.json', validation)
    print(json.dumps({'status': validation['status'], 'progress': report['progress'],
        'checked_source_inputs': len(checked_inputs), 'historical_coordinate_differences': len(historical_differences),
        'validation_sha256': digest(BATCH / 'validation.json')}, indent=2))


if __name__ == '__main__':
    main()
