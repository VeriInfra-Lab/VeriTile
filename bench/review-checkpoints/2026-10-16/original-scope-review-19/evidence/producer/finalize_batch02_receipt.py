"""Validate and pin the three frozen reviews without modifying their inputs."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo / 'bench'))
import adapter_common as adapter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = {
    'batch-02-part1-review.json': 'f2f43ea2c3cbf2f230fa597fcc777daaee2d84c17e3ade750c95b428ec82ceb9',
    'batch-02-part2-review.json': '307b7b690bb56bb418b6f17d3a8194607b555aac744dd24d5d0a2a62b74dfa52',
    'batch-02-part3-review.json': '7f13992196c904c5b055abeae4cc692511a36d224916280d400dadea78341934',
}
inventory = json.loads((root / 'inventory.json').read_text())
inventory_tasks = {row['task']: row for row in inventory['tasks']}
assert sha(root / 'baseline-scope-ledger.json') == inventory['ledger_snapshot_sha256']
assert sha(Path(inventory['catalog_path'])) == inventory['catalog_sha256']
excluded = {'destindex_copy_kv1', 'destindex_copy_kv2', 'square_matrix'}
requested = set(next(batch['tasks'] for batch in inventory['batches'] if batch['number'] == 2)) - excluded
assert len(requested) == 17
source_bindings = {}
seen = set()
parts = []
tasks = []
declarations_checked = 0
gaps_checked = 0
evidence_ranges_checked = 0
production_functions_checked = 0


def check_source(rel, digest):
    assert sha(repo / rel) == digest, ('source changed', rel)
    assert rel not in source_bindings or source_bindings[rel] == digest
    source_bindings[rel] = digest


for name, digest in expected.items():
    path = root / name
    assert sha(path) == digest, ('frozen batch changed', name)
    batch = json.loads(path.read_text())
    assert batch['catalog_sha256'] == inventory['catalog_sha256']
    assert batch['source_ledger_snapshot_sha256'] == inventory['ledger_snapshot_sha256']
    assert batch['whole_original_program_accepted'] == batch['completion_increment'] == 0
    assert not batch['catalog_declaration_metadata_discrepancies']
    assert batch['task_count'] == len(batch['tasks'])
    assert batch['public_declarations_checked'] == sum(len(row['public_declarations_checked']) for row in batch['tasks'])
    assert batch['structural_gap_count'] == sum(len(row['structural_gaps']) for row in batch['tasks'])
    for row in batch['tasks']:
        task = row['task']
        assert task in requested and task not in seen
        seen.add(task)
        assert row['status'] == 'structural_gaps_confirmed'
        assert row['whole_original_program_accepted'] is False
        assert row['completion_increment'] == 0
        assert not row['catalog_declaration_metadata_discrepancies']
        for rel, value in row['source_sha256'].items():
            check_source(rel, value)
        for group in row['all_input_sha256'].values():
            for rel, value in group.items():
                check_source(rel, value)
        for declaration in row['public_declarations_checked']:
            file = repo / declaration['file']
            check_source(declaration['file'], declaration['sha256'])
            current = adapter.declarations(file)[declaration['parser_name']]
            assert current['kind'] == 'specification'
            assert current['statement_sha256'] == declaration['statement_sha256']
            assert adapter.split_statement(current['text'])[0] == declaration['statement']
            text = file.read_text()
            assert text[:text.index('specification ' + declaration['parser_name'])].count('\n') + 1 == declaration['line']
            declarations_checked += 1
        actual_public = {
            (rel, name)
            for rel in {d['file'] for d in row['public_declarations_checked']}
            for name, d in adapter.declarations(repo / rel).items()
            if d['kind'] == 'specification'
        }
        assert actual_public == {(d['file'], d['parser_name']) for d in row['public_declarations_checked']}
        production = inventory_tasks[task]['production']
        assert {stage['function'] for stage in row['production_stage_mapping']} == {stage['name'] for stage in production}
        assert len(row['production_stage_mapping']) == len(production)
        for stage in row['production_stage_mapping']:
            original = next(p for p in production if p['name'] == stage['function'])
            assert (stage['path'], stage['line'], stage['end_line']) == (original['file'], original['line'], original['end_line'])
            check_source(stage['path'], stage['sha256'])
            production_functions_checked += 1
        for gap in row['structural_gaps']:
            assert gap['can_defer_as_numeric_assumption'] is False
            for evidence in gap['evidence']:
                check_source(evidence['path'], evidence['sha256'])
                lines = (repo / evidence['path']).read_text().splitlines()
                assert '\n'.join(lines[evidence['line'] - 1:evidence['end_line']]) == evidence['excerpt']
                evidence_ranges_checked += 1
            gaps_checked += 1
        verification = row['verification']
        for key in ('new_lean_runs', 'new_frontend_runs', 'gpu_runs', 'numerical_deferrals_registered_by_this_read_only_review'):
            assert verification[key] == 0
        tasks.append({
            'task': task,
            'frozen_batch': name,
            'public_declarations_checked': len(row['public_declarations_checked']),
            'structural_gap_count': len(row['structural_gaps']),
            'note': {'path': f'notes/{task}.json', 'sha256': sha(root / 'notes' / f'{task}.json')},
            'read_inputs': {'path': f'read-inputs/{task}.json', 'sha256': sha(root / 'read-inputs' / f'{task}.json')},
        })
    parts.append({'path': str(path), 'sha256': digest, 'task_count': batch['task_count'], 'public_declarations_checked': batch['public_declarations_checked'], 'structural_gap_count': batch['structural_gap_count']})

assert seen == requested
assert declarations_checked == 53 and gaps_checked == 36
receipt = {
    'schema': 1,
    'kind': 'remaining_batch02_original_scope_aggregate_receipt',
    'reviewer': '/root/numerical_adapter_finish',
    'validated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewed_base_commit': inventory['reviewed_base_commit'],
    'inventory': {'path': str(root / 'inventory.json'), 'sha256': sha(root / 'inventory.json')},
    'source_ledger_snapshot': {'path': str(root / 'baseline-scope-ledger.json'), 'sha256': inventory['ledger_snapshot_sha256']},
    'catalog': {'path': inventory['catalog_path'], 'sha256': inventory['catalog_sha256']},
    'frozen_parts': parts,
    'task_count': len(tasks),
    'public_declarations_checked': declarations_checked,
    'structural_gap_count': gaps_checked,
    'production_functions_checked': production_functions_checked,
    'evidence_ranges_checked': evidence_ranges_checked,
    'current_source_files_hash_checked': len(source_bindings),
    'catalog_declaration_metadata_discrepancies': [],
    'whole_original_program_accepted': 0,
    'completion_increment': 0,
    'new_lean_runs': 0,
    'new_frontend_runs': 0,
    'gpu_runs': 0,
    'numerical_deferrals_registered': 0,
    'integration_note': 'Read-only scope findings. Preserve existing proofs. Only root may integrate; structural omissions cannot be treated as numerical deferrals. No repository or git writes, no new program acceptance, and no new execution are asserted by this receipt.',
    'tasks': tasks,
    'source_sha256': source_bindings,
    'validator': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
}
out = root / 'batch-02-remaining-17-aggregate-receipt.json'
with out.open('x') as handle:
    json.dump(receipt, handle, indent=2, ensure_ascii=False)
    handle.write('\n')
print(json.dumps({'path': str(out), 'sha256': sha(out), 'task_count': len(tasks), 'public_declarations_checked': declarations_checked, 'structural_gap_count': gaps_checked, 'evidence_ranges_checked': evidence_ranges_checked, 'source_files_hash_checked': len(source_bindings)}, sort_keys=True))
