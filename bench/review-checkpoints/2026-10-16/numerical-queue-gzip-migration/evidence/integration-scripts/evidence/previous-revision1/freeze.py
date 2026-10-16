from pathlib import Path
import hashlib
import json

W = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
parent_path = W / 'evidence/previous-candidate/receipt.json'
parent_hash = '18f8b40a7718ef958edad7ef67dad8bbe919732ec1572c9a600e8cf4a7ed29d4'
assert sha(parent_path) == parent_hash
parent = load(parent_path)
for rel, h in parent['artifacts'].items():
    assert sha(parent_path.parent / rel) == h
for rel, row in parent['proposed_installed_files'].items():
    assert sha(W / 'after' / rel) == row['after']
audit = load(W / 'evidence/revision-audit.json')
checks = load(W / 'evidence/revision-checks.json')
assert audit['status'] == checks['status'] == 'passed'
assert checks['integration_tests_run'] == 16 and checks['new_revision_tests'] == 8
for row in checks['commands']:
    assert row['exit_code'] == 0 and sha(W / row['log']) == row['log_sha256']
artifacts = {str(p.relative_to(W)): sha(p) for p in sorted(W.rglob('*')) if p.is_file() and p != W / 'receipt.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}
receipt = {key: value for key, value in parent.items() if key not in ['artifacts', 'unit_tests', 'status']}
receipt.update(
    revision=1, status='integration_script_revision_pending_independent_delta_review',
    previous_scripts_receipt_sha256=parent_hash,
    exact_previous_package_preserved='evidence/previous-candidate',
    six_installed_candidates_byte_identical=True,
    checkpoint_script_receipt_and_all_artifacts_copied_and_reverified=True,
    run_directory_resolved_outside_repository_and_checkpoint=True,
    current_integration_tests=16, new_revision_tests=8,
    retained_failed_revision_audit_attempts=2,
    inherited_consumer_tests=14, consumer_tests_rerun=False,
    fresh_baseline_checks_retained=True, baseline_validation_reuse_added=False,
    broader_Lean_or_global_test_reruns_required_by_this_storage_delta=False,
    artifacts=artifacts,
)
target = W / 'receipt.json'
assert not target.exists()
target.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': str(target), 'sha256': sha(target), 'artifacts': len(artifacts), 'files_bytes': sum(p.stat().st_size for p in W.rglob('*') if p.is_file()), 'current_integration_tests': 16, 'live_installation': False}))
