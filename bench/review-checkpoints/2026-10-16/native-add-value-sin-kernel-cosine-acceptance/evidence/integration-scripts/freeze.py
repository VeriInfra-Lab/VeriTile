from pathlib import Path
import hashlib,json

W=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
checks=load(W/'evidence/final-checks.json');assert checks['status']=='passed_for_script_preparation_only'
assert checks['test_count']==7 and checks['input_sha256_before']==checks['input_sha256_after']
for name,h in checks['input_sha256_after'].items():assert sha(W/name)==h
for row in checks['commands']:assert row['exit_code']==0 and sha(W/row['log'])==row['log_sha256']
preflight=load(W/checks['frozen_input_preflight']);lint=load(W/checks['lint_report'])
assert lint['status']=='passed' and sum(row['module_sources'] for row in preflight['packages'])==24
assert sum(row['families'] for row in preflight['packages'])==34 and sum(row['bindings'] for row in preflight['packages'])==577 and sum(row['cases'] for row in preflight['packages'])==558
assert sha(W/'specs.json')=='50490b7af9c7b81899ec872b0c3e4c158677a37bce2cc6ba20d8f3165a8f81b8'
assert load(W/'evidence/stage-controls/results.json')['status']=='passed'
artifacts={p.relative_to(W).as_posix():sha(p) for p in sorted(W.rglob('*')) if p.is_file() and p!=W/'candidate-receipt.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
receipt={'schema':1,'frozen':True,'kind':'next_three_source_integration_script_candidate_revision2','revision1_receipt_sha256':sha(W/'revision1-candidate-receipt.json'),'staging_checkpoint_set_sealed':True,'literal_no_renames_path_sets':True,'automatic_staging_reset_or_rollback':False,'focused_stage_controls':load(W/'evidence/stage-controls/results.json')['test_count'],'status':'pending_independent_root_script_review_and_actual_post_migration_base','spec_sha256':sha(W/'specs.json'),'baseline_legacy_scopes':{'accepted':8,'pending':175},'expected_legacy_scopes_after_integration':{'accepted':11,'pending':172},'new_source_modules':24,'new_numerical_rules':34,'new_exact_bindings':577,'new_declaration_cases':558,'current_candidate_tests':7,'adapter_logic_preserved_except_archive_paths':True,'old_adversarial_test_logic_preserved':True,'future_base_commit':None,'base_must_be_explicit_clean_commit_after_gzip_migration':True,'active_queue_ledger_catalog_supplement_use_gzip':True,'storage_receipt_manifest_plan_script_chain_required':True,'historical_deletions_authorized':False,'repository_or_Git_modified':False,'live_plan_or_integration_executed':False,'Lean_or_GPU_executed':False,'large_dependency_files_copied':0,'new_dtype_profiles':0,'legacy_acceptance_increment':0,'primary_version_completion_increment':0,'artifacts':artifacts}
path=W/'candidate-receipt.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'sha256':sha(path),'artifacts':len(artifacts),'package_bytes':sum(p.stat().st_size for p in W.rglob('*') if p.is_file()),'live_plan_executed':False,'acceptance_increment':0}))
