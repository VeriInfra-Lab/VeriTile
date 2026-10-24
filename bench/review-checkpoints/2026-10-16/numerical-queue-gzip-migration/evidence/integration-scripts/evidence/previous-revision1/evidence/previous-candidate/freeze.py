from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
source=load(W/'evidence/source-plan.json');audit=load(W/'evidence/candidate-audit.json');checks=load(W/'evidence/checks.json')
assert audit['status']==checks['status']=='passed' and checks['unit_tests']==22
for rel,row in source['changes'].items():assert sha(W/'after'/rel)==row['after'] and (sha(R/rel) if (R/rel).exists() else None)==row['before']
for rel,h in {**source['unchanged_consumers'],**source['policy_inputs']}.items():assert sha(R/rel)==h
for row in checks['commands']:assert row['exit_code']==0 and sha(W/row['log'])==row['log_sha256']
artifacts={str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and p.name!='receipt.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
r={'schema':1,'frozen':True,'status':'integration_scripts_and_consumer_delta_pending_independent_review','producer':'/root/matrix_transpose_acceptance','storage_candidate_receipt_sha256':source['original_candidate_receipt_sha256'],'storage_independent_review_receipt_sha256':source['independent_receipt_sha256'],'proposed_installed_files':source['changes'],'unchanged_consumers':source['unchanged_consumers'],'policy_inputs':source['policy_inputs'],'expected_base_commit':None,'base_commit_must_be_supplied_after_current_batch_commit':True,'live_plan_executed':False,'migration_performed':False,'repository_source_modified':False,'frozen_packages_modified':False,'primary_gate_installed_or_changed':False,'legacy_task_rows_preserved_by_candidate':183,'legacy_ledger_fields_allowed_to_change':['/deferred_queue/path','/deferred_queue/sha256'],'real_ledger_rebind_test':'in-memory only; all bytes outside the deferred_queue block preserved','unit_tests':22,'retained_failed_harness_attempts':1,'actual_plain_queue_bytes':audit['checks']['live_plain_queue_bytes'],'new_dtype_profiles':0,'dependency_files_copied':0,'acceptance_increment':0,'artifacts':artifacts}
p=W/'receipt.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'receipt':str(p),'sha256':sha(p),'artifacts':len(artifacts),'installed_candidate_files':len(source['changes']),'tests':22,'live_installation':False}))
