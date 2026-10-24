"""Freeze successful selected-profile source work and its honest scope boundary."""
from datetime import datetime,timezone
import hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(REPO/'bench'))
from adapter_common import declarations
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=ROOT/'evidence/receipt.json'
assert not out.exists(),'refuse frozen receipt overwrite'
modules=read(ROOT/'modules.json')
before=read(ROOT/'evidence/final-dependencies-before.json')
assert before==read(ROOT/'evidence/final-dependencies-after.json')
for row in before['files'].values():
    assert sha(Path(row['source_path']))==row['source_sha256']
    if 'olean_path' in row:assert sha(Path(row['olean_path']))==row['olean_sha256']
for p,h in before['preexisting_task_files_unchanged'].items():assert sha(REPO/p)==h
history=read(ROOT/'evidence/verification-attempts.json')
final=read(ROOT/'evidence/verification-results.json')
assert [r['module'] for r in final]==modules and history[-len(final):]==final
axioms=noncircular=audits=0
for rel in modules:
    text=(ROOT/rel).read_text()
    assert all(s==s.rstrip(' \t') for s in text.splitlines()) and text.endswith('\n')
    assert not re.search(r'\b(sorry|admit|axiom|unsafe)\b',text)
    axioms+=text.count('#axiomsClean ');noncircular+=text.count('#specNonCircular ');audits+=text.count('#auditModuleAxioms')
for row in history:
    assert sha(ROOT/row['log'])==row['log_sha256']
    assert sha(ROOT/row['source_snapshot'])==row['source_sha256_before']==row['snapshot_sha256']
for row in final:
    assert row['exit_code']==0 and sha(ROOT/row['module'])==row['source_sha256_before']==row['source_sha256_after']
    assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b|warning:',(ROOT/row['log']).read_text())
frontend=read(ROOT/'evidence/frontend/receipt.json')
assert frontend['frontend_successes']==18 and frontend['kernel_ast_identical_between_tasks']
for row in frontend['profiles']:assert sha(ROOT/row['ttir_path'])==row['ttir_sha256']
scope=read(ROOT/'evidence/scope-review.json')
public=[]
for task in scope['tasks']:
    rel=scope['tasks'][task]['public_file'];name='token_softmax_mixed_output_typed_correctness'
    parsed=declarations(ROOT/rel)[name]
    assert parsed['kind']=='specification'
    public.append({'task':task,'file':rel,'declaration':name,'kind':'specification','statement_sha256':parsed['statement_sha256'],
        'role':'parametric_value_contract','model_kind':'backend_value_contract','scope_relation':'full_declared_stage',
        'dtype_scope_relation':'selected_explicit_profiles_only','selected_storage_pairs':scope['selected_storage_pairs'],
        'covers_stages':['complete_per_program_mixed_floating_row_value_and_frame'],'whole_original_program_accepted':False})
artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*'))
           if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean','.ilean','.pyc'} and p!=out}
receipt={'schema':1,'kind':'selected_mixed_output_token_softmax_source_package',
    'status':'compiled_partial_selected_routes_pending_root_review','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
    'head_at_freeze':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    'modules':{p:sha(ROOT/p) for p in modules},'public_declarations':public,
    'scope_review_sha256':sha(ROOT/'evidence/scope-review.json'),
    'verification':{'final_successful_modules':len(final),'total_attempts':len(history),
        'failed_compilation_attempts':sum(r['exit_code']!=0 for r in history),
        'named_axiom_checks':axioms,'noncircularity_checks':noncircular,'module_audits':audits,
        'dependency_source_files':len(before['files']),'imported_project_oleans':sum('olean_path' in r for r in before['files'].values()),
        'before_after_dependency_snapshots_identical':True},
    'source_evidence':{'exact_original_kernel_extractions':2,'original_kernel_asts_identical':True,
        'successful_raw_frontend_profiles':18,'connected_scalar_graph_audits':18,
        'target':'sm90','block_size':1024,'num_warps':4,'native_backend_compilations':0,'gpu_executions':0},
    'existing_task_files_preserved':len(before['preexisting_task_files_unchanged']),
    'descriptive_numerical_followup_families':9,'exact_numerical_registrations_added':0,
    'current_repository_integrated':False,'whole_program_completion_increment':0,
    'independent_review':'pending','artifacts':artifacts,
    'acceptance_boundary':'Two successful source-coupled typed value/frame contracts close the independent destination dtype gap for the selected nine floating pairs. Original host/grid coverage and remaining dtype/admission routes are not accepted; native numerical validation and compiler refinement are not inferred.'}
out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'receipt':str(out),'sha256':sha(out),'verification':receipt['verification'],'public_declarations':public},indent=2))
