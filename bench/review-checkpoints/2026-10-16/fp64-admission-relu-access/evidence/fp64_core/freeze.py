import datetime,hashlib,json,re,shutil,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(rel,o):
 p=root/rel;p.write_text(json.dumps(o,indent=2)+'\n');return p
assert not (root/'evidence/receipt.json').exists(), 'Frozen receipt exists; do not modify this package'
subprocess.run(['python3',str(root/'dependencies.py'),'final-dependencies-after.json'],check=True)
a=json.loads((root/'evidence/final-dependencies-before.json').read_text());z=json.loads((root/'evidence/final-dependencies-after.json').read_text());assert a==z
results=json.loads((root/'evidence/verification-results.json').read_text());assert len(results)==3
modules=sorted(root.glob('bench/**/*.lean'));assert len(modules)==3
assert all(r['exit_code']==0 and r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module']) and r['log_sha256']==sha(root/r['log']) for r in results)
logs='\n'.join((root/r['log']).read_text() for r in results)
assert 'warning:' not in logs and 'error:' not in logs
assert logs.count('axiom footprint ⊆ standard base ✓')==22
assert logs.count('non-circular ✓')==2
assert sum(p.read_text().count('#auditModuleAxioms') for p in modules)==3
obligations=[
('signedTo64','Interpret each8/16/32/64-bit signed input in two\'s complement and round directly to binary64; no fp32 intermediate.'),
('unsignedTo64','Interpret each8/16/32/64-bit unsigned input and round directly to binary64; no fp32 intermediate.'),
('halfTo32','Bind binary16-to-binary32 extension including signed zero, subnormals, infinities and NaN/payload policy.'),
('bfloatTo32','Bind bfloat16-to-binary32 extension including signed zero, subnormals, infinities and NaN/payload policy.'),
('singleTo64','Bind binary32-to-binary64 extension including specials and NaN/payload policy.'),
('e4To64','Bind direct E4M3FN-to-binary64 tt.fp_to_fp, including finite-only format and NaN policy.'),
('e5To64','Bind direct E5M2-to-binary64 tt.fp_to_fp, including subnormals, infinities and NaN policy.'),
('toSigned','Bind direct f64-to-signed8/16/32/64 rounding toward zero on its exact finite/range domain. None is explicit outside that domain; native poison/traps are a separate structural gap. Bool output uses signed8.'),
('toUnsigned','Bind direct f64-to-unsigned8/16/32/64 rounding toward zero on its exact finite/range domain. None is explicit outside that domain; native poison/traps are a separate structural gap.'),
('toHalf','Bind direct f64-to-binary16 truncf with specified rounding/overflow/underflow/NaN policy; no fp32 double-rounding route.'),
('toBFloat','Bind direct f64-to-bfloat16 truncf with specified rounding/overflow/underflow/NaN policy; no fp32 double-rounding route.'),
('toSingle','Bind direct f64-to-binary32 truncf with specified rounding/overflow/underflow/NaN policy.'),
('toE4RTNE','Bind direct f64-to-E4M3FN tt.fp_to_fp RTNE, including overflow/saturation and NaN policy. Existing fp32-input E4M3 software narrow does not establish this.'),
('toE5RTNE','Bind direct f64-to-E5M2 tt.fp_to_fp RTNE, including subnormal/overflow/infinity/NaN policy.'),
('add64','Bind binary64 addition with exact rounding, signed-zero, subnormal/flush, overflow, infinity and NaN behavior for the intended target configuration.')]
core=root/'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean'
block=core.read_text().split('structure Numerics where\n')[1].split('\ninductive Cast')[0]
assert re.findall(r'^  ([A-Za-z0-9]+) :',block,re.M)==[n for n,_ in obligations]
num=write('evidence/numerical-correspondence.json',{'schema':1,'kind':'explicit_deferred_numerical_correspondence_obligations','callback_declaration':str(core.relative_to(root)),'callback_declaration_source_sha256':sha(core),'status':'all_pending_native_numerical_binding','lean_assumption_status':'Theorems universally quantify callbacks; there is no native-conformance axiom or silently accepted numerical law.','families':[{'id':f'VA-F64-N{i:02d}','callback':n,'required_correspondence':d,'status':'pending_explicitly_deferred','may_be_deferred_under_user_standard':True} for i,(n,d) in enumerate(obligations,1)],'concrete_exceptions':['Binary64 word field encode/decode is concrete and kernel checked.','Bool input byte!=0 and unsignedi1-to-binary64 zero/one are concrete.','Bool output preserves the entire signed8 result byte.'],'not_numerical_exceptions':['typed-memory/grid/source/compiler/native storage/address/scheduling/host/poison/trap/sanitizer relations remain structural blockers.']})
gaps=[
{'id':'VA-F64-S1','gap':'Source/interpreter/compiler refinement','detail':'Task-local Expr is separate from shared ComputeOp/FP.Structural. Raw-SSA dependency checks are observations, not a formal source extraction or compiler theorem.','may_be_deferred_as_numerical':False},
{'id':'VA-F64-S2','gap':'Typed logical memory, grid and ordering','detail':'runLane returns a requested write; it has no memory state, frame, alias model, block/grid iteration or scheduling theorem. The mask is externally supplied.','may_be_deferred_as_numerical':False},
{'id':'VA-F64-S3','gap':'Native byte layout, allocations and finite addresses','detail':'Indexed bit widths do not bind physical storage/pointers/endianness, allocation extent, i32 pid/offset/mask arithmetic or address overflow.','may_be_deferred_as_numerical':False},
{'id':'VA-F64-S4','gap':'Native undefined/error/trap and host behavior','detail':'Option and absent-input outcomes do not prove native poison propagation, trap/sanitizer behavior, wrapper dispatch, allocations or Python/Torch composition.','may_be_deferred_as_numerical':False},
{'id':'VA-F64-S5','gap':'Remaining unsupported original dtype branches','detail':'The two fnuz types remain outside these29 successful frontend routes. Their28 semantically promoted frontend rejection pairs still require explicit original-program contracts.','may_be_deferred_as_numerical':False}]
scope=write('evidence/scope-review.json',{'schema':1,'reviewer':'producer /root/full_scope_audit; independent review pending','status':'partial_foundation_ready_for_independent_review','whole_original_program_accepted':False,'original_program_completion_increment':0,'complete_dtype_grid_profile_increment':0,'public_specification_count':0,'three_modules':[str(p.relative_to(root)) for p in modules],'proved_scope':['Full64-bit fields encode/extract inverse and exact complete class decoder.','Typed per-lane source expression equals independent word result for arbitrary numerical callbacks.','No-other masked lane has no write request; active absent loads and undefined integer conversion are explicit.','All29 successful fp64 supported pairs enumerated exactly and their dependency labels match the observed raw-SSA inventory.'],'numerical_obligations_file':str(num.relative_to(root)),'numerical_obligations_sha256':sha(num),'structural_gaps':gaps,'source_and_dependency_hashes_stable':True,'verification':{'actual_lean_exits':[r['exit_code'] for r in results],'named_axiom_checks':22,'noncircularity_checks':2,'whole_module_axiom_audits':3,'warnings':0,'project_source_dependencies':4,'imported_project_olean_hashes_checked':True,'external_library_rebuild_claimed':False},'coverage':{'prior_logical_value_grid_pairs':160,'promoted_pairs':217,'still_without_complete_logical_grid_value':57,'whole_original_programs_accepted':0,'whole_original_program_total':183}})
# Stage the actual deliverable text files in a disposable Git repository.
files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix!='.olean' and p.name not in {'receipt.json','whitespace-check.json'})
with tempfile.TemporaryDirectory(prefix='veritile-fp64-staged-') as tmp:
 stage=Path(tmp)
 for p in files:
  dest=stage/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 subprocess.run(['git','init','-q',str(stage)],check=True)
 subprocess.run(['git','-C',str(stage),'add','--','.'],check=True)
 cmd=['git','-C',str(stage),'diff','--cached','--check']
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 assert r.returncode==0,r.stdout
 whitespace=write('evidence/whitespace-check.json',{'command':cmd,'actual_exit_code':r.returncode,'output':r.stdout,'staged_files_sha256':{str(p.relative_to(root)):sha(p) for p in files},'temporary_git_directory_removed':True})
receipt={'schema':1,'task':'vector_addition genuine binary64 word and typed scalar-route foundation','package_root':str(root),'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'frozen_ready_for_independent_review','whole_original_program_accepted':False,'original_program_completion_increment':0,'complete_dtype_grid_profile_increment':0,'public_specification_count':0,'modules':{str(p.relative_to(root)):sha(p) for p in modules},'compiled_olean_sha256':{str(p.with_suffix('.olean').relative_to(root)):sha(p.with_suffix('.olean')) for p in modules},'verification':{'actual_lean_exit_codes':[r['exit_code'] for r in results],'named_axiom_checks':22,'noncircularity_checks':2,'whole_module_axiom_audits':3,'warnings':0,'project_dependencies':4,'source_and_imported_project_olean_hashes_unchanged':True,'staged_whitespace_exit':0,'new_frontend_compilations':0,'native_backend_compilations':0,'gpu_executions':0},'source_route_audit':{'profiles':29,'receipt_sha256':sha(root/'evidence/source-routes/receipt.json'),'source_sha256':json.loads((root/'evidence/source-routes/receipt.json').read_text())['source_sha256']},'numerical_callback_families':15,'numerical_correspondence_sha256':sha(num),'scope_review_sha256':sha(scope),'structural_gaps':gaps,'coverage':{'logical_value_grid_pairs':160,'promoted_pairs':217,'remaining_logical_grid_value_pairs':57,'fp64_foundation_pairs':29,'fp64_foundation_counted_as_completed_profiles':False,'whole_original_program_accepted':0,'whole_original_program_total':183},'artifacts':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix!='.olean' and p.name!='receipt.json'}}
p=write('evidence/receipt.json',receipt)
print(json.dumps({'receipt':str(p),'receipt_sha256':sha(p),'scope_sha256':sha(scope),'modules':receipt['modules']},indent=2))
