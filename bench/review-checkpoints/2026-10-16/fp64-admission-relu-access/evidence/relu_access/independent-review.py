"""Read-only independent review of frozen ReLU partial byte execution."""
import hashlib,json,re,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
base=Path('/tmp/veritile-standardization-20261015');root=base/'relu-partial-byte-execution-wip';ev=root/'evidence'
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
assert sha(ev/'receipt.json')=='0f845c08f838bb9f2f36c9c0fbcdbd799ba2ec7d5e394d41afa13ef9b4aa8228'
assert sha(ev/'scope-review.json')=='375a7f945b9d020babf5bdec78f0bee589401121a7f773b01f8f62f4269413ca'
r=read(ev/'receipt.json');scope=read(ev/'scope-review.json')
for rel,h in r['artifacts'].items():assert sha(root/rel)==h,rel
for rel,h in r['modules'].items():assert sha(root/rel)==h,rel
before=read(ev/'final-dependencies-before.json');after=read(ev/'final-dependencies-after.json');assert before==after
assert len(before['files'])==184
for rel,v in before['files'].items():
 assert sha(Path(v['source_path']))==v['sha256'],rel
 if 'olean_path' in v:
  assert sha(Path(v['olean_path']))==v['olean_sha256'],rel
  if rel.startswith('bench/'):
   actual=(root/rel).with_suffix('.olean');assert actual.is_file() and sha(actual)==v['olean_sha256'],rel
assert sum('olean_path' in v for v in before['files'].values())==176
prior_source_count=0;prior_artifact_count=0
assert r['prior_frozen_receipts']==before['frozen_receipts']
for path,h in before['frozen_receipts'].items():
 prior=Path(path);assert sha(prior/'evidence/receipt.json')==h['receipt'] and sha(prior/'evidence/scope-review.json')==h['scope']
 old=read(prior/'evidence/receipt.json')
 for rel,hash_value in old['modules'].items():assert sha(prior/rel)==hash_value;prior_source_count+=1
 for rel,hash_value in old['artifacts'].items():assert sha(prior/rel)==hash_value;prior_artifact_count+=1
assert prior_source_count==16
route=before['inherited_route_review'];assert sha(Path(route['path']))==route['sha256']==scope['inherited_route_review']['sha256']
for path,h in before['audited_ttir_sha256'].items():assert sha(Path(path))==h
assert len(before['audited_ttir_sha256'])==13
assert sha(repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py')==scope['source_sha256']
attempts=read(ev/'verification-attempts.json');results=read(ev/'verification-results.json')
assert len(attempts)==21 and len(results)==4 and attempts[-4:]==results
failed=[(i+1,a) for i,a in enumerate(attempts) if a['exit_code']!=0]
assert [i for i,_ in failed]==[1,2,3,5,7,8,11]
assert [a for _,a in failed]==r['failed_attempts']
for a in attempts:
 assert sha(root/a['log'])==a['log_sha256']
 assert a['source_sha256_before']==a['source_sha256_after']==a['snapshot_sha256']==sha(root/a['source_snapshot'])
 if a['exit_code']!=0:assert re.search(r'\berror:',(root/a['log']).read_text())
logs=[]
for a in results:
 assert a['exit_code']==0 and sha(root/a['module'])==a['source_sha256_before']
 s=(root/a['log']).read_text();assert not re.search(r'\berror:|\bwarning:|sorryAx|DISALLOWED',s);logs.append(s)
assert sum(s.count('axiom footprint ⊆ standard base ✓') for s in logs)==29
assert sum(s.count('non-circular ✓') for s in logs)==3
sources={Path(rel).stem:(root/rel).read_text() for rel in r['modules']}
assert sum(s.count('#auditModuleAxioms') for s in sources.values())==4
assert sum(s.count('#axiomsClean ') for s in sources.values())==29
assert sum(s.count('#specNonCircular ') for s in sources.values())==3
for s in sources.values():assert s.endswith('\n') and not re.search(r'[ \t]+$',s,re.M)
sys.path.insert(0,str(repo/'bench'));from adapter_common import declarations
public=r['public_contract'];parsed=declarations(root/public['file'])
assert parsed[public['parser_name']]['statement_sha256']==public['statement_sha256']
assert parsed[public['parser_name']]['kind']==public['kind']=='specification'
assert len([v for v in parsed.values() if v['kind']=='specification'])==1
# Guard the reviewed control choices. These are reviewer source checks, not new
# execution proofs or a source-to-native semantic refinement.
mem=sources['AccessMemory'];loads=sources['AccessLoads'];grid=sources['AccessGrid'];pub=sources['AccessGridCorrect']
checks={
 'read_requires_allocated_and_readable':'if mem.allocated address = Bool.true ∧ mem.readable address = Bool.true then' in mem,
 'write_requires_allocated_and_writable':'if mem.allocated address = Bool.true ∧ mem.writable address = Bool.true then' in mem,
 'store_fault_retains_memory':'.error (.storeUnavailable address, mem)' in mem and 'write_prefix_unavailable' in mem,
 'loads_bind_low_byte_then_remaining_high':'let low ← readByte mem base\n    let high ← loadLE count mem (base + 1)' in mem,
 'all_active_words_bind_in_order':'let previous ← loadWords encoding mem base start count\n    let word ← loadInput encoding mem base (start + count)' in loads,
 'program_load_result_is_matched_before_store_gate':grid.index('match loadWords')<grid.index('if pid = 0 then'),
 'nonzero_load_failure_is_explicit':'theorem nonzero_program_unavailable' in grid,
 'masked_lanes_excluded_by_active_count':'def activeCount (N pid : Nat) : Nat := min blockSize (N - pid * blockSize)' in loads,
 'active_store_values_only_use_loaded_lanes':'activeCount N pid)' in grid and 'loadedPrefix' in grid and 'store_prefix_congr' in grid,
 'nonzero_program_omits_scalar_result_evaluation':'else .ok mem' in grid,
 'schedule_threads_previous_memory':'let current ← program M encoding inputBase outputBase N pid mem\n    runSchedule M encoding inputBase outputBase N rest current' in grid,
 'public_output_permission_is_first_tile_only':'(min N blockSize * 4)' in pub,
 'public_input_permission_is_entire_N':'(N * elementBytes encoding)' in pub,
 'public_has_no_execution_success_premise':all(w not in pub.split(' :\n    ∃ final logicalFinal,')[0] for w in [' = .ok ', ' = some logicalFinal']),
 'full_output_domain_is_stronger_than_permission_domain':'(outputDomain : ByteDomain outputBase 4 N)' in pub,
 'public_composes_old_actual_typed_grid_and_address_certificate':'ByteGridCorrect.relu_byte_grid_correctness' in pub and 'AddressGridCorrect.AddressCertificate' in pub,
}
assert all(checks.values()),checks
# Independently stage immutable package artifacts and compare every index blob.
staged=Path(tempfile.mkdtemp(prefix='veritile-relu-access-independent-stage-'))
for rel in r['artifacts']:
 target=staged/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,target)
subprocess.run(['git','init','-q',str(staged)],check=True)
subprocess.run(['git','add','--','.'],cwd=staged,check=True)
for rel,h in r['artifacts'].items():assert hashlib.sha256(subprocess.check_output(['git','show',':'+rel],cwd=staged)).hexdigest()==h
check=subprocess.run(['git','diff','--cached','--check'],cwd=staged,capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
report={'schema':1,'status':'accepted_partial_byte_access_and_complete_program_permutation_refinement',
 'reviewer':'/root/comparator_restore','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
 'receipt_sha256':sha(ev/'receipt.json'),'scope_review_sha256':sha(ev/'scope-review.json'),'reviewed_source_sha256':r['modules'],
 'proof_defects_found':[],'whole_original_program_accepted':False,'original_program_completion_increment':0,
 'new_public_contracts':1,'new_dtype_profiles':0,'native_backend_runs':0,'gpu_runs':0,
 'checker_path':str(Path(__file__).resolve()),'checker_sha256':sha(Path(__file__)),
 'verification':{'frozen_artifacts_checked':len(r['artifacts']),'source_dependency_and_anchor_hashes_checked':184,
 'imported_project_oleans_checked':176,'inherited_source_modules_checked':16,'inherited_artifacts_checked':prior_artifact_count,
 'inherited_ttir_hashes_checked':13,'preserved_compile_attempts':21,'preserved_failed_attempts':[i for i,_ in failed],
 'final_actual_lean_exits':[0,0,0,0],'named_axiom_checks':29,'noncircularity_checks':3,'module_axiom_audit_commands':4,'warnings':0,
 'independent_lean_rerun':False,'public_statement_hashes_checked':1,'independent_staged_blobs_checked':len(r['artifacts']),
 'independent_staged_whitespace_exit':0,'temporary_stage':str(staged),'external_library_build_rechecked':False},
 'source_checks':checks,
 'semantic_review':[
 'All four new Lean sources read and their imported ByteGridCorrect/IntegerWords contracts inspected. readByte checks allocation and read permission before exposing a byte; loadLE uses ordered Except binds and exact little-endian assembly.',
 'writePrefix performs increasing-address byte writes. A failed later byte returns the precise memory after the successful prefix, with allocation/read/write maps preserved. This is a mathematical partial-write policy, not a native fault-order or trap result.',
 'program matches loadWords for every active lane of every pid before its pid-zero store gate. The nonzero_program_unavailable theorem proves a previously invisible active input fault even when the old total memory-effect projection is identity.',
 'loadWords and loadedPrefix fill unloaded container entries with zero. Only the actually loaded prefix supplies active stores. These placeholders do not assign zero to original masked loads without other, and do not prove inactive native values.',
 'The inherited scalar callbacks are pure total functions. Nonzero-pid comparison/conversion results are not separately evaluated in this access interpreter. It proves observable memory/access behavior under that model; it is not a stepwise source/TTIR trace.',
 'Readable covers all N input words. Writable covers exactly min(N,1024) output words and needs no output read permission. No tail allocation/read/write permission is required. LogicalView over the latent total byte payload does not imply tail accessibility.',
 'runSchedule and runPrefix thread actual previous memory. Permission maps are unchanged, so all required later loads remain permitted. Only pid0 writes; complete whole-program permutations therefore have the same byteGrid effect. No instruction interleaving or native race claim follows.',
 'The public specification composes successful partial-access schedules, the actual prior typed-grid execution, independent wordSpec/byteSpec, final complete tagged view, total-payload and allocated-content frames, active finite-pointer input loads and the inherited address certificate.',
 'Distinct logical input/output names, full-N interval separation, B1024, N<2^31 and finite64-bit base/end domains remain explicit selected-domain premises. They do not prove allocation freshness, native layouts/provenance, zero-grid launch behavior or general overlapping aliases.',
 'Nine finite-word integer/bool encodings are refined with zero new dtype names. fromInt/fromNat fp32 conversion and signed-negative fp32 zero remain explicit inherited numerical callbacks; native memory/compiler/host/error correspondence stays structurally open.'
 ],'numerical_followups':scope['numerical_followups'],
 'numerical_registration_status':'Producer followups are descriptive pending exact queue-scope binding; parent integration must bind the inherited callbacks to this new public statement before claiming queue registration.',
 'remaining_structural_obligations':scope['structural_remaining']}
p=base/'relu-partial-byte-execution-independent-review.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'review_path':str(p),'review_sha256':sha(p),'verification':report['verification']}))
