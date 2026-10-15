import hashlib,json,re,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parent;repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
evidence=root/'evidence';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def write(p,d):assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
results=read(evidence/'verification-results.json');assert len(results)==3 and all(r['exit_code']==0 for r in results)
modules={r['module']:r['source_sha256_after'] for r in results}
before=read(evidence/'final-dependencies-before.json');assert before==read(evidence/'final-dependencies-after.json')
for p,r in before['files'].items():
 assert sha(Path(r['source_path']))==r['sha256']
 if 'olean_path' in r:assert sha(Path(r['olean_path']))==r['olean_sha256']
logs=[]
for r in results:
 assert r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module'])
 assert sha(root/r['log'])==r['log_sha256'];logs.append((root/r['log']).read_text())
s='\n'.join(logs);assert s.count('axiom footprint ⊆ standard base ✓')==16 and s.count('non-circular ✓')==2
assert not re.search(r'error:|warning:|sorryAx|DISALLOWED',s)
for p in modules:
 source=(root/p).read_text();assert source.endswith('\n') and not re.search(r'[ \t]+$',source,re.M)
 assert '#auditModuleAxioms' in source
checkroot=Path(tempfile.mkdtemp(prefix='veritile-pointwise-whitespace-'))
for p in modules:
 target=checkroot/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/p,target)
subprocess.run(['git','init','--quiet',str(checkroot)],check=True)
subprocess.run(['git','add','--',*modules],cwd=checkroot,check=True)
check=subprocess.run(['git','diff','--cached','--check'],cwd=checkroot,capture_output=True,text=True)
assert check.returncode==0,check.stdout+check.stderr
write(evidence/'whitespace-check.json',{'kind':'actual_isolated_staged_whitespace_check','exit_code':check.returncode,
 'command':['git','diff','--cached','--check'],'source_sha256':modules,'temporary_index_root':str(checkroot),
 'stdout':check.stdout,'stderr':check.stderr})
publicpath='bench/tritonbench_g/vector_addition/ValueScheduleCorrect.lean'
parsed=declarations(root/publicpath)
names=['vector_addition_integer_program_order_correctness','vector_addition_mixed_fp32_total_program_order_correctness',
'vector_addition_mixed_fp32_defined_program_order_correctness','vector_addition_mixed_small_total_program_order_correctness',
'vector_addition_mixed_small_defined_program_order_correctness']
assert set(parsed)==set(names)
public=[]
for name in names:
 assert parsed[name]['kind']=='specification'
 integral='integer_program_order' in name
 total='_total_' in name
 public.append({'file':publicpath,'parser_name':name,'declaration':'VeriTile.Bench.TritonBenchG.VectorAddition.ValueScheduleCorrect.'+name,
 'kind':'specification','statement_sha256':parsed[name]['statement_sha256'],
 'model_kind':'bit_interpreter_contract' if integral else 'backend_value_contract','role':'parametric_value_contract',
 'dtype_scope_relation':'selected_explicit_profiles_only','native_binding_status':'not_established','bit_interpreter_closed_command':False,
 'unclosed_numeric_callbacks':['Explicit us/uu policies on undefined fp32-to-integer output conversions.'] if total else [],
 'premises':'Positive natural block and a complete permutation of original logical program ids.'+(' Active integer-output OutputDefined domain.' if '_defined_' in name else '')})
gaps=[
 {'id':'VA-ORDER-S1','gap':'Native source, compiler and dtype/error correspondence','detail':'Actual existing manually materialized typed source programs are executed. All original Python/compiler admission, unsupported profiles, errors/poison/sanitizers and noncanonical dtype families retain their prior obligations.'},
 {'id':'VA-ORDER-S2','gap':'Native bytes, physical views and finite addresses','detail':'Whole tagged logical cells and Nat element offsets do not establish raw byte decoding, memory validity, mixed-width or shifted aliases, finite-width overflow or native allocation/layout.'},
 {'id':'VA-ORDER-S3','gap':'Native instruction interleaving','detail':'Any complete permutation of whole programs is proved. Operations within each program remain sequential/atomic at this scheduling layer; native load/store interleaving, races and physical scheduling require separate refinement.'},
 {'id':'VA-ORDER-S4','gap':'Remaining host, harness, module and dtype scope','detail':'The same135 existing integer/mixed logical profiles are refined. No new dtype pairs, original error branches, native host allocation/dispatch, harness or module execution are accepted.'}]
counts={'refined_existing_integer_bool_pairs':81,'refined_existing_mixed_fp32_pairs':18,'refined_existing_mixed_fp16_bf16_pairs':36,
 'refined_existing_pairs_total':135,'new_dtype_pairs':0,'cumulative_value_grid_pairs_unchanged':160,
 'prior_floating_permutation_pairs':25,'cumulative_logical_complete_program_permutation_pairs':160,
 'new_public_contracts':5,'native_backend_runs':0,'gpu_runs':0,'whole_original_program_completion_increment':0}
verification={'module_count':3,'actual_exit_codes':[r['exit_code'] for r in results],'named_axiom_checks':16,
 'spec_noncircularity_checks':2,'module_axiom_audits':3,'warnings':0,'dependency_file_count':len(before['files']),
 'imported_project_olean_hashes_rechecked':sum('olean_path' in r for r in before['files'].values()),
 'source_and_dependency_hashes_unchanged':True,'actual_staged_whitespace_check_exit_code':0,'external_library_build_rechecked':False}
scope={'task':'vector_addition integer/mixed complete program permutations','standard':'All original stages/dtypes/branches; only explicitly registered numerical correspondence may be deferred.',
 'whole_original_program_accepted':False,'completed_original_program_increment':0,
 'source':{'file':'bench/tritonbench_g/vector_addition/vector_addition.py','sha256':sha(repo/'bench/tritonbench_g/vector_addition/vector_addition.py')},
 'public_contracts':public,'counts':counts,'structural_remaining':gaps,'numerical_followups':[],
 'inherited_numeric_obligations':'Existing mixed-fp32 and mixed-small numerical/native bindings retain their exact scope. Total us/uu policies remain visible and defined-domain policy independence is unchanged.',
 'proof_composition':'Generic internal schedule lemma requires a real single-program success/memory theorem and same-offset locality. ValueSchedules supplies both using exact existing IntegerSource/MixedFP32/MixedSmall complete_correct and expected_cell theorems. No public specification assumes desired correctness or successful execution.',
 'state_threading':'Each whole program receives actual preceding memory and resets local registers/pid from the same initial context used by existing ascending grids.',
 'alias_boundary':'Equal logical region names are allowed because different program ids own disjoint element offsets. This does not imply physical mixed-width/shifted memory disjointness.',
 'zero_N':'The complete permutation of an empty grid is empty; only logical no-op behavior is proved.',
 'source_views':'Preserve all original source view records; these scheduling proofs do not establish an extraction or compiler bridge.',
 'independent_review':'pending independent source and receipt review'}
write(evidence/'scope-review.json',scope)
(root/'README.md').write_text('''# Complete integer and mixed program permutations

Three modules and five public specifications refine the existing 135 integer/bool
and mixed fp32/fp16/bf16 profiles. Alongside the prior 25 floating profiles, all
160 currently represented value/grid pairs admit arbitrary complete whole-program
permutations. There are zero new dtype pairs and zero accepted original programs.

The generic schedule argument is instantiated with actual source execution and
same-offset locality proofs. Public premises contain no execution-success or
output-correctness assumption. Programs read actual preceding memory. References
use initial memory and preserve the complete outside-output frame. Equal logical
region names are allowed; physical byte aliases and native interleavings remain
unproved. Total undefined conversion policies and active defined-domain results
keep their prior numerical boundaries. This package adds no numerical rules.

All three final compilations passed: 16 axiom checks, 2 noncircular checks, 3 module
audits, no warnings. Exact dependency fingerprints and an isolated staged
whitespace check accompany the receipt. No new native/compiler/GPU run is claimed.
''')
artifacts={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and not p.is_symlink() and
 (p.suffix in ('.lean','.py','.json','.log','.md')) and p.name!='receipt.json'}
receipt={'task':scope['task'],'package_root':str(root),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'whole_original_program_accepted':False,'original_program_completion_increment':0,'modules':modules,'public_contracts':public,
 'verification':verification,'coverage_summary':counts,'acceptance_limits':gaps,'artifacts':dict(sorted(artifacts.items()))}
write(evidence/'receipt.json',receipt)
print(json.dumps({'receipt_sha256':sha(evidence/'receipt.json'),'scope_sha256':sha(evidence/'scope-review.json'),
 'source_sha256':modules,'verification':verification},indent=2))
