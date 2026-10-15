"""Freeze only actual successful results and the exact partial scope."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

WORK=Path(__file__).resolve().parent
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import checked, declarations, hash_file, load, write_json

assert not (WORK/'evidence/receipt.json').exists()
before,after=[load(WORK/('evidence/final-dependencies-'+x+'.json')) for x in ('before','after')]
assert before==after and len(before['files'])==162
modules={p:r['sha256'] for p,r in before['files'].items() if r['source_kind']=='package'}
assert len(modules)==3
for p,r in before['files'].items():
    checked(WORK if r['source_kind']=='package' else ROOT,p,r['sha256'])
results=load(WORK/'evidence/verification-results.json')
assert len(results)==3
logs=[]
for r in results:
    assert r['exit_code']==0
    assert r['source_sha256_before']==r['source_sha256_after']==modules[r['module']]
    text=checked(WORK,r['log'],r['log_sha256']).read_text();logs.append(text)
    assert not re.search(r'\b(error|warning|sorryAx|DISALLOWED)\b',text)
assert sum(x.count('axiom footprint') for x in logs)==17
assert sum(x.count('non-circular ✓') for x in logs)==2
for p in modules:
    s=(WORK/p).read_text()
    assert s.endswith('\n') and not re.search(r'[ \t]+$',s,re.M)
    assert s.count('#auditModuleAxioms')==1
with tempfile.TemporaryDirectory(prefix='veritile-integer-words-whitespace-') as tmp:
    path=Path(tmp)
    subprocess.run(['git','init','-q',str(path)],check=True)
    for p in modules:
        target=path/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(WORK/p,target)
    subprocess.run(['git','-C',str(path),'add','--',*modules],check=True)
    check=subprocess.run(['git','-C',str(path),'diff','--cached','--check'],capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    for p,h in modules.items():
        data=subprocess.check_output(['git','-C',str(path),'show',':'+p])
        assert hashlib.sha256(data).hexdigest()==h
    write_json(WORK/'evidence/whitespace-check.json',{'actual_diff_cached_check_exit_code':check.returncode,
        'source_sha256':modules,'staged_bytes_equal':True,'stdout':check.stdout,'stderr':check.stderr,
        'boundary':'Isolated temporary index only; root repository has not staged this package.'})
old=ROOT/'bench/review-checkpoints/2026-10-16/vector-integers-relu-program-order'
prior=load(old/'evidence/integer/evidence/receipt.json')
assert hash_file(old/'evidence/integer/evidence/receipt.json')=='85dc08106852c4a3f69516046182954693eb5afd0158064db358c708a70b9100'
for p,h in prior['modules'].items():
    checked(ROOT,p,h)
audit=old/'evidence/integer/vector-addition-integer-independent-audit.json'
assert audit.is_file()
audited=load(audit)
assert len(audited['independently_traced_integer_store_graphs'])==81
source='bench/tritonbench_g/vector_addition/vector_addition.py'
assert hash_file(ROOT/source)=='cfcb749bd562924106c1e0e95e18a9bd987238be3a3b522abac89e099de3d098'
public='bench/tritonbench_g/vector_addition/IntegerWordGridCorrect.lean'
name='vector_addition_integer_word_grid_correctness'
decl=declarations(WORK/public)[name];assert decl['kind']=='specification'
claim={'file':public,'parser_name':name,'declaration':'VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWordGridCorrect.'+name,
    'kind':'specification','statement_sha256':decl['statement_sha256'],
    'model_kind':'bit_interpreter_contract','role':'parametric_value_contract',
    'dtype_scope_relation':'selected_explicit_profiles_only','unclosed_numeric_callbacks':[],
    'bit_interpreter_closed_command':False,'native_binding_status':'not_established',
    'premises':'Positive natural block and InputWords for both complete tagged logical input cells at every active offset.',
    'alias_boundary':'No distinct-region premise. If x=y the two input word decodings must agree; withInputs_y makes this compatibility explicit.',
    'unused_algebra':'M occurs in execution but no floating operation is used; resultWord and wordSpec have no Algebra parameter.'}
structural=[
    {'id':'VA-W-S1','gap':'Native memory to finite word input relation',
     'detail':'BitVec source widths and complete logical input tags are formal. Actual byte assembly/endianness, native load validity and source dtype dispatch satisfying InputWords remain unproved.'},
    {'id':'VA-W-S2','gap':'Native physical storage and aliasing',
     'detail':'Input compatibility handles identical logical names. Shifted/misaligned/mixed-width physical byte views, allocation bounds/layout/freshness and race behavior remain outside the logical-cell model.'},
    {'id':'VA-W-S3','gap':'Finite addresses, native errors and scheduling',
     'detail':'Positive natural block and ascending grid do not prove native block domain, zero-grid behavior, signed offset/pointer overflow, sanitizer/compiler admission/errors or native interleavings.'},
    {'id':'VA-W-S4','gap':'Remaining original stages and dtypes',
     'detail':'This refines the same81 integer/bool profiles and adds no dtype pairs. All other dtype/error families, original host/harness/module behavior and their native correspondence remain required.'}]
scope={'task':'vector_addition finite integer/bool input and output words',
    'standard':'All original stages/dtypes/branches; only explicit numerical correspondence may be deferred.',
    'whole_original_program_accepted':False,'completed_original_program_increment':0,
    'source':{'file':source,'sha256':hash_file(ROOT/source)},'public_contract':claim,
    'counts':{'refined_existing_integer_bool_pairs':81,'new_dtype_pairs':0,'new_logical_profile_names':0,
        'raw_bool_input_bytes':256,'original_ttir_store_graphs_reused':81,'new_frontend_runs':0,'native_backend_runs':0,'gpu_runs':0},
    'new_proved':[
        'Finite source-width words decode into complete tagged logical cells, preventing reliance on arbitrary unbounded or mismatched-tag defaults.',
        'Bool input normalizes all256 raw signed i8 patterns with !=0; both nonzero bool inputs wrap at compute i1 before output i8.',
        'Every input/output word has an exact storage roundtrip, including noncanonical bool output3. Store bytes are not normalized.',
        'Concrete promotion/add/store resultWord is independent of memory, execution and Algebra; both InputWords relations refine the prior memory formula.',
        'Actual full ascending integer grid success composes with independent initial-word memory, complete frame and exact output-word readback.',
        'A constructive initial memory satisfies both relations under precisely the decoded-cell agreement needed for equal logical names.'],
    'suggested_covered_stage':'canonical_integer_bool_finite_input_output_word_grid_refinement',
    'structural_remaining':structural,'numerical_followups':[],
    'prior_source_views':'Preserve the original add_kernel shared-DSL view; no source/compiler refinement is established by this logical word relation.',
    'inherited_route_evidence':{'path':audit.relative_to(ROOT).as_posix(),'sha256':hash_file(audit),
        'original_integer_receipt_sha256':hash_file(old/'evidence/integer/evidence/receipt.json')},
    'masked_lanes':'No input representation premise outside N; unspecified masked loads retain the existing source behavior.',
    'independent_review':'pending independent source/evidence review'}
write_json(WORK/'evidence/scope-review.json',scope)
artifact_paths=[WORK/p for p in modules]
artifact_paths += [WORK/p for p in ('verify.py','freeze.py','finalize.py','README.md')]
artifact_paths += list((WORK/'evidence').glob('*.json'))+list((WORK/'evidence').glob('*.log'))
assert all(p.is_file() and not p.is_symlink() for p in artifact_paths)
receipt={'task':scope['task'],'package_root':str(WORK),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
    'whole_original_program_accepted':False,'original_program_completion_increment':0,
    'modules':modules,'public_contract':claim,'coverage_summary':scope['counts'],
    'verification':{'module_count':3,'actual_exit_codes':[r['exit_code'] for r in results],
        'named_axiom_checks':17,'spec_noncircularity_checks':2,'module_axiom_audits':3,'warnings':0,
        'dependency_file_count':162,'source_and_dependency_hashes_unchanged':True,'actual_lakefile_toml_bound':True,
        'external_library_build_rechecked':False,'actual_staged_whitespace_check_exit_code':check.returncode},
    'prior_modules_unchanged':prior['modules'],'inherited_route_evidence':scope['inherited_route_evidence'],
    'acceptance_limits':structural,'artifacts':{p.relative_to(WORK).as_posix():hash_file(p) for p in sorted(set(artifact_paths))}}
write_json(WORK/'evidence/receipt.json',receipt)
print(json.dumps({'receipt_sha256':hash_file(WORK/'evidence/receipt.json'),
    'scope_sha256':hash_file(WORK/'evidence/scope-review.json'),'statement_sha256':decl['statement_sha256'],
    'modules':3,'dependencies':162,'axioms':17,'noncircular':2,'warnings':0},indent=2))
