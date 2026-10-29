"""Root review of frozen mixed36 evidence after independent source reading."""
import ast, hashlib, json, re, sys
from pathlib import Path
from datetime import datetime, timezone
base=Path('/tmp/veritile-standardization-20261015')
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
pkg=base/'vector-addition-mixed-half-bfloat-wip'
integer=base/'vector-addition-dtype-extension-wip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
out=base/'vector-addition-mixed-small-independent-review.json'
audit_out=base/'vector-addition-mixed-small-independent-audit.json'
assert not out.exists() and not audit_out.exists()
assert sha(pkg/'evidence/receipt.json')=='30c0d22be4fb44580f64ce9d5d5274e731d55852cc1f426f5c63baa39a21871d'
assert sha(pkg/'evidence/scope-review.json')=='f8c4318a37789b1d9880ff49b91e4a6598d2283c8507b1e8ed87e6c57a273eec'
receipt=read(pkg/'evidence/receipt.json');scope=read(pkg/'evidence/scope-review.json')
for p,h in receipt['artifacts'].items():
    assert (pkg/p).is_file() and not (pkg/p).is_symlink() and sha(pkg/p)==h,p
before=read(pkg/'evidence/final-dependencies-before.json')
assert before==read(pkg/'evidence/final-dependencies-after.json') and len(before['files'])==172
inputs={};oleans=0
allowed=[repo,pkg,integer,base/'vector-addition-mixed-fp32-revision1-wip',base/'vector-addition-wip']
for p,r in before['files'].items():
    source=Path(r['source_path']);assert any(source==root/p for root in allowed),p
    assert sha(source)==r['sha256'];inputs[p]=r['sha256']
    if p not in receipt['modules']:assert sha(repo/p)==r['sha256']
    if 'olean_path' in r:
        assert sha(Path(r['olean_path']))==r['olean_sha256'];oleans+=1
for k in ('prior_receipts','prior_scopes'):
    for p,h in before[k].items(): assert sha(Path(p))==h
results=read(pkg/'evidence/verification-results.json');assert len(results)==6
logs=[]
for r in results:
    assert r['exit_code']==0 and r['source_sha256_before']==r['source_sha256_after']==receipt['modules'][r['module']]
    assert sha(pkg/r['log'])==r['log_sha256'];logs.append((pkg/r['log']).read_text())
s='\n'.join(logs)
assert s.count('axiom footprint ⊆ standard base ✓')==32 and s.count('non-circular ✓')==4
assert not re.search(r'error:|warning:|sorryAx|DISALLOWED',s)
for p in receipt['modules']:
    s=(pkg/p).read_text();assert s.endswith('\n') and not re.search(r'[ \t]+$',s,re.M)
    assert '#auditModuleAxioms' in s and not re.search(r'\b(?:sorry|admit|axiom)\b',s)
public=declarations(pkg/'bench/tritonbench_g/vector_addition/MixedSmallValueCorrect.lean')
expected_public={'vector_addition_mixed_small_total_correctness':'e02908380f5b18c3d895976ee833f255a4630686d231ff46ae1620f7c454c4dc',
'vector_addition_mixed_small_defined_correctness':'928ea329e426e0e4589e671fea1207f97897fc4bc060c499b95db4d2c33eb930'}
for n,h in expected_public.items(): assert public[n]['kind']=='specification' and public[n]['statement_sha256']==h
failed=pkg/'evidence/failed-attempts/value-stack-overflow'
failure=read(failed/'verification.json')
assert failure['exit_code']==-6 and sha(failed/'compile.log')==failure['log_sha256']
assert sha(failed/'MixedSmallValue.lean.txt')==failure['source_sha256_before']==failure['source_sha256_after']
assert 'Stack overflow' in (failed/'compile.log').read_text() or 'Stack overflow' in (failed/'compile.log').read_text().capitalize()
import difflib
assert ''.join(difflib.unified_diff((failed/'MixedSmallValue.lean.txt').read_text().splitlines(True),
 (pkg/'bench/tritonbench_g/vector_addition/Proofs/MixedSmallValue.lean').read_text().splitlines(True),
 fromfile='actual-failed-MixedSmallValue',tofile='final-MixedSmallValue'))==(failed/'structural-proof-fix.diff').read_text()
source=repo/'bench/tritonbench_g/vector_addition/vector_addition.py'
prior_audit=integer/'evidence/extension-audit'
profile=read(pkg/'evidence/source-profile/receipt.json')
assert sha(pkg/'evidence/source-profile/receipt.json')=='0c9d44fb01be270f42b93c23dcfd5f824d239a7d484ab4d1d0be93c0587bb3d1'
assert sha(source)==profile['source_sha256']==scope['source_sha256']
assert sha(prior_audit/'receipt.json')==profile['prior_audit_sha256']
assert sha(prior_audit/'frontend-probes.json')==profile['prior_profiles_sha256']
def kernel_ast(path):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name=='add_kernel'),include_attributes=False)
assert kernel_ast(source)==kernel_ast(prior_audit/'extracted_add_kernel.py')
prior_rows={(r['x'],r['y']):r for r in read(prior_audit/'frontend-probes.json')}
ints={'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
floats={'float16','bfloat16'}
expected_pairs={(d,f) for d in ints for f in floats}|{(f,d) for d in ints for f in floats}
assert len(profile['profiles'])==36
assert {(r['x'],r['y']) for r in profile['profiles']}==expected_pairs
graphs=[]
for row in profile['profiles']:
    x,y=row['x'],row['y'];first=x in ints;d=x if first else y
    floating=y if first else x;storage_dtype='f16' if floating=='float16' else 'bf16';compute='f16' if floating=='float16' else 'f32'
    path=pkg/'evidence/source-profile'/row['ttir_path']
    assert sha(path)==row['ttir_sha256']==prior_rows[x,y]['ttir_sha256']
    assert path.read_bytes()==(prior_audit/prior_rows[x,y]['ttir_path']).read_bytes()
    ir=path.read_text()
    definitions_list=re.findall(r'^\s*(%\w+) = (.*)$',ir,re.M)
    definitions=dict(definitions_list);assert len(definitions)==len(definitions_list)
    stores=re.findall(r'tt.store (%\w+), (%\w+), (%\w+) : tensor<1024x!tt.ptr<(i\d+|f\d+|bf16)>>',ir)
    assert len(stores)==1
    ptr,value,mask,storage=stores[0]
    width=1 if d=='bool' else int(re.search(r'\d+',d).group())
    destwidth=8 if d=='bool' else width
    assert storage==('i'+str(destwidth) if first else storage_dtype)
    offsets=set();masks={mask};load_positions=[]
    def pointer(v):
        if v in ('%x_ptr','%y_ptr','%output_ptr'): return v
        rhs=definitions[v];op=rhs.split()[0];args=re.findall(r'%\w+',rhs)
        assert op in ('tt.splat','tt.addptr','tt.bitcast'),rhs
        if op=='tt.addptr': offsets.add(args[1])
        return pointer(args[0])
    assert pointer(ptr)=='%output_ptr'
    def term(v):
        rhs=definitions[v];op=rhs.split()[0];args=re.findall(r'%\w+',rhs)
        if op=='tt.load':
            assert len(args)==2,'masked load must omit other'
            side=pointer(args[0]);assert side in ('%x_ptr','%y_ptr')
            masks.add(args[1]);load_positions.append(ir.index(v+' = '+rhs))
            dtype=re.search(r'tt.ptr<(i\d+|f\d+|bf16)>',rhs).group(1)
            return ('load',side[1],dtype)
        if op=='arith.constant':
            m=re.search(r'dense<(-?\d+)> : tensor<1024x(i\d+)>',rhs);assert m,rhs
            return ('constant',m.group(2),int(m.group(1)))
        if op=='arith.cmpi':
            assert rhs.startswith('arith.cmpi ne,'),rhs
            return ('ne',term(args[0]),term(args[1]))
        if op in ('arith.sitofp','arith.uitofp','arith.fptosi','arith.fptoui','arith.extf','arith.truncf'):
            types=re.findall(r'tensor<1024x(i\d+|f\d+|bf16)>',rhs);assert len(types)==2
            return (op.split('.')[1],types[0],types[1],term(args[0]))
        if op=='arith.addf':
            assert f'tensor<1024x{compute}>' in rhs
            return ('addf',compute,term(args[0]),term(args[1]))
        raise AssertionError(rhs)
    def operand(side, dtype):
        if dtype=='float16': return ('load',side,'f16')
        if dtype=='bfloat16': return ('extf','bf16','f32',('load',side,'bf16'))
        node=('load',side,'i8' if dtype=='bool' else 'i'+str(width))
        if dtype=='bool':node=('ne',node,('constant','i8',0))
        return ('sitofp' if dtype.startswith('int') else 'uitofp','i'+str(width),compute,node)
    expected=('addf',compute,operand('x',x),operand('y',y))
    if first:
        if floating=='float16': expected=('extf','f16','f32',expected)
        expected=('fptoui' if d.startswith('uint') else 'fptosi','f32','i'+str(destwidth),expected)
    elif floating=='bfloat16': expected=('truncf','f32','bf16',expected)
    actual=term(value);assert actual==expected,(x,y,actual,expected)
    assert len(offsets)==len(masks)==1
    assert len(load_positions)==2 and max(load_positions)<ir.index('tt.store ')
    offset=next(iter(offsets));assert next(iter(masks))==mask
    def address_term(v):
        if v=='%n_elements':return ('argument','i32','n_elements')
        rhs=definitions[v];op=rhs.split()[0];args=re.findall(r'%\w+',rhs)
        if op=='tt.get_program_id':
            assert rhs.startswith('tt.get_program_id x : i32 ')
            return ('pid','x','i32')
        if op=='arith.constant':
            m=re.match(r'arith.constant (-?\d+) : (i\d+) ',rhs);assert m,rhs
            return ('constant',m.group(2),int(m.group(1)))
        if op=='tt.make_range':
            assert '{end = 1024 : i32, start = 0 : i32}' in rhs
            return ('arange',0,1024,'i32')
        if op=='tt.splat':return ('splat',address_term(args[0]))
        if op in ('arith.muli','arith.addi'):
            assert (' : i32 ' in rhs) or ('tensor<1024xi32>' in rhs),rhs
            return (op.split('.')[1],'i32',address_term(args[0]),address_term(args[1]))
        if op=='arith.cmpi':
            assert rhs.startswith('arith.cmpi slt,') and 'tensor<1024xi32>' in rhs,rhs
            return ('slt','i32',address_term(args[0]),address_term(args[1]))
        raise AssertionError(rhs)
    expected_offset=('addi','i32',('splat',('muli','i32',('pid','x','i32'),('constant','i32',1024))),('arange',0,1024,'i32'))
    expected_mask=('slt','i32',expected_offset,('splat',('argument','i32','n_elements')))
    assert address_term(offset)==expected_offset
    assert address_term(mask)==expected_mask
    graphs.append({'x':x,'y':y,'store_dataflow':actual,'offset_dataflow':expected_offset,
                   'mask_dataflow':expected_mask,'two_masked_loads_precede_store':True,
                   'same_offset_and_mask':True,'masked_other_absent':True,'ttir_sha256':sha(path)})


prior_scope=read(base/'vector-addition-mixed-fp32-revision1-wip/evidence/scope-review.json')
oldrows={(r['x'],r['y']):r for r in prior_scope['dtype_pairs']}
newrows={(r['x'],r['y']):r for r in scope['dtype_pairs']}
assert len(newrows)==289 and set(oldrows)==set(newrows)
assert {k for k in oldrows if oldrows[k]!=newrows[k]}==expected_pairs
assert scope['counts']['cumulative_logical_value_and_grid_pairs']==160 and scope['counts']['promoted_pairs_without_logical_value_execution']==57
assert not scope['whole_original_program_accepted'] and len(scope['numerical_assumptions'])==4
record={'schema':1,'read_only':True,'source_ast_unchanged':True,'complete_store_ssa_graphs':graphs,
 'only_36_dtype_pair_rows_change_coverage':True,'receipt_sha256':sha(pkg/'evidence/receipt.json'),
 'script_sha256':sha(Path(__file__)),'boundary':'Independent observed SSA dataflow audit, not native compiler refinement or new native execution.'}
audit_out.write_text(json.dumps(record,indent=2)+'\n')
review={'schema':1,'reviewer':'root','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'accepted_partial_concrete_mixed_small_profile_extensions','proof_defects_found':[],
 'whole_original_program_accepted':False,'original_program_completion_increment':0,
 'receipt_sha256':sha(pkg/'evidence/receipt.json'),'scope_review_sha256':sha(pkg/'evidence/scope-review.json'),
 'reviewed_source_sha256':receipt['modules'],'reviewed_input_sha256':inputs,
 'script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),
 'audit_path':str(audit_out),'audit_sha256':sha(audit_out),
 'verification':{'actual_exit_codes':[r['exit_code'] for r in results],'axiom_checks':32,'noncircular_checks':4,
 'module_axiom_audits':6,'warnings':0,'dependency_fingerprints':172,'imported_oleans_checked':oleans,
 'actual_failed_attempt_exit':-6,'failed_source_and_log_preserved':True,'independent_lean_rerun':False},
 'public_statement_sha256':expected_public,
 'semantic_review':[
  'Read all six final sources plus shared IntegerConversion and Binary16/BinaryConversion primitive definitions. Own-width integer normalization precedes direct Binary16.round for half; no preliminary fp32 rounding occurs. The bf16 mixed route rounds integers directly to fp32.',
  'The carrier low16 bits encode half/bfloat values, not native fp32 numerical inputs. Half add is Binary16.add; bf16 extends by field embedding and uses FP.add. Half sums widen before integer conversion; bf16 floating outputs use FP.cast. High16 input carrier bits are ignored by half/bfloat decoding.',
  'All source loads, ordered operands and destination conversions match36 frozen SSA routes; output type is original x type. Bool load is nonzero; signed8 bool stores retain3 and undefined conversions keep explicit policies.',
  'Execution constructs actual logical success. Complete ascending programs thread prior memory and equal an independent initial-memory reference with full frame, even for equal logical region names. Physical shifted/mixed-width aliases remain open.',
  'Defined theorem constrains only active integer-output addresses and makes their stored values policy-independent. Floating output orientation is unconditionally in this domain; masked addresses are unconstrained. Undefined native poison/trap/sanitizer execution is not proved by totalization.',
  'Value proof local irreducible attributes prevent elaborator expansion of scalar algorithms; they do not add axioms or change arithmetic definitions/theorem conclusions. Failed stack-overflow evidence and actual successful final compiles remain separate.',
  'Both public statements conservatively remain backend_value_contract. Neither native memory/compiler/host correspondence nor whole-original-program acceptance is established.'
 ],'coverage':scope['counts'],'remaining_structural_obligations':scope['structural_remaining'],
 'numerical_boundary':'Four narrow scalar-native followups need exact unified-queue scope binding. Native bytes/tags/addresses/errors and scheduling are structural, never numerical deferrals.'}
out.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'review':str(out),'sha256':sha(out),'audit_sha256':sha(audit_out),'ssa_graphs':len(graphs),'dependencies':len(inputs),'imported_oleans':oleans},indent=2))
