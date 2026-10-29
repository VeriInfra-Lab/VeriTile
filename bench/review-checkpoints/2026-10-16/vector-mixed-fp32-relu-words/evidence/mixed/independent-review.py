"""Independent read-only mixed18 revision/source/evidence/SSA verification."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import sys

base=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
pkg=base/'vector-addition-mixed-fp32-revision1-wip'
old=base/'vector-addition-mixed-fp32-wip'
integer=base/'vector-addition-dtype-extension-wip'
floating=base/'vector-addition-wip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
sys.path.insert(0,str(repo/'bench'))
from adapter_common import declarations
receipt=read(pkg/'evidence/receipt.json')
scope=read(pkg/'evidence/scope-review.json')
assert sha(pkg/'evidence/receipt.json')=='4a08b8707bbd75e17b2f1b19e4e6982827d0f451a629fcce3ed6a97fbf2b80e9'
assert sha(pkg/'evidence/scope-review.json')=='8f7727de446c911745d5837862f61654c80ea1caf0f45cf6945c42c1a40fe7b2'
for rel,h in receipt['artifacts'].items(): assert sha(pkg/rel)==h,rel
provenance=read(pkg/'evidence/revision-provenance.json')
assert sha(old/'evidence/receipt.json')==provenance['original_receipt_sha256']
assert sha(old/'evidence/scope-review.json')==provenance['original_scope_sha256']
oldreceipt=read(old/'evidence/receipt.json')
for rel,h in oldreceipt['artifacts'].items(): assert sha(old/rel)==h,rel
assert (old/'evidence/scope-review.json').read_bytes()==(pkg/'evidence/scope-review.json').read_bytes()
changes=[]
for rel,h in receipt['modules'].items():
    assert sha(pkg/rel)==h
    past=(old/rel).read_text();current=(pkg/rel).read_text()
    assert current=='\n'.join(line.rstrip(' \t') for line in past.splitlines())+'\n'
    assert all(line==line.rstrip(' \t') for line in current.splitlines())
    if current!=past: changes.append((rel,sum(line!=line.rstrip(' \t') for line in past.splitlines())))
assert changes==[('bench/tritonbench_g/vector_addition/Proofs/MixedFP32Grid.lean',5)]
before=read(pkg/'evidence/final-dependencies-before.json')
after=read(pkg/'evidence/final-dependencies-after.json')
assert before==after and len(before['files'])==166
roots={'repository':repo,'package':pkg,'frozen_vector_package':floating,'frozen_integer_package':integer}
for rel,info in before['files'].items(): assert sha(roots[info['source_kind']]/rel)==info['sha256'],rel
for key,path in [('frozen_vector_receipt_sha256',floating/'evidence/receipt.json'),
                 ('frozen_vector_scope_sha256',floating/'evidence/scope-review.json'),
                 ('frozen_integer_receipt_sha256',integer/'evidence/receipt.json'),
                 ('frozen_integer_scope_sha256',integer/'evidence/scope-review.json')]:
    assert sha(path)==before[key]
results=read(pkg/'evidence/verification-results.json')
assert len(results)==6 and set(r['module'] for r in results)==set(receipt['modules'])
logs=[]
for row in results:
    assert row['exit_code']==0
    assert row['source_sha256_before']==row['source_sha256_after']==receipt['modules'][row['module']]
    assert row['log_sha256']==sha(pkg/row['log'])
    log=(pkg/row['log']).read_text();logs.append(log)
    assert not re.search(r'error:|sorryAx|DISALLOWED',log)
assert sum(s.count('axiom footprint') for s in logs)==25
assert sum(s.count('non-circular ✓') for s in logs)==4
publicrel='bench/tritonbench_g/vector_addition/MixedFP32ValueCorrect.lean'
public=declarations(pkg/publicrel)
oldpublic=declarations(old/publicrel)
expected_public={
 'vector_addition_mixed_fp32_total_correctness':'577378df5b430db1d4aa7232e29676e0e650943ccda92c648af2c55b6726ff61',
 'vector_addition_mixed_fp32_defined_correctness':'83d3303a6f2aa1bc7ab335a2e9ac45380dd33af6972561ec696e89d5fbb14272'}
for name,h in expected_public.items():
    assert public[name]['kind']=='specification' and public[name]['statement_sha256']==h
    assert public[name]==oldpublic[name]

source=repo/'bench/tritonbench_g/vector_addition/vector_addition.py'
prior_audit=integer/'evidence/extension-audit'
profile=read(pkg/'evidence/source-profile/receipt.json')
assert sha(pkg/'evidence/source-profile/receipt.json')=='1b4506533f379e3db95eedd11f8a1cb3781d83b28febef7c3a63ace42204f982'
assert sha(source)==profile['source_sha256']==scope['source_sha256']
assert sha(integer/'evidence/receipt.json')==profile['frozen_integer_package_receipt_sha256']
assert sha(prior_audit/'receipt.json')==profile['frozen_frontend_receipt_sha256']
assert sha(prior_audit/'frontend-probes.json')==profile['frozen_frontend_profiles_sha256']
def kernel_ast(path):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name=='add_kernel'),include_attributes=False)
assert kernel_ast(source)==kernel_ast(prior_audit/'extracted_add_kernel.py')
prior_rows={(r['x'],r['y']):r for r in read(prior_audit/'frontend-probes.json')}
ints={'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
expected_pairs={(d,'float32') for d in ints}|{('float32',d) for d in ints}
assert len(profile['profiles'])==18
assert {(r['x'],r['y']) for r in profile['profiles']}==expected_pairs
graphs=[]
for row in profile['profiles']:
    x,y=row['x'],row['y'];first=x!='float32';d=x if first else y
    path=pkg/'evidence/source-profile'/row['ttir_path']
    assert sha(path)==row['ttir_sha256']==prior_rows[x,y]['ttir_sha256']
    assert path.read_bytes()==(prior_audit/prior_rows[x,y]['ttir_path']).read_bytes()
    ir=path.read_text()
    definitions_list=re.findall(r'^\s*(%\w+) = (.*)$',ir,re.M)
    definitions=dict(definitions_list);assert len(definitions)==len(definitions_list)
    stores=re.findall(r'tt.store (%\w+), (%\w+), (%\w+) : tensor<1024x!tt.ptr<([if]\d+)>>',ir)
    assert len(stores)==1
    ptr,value,mask,storage=stores[0]
    width=1 if d=='bool' else int(re.search(r'\d+',d).group())
    destwidth=8 if d=='bool' else width
    assert storage==('i'+str(destwidth) if first else 'f32')
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
            dtype=re.search(r'tt.ptr<([if]\d+)>',rhs).group(1)
            return ('load',side[1],dtype)
        if op=='arith.constant':
            m=re.search(r'dense<(-?\d+)> : tensor<1024x(i\d+)>',rhs);assert m,rhs
            return ('constant',m.group(2),int(m.group(1)))
        if op=='arith.cmpi':
            assert rhs.startswith('arith.cmpi ne,'),rhs
            return ('ne',term(args[0]),term(args[1]))
        if op in ('arith.sitofp','arith.uitofp','arith.fptosi','arith.fptoui'):
            types=re.findall(r'tensor<1024x([if]\d+)>',rhs);assert len(types)==2
            return (op.split('.')[1],types[0],types[1],term(args[0]))
        if op=='arith.addf':
            assert 'tensor<1024xf32>' in rhs
            return ('addf','f32',term(args[0]),term(args[1]))
        raise AssertionError(rhs)
    def operand(side, dtype):
        if dtype=='float32':return ('load',side,'f32')
        node=('load',side,'i8' if dtype=='bool' else 'i'+str(width))
        if dtype=='bool':node=('ne',node,('constant','i8',0))
        return ('sitofp' if dtype.startswith('int') else 'uitofp','i'+str(width),'f32',node)
    expected=('addf','f32',operand('x',x),operand('y',y))
    if first:expected=('fptoui' if d.startswith('uint') else 'fptosi','f32','i'+str(destwidth),expected)
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

oldrows={(r['x'],r['y']):r for r in read(integer/'evidence/scope-review.json')['dtype_pairs']}
newrows={(r['x'],r['y']):r for r in scope['dtype_pairs']}
assert set(oldrows)==set(newrows) and len(newrows)==289
assert {k for k in oldrows if oldrows[k]!=newrows[k]}==expected_pairs
assert scope['counts']['cumulative_logical_value_and_grid_pairs']==124
assert scope['counts']['promoted_pairs_without_logical_value_execution']==93
result={'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,
 'receipt_sha256':sha(pkg/'evidence/receipt.json'),'scope_sha256':sha(pkg/'evidence/scope-review.json'),
 'old_receipt_sha256':sha(old/'evidence/receipt.json'),
 'new_artifact_hashes_verified':len(receipt['artifacts']),'old_artifact_hashes_verified':len(oldreceipt['artifacts']),
 'dependency_hashes_verified':166,'dependency_sources':dict(Counter(v['source_kind'] for v in before['files'].values())),
 'before_after_dependencies_equal':True,'actual_exit_codes':[r['exit_code'] for r in results],
 'named_axiom_checks':25,'noncircularity_checks':4,'warnings':sum(s.count('warning:') for s in logs),
 'warning_details':[line for s in logs for line in s.splitlines() if 'warning:' in line],
 'only_source_change':changes,'all_sources_trailing_whitespace_free':True,'old_new_scope_byte_identical':True,
 'public_specification_statement_sha256':expected_public,'old_new_public_declarations_identical':True,
 'original_kernel_ast_identical':True,'complete_store_ssa_graphs':graphs,
 'source_dtype_pair_rows':289,'only_18_new_pairs_change_coverage':True,
 'actual_lakefile_sha256':sha(repo/'lakefile.toml'),
 'lakefile_limit':'Producer freeze.py probes absent lakefile.lean and omits actual lakefile.toml. This independent audit binds the actual current file; root integration build must bind it explicitly.',
 'no_recompilation_or_native_execution_performed':True,'script_sha256':sha(Path(__file__))}
out=base/'vector-addition-mixed-fp32-independent-audit.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='complete_store_ssa_graphs'},indent=2))
print(json.dumps({'ssa_graph_count':len(graphs),'output':str(out),'sha256':sha(out)}))
