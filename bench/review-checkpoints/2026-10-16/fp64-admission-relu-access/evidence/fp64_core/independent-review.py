"""Independent immutable-evidence and dataflow review after reading all3 sources."""
import hashlib, json, re
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP=Path('/tmp/veritile-standardization-20261015')
PKG=TMP/'vector-addition-fp64-core-wip'
DEST=TMP/'vector-addition-fp64-core-independent-review.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def check(p,h):
    assert p.is_file() and not p.is_symlink() and sha(p)==h,str(p)
    return p
assert not DEST.exists()
r=load(check(PKG/'evidence/receipt.json','de82b457d5717243302331868d49aed72430991a03c24362ba570a3d2d132448'))
s=load(check(PKG/'evidence/scope-review.json','6b8e4b5ce3fd501d86db34c3c259ab88dce4e169d2ae0ede219578acdf358e78'))
for p,h in r['artifacts'].items():check(PKG/p,h)
for p,h in r['compiled_olean_sha256'].items():check(PKG/p,h)
before=load(PKG/'evidence/final-dependencies-before.json')
assert before==load(PKG/'evidence/final-dependencies-after.json')
assert before['package_sources']==r['modules']
for p,d in before['project_dependencies'].items():
    check(ROOT/p,d['source_sha256']);check(Path(d['olean_path']),d['olean_sha256'])
check(ROOT/'lean-toolchain',before['toolchain']['lean_toolchain_source_sha256'])
check(Path('/tmp/veritile-lean-toolchain/bin/lean'),before['toolchain']['lean_executable_sha256'])
results=load(PKG/'evidence/verification-results.json')
assert len(results)==3 and all(x['exit_code']==0 for x in results)
logs=[];axioms=audits=noncirc=0
for x in results:
    assert x['source_sha256_before']==x['source_sha256_after']==r['modules'][x['module']]
    logs.append(check(PKG/x['log'],x['log_sha256']).read_text())
    text=(PKG/x['module']).read_text()
    axioms+=len(re.findall(r'^#axiomsClean ',text,re.M))
    audits+=len(re.findall(r'^#auditModuleAxioms$',text,re.M))
    noncirc+=len(re.findall(r'^#specNonCircular ',text,re.M))
    assert not re.search(r'\b(sorry|admit)\b|^axiom\s',text,re.M)
    assert text.endswith('\n') and not re.search(r'[ \t]+$',text,re.M)
log='\n'.join(logs)
assert axioms==22==log.count('axiom footprint ⊆ standard base ✓')
assert noncirc==2==log.count('non-circular ✓') and audits==3
assert not re.search(r'\b(?:warning|error):',log)
w=load(PKG/'evidence/whitespace-check.json');assert w['actual_exit_code']==0 and w['output']==''
for p,h in w['staged_files_sha256'].items():check(PKG/p,h)
n=load(PKG/'evidence/numerical-correspondence.json')
assert len(n['families'])==15 and all(x['status']=='pending_explicitly_deferred' for x in n['families'])
assert len({x['id'] for x in n['families']})==15
core=(PKG/'bench/tritonbench_g/vector_addition/Proofs/FP64TypedCore.lean').read_text()
fields=re.findall(r'^  (\w+) :',core.split('structure Numerics where\n')[1].split('\ninductive Cast')[0],re.M)
assert fields==[x['callback'] for x in n['families']]
assert n['callback_declaration_source_sha256']==sha(PKG/n['callback_declaration'])

# Independently reconstruct postorder conversion routes from raw SSA, preserving
# operand order, load roots, bool-byte interpretation and direct f64 narrowing.
a=load(PKG/'evidence/source-routes/receipt.json');assert a['count']==len(a['profiles'])==29
prior=TMP/'vector-addition-dtype-extension-wip/evidence/extension-audit'
check(prior/'receipt.json',a['prior_audit_sha256'])
check(prior/'frontend-probes.json',a['prior_frontend_profiles_sha256'])
check(ROOT/'bench/tritonbench_g/vector_addition/vector_addition.py',a['source_sha256'])
rawrows=load(prior/'frontend-probes.json')
selected={(x['x'],x['y']):x for x in rawrows if x['status']=='frontend_generated' and 'float64' in (x['x'],x['y'])}
assert len(selected)==29
var=r'%[A-Za-z0-9_]+'
routes=[]
for row in a['profiles']:
    pair=(row['x'],row['y']);orig=selected[pair]
    assert row['ttir_sha256']==orig['ttir_sha256']
    raw=check(PKG/'evidence/source-routes'/row['ttir_path'],row['ttir_sha256']).read_text()
    defs={m[1]:(m[2],m[3],re.findall(var,m[3]),i) for i,line in enumerate(raw.splitlines())
          if (m:=re.match(r'\s*('+var+r') = ([\w.]+) (.*)',line))}
    loads=[v for v,d in defs.items() if d[0]=='tt.load'];assert len(loads)==2
    assert set(loads)==set(row['loads'].values())
    for side in ('x','y'):
        v=row['loads'][side];op,body,args,pos=defs[v]
        assert len(args)==2 and args[1]==row['mask_ssa']
        ptr=args[0]
        if defs[ptr][0]=='tt.bitcast':
            assert row[side]=='bool'
            ptr=defs[ptr][2][0]
        assert defs[ptr][0]=='tt.addptr' and defs[ptr][2][1]==row['offset_ssa']
        base=defs[ptr][2][0]
        assert defs[base][0]=='tt.splat' and defs[base][2]==['%'+side+'_ptr']
    add=row['add_ssa'];assert defs[add][0]=='arith.addf'
    assert defs[add][2]==row['ordered_add_operands'] and 'tensor<1024xf64>' in defs[add][1]
    def walk(v):
        op,body,args,pos=defs[v]
        if op=='tt.load':return [],v
        assert all(defs[k][3]<pos for k in args)
        if op=='arith.cmpi':
            assert body.startswith('ne,') and 'tensor<1024xi8>' in body
            assert defs[args[1]][0]=='arith.constant' and 'dense<0>' in defs[args[1]][1]
            out,root=walk(args[0]);return out+['boolLoad:i8!=0->i1'],root
        assert len(args)==1 and op in {'arith.sitofp','arith.uitofp','arith.extf','arith.fptosi','arith.fptoui','arith.truncf','tt.fp_to_fp'}
        types=re.findall(r'tensor<1024x([^>]+)>',body);assert len(types)==2
        label=op+':'+types[0]+'->'+types[1]
        if op=='tt.fp_to_fp' and types[0]=='f64':
            assert 'rounding = rtne' in body;label+=':rtne'
        out,root=walk(args[0]);return out+[label],root
    left,lroot=walk(defs[add][2][0]);right,rroot=walk(defs[add][2][1])
    assert lroot==row['loads']['x'] and rroot==row['loads']['y']
    assert left==row['input_routes']['x'] and right==row['input_routes']['y']
    v=row['stored_ssa'];out=[]
    while v!=add:
        op,body,args,pos=defs[v];assert len(args)==1 and defs[args[0]][3]<pos
        types=re.findall(r'tensor<1024x([^>]+)>',body);assert types[0]=='f64' and len(types)==2
        assert op in {'arith.fptosi','arith.fptoui','arith.truncf','tt.fp_to_fp'}
        label=op+':'+types[0]+'->'+types[1]
        if op=='tt.fp_to_fp':assert 'rounding = rtne' in body;label+=':rtne'
        out.insert(0,label);v=args[0]
    if row['x']=='bool':out+=['boolStore:preserve-i8-byte']
    assert out==row['output_route']
    stores=[x for x in raw.splitlines() if x.strip().startswith('tt.store ')];assert len(stores)==1
    ptr,value,mask=re.findall(var,stores[0]);assert value==row['stored_ssa'] and mask==row['mask_ssa']
    if defs[ptr][0]=='tt.bitcast':assert row['x']=='bool';ptr=defs[ptr][2][0]
    assert defs[ptr][0]=='tt.addptr' and defs[ptr][2][1]==row['offset_ssa']
    assert defs[defs[ptr][2][0]][2]==['%output_ptr']
    off=row['offset_ssa'];assert defs[off][0]=='arith.addi' and 'tensor<1024xi32>' in defs[off][1]
    bs,lanes=defs[off][2];assert defs[bs][0]=='tt.splat'
    prod=defs[bs][2][0];assert defs[prod][0]=='arith.muli' and ': i32' in defs[prod][1]
    pid,block=defs[prod][2]
    assert defs[pid][0]=='tt.get_program_id' and defs[pid][1].startswith('x : i32')
    assert defs[block][1].startswith('1024 : i32')
    assert defs[lanes][0]=='tt.make_range' and 'end = 1024 : i32, start = 0 : i32' in defs[lanes][1]
    assert defs[mask][0]=='arith.cmpi' and defs[mask][1].startswith('slt,') and defs[mask][2][0]==off
    assert defs[defs[mask][2][1]][2]==['%n_elements']
    routes.append(left+right+['arith.addf:f64']+out)
profiles=(PKG/'bench/tritonbench_g/vector_addition/Proofs/FP64Profiles.lean').read_text()
expected=profiles.split('def expectedRoutes : List (List String) := ')[1].split('\n\ntheorem all_routes_match')[0]
assert json.loads(expected)==routes
assert all(r[k]==0 for k in ['public_specification_count','complete_dtype_grid_profile_increment','original_program_completion_increment'])
assert not r['whole_original_program_accepted']
assert all(x['may_be_deferred_as_numerical'] is False for x in s['structural_gaps'])
review={'schema':1,'status':'accepted_partial_binary64_word_and_typed_route_foundation','reviewer':'root',
'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'receipt_sha256':sha(PKG/'evidence/receipt.json'),
'scope_review_sha256':sha(PKG/'evidence/scope-review.json'),'reviewed_source_sha256':r['modules'],
'proof_defects_found':[],'whole_original_program_accepted':False,'public_specification_increment':0,
'complete_dtype_grid_increment':0,'original_program_completion_increment':0,
'checker_sha256':sha(Path(__file__)),'verification':{'actual_producer_lean_exits':[x['exit_code'] for x in results],
'independent_lean_rerun':False,'axiom_checks':axioms,'noncircular_checks':noncirc,'module_axiom_audits':audits,
'project_sources_and_oleans_checked':len(before['project_dependencies']),'frozen_artifacts_checked':len(r['artifacts']),
'raw_ssa_routes_independently_traced':29,'warnings':0,'numerical_families_explicitly_pending':15},
'semantic_review':[
'All three sources read. BitVec64 fields are bijective and complete; finite decode scales are -1074 for exponent0 and exponent-1075 otherwise. Signed zeros, NaN payload/signaling and infinities preserved.',
'Genuine dependent word types distinguish all storage formats. Expr evaluation is compared with an independent result formula, universally quantified over explicit numerical callbacks. Boolean input normalization and full i8 store bytes are concrete.',
'All29 observed SSA profiles preserve ordered operands, direct integer-to-f64 and f64-output conversions, fp16/bf16 via fp32 on input, direct fp8 routes and fp8 RTNE. Literal route labels agree with Lean expectedRoutes.',
'runLane is a per-lane write request only, with explicit active absent-input and undefined-conversion outcomes. Externally provided masks are not a proved grid/address calculation.',
'No public specification or complete dtype/grid is added. Logical coverage remains160/217, and all57 remaining pairs remain pending. Native byte layout, memory, host, compiler, interleavings and fnuz rejected paths remain structural obligations.'],
'numerical_obligations':n['families'],'remaining_structural_obligations':s['structural_gaps']}
DEST.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'review':str(DEST),'sha256':sha(DEST),'status':review['status'],'ssa_routes':29},indent=2))
