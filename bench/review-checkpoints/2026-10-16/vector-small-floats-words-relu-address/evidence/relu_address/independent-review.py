"""Independent root binding checks after reading all three frozen address sources."""
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP=Path('/tmp/veritile-standardization-20261015')
PKG=TMP/'relu-finite-address-wip'
DEST=TMP/'relu-finite-address-independent-review.json'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import declarations

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def check(p,h):
    assert p.is_file() and not p.is_symlink() and sha(p)==h, str(p)
    return p

assert not DEST.exists()
receipt=load(check(PKG/'evidence/receipt.json','fc98bfa9d776e2b5592506905eda9241efb16a4809c7ad112f500afc83d815f3'))
scope=load(check(PKG/'evidence/scope-review.json','6eaaf3bfa779c674ad6a6a3436354b9684a8d67e55c4e1d7d58ce892491c6e6e'))
for p,h in receipt['artifacts'].items(): check(PKG/p,h)
before=load(PKG/'evidence/final-dependencies-before.json')
assert before==load(PKG/'evidence/final-dependencies-after.json')
roots={'repository':ROOT,'package':PKG,'frozen_original_package':TMP/'relu-original-wip',
       'frozen_words_package':TMP/'relu-integer-word-refinement-wip'}
inputs={}
for p,r in before['files'].items():
    check(roots[r['source_kind']]/p,r['sha256']);inputs[p]=r['sha256']
    if r['source_kind'].startswith('frozen_'): check(ROOT/p,r['sha256'])
assert len(inputs)==176
for p,h in before['audited_ttir_sha256'].items(): check(Path(p),h)
for p in ('lakefile.toml','lake-manifest.json','lean-toolchain'):
    assert inputs[p]==sha(ROOT/p)
results=load(PKG/'evidence/verification-results.json')
assert len(results)==3 and [r['exit_code'] for r in results]==[0,0,0]
logs=[]
for r in results:
    assert r['source_sha256_before']==r['source_sha256_after']==receipt['modules'][r['module']]
    logs.append(check(PKG/r['log'],r['log_sha256']).read_text())
text='\n'.join(logs)
assert text.count('axiom footprint ⊆ standard base ✓')==21
assert text.count('non-circular ✓')==3
assert not re.search(r'\b(?:warning|error):',text)
for p in receipt['modules']:
    s=(PKG/p).read_text()
    assert s.endswith('\n') and not re.search(r'[ \t]+$',s,re.M)
    assert '#auditModuleAxioms' in s and not re.search(r'\b(?:sorry|admit|axiom)\b',s)
public=receipt['public_contract']
assert declarations(PKG/public['file'])[public['parser_name']]['statement_sha256']==public['statement_sha256']
assert public['statement_sha256']=='4c4b5b99f314eef7d15e1b3bd039b066a1574d4b24a7bc460cf330864a3f44fd'
assert not receipt['whole_original_program_accepted'] and receipt['original_program_completion_increment']==0
assert scope['counts']['new_profile_names']==0 and len(scope['structural_remaining'])==4

# Independently follow the address operands in all thirteen immutable TTIRs.
source=ROOT/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'
assert sha(source)==scope['source_sha256']
tree=ast.parse(source.read_text())
wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relu')
assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BLOCK_SIZE' for t in n.targets)
           and isinstance(n.value,ast.Constant) and n.value.value==1024 for n in wrapper.body)
route=load(PKG/'evidence/address-route-review.json')
assert len(route['profiles'])==13
for r in route['profiles']:
    t=check(Path(r['path']),r['sha256']).read_text()
    defs={m[1]:(m[2],m[3],re.findall(r'%[A-Za-z0-9_]+',m[3]))
          for line in t.splitlines() if (m:=re.match(r'\s*(%[A-Za-z0-9_]+) = ([A-Za-z0-9_.]+) (.*)',line))}
    pid,prod,lane,off,mask,gate=[r[k] for k in ('pid','product','lane','offset','mask','store_gate')]
    assert defs[pid][0]=='tt.get_program_id' and defs[pid][1].startswith('x : i32')
    assert defs[prod][0]=='arith.muli' and defs[prod][2][0]==pid
    assert defs[defs[prod][2][1]][1].startswith('1024 : i32')
    assert defs[lane][0]=='tt.make_range' and 'end = 1024' in defs[lane][1] and 'start = 0' in defs[lane][1]
    assert defs[off][0]=='arith.addi' and defs[off][2][1]==lane
    assert defs[defs[off][2][0]][0]=='tt.splat' and defs[defs[off][2][0]][2]==[prod]
    assert defs[mask][0]=='arith.cmpi' and defs[mask][1].startswith('slt,') and defs[mask][2][0]==off
    assert 'dense<1025>' in defs[defs[mask][2][1]][1]
    assert defs[gate][0]=='arith.cmpi' and defs[gate][1].startswith('eq,') and defs[gate][2][0]==pid
    assert defs[defs[gate][2][1]][1].startswith('0 : i32')
    inp,out=r['input_pointer_before_possible_bool_bitcast'],r['output_pointer']
    for ptr,base in ((inp,'%x_ptr'),(out,'%out_ptr')):
        assert defs[ptr][0]=='tt.addptr' and defs[ptr][2][1]==off
        assert defs[defs[ptr][2][0]][0]=='tt.splat' and defs[defs[ptr][2][0]][2]==[base]
    stores=[line.strip() for line in t.splitlines() if line.strip().startswith('tt.store ')]
    assert len(stores)==1 and re.findall(r'%[A-Za-z0-9_]+',stores[0])[::2]==[out,mask]
    assert t.index('scf.if '+gate+' {')<t.index(stores[0])<t.index('} else {')

review={'schema':1,'reviewer':'root','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'accepted_partial_finite_address_arithmetic_extension','proof_defects_found':[],
 'whole_original_program_accepted':False,'original_program_completion_increment':0,
 'receipt_sha256':sha(PKG/'evidence/receipt.json'),'scope_review_sha256':sha(PKG/'evidence/scope-review.json'),
 'reviewed_source_sha256':receipt['modules'],'reviewed_input_sha256':inputs,
 'script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),
 'verification':{'actual_compile_exit_codes':[r['exit_code'] for r in results],
 'axiom_checks':21,'noncircular_checks':3,'module_axiom_audits':3,'warnings':0,
 'dependency_fingerprints_checked':len(inputs),'artifacts_checked':len(receipt['artifacts']),
 'independently_traced_ttir_address_graphs':13,'independent_lean_rerun':False},
 'semantic_review':[
  'Read all three sources and the frozen original Python/TypedValue source. Fixed1024 ceiling grid fits at most2^31 for N<2^31; every lane, including masked tail, has signed-positive multiply/add and exact signed<N and pid==0.',
  'source_prefix_registers actually runs source.surfaceBody.take4 and relates logical offsets/mask registers to BitVec expressions while preserving arbitrary memory. This is a separate proved lemma, not a native compiler claim.',
  'Active full-element footprints, including last bytes, fit explicit64bit mathematical base/end domains. Sign-extension, scale/add, byte coordinate injectivity and separated ranges are proved without claiming allocation or byte storage.',
  'Public theorem constructs prior word-grid success and exact memory/frame from InputWords, conjoined with independent address arithmetic. N0 has an empty logical grid; N>=2^31 and non1024 are outside this subtheorem.',
  'All nine word profiles retain arbitrary conversion/zero callbacks and original pid0-only stores. No new dtype, successful native execution, physical memory correspondence or complete original-program acceptance follows.'
 ],'remaining_structural_obligations':scope['structural_remaining'],
 'numerical_boundary':'No new numerical callbacks. The two prior descriptive ReLU word followups remain pending exact unified-queue scope binding.'}
DEST.write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'review':str(DEST),'sha256':sha(DEST),'dependencies':len(inputs),'ttir_graphs':13,'status':review['status']},indent=2))
