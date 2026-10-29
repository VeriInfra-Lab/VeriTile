"""Read-only exact SSA address review of the thirteen frozen ReLU TTIRs."""
import ast
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
prior = root.parent / 'relu-original-wip'
audit = prior / 'evidence/dtype-audit'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
assert sha(prior/'evidence/receipt.json') == 'ed4a852afafce5efd027f62565e32afe42146eb9c0b140d872990a5da211fd90'
source = repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'
receipt = read(audit/'receipt.json')
assert sha(source) == receipt['source_sha256']
assert receipt['constexprs'] == {'N':1025,'block_size':1024}
for p,h in receipt['artifacts'].items(): assert sha(audit/p)==h
tree = ast.parse(source.read_text())
wrapper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relu')
assignment = next(n for n in wrapper.body if isinstance(n,ast.Assign) and
    any(isinstance(t,ast.Name) and t.id=='BLOCK_SIZE' for t in n.targets))
assert isinstance(assignment.value,ast.Constant) and assignment.value.value==1024
kernel = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relu_kernel')
assert any(a.arg=='N' and isinstance(a.annotation,ast.Attribute) and a.annotation.attr=='constexpr'
           for a in kernel.args.args)
rows = read(audit/'frontend-profiles.json')
checked = []
var = r'%[A-Za-z0-9_]+'
for row in rows:
    if row['status']!='frontend_generated': continue
    path=audit/row['ttir']; assert sha(path)==row['ttir_sha256']
    text=path.read_text(); lines=text.splitlines(); defs={}
    for line in lines:
        m=re.match(r'\s*('+var+r') = ([A-Za-z0-9_.]+) (.*)',line)
        if m: defs[m[1]]={'op':m[2],'text':m[3],'vars':re.findall(var,m[3]),'line':line.strip()}
    def unique(pred):
        found=[name for name,d in defs.items() if pred(name,d)]
        assert len(found)==1,(row['input'],found)
        return found[0]
    pid=unique(lambda _,d:d['op']=='tt.get_program_id' and d['text'].startswith('x : i32'))
    product=unique(lambda _,d:d['op']=='arith.muli' and d['text'].endswith(': i32 loc(#loc18)'))
    assert defs[product]['vars'][0]==pid
    factor=defs[product]['vars'][1]
    assert defs[factor]['op']=='arith.constant' and defs[factor]['text'].startswith('1024 : i32')
    splat=unique(lambda _,d:d['op']=='tt.splat' and d['vars']==[product])
    lane=unique(lambda _,d:d['op']=='tt.make_range')
    assert '{end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>' in defs[lane]['text']
    offset=unique(lambda _,d:d['op']=='arith.addi' and d['vars']==[splat,lane])
    assert ': tensor<1024xi32>' in defs[offset]['text']
    mask=unique(lambda _,d:d['op']=='arith.cmpi' and d['text'].startswith('slt,') and d['vars'][0]==offset)
    constant=defs[mask]['vars'][1]
    assert defs[constant]['op']=='arith.constant' and 'dense<1025> : tensor<1024xi32>' in defs[constant]['text']
    load=unique(lambda _,d:d['op']=='tt.load')
    assert len(defs[load]['vars'])==2 and defs[load]['vars'][1]==mask
    inp=defs[load]['vars'][0]
    raw_input_pointer=inp
    if defs[inp]['op']=='tt.bitcast':
        assert row['input']=='bool' and 'ptr<i1>' in defs[inp]['text'] and 'ptr<i8>' in defs[inp]['text']
        inp=defs[inp]['vars'][0]
    assert defs[inp]['op']=='tt.addptr' and defs[inp]['vars'][1]==offset
    base=defs[inp]['vars'][0]
    assert defs[base]['op']=='tt.splat' and defs[base]['vars']==['%x_ptr']
    stores=[line.strip() for line in lines if line.strip().startswith('tt.store ')]
    assert len(stores)==1
    out,_,stored_mask=re.findall(var,stores[0]); assert stored_mask==mask
    assert defs[out]['op']=='tt.addptr' and defs[out]['vars'][1]==offset
    out_base=defs[out]['vars'][0]
    assert defs[out_base]['op']=='tt.splat' and defs[out_base]['vars']==['%out_ptr']
    assert 'ptr<f32>' in stores[0]
    gate=unique(lambda _,d:d['op']=='arith.cmpi' and d['text'].startswith('eq,') and d['vars'][0]==pid)
    zero=defs[gate]['vars'][1]
    assert defs[zero]['op']=='arith.constant' and defs[zero]['text'].startswith('0 : i32')
    if_lines=[line.strip() for line in lines if line.strip().startswith('scf.if ')]
    assert if_lines==['scf.if '+gate+' {']
    assert text.index('scf.if '+gate+' {')<text.index(stores[0])<text.index('} else {')
    checked.append({'input':row['input'],'path':str(path),'sha256':sha(path),
        'pid':pid,'product':product,'lane':lane,'offset':offset,'mask':mask,'store_gate':gate,
        'input_pointer_before_possible_bool_bitcast':inp,'actual_load_pointer':raw_input_pointer,
        'output_pointer':out,'masked_load_has_no_other':True,
        'formula':'i32 pid.x * 1024 + i32 arange(0,1024); signed i32 < 1025; pid i32 == 0 gates store'})
assert len(checked)==13
result={'kind':'read_only_frozen_relu_address_ssa_review','source_sha256':sha(source),
    'original_receipt_sha256':sha(prior/'evidence/receipt.json'),'dtype_receipt_sha256':sha(audit/'receipt.json'),
    'profiles_sha256':sha(audit/'frontend-profiles.json'),'profiles':checked,
    'original_N_is_constexpr':True,'wrapper_block_size':1024,'observed_frontend_N':1025,
    'formal_index_domain':'0 <= N < 2^31; no native behavior claim for N>=2^31',
    'frontend_regeneration':False,'native_backend_compilations':0,'gpu_executions':0,
    'boundary':'TTIR observations bind the i32 offset/mask/gate route at N=1025 only. The Lean arithmetic is generic in its explicit N<2^31 domain. Native constant specialization, 64-bit pointer lowering, bool i1 pointer byte scaling, allocation and load correctness remain unproved.'}
dest=root/'evidence/address-route-review.json'
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'profiles':len(checked),'sha256':sha(dest)}))
