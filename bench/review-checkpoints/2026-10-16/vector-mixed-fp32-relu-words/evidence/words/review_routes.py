"""Bind all 13 prior IRs and annotate the nine finite integer/bool routes."""
from pathlib import Path
import ast
import hashlib
import json
import re

root = Path(__file__).resolve().parent
prior = root.parent / 'relu-original-wip'
audit = prior / 'evidence/dtype-audit'
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
rows = read(audit/'frontend-profiles.json')
successes = [r for r in rows if r['status']=='frontend_generated']
assert len(rows)==17 and len(successes)==13
for row in successes: assert sha(audit/row['ttir'])==row['ttir_sha256']
selected = []
for row in successes:
    name=row['input']
    if name not in {'bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64'}: continue
    ir=(audit/row['ttir']).read_text()
    lines=ir.splitlines()
    load=next(l.strip() for l in lines if 'tt.load' in l)
    compare=next(l.strip() for l in lines if '%result_' in l and 'arith.cmpi ' in l)
    convert=next(l.strip() for l in lines if 'arith.sitofp ' in l or 'arith.uitofp ' in l)
    select=next(l.strip() for l in lines if 'arith.select ' in l)
    store=next(l.strip() for l in lines if 'tt.store ' in l)
    width=8 if name=='bool' else int(re.search(r'\d+',name).group())
    assert 'tt.ptr<i'+str(width)+'>' in load
    assert 'tensor<1024xf32>' in convert and 'tt.ptr<f32>' in store
    assert 'arith.cmpi eq, %pid' in ir and 'scf.if' in ir
    if name=='bool':
        assert 'arith.cmpi ne,' in ir and 'arith.extui ' in ir
        assert 'arith.cmpi sge,' in compare and 'tensor<1024xi32>' in compare
        assert 'arith.uitofp ' in convert and 'tensor<1024xi1> to tensor<1024xf32>' in convert
        route='i8 load -> !=0 to i1; zero-extend i1 to signed i32 for >=0; original unsigned i1 -> fp32; select; pid0 fp32 store'
    else:
        unsigned=name.startswith('uint')
        assert ('arith.uitofp ' if unsigned else 'arith.sitofp ') in convert
        assert 'tensor<1024xi'+str(width)+'> to tensor<1024xf32>' in convert
        if width<32:
            assert ('arith.extui ' if unsigned else 'arith.extsi ') in ir
            assert 'arith.cmpi sge,' in compare and 'tensor<1024xi32>' in compare
            comparison=('zero' if unsigned else 'sign')+'-extend to signed i32 comparison'
        else:
            assert ('arith.cmpi uge,' if unsigned else 'arith.cmpi sge,') in compare
            assert 'tensor<1024xi'+str(width)+'>' in compare
            comparison=('unsigned' if unsigned else 'signed')+' comparison at source width'
        route=f'i{width} load -> {comparison}; original source-width '+('unsigned' if unsigned else 'signed')+' conversion to fp32; select; pid0 fp32 store'
    relevant=[l.strip() for l in lines if any(k in l for k in ['%result','%x_']) and
              any(k in l for k in ['arith.extsi','arith.extui','arith.cmpi','arith.uitofp','arith.sitofp','arith.select','tt.load'])]
    selected.append({'input':name,'source_width':width,'ttir_path':str(audit/row['ttir']),
                     'ttir_sha256':row['ttir_sha256'],'route':route,'dataflow_lines':relevant,'store':store})
assert len(selected)==9
result={'kind':'read_only_frozen_relu_integer_word_route_review',
    'prior_receipt_sha256':sha(prior/'evidence/receipt.json'),
    'prior_dtype_receipt_sha256':sha(audit/'receipt.json'),
    'prior_profiles_sha256':sha(audit/'frontend-profiles.json'),
    'prior_source_sha256':sha(repo/'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'),
    'all_original_frontend_profiles':17,'all_successful_ttir_hashes_verified':13,
    'finite_word_routes':selected,'native_backend_compiled':False,'gpu_executed':False,
    'boundary':'Pinned original-source sm90 frontend evidence. The new Lean theorems prove these comparison/decoding routes within their finite-word definitions and relate them to existing typed execution; no compiler/native physical-memory refinement is inferred.'}
out=root/'evidence/route-review.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'integer_word_routes':len(selected),'all_ttir_hashes_verified':13,'sha256':sha(out)}))
