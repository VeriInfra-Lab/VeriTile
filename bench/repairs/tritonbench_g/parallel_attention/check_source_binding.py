"""Exact source/AST binding and semantic repair controls, without repository writes."""
from pathlib import Path
import ast,hashlib,json,copy,re,subprocess
D=Path(__file__).resolve().parent
P=D
S=P/'parallel_attention_repaired.py'
O=D.parents[2]/'tritonbench_g/parallel_attention/parallel_attention.py'
sha=lambda b:hashlib.sha256(b).hexdigest()
source=S.read_text();tree=ast.parse(source)
def callname(node):return ast.unparse(node.func) if isinstance(node,ast.Call) else ''
def validate(text):
    t=ast.parse(text);nodes=list(ast.walk(t));loads=[n for n in nodes if callname(n)=='tl.load'];assert len(loads)==20
    for n in loads:
        kw={k.arg:ast.literal_eval(k.value) for k in n.keywords if k.arg in ['padding_option','other']}
        assert kw.get('padding_option')=='zero' or kw.get('other')==0,'zero padding'
    defs={n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    guard=ast.unparse(defs['_check_dense_fp32_inputs']);assert 'torch.float32' in guard and '.is_contiguous()' in guard and '128' in guard
    assert 'B * H > 65535 or triton.cdiv(T, 64) > 65535' in guard,'axis guard'
    assert 'formed_limit = (NV * B * H * T + 256) * max(K, V) + 128' in guard,'formed index envelope'
    assert 'formed_limit >= 2 ** 31' in guard,'native width guard'
    methods={n.name:n for n in defs['ParallelBasedFunction'].body if isinstance(n,ast.FunctionDef)}
    for label,method in methods.items():
        empty=[n for n in ast.walk(method) if isinstance(n,ast.If) and ast.unparse(n.test)=='B == 0 or H == 0 or T == 0']
        assert len(empty)==1 and any(isinstance(n,ast.Return) for n in empty[0].body),'empty no-launch branch'
        copies=[ast.unparse(n.func.value) for n in ast.walk(method) if callname(n).endswith('.contiguous')]
        assert copies==([] if label=='forward' else ['do','dz']),'cotangent-only copies'
        launches=[n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)]
        assert len(launches)==1
        args=launches[0].args;offset=5 if label=='forward' else 8
        assert [ast.unparse(n) for n in args[offset:offset+6]]==['T * K','K','1','T * V','V','1'],'canonical dense strides'
        for n in ast.walk(method):
            if callname(n)=='torch.empty':assert {k.arg:ast.unparse(k.value) for k in n.keywords}.get('dtype')=='q.dtype','explicit FP32 scratch'
    backward_guard=ast.unparse(defs['_check_backward_inputs'])
    prepared_guard=ast.unparse(defs['_check_prepared_cotangents'])
    storage_guard=ast.unparse(defs['_check_cotangent_storage'])
    for fragment in ['_check_dense_fp32_inputs(q, k, v)', 'type(x) is not torch.Tensor',
                     'x.layout != torch.strided or not x.is_cuda or x.dtype != torch.float32',
                     'x.is_neg() or x.is_conj()', 'do.device != q.device or dz.device != q.device',
                     'do.shape != v.shape or dz.shape != q.shape[:3]', '_check_cotangent_storage(x)']:
        assert fragment in backward_guard, 'backward input guard: '+fragment
    for fragment in ['stride < 0', '(size - 1) * stride', 'pointer % 4', 'pointer < base',
                     'pointer + 4 * (top + 1) > base + capacity', 'base + capacity > 2 ** 63',
                     'pointer + 4 * formed_limit >= 2 ** 63']:
        assert fragment in storage_guard, 'cotangent storage guard: '+fragment
    for fragment in ['_check_backward_inputs(q, k, v, do, dz)', 'not x.is_contiguous()',
                     'size > 1 and stride != expected_stride', '_check_cotangent_storage(x, formed_limit)',
                     'formed_limit = (NV * B * H * T + 256) * max(K, V) + 128']:
        assert fragment in prepared_guard, 'prepared cotangent guard: '+fragment
    bnodes=list(ast.walk(methods['backward']))
    pre=[n for n in bnodes if callname(n)=='_check_backward_inputs']
    post=[n for n in bnodes if callname(n)=='_check_prepared_cotangents']
    copies=[n for n in bnodes if callname(n).endswith('.contiguous')]
    empty=[n for n in bnodes if isinstance(n,ast.If) and ast.unparse(n.test)=='B == 0 or H == 0 or T == 0']
    launch=[n for n in bnodes if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript)]
    assert len(pre)==len(post)==1 and pre[0].lineno<empty[0].lineno<min(n.lineno for n in copies) and max(n.lineno for n in copies)<post[0].lineno<launch[0].lineno, 'backward validation ordering'
    assert not any(callname(n).endswith(('.resolve_neg','.resolve_conj')) for n in nodes), 'no lazy-flag resolution copies'
    fwd=defs['parallel_rebased_fwd_kernel']
    zguards=[n for n in ast.walk(fwd) if isinstance(n,ast.If) and ast.unparse(n.test)=='i_v == 0' and any(callname(c)=='tl.store' and ast.unparse(c.args[0])=='p_z' for c in ast.walk(n))]
    assert len(zguards)==1,'unique normalizer owner'
    dkv=ast.unparse(defs['_parallel_rebased_bwd_dkv'])
    assert 'valid_query' in dkv and dkv.count('tl.where(valid_query[None, :],')>=2,'future valid-query mask'
    assert 'm_s = (o_k[:, None] <= o_q[None, :]) & ((i + tl.arange(0, BTS))[None, :] < T)' in dkv,'diagonal valid-query mask'
    assert all(any(k.arg=='allow_tf32' and isinstance(k.value,ast.Constant) and k.value.value is False for k in n.keywords) for n in nodes if callname(n)=='tl.dot'),'FP32 dot mode'
    wrapper=defs['parallel_rebased'];branches=[ast.unparse(n.test) for n in ast.walk(wrapper) if isinstance(n,ast.If) and ast.unparse(n.test) in ['use_scale','return_both','use_normalize']]
    assert branches==['use_scale','return_both','use_normalize'],'wrapper branch priority'
    assert 'type(x) is not torch.Tensor' in guard and 'x.is_neg() or x.is_conj()' in guard
    assert 'x.layout != torch.strided or not x.is_cuda' in guard
    assert 'pointer + 4 * x.numel() > base + capacity' in guard
    assert 'size > 1 and stride != expected_stride' in guard
    assert 'x.data_ptr() + 4 * formed_limit >= 2 ** 63' in guard
    for method in [*methods.values(),wrapper]:
        assert any(isinstance(n,ast.With) and any(ast.unparse(i.context_expr)=='torch.cuda.device(q.device)' for i in n.items) for n in ast.walk(method)),'owning CUDA device context'
    public=ast.unparse(wrapper)
    assert 'type(flag) is not bool' in public and 'type(eps) not in (int, float)' in public and 'eps = float(eps)' in public
    return {'loads':len(loads),'dots':sum(callname(n)=='tl.dot' for n in nodes),'stores':sum(callname(n)=='tl.store' for n in nodes),'zero_padding':True,'canonical_dense_strides':True,'unique_z_writer':True,'future_masks':True,'wrapper_branches':branches,'signed_i32_guard':True,'empty_no_launch':True,'backward_inputs_checked_before_empty_and_copy':True,'prepared_actual_storage_checked_before_launch':True}
positive=validate(source)
mutations={
    'missing_native_guard':source.replace('formed_limit >= 2**31','formed_limit >= 2**63',1),
    'missing_empty_branch':source.replace('if B == 0 or H == 0 or T == 0:', 'if False:',2),
    'missing_padding':source.replace('padding_option="zero"','padding_option="nan"',1),
    'duplicate_z_writer':source.replace('if i_v == 0:\n        tl.store(p_z','if i_v >= 0:\n        tl.store(p_z',1),
    'borrowed_q_stride':source.replace('T * K, K, 1,','q.stride(1), q.stride(2), q.stride(3),',1),
    'implicit_dtype':source.replace('dtype=q.dtype, device=q.device','device=q.device',1),
    'missing_future_mask':source.replace('tl.where(valid_query[None, :],','tl.where(True,',1),
    'wrong_priority':source.replace('if return_both:','if use_normalize:',1),
    'wrong_device':source.replace('torch.cuda.device(q.device)','torch.cuda.device(0)',1),
    'missing_storage':source.replace('pointer + 4 * x.numel() > base + capacity','False',1),
    'wrong_stride':source.replace('size > 1 and stride != expected_stride','False',1),
    'missing_backward_lazy_guard':source.replace('x.is_neg() or x.is_conj() for x in (do, dz)','False for x in (do, dz)',1),
    'wrong_backward_dtype':source.replace('x.dtype != torch.float32','x.dtype != torch.float16',1),
    'wrong_backward_device':source.replace('do.device != q.device or dz.device != q.device','False',1),
    'wrong_backward_shape':source.replace('do.shape != v.shape or dz.shape != q.shape[:3]','False',1),
    'missing_cotangent_span':source.replace('pointer + 4 * (top + 1) > base + capacity','False',1),
    'missing_prepared_guard':source.replace('            _check_prepared_cotangents(q, k, v, do, dz)\n','',1),
    'missing_backward_guard':source.replace('        _check_backward_inputs(q, k, v, do, dz)\n','',1),

}
controls=[]
for name,mutant in mutations.items():
    assert mutant!=source,name
    try:validate(mutant)
    except AssertionError as e:controls.append({'name':name,'rejected':True,'reason':str(e),'mutant_sha256':sha(mutant.encode())})
    else:raise AssertionError('mutation not rejected: '+name)
functions=[]
for n in tree.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
        functions.append({'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'AST_sha256':sha(ast.dump(n,include_attributes=False).encode())})
canonical=ast.dump(tree,include_attributes=False)
assert sha(O.read_bytes())=='a38eb94742c262088ad9107b8eeb64f551476964f93eee2ed612d0890e449f31'
manifest={'source':str(S),'source_sha256':sha(S.read_bytes()),'original_sha256':sha(O.read_bytes()),'AST_sha256':sha(canonical.encode()),'functions':functions,'checks':positive,'negative_controls':controls,'scope':'Exact candidate binding plus structural repair invariants; no native scalar conformance assertion','script_sha256':sha(Path(__file__).read_bytes()),'lean_files':{str(p.relative_to(D)):sha(p.read_bytes()) for p in sorted(D.rglob('*.lean')) if p.name!='Source.lean'}}
(D/'source-binding.json').write_text(json.dumps(manifest,indent=2)+'\n')

lean='import bench.repairs.tritonbench_g.parallel_attention.Proofs.Contract\n\nnamespace VeriTile.Bench.Repairs.ParallelAttention\n\n/-- Exact candidate provenance; semantic scope and compiler binding are in README.md. -/\ndef sourceSHA256 : String := '+json.dumps(manifest['source_sha256'])+'\ndef sourceASTSHA256 : String := '+json.dumps(manifest['AST_sha256'])+'\ndef sourceASTLiteral : String := '+json.dumps(canonical)+'\n\nend VeriTile.Bench.Repairs.ParallelAttention\n'
(P/'Proofs/Source.lean').write_text(lean)

manifest['lean_files']['Proofs/Source.lean']=sha((P/'Proofs/Source.lean').read_bytes())
workspace=D.parents[3]
original_kernel=(workspace/'bench/tritonbench_g/parallel_attention/Kernels.lean').read_text()
def declaration(text,name):
    start=text.index('def '+name+'\n')
    end=text.index('\n}\n',start)+3
    return text[start:end]
original_declarations=[]
for path,names in [
 ('bench/tritonbench_g/parallel_attention/ParallelAttention.lean',['pa_fwd_surface','pa_bwd_dkv_surface']),
 ('bench/tritonbench_g/parallel_attention/Proofs/FPSources.lean',['forwardSource','backwardSource'])]:
    historical=subprocess.check_output(['git','-C',str(workspace),'show','HEAD:'+path],text=True)
    for name in names:
        before=declaration(historical,name);after=declaration(original_kernel,name)
        assert before==after,('historical source body changed',name)
        original_declarations.append({'name':name,'body_sha256':sha(before.encode()),'identical_body':True})
manifest['preserved_original_declarations']=original_declarations
(D/'source-binding.json').write_text(json.dumps(manifest,indent=2)+'\n')

print(json.dumps({'positive':positive,'negative_controls':controls,'source_sha256':manifest['source_sha256'],'AST_sha256':manifest['AST_sha256']}))
