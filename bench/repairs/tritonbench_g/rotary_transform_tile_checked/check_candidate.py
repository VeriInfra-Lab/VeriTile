"""Actual wrapper and JIT binder captures, with no kernel/compiler execution."""
import ast
import contextlib
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ.pop('TRITON_INTERPRET', None)
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from triton.runtime.jit import create_function_from_signature
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name('rotary_transform_tile_checked.py')
OLD = SOURCE.parent.parent/'rotary_transform/rotary_transform_repaired.py'
OPS = SOURCE.parent.parent/'rotary_transform_ops/rotary_transform_ops_repaired.py'
OUT = ROOT/'.veritile-cache/kernel-evidence/rotary_transform_tile_checked'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
before, after, ops = functions(OLD), functions(SOURCE), functions(OPS)
assert ast.dump(before['rotary_kernel']) == ast.dump(after['rotary_kernel'])
assert ast.dump(ast.Module(body=before['rotary_kernel'].body,type_ignores=[])) == ast.dump(ast.Module(body=ops['rotary_kernel'].body,type_ignores=[]))
new_guard = [n for n in after['apply_rotary'].body if isinstance(n,ast.If) and 'source tile exceeds' in ast.unparse(n)]
assert len(new_guard)==1
after['apply_rotary'].body.remove(new_guard[0])
assert set(before)==set(after)
assert all(ast.dump(before[n])==ast.dump(after[n]) for n in before)
old_module,new_module=ast.parse(OLD.read_text()),ast.parse(SOURCE.read_text())
wrapper=next(n for n in new_module.body if isinstance(n,ast.FunctionDef) and n.name=='apply_rotary')
wrapper.body=[n for n in wrapper.body if not (isinstance(n,ast.If) and 'source tile exceeds' in ast.unparse(n))]
assert ast.dump(old_module)==ast.dump(new_module)
assert ast.unparse(new_guard[0].test) == 'BLOCK_M * (BLOCK_K if interleaved else BLOCK_K // 2) > 1 << 20'

spec=importlib.util.spec_from_file_location('rotary_transform_tile_checked_capture',SOURCE)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
fn=module.rotary_kernel
backend=CUDABackend(GPUTarget('cuda',80,32))
binder=create_function_from_signature(fn.signature,fn.params,backend)
torch.set_num_threads(1)
cases=[]
for packed,tensor,inter,inplace,conj in itertools.product((False,True),repeat=5):
    for dtype in (torch.int32,torch.int64):
        cases.append(dict(name=f'flags_{int(packed)}{int(tensor)}{int(inter)}{int(inplace)}{int(conj)}_{dtype}',
                          packed=packed,tensor=tensor,inter=inter,inplace=inplace,conj=conj,dtype=dtype,width=10,half=3))
for inter,limit in [(True,262144),(False,524288)]:
    for label,width,empty in [('boundary',limit,None),('oversize',limit+2,None),
                              ('empty_batch',limit+2,'batch'),('empty_sequence',limit+2,'sequence'),
                              ('empty_packed',limit+2,'packed'),('zero_rotary',limit+2,'rotary')]:
        cases.append(dict(name=f'{"interleaved" if inter else "split"}_{label}',inter=inter,
                          width=width,half=width//2,empty=empty,reject=label=='oversize'))
for width in [32,64,66,514]:
    cases.append(dict(name=f'split_block_selection_{width}',width=width,half=width//2))
cases.extend([dict(name='mixed_cu32_offsets64',packed=True,tensor=True,width=10,half=3,dtype=torch.int32,offset_dtype=torch.int64),
              dict(name='mixed_cu64_offsets32',packed=True,tensor=True,width=10,half=3,dtype=torch.int64,offset_dtype=torch.int32),
              dict(name='strided_preparation',packed=True,tensor=True,width=514,half=257,strided=True),
              dict(name='empty_coefficients',width=18,half=8,cache_rows=0)])
rows=[]
for case in cases:
    packed=case.get('packed',False) or case.get('empty')=='packed'
    inter=case.get('inter',False);inplace=case.get('inplace',False);conj=case.get('conj',False)
    width=case['width'];half=0 if case.get('empty')=='rotary' else case['half'];dtype=case.get('dtype',torch.int64)
    small=case['name'].startswith(('flags_','mixed_')) or case.get('strided',False)
    batch=3 if small else 1;length=3 if small else 1;heads=2 if small else 1
    if case.get('empty')=='batch':batch=0
    if case.get('empty')=='sequence':length=0
    cu_values=[0,2,2,5] if small else [0,0] if packed else None
    if packed and not small:total=0
    else:total=5 if packed else batch*length
    shape=(total,heads,width) if packed else (batch,length,heads,width)
    x=torch.zeros(shape)
    cache_rows=case.get('cache_rows',4 if small else 1)
    stride=2 if case.get('strided') else 1
    co=torch.ones(cache_rows,half*stride)[:,::stride]
    si=torch.zeros(cache_rows,half*stride)[:,::stride]
    cu=torch.tensor(cu_values,dtype=dtype) if packed else None
    shifts=[0,9,2] if small else [0]*batch
    offsets=torch.tensor([v for v in shifts for _ in range(stride)],dtype=case.get('offset_dtype',dtype))[::stride] if case.get('tensor') else 0
    inputs=[v for v in [x,co,si,cu,offsets] if isinstance(v,torch.Tensor)]
    snapshots=[v.clone() for v in inputs];launches=[]
    class Capture:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                env=dict(zip(fn.arg_names,args));env.update(kwargs)
                bound,specialization,options=binder(*args,**kwargs)
                _,signature,constants,attrs=fn._pack_args(backend,options,bound,specialization,options)
                assert env['X'] is x and env['CU_SEQLENS'] is cu
                assert env['COS'].is_contiguous() and env['SIN'].is_contiguous()
                assert torch.equal(env['COS'],co) and torch.equal(env['SIN'],si)
                if isinstance(offsets,torch.Tensor):
                    assert env['SEQLEN_OFFSETS'].is_contiguous() and torch.equal(env['SEQLEN_OFFSETS'],offsets)
                bm,bk=env['BLOCK_M'],env['BLOCK_K']
                assert bm*(bk if inter else bk//2)<=1<<20
                launches.append({'grid':list(grid(env)),'block_m':bm,'block_k':bk,
                                 'pointer_elements':bm*(bk if inter else bk//2),
                                 'actual_binder':True,'signature':signature})
            return launch
    error=None;result=None
    try:
        with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)),patch.object(torch.cuda,'device',lambda _:contextlib.nullcontext()),patch.object(module,'rotary_kernel',Capture()):
            result=module.apply_rotary(x,co,si,offsets,cu,length if packed else None,inter,inplace,conj)
    except ValueError as e:error=str(e)
    assert all(torch.equal(a,b) for a,b in zip(inputs,snapshots))
    if case.get('reject'):
        assert error=='source tile exceeds Triton static tensor element limit' and not launches
    else:
        assert error is None,(case,error)
        assert (result is x)==inplace and result.shape==x.shape and result.dtype==torch.float32
        assert len(launches)==int(x.numel()!=0 and half!=0)
        assert torch.equal(result[...,2*half:],x[...,2*half:])
    rows.append({'case':case['name'],'launches':launches,'error':error,'empty':x.numel()==0 or half==0})

old_report=OLD.parent/'source-defect-evidence/report.json'
report=json.loads(old_report.read_text());assert report['source']['sha256']==sha(OLD)
controls=[]
for result in report['results']:
    if result['failure_expected']:continue
    artifact=OLD.parent/'source-defect-evidence'/(result['name']+'.ptx')
    controls.append({'case':result['name'],'ptx':{'path':str(artifact.relative_to(ROOT)),'sha256':sha(artifact)}})
assert len(controls)==4
packet={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':sha(SOURCE),
        'only_change':'branch-specific static tensor element check after original empty/zero-rotary return',
        'unchanged_complete_jit':True,'rotary_ops_body_ast_identical':True,
        'rotary_ops_signature_difference':'rotary_transform has two unused runtime parameters nheads and CACHE_KEY_SEQLEN; capture uses actual rotary_transform binder',
        'cases':rows,'reused_compiler_controls':controls,'original_report_sha256':sha(old_report),
        'gpu_execution':False,'kernel_interpreter_execution':False,'new_compilations':0,'numerical_experiments':[],
        'boundary':'Actual Torch objects and wrapper; only CUDA classification/context and launcher replaced. Actual installed JIT binder receives captured arguments. Admission checks do not certify device resources.'}
(OUT/'candidate-checks.json').write_text(json.dumps(packet,indent=2)+'\n')
print(f'PASS {len(rows)} wrapper cases; {sum(len(r["launches"]) for r in rows)} actual binder captures; 4 saved SM80/SM90 controls reused; no GPU/compiler/interpreter run.')
