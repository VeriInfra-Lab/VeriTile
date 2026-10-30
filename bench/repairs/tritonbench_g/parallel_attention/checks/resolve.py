"""Admit one actual specialization pair. Reads artifacts; never invokes a compiler/GPU.

This checker is an audited ordinary compiler-boundary tool, not a Lean theorem
about a universal Python/LLVM/PTX parser. See COMPILER-CONTRACT.md for its scope.
"""
from pathlib import Path
import ast,hashlib,json,re,sys
from check_dot import audit_dot
from check_norm import audit_norm
from check_scalar import audit_scalar
from inspect_ptx import scan

ROOT=Path(__file__).resolve().parent
PRODUCER=ROOT.parent
SOURCE=PRODUCER/'parallel_attention_repaired.py'
KERNELS=['parallel_rebased_fwd_kernel','_parallel_rebased_bwd_dq','_parallel_rebased_bwd_dkv','parallel_rebased_bwd_kernel']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return {'path':str(path),'sha256':sha(path)}
def kernel_ast(path):
    nodes={n.name:n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    return {name:hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest() for name in KERNELS}

def check_receipt(path,kind,current_source):
    r=json.loads(path.read_text())
    assert r['kind']==kind and r['success'] is True
    assert r['target']==['cuda',90,32] and r['num_warps']==4 and r['num_stages']==2
    assert r['fp_fusion'] is True
    assert all(v=='*fp32' for v in r['signature'].values() if v.startswith('*'))
    for row in r['artifacts'].values():assert sha(row['path'])==row['sha256'],row['path']
    expected_ast=json.loads((ROOT/'kernel-ast.json').read_text())
    assert kernel_ast(r['artifacts']['source']['path'])==expected_ast,'compiled kernel AST changed'
    assert kernel_ast(current_source)==expected_ast,'current kernel AST changed'
    c=r['constants'];shape=r['shape']
    assert c['BTS']==32 and c['BTL']==(128 if kind=='forward' else 64)
    assert c['K']==shape[3] and c['V']==shape[4]
    assert c['BK']==min(128,max(16,1<<(shape[3]-1).bit_length()))
    assert c['BV']==min(128,max(16,1<<(shape[4]-1).bit_length()))
    assert c['s_qk_d']==c['s_vo_d']==1
    assert c.get('scale',1)==1
    ttgir=Path(r['artifacts']['ttgir']['path']).read_text()
    assert 'ttng.warp_group_dot' not in ttgir and '#ttg.nvidia_mma' not in ttgir
    if kind=='forward':
        assert 'sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]' in ttgir
    return r

def route_report(receipt,cached=False):
    path=Path(receipt['artifacts']['llir']['path'])
    cache=ROOT/'routing'/(path.parent.name+'.json')
    if cached:
        r=json.loads(cache.read_text())
        bound=json.loads((ROOT/'routing-bindings.json').read_text())
        assert bound['checker_sha256']==sha(ROOT/'check_dot.py')
        assert bound['reports'][path.parent.name]['llir_sha256']==sha(path)
        assert bound['reports'][path.parent.name]['report_sha256']==sha(cache)
    else:r=audit_dot(path,receipt['constants'])
    assert not r['failures']
    # Reject omitted scalar or vector FMA intrinsics, even if no recognized
    # source site happened to use their result.
    calls=re.findall(r'@llvm\.(?:fmuladd|fma)\.(f32|v(\d+)f32)\([^\n]*!dbg',path.read_text())
    count=sum(int(width) if width else 1 for _,width in calls)
    assert count==r['scalar_fma_instruction_count'],(path,count,r['scalar_fma_instruction_count'])
    ptx=Path(receipt['artifacts']['ptx']['path']);events=scan(ptx)
    ptx_fma=sum(e['op']=='fma.rn.f32' for e in events)
    scalar_fma=sum(e['op']=='fma.rn.f32' and e['loc'] in ([57,84] if receipt['kind']=='forward' else [184,238,254,271]) for e in events)
    assert ptx_fma-scalar_fma==count,(ptx,ptx_fma,scalar_fma,count)
    return {'llvm_route_report':pin(cache) if cached else r,'scalar_dot_fmas':count,
            'ptx_dot_fmas':ptx_fma-scalar_fma,'ptx_other_fmas':scalar_fma,
            'scalar_chains':len(r['roots']),'threads':128}

def resolve_pair(forward_receipt,backward_receipt,current_source=SOURCE,cached=False):
    fr=check_receipt(Path(forward_receipt),'forward',current_source)
    br=check_receipt(Path(backward_receipt),'backward',current_source)
    assert fr['shape']==br['shape']
    assert fr['constants']['BK']==br['constants']['BK'] and fr['constants']['BV']==br['constants']['BV']
    use_scale='scale' not in fr['constants']
    assert use_scale==('scale' not in br['constants'])
    post_seed=fr['shape'][4]>fr['constants']['BV']
    norm_path=Path(fr['artifacts']['ptx']['path'])
    # Resolve from syntax, never from a profile-name or input-shape whitelist.
    norms=[]
    for tiny in [False,True]:
        try:norms.append((tiny,audit_norm(norm_path,tiny=tiny)))
        except AssertionError:pass
    assert len(norms)==1,'unrecognized or ambiguous normalizer DAG'
    tiny,norm=norms[0]
    scalar=audit_scalar(Path(br['artifacts']['ptx']['path']),use_scale,post_seed,post_seed)
    padded_direct=next(x['count']>0 for x in scalar if x['site']=='constant-padded-DK')
    if padded_direct:assert br['shape'][2]==1 and use_scale
    # Forward query scale must be present only for runtime float scale and
    # retain its left-operand form at every surviving scalar instruction.
    fs=scan(norm_path);scales=[e for e in fs if e['loc']==42 and e['op']=='mul.f32']
    if use_scale:
        assert scales and all(e['node'][1]==scales[0]['node'][1] for e in scales)
    else:assert not scales
    return {'status':'ADMITTED_UNDER_TYPED_COMPILER_CONTRACT','shape':fr['shape'],
      'descriptor':{'BK':fr['constants']['BK'],'BV':fr['constants']['BV'],'useScale':use_scale,
                    'postSeed':post_seed,'tinyDiagonal':tiny,'scaledSumOnZ':post_seed,
                    'paddedCoefficientDirect':padded_direct},
      'source':pin(current_source),'kernel_ast_sha256':kernel_ast(current_source),
      'receipts':[pin(Path(forward_receipt)),pin(Path(backward_receipt))],
      'routing':[route_report(fr,cached),route_report(br,cached)],'normalizer':norm,'scalar':scalar,
      'compiler_contract':pin(ROOT/'COMPILER-CONTRACT.md'),
      'excludes':['native numerical conformance','host Torch reduction lowering','unseen specialization admission'],
      'compiler_or_gpu_invocations':0}

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--saved':
        dest=ROOT/'admissions';dest.mkdir(exist_ok=True)
        for f in sorted((PRODUCER/'evidence').glob('actual-binder-*-forward/receipt.json')):
            b=f.parent.with_name(f.parent.name.replace('-forward','-backward'))/'receipt.json'
            r=resolve_pair(f,b,cached=True)
            output=dest/(f.parent.name.removesuffix('-forward')+'.json')
            output.write_text(json.dumps(r,indent=2)+'\n')
            print(output.name,r['status'],flush=True)
    else:
        assert len(sys.argv)==4,'usage: resolve.py FORWARD_RECEIPT BACKWARD_RECEIPT CURRENT_SOURCE'
        print(json.dumps(resolve_pair(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])),indent=2))
