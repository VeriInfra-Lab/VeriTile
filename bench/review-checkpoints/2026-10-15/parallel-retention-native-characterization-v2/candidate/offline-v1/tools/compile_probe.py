#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,os,sys,types,traceback,argparse
sys.dont_write_bytecode=True
NAMES=['parallel_retention_fwd_kernel','_parallel_retention_bwd_dq','_parallel_retention_bwd_dkv','parallel_retention_bwd_kernel']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def serial(x):
    if x is None or isinstance(x,(bool,int,float,str)):return x
    if hasattr(x,'_asdict'):return serial(x._asdict())
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return repr(x)
def extract(data):
    text=data.decode();tree=ast.parse(text);nodes=[]
    for name in NAMES:
        matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(matches)==1;fn=matches[0]
        assert [ast.unparse(x) for x in fn.decorator_list]==['triton.jit'];nodes.append(fn)
    out='import triton\nimport triton.language as tl\n\n'+'\n\n'.join(''.join(text.splitlines(keepends=True)[n.decorator_list[0].lineno-1:n.end_lineno]) for n in nodes)
    new=[n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef)]
    assert [ast.dump(x,include_attributes=False) for x in new]==[ast.dump(x,include_attributes=False) for x in nodes]
    return out.encode(),{'function_AST_equal':True,'functions':NAMES,'arguments':{n.name:[a.arg for a in n.args.args] for n in nodes}}
def signature(fn,constants):
    return {n:('*fp32' if n in ('q','k','v','o','do','dq','dk','dv') else 'fp32' if n=='scale' else 'i32') for n in fn.arg_names if n not in constants}
def profiles():
    out={}
    for shape,dk,dv,bk,bv,stages in [('default',64,64,64,64,3),('ragged_split',160,96,128,128,2)]:
        for stage,fn,btl in [('forward',NAMES[0],128),('backward_helper',NAMES[2],64),('backward_shell',NAMES[3],64)]:
            out[shape+'_'+stage]={'entrypoint':fn,'constexprs':{'BTL':btl,'BTS':32,'BK':bk,'BV':bv,'DK':dk,'DV':dv},'options':{'num_warps':4,'num_stages':stages},'accepted_stage':stage!='backward_shell'}
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=a.source.read_bytes();jit,meta=extract(data);(out/'original.py').write_bytes(data);(out/'original_jit.py').write_bytes(jit);dump(out/'input-binding.json',{**meta,'source_sha256':sha(a.source),'jit_sha256':sha(out/'original_jit.py'),'profiles':profiles()})
    os.environ['TRITON_CACHE_DIR']=str(out/'cache');sys.pycache_prefix=str(out/'pycache')
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    mod=types.ModuleType('pra_original_checked');mod.__file__=str(out/'original_jit.py');sys.modules[mod.__name__]=mod;exec(compile(jit,mod.__file__,'exec'),mod.__dict__)
    rows=[]
    for arch in [80,90]:
      for name,prof in profiles().items():
       for fusion in [False,True]:
        label=f'sm{arch}-{name}-fusion-{str(fusion).lower()}';fn=getattr(mod,prof['entrypoint']);opts={**prof['options'],'enable_fp_fusion':fusion};sig=signature(fn,prof['constexprs'])
        row={'label':label,'profile':prof,'target':arch,'signature':sig,'options':opts,'native_execution':False}
        try:
            k=triton.compile(ASTSource(fn,sig,prof['constexprs']),target=GPUTarget('cuda',arch,32),options=opts);row['artifact_sha256']={}
            for ext,b in k.asm.items():
                if isinstance(b,(str,bytes)):
                    q=out/(label+'.'+ext);q.write_bytes(b.encode() if isinstance(b,str) else b);row['artifact_sha256'][q.name]=sha(q)
            row.update(status='COMPILED_ONLY',exit_code=0,resolved_metadata=serial(k.metadata))
        except Exception as e:row.update(status='COMPILE_FAILED',exit_code=1,error=repr(e),traceback=traceback.format_exc())
        rows.append(row);dump(out/'running-results.json',rows);print(label,row['status'],flush=True)
    r={'source_sha256':sha(a.source),'input_binding_sha256':sha(out/'input-binding.json'),'records':rows,'compile_successes':sum(r['exit_code']==0 for r in rows),'native_execution':False};dump(out/'receipt.json',r);raise SystemExit(int(r['compile_successes']!=24))
