#!/usr/bin/env python3
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,sys,types,traceback
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def extract(data):
    text=data.decode();tree=ast.parse(text);names=['get_freq_multi_tokens','rbe_triton'];nodes=[]
    for name in names:
        found=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name]
        assert len(found)==1;f=found[0];assert [ast.unparse(x) for x in f.decorator_list]==['triton.jit'];nodes.append(f)
    result='import triton\nimport triton.language as tl\n\n'+'\n\n'.join(''.join(text.splitlines(keepends=True)[f.decorator_list[0].lineno-1:f.end_lineno]) for f in nodes)
    new=[x for x in ast.parse(result).body if isinstance(x,ast.FunctionDef)]
    assert [ast.dump(x,include_attributes=False) for x in new]==[ast.dump(x,include_attributes=False) for x in nodes]
    return result.encode(),{'functions':names,'function_AST_equal':True,'kernel_arguments':[x.arg for x in nodes[1].args.args]}
def serial(x):
    if x is None or isinstance(x,(bool,int,float,str)):return x
    if hasattr(x,'_asdict'):return serial(x._asdict())
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return repr(x)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=a.source.resolve();data=source.read_bytes();jit,binding=extract(data)
    (out/'original.py').write_bytes(data);(out/'original_jit.py').write_bytes(jit);binding.update(source_path=str(source),source_sha256=sha(source),jit_sha256=sha(out/'original_jit.py'))
    dump(out/'input-binding.json',binding);os.environ['TRITON_CACHE_DIR']=str(out/'cache');sys.pycache_prefix=str(out/'pycache')
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    mod=types.ModuleType('rbe_original_checked');mod.__file__=str(out/'original_jit.py');sys.modules[mod.__name__]=mod
    exec(compile(jit,mod.__file__,'exec'),mod.__dict__)
    profiles=[('wrapper_default',2,1024),('old_backend_witness',1,2),('nonperiod_tile',2,64),('period_tile',4,128)]
    records=[]
    for arch in [80,90]:
        for label,bm,bk in profiles:
            for fusion in [False,True]:
                name=f'sm{arch}-{label}-fusion-{str(fusion).lower()}'
                constants={'THETA':10000.,'BLOCK_SIZE_M':bm,'BLOCK_SIZE_K':bk}
                signature={n:('*fp16' if n.endswith('_ptr') else 'i32') for n in mod.rbe_triton.arg_names if n not in constants}
                row={'label':name,'target':{'backend':'cuda','arch':arch,'warp_size':32},'constexprs':constants,'signature':signature,'options':{'enable_fp_fusion':fusion},'native_execution':False}
                try:
                    kernel=triton.compile(ASTSource(mod.rbe_triton,signature,constants),target=GPUTarget('cuda',arch,32),options=row['options'])
                    row['artifact_sha256']={}
                    for kind,value in kernel.asm.items():
                        if isinstance(value,(bytes,str)):
                            q=out/(name+'.'+kind);q.write_bytes(value.encode() if isinstance(value,str) else value);row['artifact_sha256'][q.name]=sha(q)
                    row['resolved_metadata']=serial(kernel.metadata);row['status']='COMPILED_ONLY';row['exit_code']=0
                except Exception as e:row.update(status='COMPILE_FAILED',exit_code=1,error=repr(e),traceback=traceback.format_exc())
                records.append(row);dump(out/'running-results.json',records);print(name,row['status'],flush=True)
    r={'source_sha256':sha(source),'input_binding_sha256':sha(out/'input-binding.json'),'records':records,'compile_successes':sum(x['exit_code']==0 for x in records),'native_execution':False}
    dump(out/'receipt.json',r);return int(r['compile_successes']!=16)
if __name__=='__main__':raise SystemExit(main())
