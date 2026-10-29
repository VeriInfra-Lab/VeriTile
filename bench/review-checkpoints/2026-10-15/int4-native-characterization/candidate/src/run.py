#!/usr/bin/env python3
"""Exact original int4 kernel capture. Successful GPU capture is CHARACTERIZATION, never overall numerical PASS."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,platform,shutil,struct,subprocess,sys,traceback,types
HERE=Path(__file__).resolve().parent
PROOF_SHA='e9aac373ac26355faad834a2263cbe80bb06c62b465f834ed72a68f260b76371'
NATIVE_SHA='61b327833b2c5136ee6f232a34f881c06952d63d9f229b415d2937ea99a35c7f'
SOURCE_SHA='f8b62e30bed71855e5e40603806f6e3f34cc8548dc4be923ef1168c97f92489b'
PROFILES=('ascending_fma','descending_fma','source_separate_dot_add')
POINTER_TYPES={'a_ptr':'*fp16','b_ptr':'*i32','c_ptr':'*fp16','bs_ptr':'*fp16','bzp_ptr':'*i32'}

def digest(data):return hashlib.sha256(data).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def load(p):return json.loads(p.read_text())
def need(ok,msg):
    if not ok:raise ValueError(msg)
def checked_module(path,name,expected):
    data=path.read_bytes();need(digest(data)==expected,'checked source bytes changed: '+str(path))
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(data,str(path),'exec'),m.__dict__)
    return m
_ORACLE_LOADED_SHA=sha(HERE/'oracle.py')
oracle=checked_module(HERE/'oracle.py','int4_characterization_checked_oracle',_ORACLE_LOADED_SHA)
def tools_binding():
    need(sha(HERE/'oracle.py')==_ORACLE_LOADED_SHA,'oracle changed after load')
    return {n:sha(HERE/n) for n in ['run.py','oracle.py']}
def serial(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if hasattr(v,'_asdict'):return serial(v._asdict())
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return repr(v)
def extract(data):
    text=data.decode();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='matmul_kernel']
    need(len(nodes)==1,'unique original kernel required');fn=nodes[0]
    jit=[d for d in fn.decorator_list if isinstance(d,ast.Attribute) and ast.unparse(d)=='triton.jit']
    tune=[d for d in fn.decorator_list if isinstance(d,ast.Call) and ast.unparse(d.func)=='triton.autotune']
    need(len(jit)==len(tune)==1 and len(fn.decorator_list)==2,'original decorators changed')
    segment=''.join(text.splitlines(keepends=True)[jit[0].lineno-1:fn.end_lineno])
    body=('import triton\nimport triton.language as tl\n\n'+segment).encode()
    generated=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef));a=copy.deepcopy(fn);b=copy.deepcopy(generated);a.decorator_list=[];b.decorator_list=[]
    need(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'function AST mismatch')
    configs=[]
    for n in ast.walk(tune[0]):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='triton.Config':
            configs.append({'constants':ast.literal_eval(n.args[0]),'options':{k.arg:ast.literal_eval(k.value) for k in n.keywords}})
    return body,{'first_line':jit[0].lineno,'last_line':fn.end_lineno,'segment_sha256':digest(segment.encode()),'function_AST_equal':True,'configs':configs,'arguments':[a.arg for a in fn.args.args]}
def validate_proof(package):
    need(sha(package/'receipt.json')==PROOF_SHA,'unexpected proof receipt')
    r=load(package/'receipt.json')
    for rel,h in r['artifacts_sha256'].items():
        q=(package/rel).resolve();need(package.resolve() in q.parents and sha(q)==h,'proof artifact changed '+rel)
    need(sha(package/'native/receipt.json')==NATIVE_SHA,'native receipt pin differs')
    nr=load(package/'native/receipt.json')
    need(nr['records']==load(package/'native/running-results.json'),'native records differ from bound running results')
    need(nr['native_launches']==0 and nr['compile_successes']==8,'wrong parent native evidence')
    return r,nr

def prepare(package,out):
    r,nr=validate_proof(package);need(not out.exists(),'fresh output required');out.mkdir(parents=True)
    (out/'evidence').mkdir()
    for name in ['receipt.json','final-build.json','theorems.json','public-entry-candidate.json','experiment-case.json','native/receipt.json','native/input-binding.json','native/running-results.json']:
        q=out/'evidence'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(package/name,q)
    for row in load(package/'final-build.json'):
        source=Path(row['source']);rel=source.relative_to(package);q=out/'evidence'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,q)
        q=out/'evidence/logs'/Path(row['log']).name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['log'],q)
    original=(package/'native/original.py').read_bytes();need(digest(original)==SOURCE_SHA,'original changed')
    jit,location=extract(original);need(jit==(package/'native/original_jit.py').read_bytes(),'extraction differs from compiled evidence')
    (out/'original.py').write_bytes(original);(out/'original_jit.py').write_bytes(jit)
    case=load(package/'experiment-case.json');need(case['original_source_sha256']==SOURCE_SHA,'case source differs')
    selected={'constants':case['launch']['constexprs'],'options':{'num_warps':4,'num_stages':4}}
    need(selected in location['configs'] and len(location['configs'])==36,'not an original autotune member')
    refs={name:oracle.expected(case,name) for name in PROFILES}
    need(refs==case['expected_full_c_allocations_by_software_graph'],'CPU exact-rational oracle differs from Lean witnesses')
    for arr in case['allocations'].values():need(arr['logical_origin_elements']>0,'nonzero pointer origins required')
    dump(out/'case.json',case);dump(out/'oracle-reference.json',refs)
    reference={'schema':1,'task':'int4_matmul','purpose':'CHARACTERIZATION_ONLY','proof_package_receipt_sha256':PROOF_SHA,
      'parent_native_receipt_sha256':NATIVE_SHA,'source_sha256':SOURCE_SHA,'jit_sha256':digest(jit),'source_extraction':location,
      'tools_sha256':tools_binding(),'case_sha256':sha(out/'case.json'),'oracle_reference_sha256':sha(out/'oracle-reference.json'),
      'evidence_sha256':{p.relative_to(out).as_posix():sha(p) for p in (out/'evidence').rglob('*') if p.is_file()},
      'software_profile_outputs':{k:v[case['allocations']['c_ptr']['logical_origin_elements']] for k,v in refs.items()},
      'original_gpu_execution':False,'universal_native_equality_proved':False,'numeric_acceptance':False,
      'scope':'One original autotune member, complete two-block packed input, one active output from a full 64x128 tile, full physical allocation frame and input preservation; exact three distinct software graphs. No broader input/dtype/native claim.'}
    dump(out/'reference.json',reference);validate(out)
    return {'status':'REFERENCE_PREPARED_CPU_ONLY','native_execution':False,'software_profile_outputs':reference['software_profile_outputs']}

def validate(ref):
    r=load(ref/'reference.json');need(r['schema']==1 and r['task']=='int4_matmul' and r['purpose']=='CHARACTERIZATION_ONLY','unknown reference')
    need(r['proof_package_receipt_sha256']==PROOF_SHA and r['parent_native_receipt_sha256']==NATIVE_SHA,'producer pins changed')
    need(r['tools_sha256']==tools_binding(),'tools differ from prepared reference')
    need(sha(ref/'original.py')==r['source_sha256']==SOURCE_SHA,'original source changed')
    jit,location=extract((ref/'original.py').read_bytes());need(digest(jit)==sha(ref/'original_jit.py')==r['jit_sha256'] and location==r['source_extraction'],'JIT body/config binding changed')
    for name,h in r['evidence_sha256'].items():
        q=(ref/name).resolve();need(ref.resolve() in q.parents and sha(q)==h,'evidence changed '+name)
    need(sha(ref/'evidence/receipt.json')==PROOF_SHA and sha(ref/'evidence/native/receipt.json')==NATIVE_SHA,'producer receipts changed')
    parent=load(ref/'evidence/receipt.json')
    for name,h in r['evidence_sha256'].items():
        rel=Path(name).relative_to('evidence').as_posix()
        if rel=='receipt.json':need(h==PROOF_SHA,'proof receipt pin mismatch')
        elif rel=='native/receipt.json':need(h==NATIVE_SHA,'native receipt pin mismatch')
        else:need(parent['artifacts_sha256'].get(rel)==h,'copy is not bound by proof receipt: '+name)
    native=load(ref/'evidence/native/receipt.json')
    need(native['records']==load(ref/'evidence/native/running-results.json'),'native records drift')
    need(native['input_binding_sha256']==sha(ref/'evidence/native/input-binding.json'),'native input binding drift')
    need(sha(ref/'case.json')==r['case_sha256'] and sha(ref/'oracle-reference.json')==r['oracle_reference_sha256'],'case/reference changed')
    case=load(ref/'case.json');need(case==load(ref/'evidence/experiment-case.json'),'case differs from Lean-bound contract')
    refs={name:oracle.expected(case,name) for name in PROFILES}
    need(refs==load(ref/'oracle-reference.json')==case['expected_full_c_allocations_by_software_graph'],'CPU oracle/reference mismatch')
    need(r['software_profile_outputs']=={name:out[case['allocations']['c_ptr']['logical_origin_elements']] for name,out in refs.items()},'displayed reference words differ')
    need(r['original_gpu_execution'] is False and r['numeric_acceptance'] is False and r['universal_native_equality_proved'] is False,'unsupported claim')
    return r,case,refs

def compiler_environment(archs):
    import triton,triton.backends.nvidia.compiler as nc
    root=Path(triton.__file__).resolve().parent
    files={p.resolve() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    files.add(Path(sys.executable).resolve());tools={}
    for arch in archs:
        tool=Path(nc.get_ptxas(arch).path).resolve();files.add(tool);tools[str(arch)]={'ptxas_path':str(tool),'version':nc.get_ptxas_version(arch)}
    libdevice=Path(nc.knobs.nvidia.libdevice_path or root/'backends/nvidia/lib/libdevice.10.bc').resolve();files.add(libdevice)
    return {'file_sha256':{str(p):sha(p) for p in sorted(files)},'ptxas':tools,'libdevice':str(libdevice),'triton':triton.__version__,
      'boundary':'Actual Triton package, resolved ptxas/libdevice, Python executable before/after. OS and all host shared libraries are not hermetically captured.'}
def save_artifacts(kernel,out,label):
    files={}
    for kind,value in kernel.asm.items():
        need(kind.replace('_','').isalnum(),'unsafe artifact name')
        if not isinstance(value,(str,bytes)):continue
        q=out/(label+'.'+kind);q.write_bytes(value.encode() if isinstance(value,str) else value);files[q.name]=sha(q)
    need(all(label+'.'+x in files for x in ['ttir','ttgir','llir','ptx','cubin']),'missing compiler artifacts')
    metadata=serial(kernel.metadata);need(isinstance(metadata,dict),'metadata field names missing')
    return {'artifact_sha256':files,'resolved_metadata':metadata}
def signature_for(args,constants):return {n:POINTER_TYPES.get(n,'i32') for n in args if n not in constants}
def runtime_arguments(names,allocations,scalars):
    need(set(names)==set(allocations)|set(scalars),'runtime signature mismatch')
    return [allocations[n] if n in allocations else scalars[n] for n in names]
def raw_bytes(words,width):return struct.pack('<'+('H' if width==16 else 'I')*len(words),*(v&((1<<width)-1) for v in words))
def save_words(out,label,name,words,width):
    q=out/(label+'-'+name+'.bin');q.write_bytes(raw_bytes(words,width));return {'path':q.name,'sha256':sha(q),'word_bits':width,'words':len(words),'byte_order':'little'}

def collect(reference,out,mode,archs,selected,device):
    r,case,refs=validate(reference);need(not out.exists(),'fresh output required');out.mkdir(parents=True)
    shutil.copytree(reference,out/'reference');(out/'tools').mkdir()
    for name in tools_binding():shutil.copy2(HERE/name,out/'tools'/name)
    os.environ['TRITON_CACHE_DIR']=str(out/'cache');sys.pycache_prefix=str(out/'pycache')
    report={'schema':1,'task':'int4_matmul','mode':mode,'status':'NOT_RUN','records':[],
      'reference_sha256':sha(reference/'reference.json'),'tools_sha256':tools_binding(),'native_launch_attempted':False,
      'native_execution':False,'universal_native_equality_proved':False,'numeric_acceptance':False,
      'selected_software_profile':selected,'environment_binding_complete':False,'python':sys.version,'platform':platform.platform(),
      'scope':r['scope'],'exit_code_meaning':'0 means compilation/capture completed, never universal numeric PASS; 1 selected-profile mismatch or capture/compiler error; 2 unavailable native execution'}
    code=2;environment=None
    try:
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        if mode=='run':
            import torch
            report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            need(torch.cuda.is_available(),'CUDA unavailable; no native launch')
            torch.cuda.set_device(device);idx=torch.cuda.current_device();prop=torch.cuda.get_device_properties(idx)
            arch=10*prop.major+prop.minor;need(arch in (80,90),'this capture is limited to previously compiled SM80/SM90 targets')
            if archs:need(archs==[arch],'requested arch does not match active device')
            archs=[arch];uuid=str(getattr(prop,'uuid','unavailable'))
            report.update(device_index=idx,device_uuid=uuid,gpu=prop.name,capability=[prop.major,prop.minor])
            try:
                q=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True)
                report['driver_query']={'command':q.args,'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
                report['environment_binding_complete']=uuid!='unavailable' and q.returncode==0 and any(line.split(',')[0].strip()==uuid for line in q.stdout.splitlines())
            except Exception as exc:report['driver_query_error']=repr(exc)
        else:need(archs and all(x in (80,90) for x in archs),'offline archs required')
        environment=compiler_environment(archs);dump(out/'compiler-environment-before.json',environment)
        module=checked_module(reference/'original_jit.py','int4_characterization_original_jit',r['jit_sha256']);fn=module.matmul_kernel
        need(fn.arg_names==r['source_extraction']['arguments'],'runtime argument names changed')
        expected_fn=(reference/'original_jit.py').read_text().split('@triton.jit\n',1)[1]
        need(fn.src==expected_fn,'Triton JIT compiled source differs from checked body')
        report['jit_function_source_sha256']=digest(fn.src.encode())
        report['jit_cache_key']=fn.cache_key
        constants=case['launch']['constexprs'];signature=signature_for(fn.arg_names,constants)
        for arch in archs:
            for fusion in (False,True):
                label=f'sm{arch}-fusion-{str(fusion).lower()}';options={'num_warps':4,'num_stages':4,'enable_fp_fusion':fusion}
                row={'label':label,'target':{'backend':'cuda','arch':arch,'warp_size':32},'signature':signature,'constexprs':constants,
                  'options':options,'status':'COMPILE_PENDING','native_execution':False};report['records'].append(row)
                kernel=triton.compile(ASTSource(fn,signature,constants),target=GPUTarget('cuda',arch,32),options=options)
                row.update(save_artifacts(kernel,out,label));need(row['resolved_metadata']['enable_fp_fusion']==fusion,'fusion metadata differs')
                row['status']='COMPILED_ONLY'
                if mode=='run':
                    allocations={};pointers={};before={};ranges=[];row['raw_captures']={}
                    for name,a in case['allocations'].items():
                        dtype=torch.uint16 if 'uint16' in a['dtype'] else torch.int32
                        buf=torch.tensor(a['raw_words'],dtype=dtype,device=device);allocations[name]=buf
                        typed=buf.view(torch.float16) if dtype==torch.uint16 else buf
                        pointers[name]=typed[a['logical_origin_elements']:]
                        start=buf.data_ptr();end=start+buf.numel()*buf.element_size();ranges.append((start,end,name))
                        words=[int(v) for v in buf.cpu().tolist()];before[name]=words
                        row['raw_captures'][name]={'before':save_words(out,label+'-before',name,words,16 if dtype==torch.uint16 else 32),
                          'allocation_byte_range':[start,end],'kernel_pointer_bytes':pointers[name].data_ptr(),'logical_origin_elements':a['logical_origin_elements']}
                        need(pointers[name].data_ptr()==start+a['logical_origin_elements']*buf.element_size() and a['logical_origin_elements']>0,'pointer-origin mismatch')
                    for i,(a,b,name) in enumerate(ranges):
                        for c,d,other in ranges[i+1:]:need(b<=c or d<=a,'unsupported overlapping allocations '+name+' '+other)
                    args=runtime_arguments(list(signature),pointers,case['runtime_args']);row['runtime_argument_order']=list(signature)
                    row['runtime_scalars']=case['runtime_args'];row['grid']=[1,1,1];row['launch_api']='CompiledKernel[(1,1,1)] with signature-ordered runtime arguments; no JIT auto-specialization'
                    report['native_launch_attempted']=True;row['status']='LAUNCH_ATTEMPTED'
                    kernel[(1,1,1)](*args);torch.cuda.synchronize();report['native_execution']=True;row['native_execution']=True
                    after={name:[int(v) for v in buf.cpu().tolist()] for name,buf in allocations.items()}
                    for name,words in after.items():row['raw_captures'][name]['after']=save_words(out,label+'-after',name,words,16 if allocations[name].dtype==torch.uint16 else 32)
                    row['before_words']=before;row['after_words']=after;row['comparison']=oracle.compare(case,before,after,refs,selected)
                    row['status']='CHARACTERIZATION'
                dump(out/'running-report.json',report)
        if mode=='compile':report['status']='COMPILED_ONLY';code=0
        else:
            report['status']='CHARACTERIZATION';code=1 if selected and any(x['comparison']['selected_profile_result']=='MISMATCH_ON_FIXED_CASE' for x in report['records']) else 0
            report['cuda_loaded_shared_objects']={}
            if Path('/proc/self/maps').exists():
                for line in Path('/proc/self/maps').read_text().splitlines():
                    path=line.split()[-1]
                    if path.startswith('/') and ('libcuda' in path or 'libnvidia' in path) and Path(path).is_file():report['cuda_loaded_shared_objects'][path]=sha(Path(path))
    except Exception as exc:
        report['status']='EXECUTION_ERROR' if report['native_launch_attempted'] else ('COMPILE_FAILED' if mode=='compile' else 'NOT_RUN')
        report['error']=repr(exc);report['traceback']=traceback.format_exc();code=1 if mode=='compile' or report['native_launch_attempted'] else 2
    finally:
        try:
            validate(reference);validate(out/'reference')
            if environment:
                changed=[p for p,h in environment['file_sha256'].items() if sha(p)!=h]
                dump(out/'compiler-environment-after.json',{'files_checked':len(environment['file_sha256']),'changed':changed,'before_sha256':sha(out/'compiler-environment-before.json')})
                need(not changed,'compiler source/tool drift')
        except Exception as exc:
            report['status']='BINDING_ERROR';report['binding_error']=repr(exc);code=1
        report['exit_code']=code
        report['artifact_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(x in p.relative_to(out).parts for x in ['cache','pycache','__pycache__'])}
        dump(out/'report.json',report)
    return code,{'status':report['status'],'native_execution':report['native_execution'],'native_launch_attempted':report['native_launch_attempted'],'records':len(report['records']),'output':str(out)}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--proof-package',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for mode in ['compile','run']:
        q=sub.add_parser(mode);q.add_argument('--reference',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        q.add_argument('--arch',type=int,action='append');q.add_argument('--compare-profile',choices=PROFILES);q.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.proof_package.resolve(),a.output.resolve())));return 0
    code,result=collect(a.reference.resolve(),a.output.resolve(),a.mode,a.arch,a.compare_profile,a.device);print(json.dumps(result));return code
if __name__=='__main__':raise SystemExit(main())
