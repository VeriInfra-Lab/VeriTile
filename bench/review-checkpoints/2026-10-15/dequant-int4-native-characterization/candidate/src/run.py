#!/usr/bin/env python3
"""Original dequant int4 characterization; no match is universal numeric PASS."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,platform,shutil,struct,subprocess,sys,traceback,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROOF_SHA='af19336662623cf097dd5e4f5482cc280d3cc2fd15ebc691b0704f92dd3b4f2e'
ROOT_REPLAY_SHA='627fe4749d214a7e9632f07ab05466d8b3eee971a92eb2d5a24dda6a208de723'
SOURCE_SHA='163c65f3c3c3c82cd0df16dd558662a8c9161d7b2af5ce9dca672384ffd140ed'
PARENT_PINS={'original_source':'1bbeb820ae43ec841b8be2a9937762f558680b2bcc1b5e40f371d81655feb621',
 'compiled_graphs':'f940faa3769207890ab65506c7f27231db20c1fe897ca0c78cc264c05f661a0c',
 'native_capture':'2b694749fcfea47f0947a738cd7e62b22f88051f1e53c078281dccc3fff05886'}
PROFILES=('source_separate_dot_add','compiled_separate_ascending_fma','compiled_fused_ascending_fma')
POINTER_TYPES={'a_ptr':'*fp16','b_ptr':'*i32','c_ptr':'*fp16','scales_ptr':'*fp16','zeros_ptr':'*i32'}
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
_RUN_LOADED_SHA=sha(HERE/'run.py')
_ORACLE_LOADED_SHA=sha(HERE/'oracle.py')
oracle=checked_module(HERE/'oracle.py','dequant_characterization_checked_oracle',_ORACLE_LOADED_SHA)
def tools_binding():
    need(sha(HERE/'run.py')==_RUN_LOADED_SHA,'runner changed after load')
    need(sha(HERE/'oracle.py')==_ORACLE_LOADED_SHA,'oracle changed after load')
    return {n:sha(HERE/n) for n in ['run.py','oracle.py']}
def serial(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if hasattr(v,'_asdict'):return serial(v._asdict())
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return repr(v)
def extract(data):
    text=data.decode();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='matmul4_kernel']
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
def validate_evidence(root):
    need(sha(root/'receipt.json')==PROOF_SHA,'wrong legal-tile proof receipt')
    parent=load(root/'receipt.json')
    need(parent['new_module_count']==4 and parent['named_axiom_checks']==25 and parent['public_specifications']==[],'wrong witness scope')
    for rel,h in parent['artifact_sha256'].items():
        q=(root/rel).resolve();need(root.resolve() in q.parents and q.is_file() and sha(q)==h,'proof artifact mismatch '+rel)
    for label,pin in PARENT_PINS.items():need(sha(root/'parents'/label/'receipt.json')==pin,'parent receipt mismatch '+label)
    bindings=load(root/'parent-bindings.json');need({k:v['sha256'] for k,v in bindings.items()}==PARENT_PINS,'parent pins changed')
    nr=load(root/'parents/native_capture/receipt.json')
    need(nr['compile_successes']==8 and nr['native_execution'] is False and nr['source_sha256']==SOURCE_SHA,'wrong Q profiles')
    need(nr['records']==load(root/'parents/native_capture/running-results.json'),'Q records drift')
    need(nr['input_binding_sha256']==sha(root/'parents/native_capture/input-binding.json'),'Q source binding drift')
    records=load(root/'last-build.json');need(len(records)==4 and all(x['exit_code']==0 and x['changed_inputs']==[] for x in records),'failed proof build')
    return parent

def oracle_references(casefile):
    need(casefile['schema']==1 and casefile['task']=='matmul_dequantize_int4' and casefile['source_sha256']==SOURCE_SHA,'unknown case')
    need(casefile['profiles']==list(PROFILES) and set(casefile['cases'])=={'nogroups_false','nogroups_true'},'incomplete case/profile set')
    refs={}
    for cid,case in casefile['cases'].items():
        need(cid==case['case_id'] and case['launch']['constexprs']['NO_GROUPS']==(cid=='nogroups_true'),'NO_GROUPS label mismatch')
        need(all(a['logical_origin_elements']==8 for a in case['allocations'].values()),'origin must be fixed at 8 words')
        need(set(case['allocations'])==set(POINTER_TYPES),'allocation coverage')
        refs[cid]={profile:oracle.expected(case,profile) for profile in PROFILES}
        need(refs[cid]==case['expected_full_c_allocations_by_software_graph'],'independent CPU oracle differs from Lean full-C theorem')
    return refs

def validate_root_replay(path,parent):
    need(sha(path)==ROOT_REPLAY_SHA,'root independent replay receipt mismatch')
    rr=load(path)
    need(rr['status']=='passed_root_independent_four_module_case_replay' and rr['reviewer']=='root','root replay failed')
    need(rr['new_modules']==4 and rr['axiom_checks']==25 and rr['public_specs']==0,'root replay scope mismatch')
    need(rr['candidate_receipt_sha256']==PROOF_SHA,'root replay candidate mismatch')
    source={p:h for p,h in parent['artifact_sha256'].items() if p.startswith('bench/') and p.endswith('.lean')}
    need(rr['reviewed_input_sha256']==source,'root independently reviewed sources differ')

def prepare(package,out,root_replay):
    parent=validate_evidence(package);validate_root_replay(root_replay,parent);need(not out.exists(),'fresh output required');out.mkdir(parents=True)
    ev=out/'evidence';ev.mkdir()
    shutil.copy2(package/'receipt.json',ev/'receipt.json')
    shutil.copy2(root_replay,out/'root-replay-receipt.json')
    for name in {**parent['artifact_sha256'],**{'parents/'+label+'/receipt.json':pin for label,pin in PARENT_PINS.items()}}:
        dst=ev/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(package/name,dst)
    original=(package/'parents/native_capture/original.py').read_bytes();need(digest(original)==SOURCE_SHA,'original changed')
    jit,location=extract(original)
    need(jit==(package/'parents/native_capture/original_jit.py').read_bytes(),'JIT differs from Q capture')
    case=load(package/'experiment-case.json');refs=oracle_references(case)
    selected={'constants':{'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},'options':{'num_warps':4,'num_stages':4}}
    need(len(location['configs'])==17 and selected in location['configs'],'not an original autotune member')
    (out/'original.py').write_bytes(original);(out/'original_jit.py').write_bytes(jit)
    dump(out/'case.json',case);dump(out/'oracle-reference.json',refs)
    r={'schema':1,'task':'matmul_dequantize_int4','purpose':'CHARACTERIZATION_ONLY',
      'proof_receipt_sha256':PROOF_SHA,'root_replay_receipt_sha256':ROOT_REPLAY_SHA,'parent_pins':PARENT_PINS,'tools_sha256':tools_binding(),
      'source_sha256':SOURCE_SHA,'jit_sha256':digest(jit),'source_extraction':location,
      'case_sha256':sha(out/'case.json'),'oracle_reference_sha256':sha(out/'oracle-reference.json'),
      'evidence_sha256':{str(q.relative_to(out)):sha(q) for q in ev.rglob('*') if q.is_file()},
      'numeric_acceptance':False,'universal_native_equality_proved':False,'native_execution':False,
      'scope':'One original 64x128x32 tile, dense M64 N128 K64, two K blocks, both NO_GROUPS, nonzero physical origins, full C8208 words plus all inputs before/after. This case discriminates half fusion but not source-dot versus unfused live-carry order.'}
    dump(out/'reference.json',r);validate(out)
    return {'status':'REFERENCE_PREPARED_CPU_ONLY','cases':2,'profiles':3,'full_C_words_per_case':8208,'native_execution':False}

def validate(ref):
    r=load(ref/'reference.json')
    need(r['schema']==1 and r['task']=='matmul_dequantize_int4' and r['purpose']=='CHARACTERIZATION_ONLY','unknown reference')
    need(r['proof_receipt_sha256']==PROOF_SHA and r['parent_pins']==PARENT_PINS,'reference parent pins changed')
    need(r['tools_sha256']==tools_binding(),'tools differ from prepared reference')
    parent=validate_evidence(ref/'evidence')
    need(r['root_replay_receipt_sha256']==ROOT_REPLAY_SHA,'root replay pin changed');validate_root_replay(ref/'root-replay-receipt.json',parent)
    expected={'evidence/receipt.json':PROOF_SHA,**{'evidence/'+rel:h for rel,h in parent['artifact_sha256'].items()},**{'evidence/parents/'+label+'/receipt.json':pin for label,pin in PARENT_PINS.items()}}
    need(r['evidence_sha256']==expected,'evidence set differs from frozen proof')
    for name,h in expected.items():need(sha(ref/name)==h,'evidence copy changed')
    need(sha(ref/'original.py')==r['source_sha256']==SOURCE_SHA,'original changed')
    jit,location=extract((ref/'original.py').read_bytes())
    need(digest(jit)==sha(ref/'original_jit.py')==r['jit_sha256'] and location==r['source_extraction'],'JIT/AST/config drift')
    need(sha(ref/'case.json')==r['case_sha256'] and sha(ref/'oracle-reference.json')==r['oracle_reference_sha256'],'case/reference hash changed')
    case=load(ref/'case.json');need(case==load(ref/'evidence/experiment-case.json'),'case differs from Lean-bound contract')
    refs=oracle_references(case);need(refs==load(ref/'oracle-reference.json'),'oracle references changed')
    need(all(r[k] is False for k in ['numeric_acceptance','universal_native_equality_proved','native_execution']),'unsupported native claim')
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

def checked_selection(selected):
    need(isinstance(selected,dict) and set(selected)=={'fusion_off','fusion_on'},'preselect both fusion profiles before execution')
    need(all(x in PROFILES for x in selected.values()),'unsupported preselected profile')
    return dict(selected)

def selected_exit(records):
    need(records and all(x['status']=='CHARACTERIZATION' and x['native_execution'] is True for x in records),'incomplete native capture')
    return int(any(x['comparison']['selected_profile_result']!='MATCH_ON_FIXED_CASE' for x in records))

def collect(reference,out,mode,archs,selected,device):
    need(mode in ('compile','run'),'unknown collection mode');selected=checked_selection(selected)
    r,casefile,refs=validate(reference);need(not out.exists(),'fresh output required');out.mkdir(parents=True)
    shutil.copytree(reference,out/'reference');(out/'tools').mkdir()
    for name in tools_binding():shutil.copy2(HERE/name,out/'tools'/name)
    os.environ['TRITON_CACHE_DIR']=str(out/'cache');sys.pycache_prefix=str(out/'pycache')
    report={'schema':1,'task':'matmul_dequantize_int4','mode':mode,'status':'NOT_RUN','records':[],
      'reference_sha256':sha(reference/'reference.json'),'tools_sha256':tools_binding(),'native_launch_attempted':False,
      'native_execution':False,'universal_native_equality_proved':False,'numeric_acceptance':False,
      'selected_profile_by_fusion':selected,'environment_binding_complete':False,'python':sys.version,'platform':platform.platform(),
      'scope':r['scope'],'exit_code_meaning':'0 means compilation/capture completed, never universal numeric PASS; 1 selected-profile mismatch or capture/compiler/binding error; 2 unavailable native execution'}
    # Persist the comparison choice before importing the compiler or observing outputs.
    dump(out/'preselected-plan.json',{'reference_sha256':report['reference_sha256'],'mode':mode,'requested_archs':archs,
      'cases':list(casefile['cases']),'fusion_values':[False,True],'selected_profile_by_fusion':selected,'native_equality_claim':False})
    report['preselected_plan_sha256']=sha(out/'preselected-plan.json')
    code=2;environment=None;compiler_started=False
    try:
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        if mode=='run':
            import torch
            report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            need(torch.cuda.is_available(),'CUDA unavailable; no native launch')
            torch.cuda.set_device(device);idx=torch.cuda.current_device();prop=torch.cuda.get_device_properties(idx)
            arch=10*prop.major+prop.minor;need(arch in (80,90),'capture is limited to the bound SM80/SM90 profiles')
            if archs:need(archs==[arch],'requested arch does not match active device')
            archs=[arch];uuid=str(getattr(prop,'uuid','unavailable'))
            report.update(device_index=idx,device_uuid=uuid,gpu=prop.name,capability=[prop.major,prop.minor])
            try:
                q=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True)
                report['driver_query']={'command':q.args,'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
                report['environment_binding_complete']=uuid!='unavailable' and q.returncode==0 and any(line.split(',')[0].strip()==uuid for line in q.stdout.splitlines())
            except Exception as exc:report['driver_query_error']=repr(exc)
        else:need(archs and len(set(archs))==len(archs) and all(x in (80,90) for x in archs),'unique SM80/SM90 offline archs required')
        environment=compiler_environment(archs);dump(out/'compiler-environment-before.json',environment)
        module=checked_module(reference/'original_jit.py','dequant_characterization_original_jit',r['jit_sha256']);fn=module.matmul4_kernel
        need(fn.arg_names==r['source_extraction']['arguments'],'runtime argument names changed')
        expected_fn=(reference/'original_jit.py').read_text().split('@triton.jit\n',1)[1]
        need(fn.src==expected_fn,'Triton JIT compiled source differs from checked body')
        report['jit_function_source_sha256']=digest(fn.src.encode());report['jit_cache_key']=fn.cache_key
        report['resolved_plan']={'archs':archs,'case_ids':list(casefile['cases']),'fusion_values':[False,True],'expected_records':len(archs)*4}
        for arch in archs:
            for cid,case in casefile['cases'].items():
                constants=case['launch']['constexprs'];signature=signature_for(fn.arg_names,constants)
                for fusion in (False,True):
                    chosen=selected['fusion_on' if fusion else 'fusion_off']
                    label=f'sm{arch}-{cid}-fusion-{str(fusion).lower()}'
                    options={'num_warps':case['launch']['num_warps'],'num_stages':case['launch']['num_stages'],'enable_fp_fusion':fusion}
                    row={'label':label,'case_id':cid,'target':{'backend':'cuda','arch':arch,'warp_size':32},'signature':signature,'constexprs':constants,
                      'options':options,'selected_profile':chosen,'status':'COMPILE_PENDING','native_execution':False};report['records'].append(row)
                    compiler_started=True
                    kernel=triton.compile(ASTSource(fn,signature,constants),target=GPUTarget('cuda',arch,32),options=options)
                    row.update(save_artifacts(kernel,out,label));need(row['resolved_metadata']['enable_fp_fusion']==fusion,'fusion metadata differs')
                    need(row['resolved_metadata']['num_warps']==4 and row['resolved_metadata']['num_stages']==4,'launch metadata differs')
                    row['status']='COMPILED_ONLY'
                    if mode=='run':
                        allocations={};pointers={};before={};ranges=[];row['raw_captures']={}
                        for name,a in case['allocations'].items():
                            dtype=torch.uint16 if a['word_bits']==16 else torch.int32
                            buf=torch.tensor(a['raw_words'],dtype=dtype,device=device);allocations[name]=buf
                            typed=buf.view(torch.float16) if dtype==torch.uint16 else buf
                            pointers[name]=typed[a['logical_origin_elements']:]
                            start=buf.data_ptr();end=start+buf.numel()*buf.element_size();ranges.append((start,end,name))
                            words=[int(v)&((1<<a['word_bits'])-1) for v in buf.cpu().tolist()];before[name]=words
                            row['raw_captures'][name]={'before':save_words(out,label+'-before',name,words,a['word_bits']),
                              'allocation_byte_range':[start,end],'kernel_pointer_bytes':pointers[name].data_ptr(),
                              'logical_origin_elements':a['logical_origin_elements'],'pointer_dtype':str(pointers[name].dtype),'element_bytes':buf.element_size()}
                            need(pointers[name].data_ptr()==start+a['logical_origin_elements']*buf.element_size() and a['logical_origin_elements']>0,'pointer-origin mismatch')
                            need(words==a['raw_words'],'device before differs from bound initial memory')
                        for i,(a,b,name) in enumerate(ranges):
                            for c,d,other in ranges[i+1:]:need(b<=c or d<=a,'overlapping allocations '+name+' '+other)
                        args=runtime_arguments(list(signature),pointers,case['runtime_args']);row['runtime_argument_order']=list(signature)
                        row['runtime_scalars']=case['runtime_args'];row['grid']=case['launch']['grid']
                        row['launch_api']='CompiledKernel[(1,1,1)] with signature-ordered runtime arguments; no JIT auto-specialization'
                        report['native_launch_attempted']=True;row['status']='LAUNCH_ATTEMPTED'
                        kernel[tuple(case['launch']['grid'])](*args);torch.cuda.synchronize();report['native_execution']=True;row['native_execution']=True
                        after={name:[int(v)&((1<<case['allocations'][name]['word_bits'])-1) for v in buf.cpu().tolist()] for name,buf in allocations.items()}
                        for name,words in after.items():row['raw_captures'][name]['after']=save_words(out,label+'-after',name,words,case['allocations'][name]['word_bits'])
                        row['before_words']=before;row['after_words']=after
                        row['comparison']=oracle.compare(case,before,after,refs[cid],chosen);row['status']='CHARACTERIZATION'
                    dump(out/'running-report.json',report)
        need(len(report['records'])==report['resolved_plan']['expected_records'],'incomplete profile coverage')
        if mode=='compile':report['status']='COMPILED_ONLY';code=0
        else:
            report['status']='CHARACTERIZATION';code=selected_exit(report['records']);report['cuda_loaded_shared_objects']={}
            if Path('/proc/self/maps').exists():
                for line in Path('/proc/self/maps').read_text().splitlines():
                    path=line.split()[-1]
                    if path.startswith('/') and ('libcuda' in path or 'libnvidia' in path) and Path(path).is_file():report['cuda_loaded_shared_objects'][path]=sha(Path(path))
    except Exception as exc:
        report['status']='EXECUTION_ERROR' if report['native_launch_attempted'] else ('COMPILE_FAILED' if compiler_started or mode=='compile' else 'NOT_RUN')
        report['error']=repr(exc);report['traceback']=traceback.format_exc();code=1 if mode=='compile' or compiler_started or report['native_launch_attempted'] else 2
    finally:
        try:
            validate(reference);validate(out/'reference')
            need(sha(out/'preselected-plan.json')==report['preselected_plan_sha256'],'comparison selection changed during capture')
            if environment:
                changed=[p for p,h in environment['file_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
                dump(out/'compiler-environment-after.json',{'files_checked':len(environment['file_sha256']),'changed':changed,'before_sha256':sha(out/'compiler-environment-before.json')})
                need(not changed,'compiler source/tool drift')
        except Exception as exc:report['status']='BINDING_ERROR';report['binding_error']=repr(exc);code=1
        report['exit_code']=code
        report['artifact_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(x in p.relative_to(out).parts for x in ['cache','pycache','__pycache__'])}
        dump(out/'report.json',report)
    return code,{'status':report['status'],'native_execution':report['native_execution'],'native_launch_attempted':report['native_launch_attempted'],'records':len(report['records']),'output':str(out)}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--proof-package',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--root-replay',type=Path,required=True)
    q=sub.add_parser('validate');q.add_argument('--reference',type=Path,required=True)
    for mode in ['compile','run']:
        q=sub.add_parser(mode);q.add_argument('--reference',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        q.add_argument('--arch',type=int,action='append');q.add_argument('--device',default='cuda:0')
        q.add_argument('--profile-fusion-off',choices=PROFILES,required=True);q.add_argument('--profile-fusion-on',choices=PROFILES,required=True)
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.proof_package.resolve(),a.output.resolve(),a.root_replay.resolve())));return 0
    if a.mode=='validate':validate(a.reference.resolve());print(json.dumps({'status':'BOUND_REFERENCE_VALIDATED','native_execution':False}));return 0
    selected={'fusion_off':a.profile_fusion_off,'fusion_on':a.profile_fusion_on}
    code,result=collect(a.reference.resolve(),a.output.resolve(),a.mode,a.arch,selected,a.device);print(json.dumps(result));return code
if __name__=='__main__':raise SystemExit(main())
