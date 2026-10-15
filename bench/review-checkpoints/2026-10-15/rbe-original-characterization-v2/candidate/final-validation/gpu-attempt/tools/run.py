#!/usr/bin/env python3
"""Full original RBE capture. Numeric output comparison remains explicitly unbound."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,traceback,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BINDING_SHA='332eaa6059d9106ff9c42995a46f8ba0e143be9ac0db00573b1b0e9b1f01bb0e'
TOOL_NAMES=['run.py','contracts.py','support.py','compile_probe.py']
LOADED={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in TOOL_NAMES}
def module(name):
    p=HERE/(name+'.py');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==LOADED[p.name]
    m=types.ModuleType('rbe_checked_'+name);m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(b,str(p),'exec'),m.__dict__);return m
u=module('support');c=module('contracts');probe=module('compile_probe')
def tools():
    u.need({n:u.sha(HERE/n) for n in TOOL_NAMES}==LOADED,'loaded tool drift');return LOADED.copy()
def validate_inputs(root):
    u.need(u.sha(root/'source-bindings.json')==BINDING_SHA,'wrong bound task input manifest')
    b=u.load(root/'source-bindings.json')
    for rel,h in b['task_source_sha256'].items():
        p=(root/'task'/rel).resolve();u.need((root/'task').resolve() in p.parents and p.is_file() and u.sha(p)==h,'task source drift '+rel)
    u.need(b['materialized_support_sha256']==u.sha(HERE/'support.py'),'reused helper bytes differ');return b

def prepare(inputs,out):
    b=validate_inputs(inputs);u.need(not out.exists(),'fresh output required');out.mkdir(parents=True);shutil.copytree(inputs,out/'inputs')
    original=(inputs/'task/rbe_triton_transform.py').read_bytes();jit,extract=probe.extract(original)
    (out/'original.py').write_bytes(original);(out/'original_jit.py').write_bytes(jit);u.dump(out/'cases.json',c.cases())
    reference={'schema':1,'task':'rbe_triton_transform','tools_sha256':tools(),'source_bindings_sha256':BINDING_SHA,
      'source_sha256':u.digest(original),'jit_sha256':u.digest(jit),'source_extraction':extract,'cases_sha256':u.sha(out/'cases.json'),
      'profiles':c.PROFILES,'numeric_acceptance':False,'native_numeric_reference':'NOT_ESTABLISHED',
      'claim':'Full original kernel with original helper; no supplied cos/sin, no replaced pow. Finite experiment corpus does not shrink the general backend theorem.'}
    u.dump(out/'reference.json',reference);validate(out);return {'status':'REFERENCE_PREPARED','cases':len(c.cases()),'numeric_reference':'NOT_ESTABLISHED'}
def validate(root):
    r=u.load(root/'reference.json');b=validate_inputs(root/'inputs')
    u.need(r['schema']==1 and r['task']=='rbe_triton_transform','bad schema');u.need(r['tools_sha256']==tools(),'tools differ')
    u.need(r['source_bindings_sha256']==BINDING_SHA and r['source_sha256']==b['task_source_sha256']['rbe_triton_transform.py']==u.sha(root/'original.py'),'original differs')
    jit,extract=probe.extract((root/'original.py').read_bytes());u.need(u.digest(jit)==u.sha(root/'original_jit.py')==r['jit_sha256'] and extract==r['source_extraction'],'JIT body/helper differs')
    cases=u.load(root/'cases.json');u.need(u.sha(root/'cases.json')==r['cases_sha256'] and cases==c.cases(),'case/footprint differs')
    u.need(r['profiles']==c.PROFILES and r['numeric_acceptance'] is False and r['native_numeric_reference']=='NOT_ESTABLISHED','unsupported numeric claim')
    return r,cases

DRIVER_QUERY_COMMAND=['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader']
def environment_binding_complete(report):
    # A complete native environment binding is separate from numerical success.
    # Offline and nonexecuted reports cannot claim it. Recompute from evidence,
    # never accept a report's environment_binding_complete flag as its premise.
    if report.get('mode')!='run' or report.get('native_execution') is not True:return False
    uuid=report.get('device_uuid');query=report.get('driver_query',{})
    if not isinstance(uuid,str) or not uuid or uuid=='unavailable' or not isinstance(query,dict):return False
    if query.get('command')!=DRIVER_QUERY_COMMAND or query.get('exit_code')!=0:return False
    output=query.get('stdout')
    if not isinstance(output,str):return False
    for line in output.splitlines():
        columns=[x.strip() for x in line.split(',')]
        if len(columns)==3 and columns[0]==uuid and columns[1] and columns[2]:return True
    return False

def collect(reference,out,mode,archs,device):
    r,cases=validate(reference);u.need(not out.exists(),'fresh output required');out.mkdir(parents=True);shutil.copytree(reference,out/'reference');(out/'tools').mkdir()
    for n in tools():shutil.copy2(HERE/n,out/'tools'/n)
    os.environ['TRITON_CACHE_DIR']=str(out/'cache');sys.pycache_prefix=str(out/'pycache')
    report={'schema':1,'task':'rbe_triton_transform','mode':mode,'status':'NOT_RUN','records':[],
      'reference_sha256':u.sha(reference/'reference.json'),'tools_sha256':tools(),'native_launch_attempted':False,'native_execution':False,
      'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED','environment_binding_complete':False,
      'python':sys.version,'platform':platform.platform(),'exit_code_meaning':'0 compile/capture complete only; 1 compile/launch/binding/frame failure; 2 no GPU execution; no exit establishes numeric correctness'}
    u.dump(out/'preselected-plan.json',{'reference_sha256':report['reference_sha256'],'mode':mode,'requested_archs':archs,'profiles':c.PROFILES,
      'case_ids':list(cases),'fusion_values':[False,True],'numeric_reference':'NOT_ESTABLISHED','no_posthoc_profile_selection':True})
    report['preselected_plan_sha256']=u.sha(out/'preselected-plan.json');code=2;environment=None;started=False
    try:
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        if mode=='run':
            import torch
            report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            u.need(torch.cuda.is_available(),'CUDA unavailable; no native launch');torch.cuda.set_device(device);idx=torch.cuda.current_device();prop=torch.cuda.get_device_properties(idx)
            arch=10*prop.major+prop.minor;u.need(arch in (80,90),'this offline source review binds SM80/SM90 only')
            if archs:u.need(archs==[arch],'requested target differs from actual device')
            archs=[arch];uuid=str(getattr(prop,'uuid','unavailable'));report.update(device_index=idx,device_uuid=uuid,gpu=prop.name,capability=[prop.major,prop.minor])
            try:
                x=subprocess.run(DRIVER_QUERY_COMMAND,capture_output=True,text=True)
                report['driver_query']={'command':x.args,'exit_code':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
            except Exception as e:report['driver_query_error']=repr(e)
        else:u.need(archs and len(set(archs))==len(archs) and all(a in (80,90) for a in archs),'unique SM80/SM90 archs required')
        environment=u.compiler_environment(archs);u.dump(out/'compiler-environment-before.json',environment)
        mod=u.checked_module(reference/'original_jit.py','rbe_original_native',r['jit_sha256']);fn=mod.rbe_triton
        u.need(fn.arg_names==r['source_extraction']['kernel_arguments'],'runtime names differ')
        # Both source functions were AST-checked; JIT source binding covers their exact executable bodies.
        report['jit_functions']={n:{'source_sha256':u.digest(getattr(mod,n).src.encode()),'cache_key':getattr(mod,n).cache_key} for n in ['get_freq_multi_tokens','rbe_triton']}
        work=list(c.PROFILES.items()) if mode=='compile' else [(cid,case['constexprs']) for cid,case in cases.items()]
        report['resolved_plan']={'archs':archs,'items':[n for n,_ in work],'expected_records':len(archs)*len(work)*2}
        for arch in archs:
          for item,constants in work:
           for fusion in [False,True]:
            label=f'sm{arch}-{item}-fusion-{str(fusion).lower()}';options={'enable_fp_fusion':fusion}
            sig={n:('*fp16' if n in ('x_ptr','out_ptr') else 'i32') for n in fn.arg_names if n not in constants}
            row={'label':label,'item':item,'target':{'backend':'cuda','arch':arch,'warp_size':32},'constexprs':constants,'signature':sig,'options':options,'status':'COMPILE_PENDING','native_execution':False};report['records'].append(row)
            started=True;kernel=triton.compile(ASTSource(fn,sig,constants),target=GPUTarget('cuda',arch,32),options=options)
            row.update(u.save_artifacts(kernel,out,label));u.need(row['resolved_metadata']['enable_fp_fusion']==fusion,'fusion metadata differs');row['status']='COMPILED_ONLY'
            if mode=='run':
                case=cases[item];alloc={};before={};row['raw_captures']={};ranges=[]
                for name,a in case['allocations'].items():
                    buf=torch.tensor(a['raw_words'],dtype=torch.uint16,device=device);alloc[name]=buf;words=[int(v) for v in buf.cpu().tolist()]
                    u.need(words==a['raw_words'],'device initial storage differs');before[name]=words;start=buf.data_ptr();end=start+2*buf.numel();ranges.append((start,end))
                    row['raw_captures'][name]={'allocation_byte_range':[start,end],'word_bits':16,'before':u.save_words(out,label+'-before',name,words,16)}
                for i,(a,b) in enumerate(ranges):
                    for x,y in ranges[i+1:]:u.need(b<=x or y<=a,'unexpected overlap of distinct allocations')
                pointers={};row['pointers']={}
                for name,p in case['pointers'].items():
                    buf=alloc[p['allocation']];ptr=buf.view(torch.float16)[p['origin_elements']:];pointers[name]=ptr
                    u.need(ptr.data_ptr()==buf.data_ptr()+2*p['origin_elements'] and p['origin_elements']==8,'physical origin mismatch')
                    row['pointers'][name]={**p,'pointer_bytes':ptr.data_ptr(),'pointer_dtype':str(ptr.dtype),'element_bytes':2}
                u.need(set(sig)==set(pointers)|set(case['runtime_args']),'runtime signature coverage mismatch')
                args=[pointers[n] if n in pointers else case['runtime_args'][n] for n in sig]
                row.update(runtime_argument_order=list(sig),runtime_scalars=case['runtime_args'],grid=case['grid'],before_words=before,
                  launch_api='CompiledKernel[full_original_grid] with explicitly ordered runtime arguments; original helper executes inside kernel')
                report['native_launch_attempted']=True;row['status']='LAUNCH_ATTEMPTED';kernel[tuple(case['grid'])](*args);torch.cuda.synchronize()
                report['native_execution']=True;row['native_execution']=True
                after={n:[int(v) for v in buf.cpu().tolist()] for n,buf in alloc.items()}
                for n,words in after.items():row['raw_captures'][n]['after']=u.save_words(out,label+'-after',n,words,16)
                row['after_words']=after;row['comparison']=c.check_capture(case,before,after);row['status']='CHARACTERIZATION_NUMERICS_UNBOUND'
            u.dump(out/'running-report.json',report)
        u.need(len(report['records'])==report['resolved_plan']['expected_records'],'incomplete plan')
        if mode=='compile':report['status']='COMPILED_ONLY';code=0
        else:
            report['status']='CHARACTERIZATION_NUMERICS_UNBOUND';code=int(any(x['comparison']['frame_violations'] for x in report['records']))
            report['cuda_loaded_shared_objects']={}
            if Path('/proc/self/maps').exists():
                for line in Path('/proc/self/maps').read_text().splitlines():
                    p=line.split()[-1]
                    if p.startswith('/') and ('libcuda' in p or 'libnvidia' in p) and Path(p).is_file():report['cuda_loaded_shared_objects'][p]=u.sha(Path(p))
    except Exception as e:
        report['status']='EXECUTION_ERROR' if report['native_launch_attempted'] else ('COMPILE_FAILED' if started or mode=='compile' else 'NOT_RUN')
        report.update(error=repr(e),traceback=traceback.format_exc());code=1 if started or mode=='compile' or report['native_launch_attempted'] else 2
    finally:
        try:
            validate(reference);validate(out/'reference');u.need(u.sha(out/'preselected-plan.json')==report['preselected_plan_sha256'],'plan drift')
            if environment:
                changed=[p for p,h in environment['file_sha256'].items() if not Path(p).is_file() or u.sha(Path(p))!=h]
                u.dump(out/'compiler-environment-after.json',{'files_checked':len(environment['file_sha256']),'changed':changed,'before_sha256':u.sha(out/'compiler-environment-before.json')});u.need(not changed,'compiler environment drift')
        except Exception as e:report['status']='BINDING_ERROR';report['binding_error']=repr(e);code=1
        report['environment_binding_complete']=environment_binding_complete(report)
        report['exit_code']=code;report['artifact_sha256']={p.relative_to(out).as_posix():u.sha(p) for p in out.rglob('*') if p.is_file() and not any(x in p.relative_to(out).parts for x in ['cache','pycache','__pycache__'])};u.dump(out/'report.json',report)
    return code,{'status':report['status'],'records':len(report['records']),'native_execution':report['native_execution'],'numeric_acceptance':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--inputs',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for mode in ['compile','run']:
        q=sub.add_parser(mode);q.add_argument('--reference',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--arch',type=int,action='append');q.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.inputs.resolve(),a.output.resolve())));return 0
    code,r=collect(a.reference.resolve(),a.output.resolve(),a.mode,a.arch,a.device);print(json.dumps(r));return code
if __name__=='__main__':raise SystemExit(main())
