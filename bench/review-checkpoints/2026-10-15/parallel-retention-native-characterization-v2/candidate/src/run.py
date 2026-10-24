#!/usr/bin/env python3
"""Capture original retention stages and explicitly unproved shell without numerical promotion."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,traceback,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BINDING_SHA='d371d79b8987a6335ff512c21269c098c3b3a5a08a14d9685ddd6282a1a4eb14'
TOOL_NAMES=['run.py','support.py','contracts.py','compile_probe.py'];LOADED={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in TOOL_NAMES}
def module(name):
    p=HERE/(name+'.py');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==LOADED[p.name]
    m=types.ModuleType('pra_checked_'+name);m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(b,str(p),'exec'),m.__dict__);return m
u=module('support');c=module('contracts');probe=module('compile_probe')
STAGES={'forward':'parallel_retention_fwd_kernel','backward_helper':'_parallel_retention_bwd_dkv','backward_shell':'parallel_retention_bwd_kernel'}
def tools():u.need({n:u.sha(HERE/n) for n in TOOL_NAMES}==LOADED,'loaded tool drift');return LOADED.copy()
def validate_inputs(root):
    u.need(u.sha(root/'source-bindings.json')==BINDING_SHA,'task manifest differs');b=u.load(root/'source-bindings.json')
    for rel,h in b['task_source_sha256'].items():
        p=(root/'task'/rel).resolve();u.need((root/'task').resolve() in p.parents and p.is_file() and u.sha(p)==h,'task source differs '+rel)
    u.need(u.sha(HERE/'support.py')==b['reused_support_sha256'],'reused support differs');return b
def prepare(inputs,out):
    b=validate_inputs(inputs);u.need(not out.exists(),'fresh output required');out.mkdir(parents=True);shutil.copytree(inputs,out/'inputs')
    data=(inputs/'task/parallel_retention_attention.py').read_bytes();jit,ex=probe.extract(data);(out/'original.py').write_bytes(data);(out/'original_jit.py').write_bytes(jit);u.dump(out/'cases.json',c.cases())
    r={'schema':1,'task':'parallel_retention_attention','source_bindings_sha256':BINDING_SHA,'source_sha256':u.digest(data),'jit_sha256':u.digest(jit),'source_extraction':ex,
      'tools_sha256':tools(),'cases_sha256':u.sha(out/'cases.json'),'profiles':probe.profiles(),'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED',
      'accepted_stages':['forward','backward_helper'],'outside_accepted_scope':['backward_shell','DQ','host partial reductions','cross-program composition'],
      'padding':'unspecified; initial buffer data do not reveal inactive load register bits'}
    u.dump(out/'reference.json',r);validate(out);return {'status':'REFERENCE_PREPARED','cases':len(c.cases()),'numeric_comparison':'NOT_ESTABLISHED'}
def validate(root):
    r=u.load(root/'reference.json');b=validate_inputs(root/'inputs');u.need(r['schema']==1 and r['task']=='parallel_retention_attention','wrong schema')
    u.need(r['tools_sha256']==tools() and r['source_bindings_sha256']==BINDING_SHA,'tool/source binding differs')
    u.need(r['source_sha256']==b['task_source_sha256']['parallel_retention_attention.py']==u.sha(root/'original.py'),'original differs')
    jit,ex=probe.extract((root/'original.py').read_bytes());u.need(u.digest(jit)==u.sha(root/'original_jit.py')==r['jit_sha256'] and r['source_extraction']==ex,'JIT AST differs')
    cs=u.load(root/'cases.json');u.need(u.sha(root/'cases.json')==r['cases_sha256'] and cs==c.cases() and r['profiles']==probe.profiles(),'experiment differs')
    u.need(r['numeric_acceptance'] is False and r['numeric_comparison']=='NOT_ESTABLISHED' and r['accepted_stages']==['forward','backward_helper'],'unsupported classification');return r,cs

def selection(cs,case_ids,stages):
    ids=case_ids or list(cs);ss=stages or list(STAGES)
    u.need(ids and len(set(ids))==len(ids) and set(ids)<=set(cs),'invalid selected cases');u.need(ss and len(set(ss))==len(ss) and set(ss)<=set(STAGES),'invalid selected stages');return list(ids),list(ss)
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

def collect(reference,out,mode,archs,device,case_ids=None,stages=None):
    r,cs=validate(reference);ids,ss=selection(cs,case_ids,stages);u.need(not out.exists(),'fresh output required');out.mkdir(parents=True);shutil.copytree(reference,out/'reference');(out/'tools').mkdir()
    for name in tools():shutil.copy2(HERE/name,out/'tools'/name)
    os.environ['TRITON_CACHE_DIR']=str(out/'cache');sys.pycache_prefix=str(out/'pycache')
    rep={'schema':1,'task':'parallel_retention_attention','mode':mode,'status':'NOT_RUN','records':[],'reference_sha256':u.sha(reference/'reference.json'),'tools_sha256':tools(),
      'native_execution':False,'native_launch_attempted':False,'native_launches_completed':0,'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED',
      'python':sys.version,'platform':platform.platform(),'environment_binding_complete':False,
      'exit_code_meaning':'0 compile/capture complete only;1 compile/launch/binding/frame failure;2 unavailable native execution;never numeric PASS'}
    plan={'reference_sha256':rep['reference_sha256'],'mode':mode,'requested_archs':archs,'selected_cases':ids,'selected_stages':ss,'fusion_values':[False,True],
      'full_corpus_selected':ids==list(cs) and ss==list(STAGES),'numeric_reference':'NOT_ESTABLISHED','host_partial_reductions':False}
    u.dump(out/'preselected-plan.json',plan);rep['preselected_plan_sha256']=u.sha(out/'preselected-plan.json');code=2;environment=None;started=False
    try:
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        if mode=='run':
            import torch
            rep.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'));u.need(torch.cuda.is_available(),'CUDA unavailable; no native launch')
            torch.cuda.set_device(device);idx=torch.cuda.current_device();prop=torch.cuda.get_device_properties(idx);arch=prop.major*10+prop.minor;u.need(arch in (80,90),'only bound SM80/SM90 profiles supported')
            if archs:u.need(archs==[arch],'requested target differs from GPU')
            archs=[arch];uuid=str(getattr(prop,'uuid','unavailable'));rep.update(device_index=idx,device_uuid=uuid,gpu=prop.name,capability=[prop.major,prop.minor])
            try:
                q=subprocess.run(DRIVER_QUERY_COMMAND,capture_output=True,text=True);rep['driver_query']={'command':q.args,'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
            except Exception as e:rep['driver_query_error']=repr(e)
        else:u.need(archs and len(set(archs))==len(archs) and all(a in (80,90) for a in archs),'unique SM80/SM90 offline targets required')
        environment=u.compiler_environment(archs);u.dump(out/'compiler-environment-before.json',environment)
        mod=u.checked_module(reference/'original_jit.py','pra_original_native',r['jit_sha256']);rep['jit_functions']={n:{'source_sha256':u.digest(getattr(mod,n).src.encode()),'cache_key':getattr(mod,n).cache_key} for n in probe.NAMES}
        if mode=='compile':work=[(n,p['entrypoint'],p['constexprs'],p['options'],None,None) for n,p in probe.profiles().items()]
        else:work=[(cid+'-'+stage,STAGES[stage],c.constants(cs[cid],stage),cs[cid]['options'],cid,stage) for cid in ids for stage in ss]
        rep['resolved_plan']={'archs':archs,'items':[w[0] for w in work],'expected_records':len(archs)*len(work)*2}
        for arch in archs:
          for item,entry,const,baseopts,cid,stage in work:
           for fusion in [False,True]:
            label=f'sm{arch}-{item}-fusion-{str(fusion).lower()}';opts={**baseopts,'enable_fp_fusion':fusion};fn=getattr(mod,entry);sig=probe.signature(fn,const)
            row={'label':label,'item':item,'entrypoint':entry,'case_id':cid,'stage':stage,'target':{'backend':'cuda','arch':arch,'warp_size':32},'constexprs':const,'signature':sig,'options':opts,'status':'COMPILE_PENDING','native_execution':False};rep['records'].append(row)
            started=True;kernel=triton.compile(ASTSource(fn,sig,const),target=GPUTarget('cuda',arch,32),options=opts);row.update(u.save_artifacts(kernel,out,label))
            for k,v in opts.items():u.need(row['resolved_metadata'][k]==v,'compiler metadata differs')
            row['status']='COMPILED_ONLY'
            if mode=='run':
                case=cs[cid];lp=c.launch_plan(case,stage);row['launch_plan']=lp;names=lp['input_regions']+lp['output_regions'];alloc={};ptrs={};before={};ranges=[];row['raw_captures']={}
                for n in names:
                    a=case['allocations'][n];buf=torch.tensor(a['raw_words'],dtype=torch.uint32,device=device);alloc[n]=buf;ptrs[n]=buf.view(torch.float32)[a['origin_elements']:]
                    words=[int(x) for x in buf.cpu().tolist()];u.need(words==a['raw_words'],'initial device words differ');before[n]=words;start=buf.data_ptr();end=start+4*buf.numel();ranges.append((start,end))
                    u.need(ptrs[n].data_ptr()==start+4*a['origin_elements'] and a['origin_elements']==8,'origin mismatch')
                    row['raw_captures'][n]={'word_bits':32,'allocation_byte_range':[start,end],'kernel_pointer_bytes':ptrs[n].data_ptr(),'origin_elements':8,'pointer_dtype':str(ptrs[n].dtype),
                      'before':u.save_words(out,label+'-before',n,words,32)}
                for i,(a,b) in enumerate(ranges):
                    for x,y in ranges[i+1:]:u.need(b<=x or y<=a,'unexpected allocation overlap')
                row['before_words']=before;row['launches']=[];row['runtime_argument_order']=list(sig)
                calls=lp['helper_calls'] if stage=='backward_helper' else [{}]
                for extra in calls:
                    scalar={**case['runtime_args'],**extra};u.need(set(sig)==set(ptrs)|set(scalar),'runtime argument coverage differs');args=[ptrs[n] if n in ptrs else scalar[n] for n in sig];grid=[1,1,1] if stage=='backward_helper' else lp['grid']
                    launch={'grid':grid,'runtime_scalars':scalar,'status':'ATTEMPTED'};row['launches'].append(launch);rep['native_launch_attempted']=True;row['status']='LAUNCH_ATTEMPTED'
                    kernel[tuple(grid)](*args);torch.cuda.synchronize();launch['status']='COMPLETED';rep['native_launches_completed']+=1
                rep['native_execution']=True;row['native_execution']=True;after={n:[int(x) for x in b.cpu().tolist()] for n,b in alloc.items()}
                for n,words in after.items():row['raw_captures'][n]['after']=u.save_words(out,label+'-after',n,words,32)
                row['after_words']=after;row['comparison']=c.check_capture(case,stage,before,after);row['status']='CHARACTERIZATION_NUMERICS_UNBOUND'
            u.dump(out/'running-report.json',rep)
        u.need(len(rep['records'])==rep['resolved_plan']['expected_records'],'incomplete planned coverage')
        if mode=='compile':rep['status']='COMPILED_ONLY';code=0
        else:rep['status']='CHARACTERIZATION_NUMERICS_UNBOUND';code=int(any(x['comparison']['frame_violations'] for x in rep['records']))
    except Exception as e:
        rep['status']='EXECUTION_ERROR' if rep['native_launch_attempted'] else ('COMPILE_FAILED' if mode=='compile' or started else 'NOT_RUN');rep.update(error=repr(e),traceback=traceback.format_exc());code=1 if started or mode=='compile' or rep['native_launch_attempted'] else 2
    finally:
        try:
            validate(reference);validate(out/'reference');u.need(u.sha(out/'preselected-plan.json')==rep['preselected_plan_sha256'],'plan drift')
            if environment:
                changed=[p for p,h in environment['file_sha256'].items() if not Path(p).is_file() or u.sha(Path(p))!=h];u.dump(out/'compiler-environment-after.json',{'files_checked':len(environment['file_sha256']),'changed':changed,'before_sha256':u.sha(out/'compiler-environment-before.json')});u.need(not changed,'compiler environment drift')
        except Exception as e:rep['status']='BINDING_ERROR';rep['binding_error']=repr(e);code=1
        rep['environment_binding_complete']=environment_binding_complete(rep)
        rep['exit_code']=code;rep['artifact_sha256']={p.relative_to(out).as_posix():u.sha(p) for p in out.rglob('*') if p.is_file() and not any(x in p.relative_to(out).parts for x in ['cache','pycache','__pycache__'])};u.dump(out/'report.json',rep)
    return code,{'status':rep['status'],'records':len(rep['records']),'native_execution':rep['native_execution'],'native_launches_completed':rep['native_launches_completed'],'numeric_acceptance':False}
def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--inputs',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for mode in ['compile','run']:
        q=sub.add_parser(mode);q.add_argument('--reference',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--arch',type=int,action='append');q.add_argument('--device',default='cuda:0');q.add_argument('--case',action='append');q.add_argument('--stage',action='append',choices=list(STAGES))
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.inputs.resolve(),a.output.resolve())));return 0
    code,r=collect(a.reference.resolve(),a.output.resolve(),a.mode,a.arch,a.device,a.case,a.stage);print(json.dumps(r));return code
if __name__=='__main__':raise SystemExit(main())
