#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
m=types.ModuleType('rbe_report_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m
exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)
u=m.u;c=m.c

def verify(out):
    r=u.load(out/'report.json');ref,cases=m.validate(out/'reference')
    u.need(r['schema']==1 and r['task']=='rbe_triton_transform','wrong task/schema')
    u.need(r['numeric_acceptance'] is False and r['numeric_comparison']=='NOT_ESTABLISHED','unsupported numeric claim')
    u.need(r['reference_sha256']==u.sha(out/'reference/reference.json') and r['tools_sha256']==m.tools(),'reference/tools differ')
    for rel,h in r['artifact_sha256'].items():
        p=(out/rel).resolve();u.need(out.resolve() in p.parents and p.is_file() and u.sha(p)==h,'artifact differs '+rel)
    for name,h in r['tools_sha256'].items():u.need(u.sha(out/'tools'/name)==h,'captured tool differs')
    plan=u.load(out/'preselected-plan.json');u.need(u.sha(out/'preselected-plan.json')==r['preselected_plan_sha256'],'preselection differs')
    u.need(plan['reference_sha256']==r['reference_sha256'] and plan['mode']==r['mode'] and plan['profiles']==c.PROFILES and plan['case_ids']==list(cases) and plan['fusion_values']==[False,True] and plan['numeric_reference']=='NOT_ESTABLISHED' and plan['no_posthoc_profile_selection'] is True,'plan incomplete/changed')
    requested=plan['requested_archs']
    u.need(requested is None or (isinstance(requested,list) and requested and
      len(set(requested))==len(requested) and all(type(a) is int and a in (80,90) for a in requested)),
      'invalid requested architectures')
    if r['mode']=='compile':u.need(requested is not None,'compile requires preselected architectures')
    u.need(r.get('environment_binding_complete') is m.environment_binding_complete(r),
      'environment binding flag lacks matching native UUID/driver evidence')
    if r['status']=='NOT_RUN':
        u.need(r['mode']=='run' and r['exit_code']==2 and not r['records'] and r['native_execution'] is False and r['native_launch_attempted'] is False,'NOT_RUN falsely contains execution')
        return {'status':'VERIFIED_NOT_RUN','records':0,'numeric_acceptance':False}
    u.need(r['mode'] in ('compile','run') and r['status'] in ('COMPILED_ONLY','CHARACTERIZATION_NUMERICS_UNBOUND'),'failed/unrecognized capture status')
    resolved=r['resolved_plan'];archs=resolved['archs'];items=list(c.PROFILES) if r['mode']=='compile' else list(cases)
    u.need(archs and len(set(archs))==len(archs) and all(a in (80,90) for a in archs),'wrong targets')
    if r['mode']=='compile' or requested is not None:
        u.need(archs==requested,'resolved architectures differ from preselected request')
    u.need(resolved['items']==items and resolved['expected_records']==len(archs)*len(items)*2,'plan coverage differs')
    expected={(a,i,f) for a in archs for i in items for f in (False,True)}
    keys=[(x['target']['arch'],x['item'],x['options']['enable_fp_fusion']) for x in r['records']]
    u.need(len(keys)==len(set(keys)) and set(keys)==expected,'missing/duplicate records')
    for row in r['records']:
        k=c.PROFILES[row['item']] if r['mode']=='compile' else cases[row['item']]['constexprs'];fusion=row['options']['enable_fp_fusion']
        u.need(row['target']=={'backend':'cuda','arch':row['target']['arch'],'warp_size':32} and row['constexprs']==k and row['options']=={'enable_fp_fusion':fusion},'source configuration differs')
        sig={n:('*fp16' if n in ['x_ptr','out_ptr'] else 'i32') for n in ref['source_extraction']['kernel_arguments'] if n not in k}
        u.need(row['signature']==sig and row['resolved_metadata']['enable_fp_fusion']==fusion,'signature/metadata differs')
        u.need(all(row['label']+'.'+x in row['artifact_sha256'] for x in ['ttir','ttgir','llir','ptx','cubin']),'missing full compiler graph')
        for rel,h in row['artifact_sha256'].items():u.need(r['artifact_sha256'].get(rel)==h and u.sha(out/rel)==h,'unbound compiler artifact')
        if r['mode']=='compile':u.need(row['status']=='COMPILED_ONLY' and row['native_execution'] is False and 'comparison' not in row,'offline promoted');continue
        case=cases[row['item']]
        u.need(row['status']=='CHARACTERIZATION_NUMERICS_UNBOUND' and row['native_execution'] is True,'missing native record')
        u.need(row['runtime_scalars']==case['runtime_args'] and row['runtime_argument_order']==list(sig) and row['grid']==case['grid'],'actual launch differs')
        u.need(set(row['raw_captures'])==set(case['allocations']) and set(row['pointers'])==set(case['pointers']),'memory capture missing')
        ranges=[]
        for name,a in case['allocations'].items():
            cap=row['raw_captures'][name];start,end=cap['allocation_byte_range'];ranges.append((start,end))
            u.need(end-start==2*len(a['raw_words']) and cap['word_bits']==16,'physical size differs')
            for phase in ['before','after']:
                raw=cap[phase];words=row[phase+'_words'][name]
                u.need(raw['word_bits']==16 and raw['words']==len(words) and raw['byte_order']=='little','raw metadata differs')
                u.need(r['artifact_sha256'].get(raw['path'])==raw['sha256'] and u.sha(out/raw['path'])==raw['sha256'] and (out/raw['path']).read_bytes()==u.raw_bytes(words,16),'raw binary differs')
        for i,(a,b) in enumerate(ranges):
            for x,y in ranges[i+1:]:u.need(b<=x or y<=a,'unexpected distinct allocation overlap')
        for name,p in case['pointers'].items():
            actual=row['pointers'][name];base=row['raw_captures'][p['allocation']]['allocation_byte_range'][0]
            u.need(actual=={**p,'pointer_bytes':base+2*p['origin_elements'],'pointer_dtype':'torch.float16','element_bytes':2},'origin/alias map differs')
        u.need(row['comparison']==c.check_capture(case,row['before_words'],row['after_words']),'comparison/frame not independently reproduced')
    env=u.load(out/'compiler-environment-before.json');after=u.load(out/'compiler-environment-after.json')
    u.need(after['changed']==[] and after['files_checked']==len(env['file_sha256']) and after['before_sha256']==u.sha(out/'compiler-environment-before.json'),'compiler environment drift')
    if r['mode']=='compile':u.need(r['status']=='COMPILED_ONLY' and r['exit_code']==0 and r['native_execution'] is False and r['native_launch_attempted'] is False,'offline claim wrong')
    else:
        code=int(any(x['comparison']['frame_violations'] for x in r['records']))
        u.need(r['status']=='CHARACTERIZATION_NUMERICS_UNBOUND' and r['exit_code']==code and r['native_execution'] is True and r['native_launch_attempted'] is True,'native status mismatch')
        u.need(len(archs)==1 and archs[0]==10*r['capability'][0]+r['capability'][1],'actual GPU target mismatch')
    return {'status':'VERIFIED_'+r['status'],'records':len(r['records']),'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();print(json.dumps(verify(a.capture.resolve())))
