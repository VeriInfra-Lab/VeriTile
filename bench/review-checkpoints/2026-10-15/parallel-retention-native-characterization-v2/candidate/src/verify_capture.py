#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
m=types.ModuleType('pra_verify_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m;exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)
u=m.u;c=m.c

def verify(out):
    r=u.load(out/'report.json');ref,cs=m.validate(out/'reference')
    u.need(r['schema']==1 and r['task']=='parallel_retention_attention' and r['numeric_acceptance'] is False and r['numeric_comparison']=='NOT_ESTABLISHED','unsupported report claims')
    u.need(r['reference_sha256']==u.sha(out/'reference/reference.json') and r['tools_sha256']==m.tools(),'reference/tool binding differs')
    for rel,h in r['artifact_sha256'].items():
        p=(out/rel).resolve();u.need(out.resolve() in p.parents and p.is_file() and u.sha(p)==h,'artifact differs '+rel)
    for n,h in r['tools_sha256'].items():u.need(u.sha(out/'tools'/n)==h,'captured tool differs')
    p=u.load(out/'preselected-plan.json');u.need(u.sha(out/'preselected-plan.json')==r['preselected_plan_sha256'],'plan differs');ids,stages=m.selection(cs,p['selected_cases'],p['selected_stages'])
    u.need(p['mode']==r['mode'] and p['reference_sha256']==r['reference_sha256'] and p['fusion_values']==[False,True] and p['numeric_reference']=='NOT_ESTABLISHED' and p['host_partial_reductions'] is False,'plan claim differs')
    u.need(p['full_corpus_selected']==(ids==list(cs) and stages==list(m.STAGES)),'partial corpus promoted')
    requested=p['requested_archs']
    u.need(requested is None or (isinstance(requested,list) and requested and
      len(set(requested))==len(requested) and all(type(a) is int and a in (80,90) for a in requested)),
      'invalid requested architectures')
    if r['mode']=='compile':u.need(requested is not None,'compile requires preselected architectures')
    u.need(r.get('environment_binding_complete') is m.environment_binding_complete(r),
      'environment binding flag lacks matching native UUID/driver evidence')
    if r['status']=='NOT_RUN':
        u.need(r['mode']=='run' and r['exit_code']==2 and not r['records'] and r['native_execution'] is False and r['native_launch_attempted'] is False and r['native_launches_completed']==0,'invalid NOT_RUN')
        return {'status':'VERIFIED_NOT_RUN','records':0,'numeric_acceptance':False}
    u.need(r['mode'] in ('compile','run') and r['status'] in ('COMPILED_ONLY','CHARACTERIZATION_NUMERICS_UNBOUND'),'failed capture retained')
    if r['mode']=='compile':work={n:(p['entrypoint'],p['constexprs'],p['options'],None,None) for n,p in m.probe.profiles().items()}
    else:work={cid+'-'+stage:(m.STAGES[stage],c.constants(cs[cid],stage),cs[cid]['options'],cid,stage) for cid in ids for stage in stages}
    rp=r['resolved_plan'];archs=rp['archs'];u.need(archs and len(set(archs))==len(archs) and set(archs)<={80,90},'invalid archs')
    if r['mode']=='compile' or requested is not None:
        u.need(archs==requested,'resolved architectures differ from preselected request')
    u.need(rp['items']==list(work) and rp['expected_records']==len(work)*len(archs)*2,'resolved coverage differs')
    actual=[(x['target']['arch'],x['item'],x['options']['enable_fp_fusion']) for x in r['records']];expected={(a,n,f) for a in archs for n in work for f in [False,True]}
    u.need(len(actual)==len(set(actual)) and set(actual)==expected,'missing/duplicate profiles');launches=0
    for row in r['records']:
        entry,k,opts,cid,stage=work[row['item']];opts={**opts,'enable_fp_fusion':row['options']['enable_fp_fusion']}
        u.need(row['entrypoint']==entry and row['constexprs']==k and row['options']==opts and row['case_id']==cid and row['stage']==stage,'entrypoint/configuration differs')
        u.need(row['target']=={'backend':'cuda','arch':row['target']['arch'],'warp_size':32},'target differs')
        sig={n:('*fp32' if n in c.POINTERS else 'fp32' if n=='scale' else 'i32') for n in ref['source_extraction']['arguments'][entry] if n not in k};u.need(row['signature']==sig,'signature differs')
        for n,v in opts.items():u.need(row['resolved_metadata'][n]==v,'resolved option differs')
        u.need(all(row['label']+'.'+ext in row['artifact_sha256'] for ext in ['ttir','ttgir','llir','ptx','cubin']),'compiler artifact missing')
        for rel,h in row['artifact_sha256'].items():u.need(r['artifact_sha256'].get(rel)==h and u.sha(out/rel)==h,'compiler artifact unbound')
        if r['mode']=='compile':u.need(row['status']=='COMPILED_ONLY' and row['native_execution'] is False and 'comparison' not in row,'offline promoted');continue
        case=cs[cid];lp=c.launch_plan(case,stage);u.need(row['launch_plan']==lp,'loop/load/store plan differs');names=lp['input_regions']+lp['output_regions']
        u.need(row['runtime_argument_order']==list(sig) and set(row['raw_captures'])==set(names),'capture/signature differs')
        ranges=[]
        for n in names:
            cap=row['raw_captures'][n];a=case['allocations'][n];start,end=cap['allocation_byte_range'];ranges.append((start,end))
            u.need(cap['word_bits']==32 and cap['origin_elements']==8 and cap['pointer_dtype']=='torch.float32' and cap['kernel_pointer_bytes']==start+32 and end-start==4*len(a['raw_words']),'physical allocation/origin differs')
            for phase in ['before','after']:
                b=cap[phase];words=row[phase+'_words'][n]
                u.need(b['word_bits']==32 and b['words']==len(words) and b['byte_order']=='little','raw metadata differs')
                u.need(r['artifact_sha256'].get(b['path'])==b['sha256'] and u.sha(out/b['path'])==b['sha256'] and (out/b['path']).read_bytes()==u.raw_bytes(words,32),'raw data differ')
        for i,(a,b) in enumerate(ranges):
            for x,y in ranges[i+1:]:u.need(b<=x or y<=a,'distinct allocations overlap')
        expected_calls=lp['helper_calls'] if stage=='backward_helper' else [{}];u.need(len(row['launches'])==len(expected_calls),'missing helper programs')
        for x,extra in zip(row['launches'],expected_calls):
            u.need(x=={'grid':[1,1,1] if stage=='backward_helper' else lp['grid'],'runtime_scalars':{**case['runtime_args'],**extra},'status':'COMPLETED'},'helper call or full grid differs')
        launches+=len(expected_calls);u.need(row['status']=='CHARACTERIZATION_NUMERICS_UNBOUND' and row['native_execution'] is True,'incomplete native stage')
        u.need(row['comparison']==c.check_capture(case,stage,row['before_words'],row['after_words']),'comparison differs')
    env=u.load(out/'compiler-environment-before.json');after=u.load(out/'compiler-environment-after.json')
    u.need(after['changed']==[] and after['files_checked']==len(env['file_sha256']) and after['before_sha256']==u.sha(out/'compiler-environment-before.json'),'compiler drift')
    if r['mode']=='compile':u.need(r['status']=='COMPILED_ONLY' and r['exit_code']==0 and r['native_execution'] is False and r['native_launch_attempted'] is False and r['native_launches_completed']==0,'offline claim differs')
    else:
        u.need(r['status']=='CHARACTERIZATION_NUMERICS_UNBOUND' and r['native_execution'] is True and r['native_launch_attempted'] is True and r['native_launches_completed']==launches,'native accounting differs')
        u.need(r['exit_code']==int(any(x['comparison']['frame_violations'] for x in r['records'])),'exit/frame differs');u.need(len(archs)==1 and archs[0]==10*r['capability'][0]+r['capability'][1],'GPU target differs')
    return {'status':'VERIFIED_'+r['status'],'records':len(r['records']),'native_launches_completed':launches,'numeric_acceptance':False,'numeric_comparison':'NOT_ESTABLISHED'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();print(json.dumps(verify(a.capture.resolve())))
