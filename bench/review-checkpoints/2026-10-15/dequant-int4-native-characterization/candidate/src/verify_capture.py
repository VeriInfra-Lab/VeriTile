#!/usr/bin/env python3
"""Recompute a characterization report's bindings and comparisons; never a native proof."""
from pathlib import Path
import argparse,hashlib,json,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
m=types.ModuleType('checked_dequant_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m
exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)

def verify(out):
    r=m.load(out/'report.json');reference,casefile,refs=m.validate(out/'reference')
    m.need(r['schema']==1 and r['task']=='matmul_dequantize_int4','wrong report schema/task')
    m.need(all(r[x] is False for x in ['numeric_acceptance','universal_native_equality_proved']),'unsupported numeric claim')
    m.need(r['reference_sha256']==m.sha(out/'reference/reference.json'),'wrong reference')
    m.need(r['tools_sha256']==m.tools_binding(),'tool binding differs')
    for rel,h in r['artifact_sha256'].items():
        p=(out/rel).resolve();m.need(out.resolve() in p.parents and p.is_file() and m.sha(p)==h,'artifact mismatch '+rel)
    for name,h in r['tools_sha256'].items():m.need(m.sha(out/'tools'/name)==h,'captured tool mismatch')
    m.need(m.sha(out/'preselected-plan.json')==r['preselected_plan_sha256'],'selection plan drift')
    plan=m.load(out/'preselected-plan.json');selected=m.checked_selection(plan['selected_profile_by_fusion'])
    m.need(plan['reference_sha256']==r['reference_sha256'] and plan['mode']==r['mode'] and selected==r['selected_profile_by_fusion'],'plan/report mismatch')
    m.need(plan['cases']==list(casefile['cases']) and plan['fusion_values']==[False,True] and plan['native_equality_claim'] is False,'wrong fixed-case plan')
    records=r['records']
    if r['status']=='NOT_RUN':
        m.need(r['mode']=='run' and r['exit_code']==2 and not records and r['native_execution'] is False and r['native_launch_attempted'] is False,'NOT_RUN contains execution or wrong exit')
        return {'status':'VERIFIED_NOT_RUN','records':0,'native_execution':False,'numeric_acceptance':False}
    m.need(r['status'] in ('COMPILED_ONLY','CHARACTERIZATION'),'failed capture retained, not verified successful')
    resolved=r['resolved_plan'];archs=resolved['archs']
    m.need(archs and len(set(archs))==len(archs) and all(a in (80,90) for a in archs),'unsupported targets')
    m.need(resolved['case_ids']==list(casefile['cases']) and resolved['fusion_values']==[False,True] and resolved['expected_records']==len(archs)*4,'wrong resolved plan')
    expected={(a,c,f) for a in archs for c in casefile['cases'] for f in (False,True)}
    actual=[(x['target']['arch'],x['case_id'],x['options']['enable_fp_fusion']) for x in records]
    m.need(len(actual)==len(set(actual)) and set(actual)==expected,'incomplete/duplicate compiler profiles')
    for row in records:
        case=casefile['cases'][row['case_id']];fusion=row['options']['enable_fp_fusion'];label=row['label']
        chosen=selected['fusion_on' if fusion else 'fusion_off']
        m.need(row['selected_profile']==chosen and row['constexprs']==case['launch']['constexprs'],'wrong selected graph or constexprs')
        m.need(row['signature']==m.signature_for(reference['source_extraction']['arguments'],row['constexprs']),'wrong signature')
        m.need(row['target']=={'backend':'cuda','arch':row['target']['arch'],'warp_size':32},'wrong target')
        m.need(row['options']=={'num_warps':4,'num_stages':4,'enable_fp_fusion':fusion},'wrong compiler options')
        for k,v in row['options'].items():m.need(row['resolved_metadata'][k]==v,'metadata differs')
        m.need(all(label+'.'+k in row['artifact_sha256'] for k in ('ttir','ttgir','llir','ptx','cubin')),'missing compiler artifacts')
        for rel,h in row['artifact_sha256'].items():m.need(r['artifact_sha256'].get(rel)==h and m.sha(out/rel)==h,'compiler artifact differs')
        if r['mode']=='compile':
            m.need(row['status']=='COMPILED_ONLY' and row['native_execution'] is False and 'comparison' not in row,'offline claim became native')
            continue
        m.need(row['status']=='CHARACTERIZATION' and row['native_execution'] is True,'incomplete native row')
        m.need(row['grid']==case['launch']['grid'] and row['runtime_scalars']==case['runtime_args'] and row['runtime_argument_order']==list(row['signature']),'actual launch differs')
        m.need(set(row['raw_captures'])==set(case['allocations']),'missing memory captures')
        ranges=[]
        for name,a in case['allocations'].items():
            capture=row['raw_captures'][name];start,end=capture['allocation_byte_range'];ranges.append((start,end))
            width=a['word_bits'];origin=a['logical_origin_elements'];size=width//8
            m.need(capture['logical_origin_elements']==origin and origin==8 and capture['element_bytes']==size,'physical origin metadata differs')
            m.need(end-start==size*len(a['raw_words']) and capture['kernel_pointer_bytes']==start+origin*size,'physical pointer/envelope differs')
            m.need(capture['pointer_dtype']==('torch.float16' if width==16 else 'torch.int32'),'pointer dtype differs')
            for phase in ('before','after'):
                words=row[phase+'_words'][name];raw=capture[phase]
                m.need(raw['word_bits']==width and raw['words']==len(words) and raw['byte_order']=='little','raw capture metadata differs')
                m.need(r['artifact_sha256'].get(raw['path'])==raw['sha256'] and m.sha(out/raw['path'])==raw['sha256'],'raw binary not bound')
                m.need((out/raw['path']).read_bytes()==m.raw_bytes(words,width),'raw bytes differ from JSON words')
        for i,(a,b) in enumerate(ranges):
            for c,d in ranges[i+1:]:m.need(b<=c or d<=a,'allocation overlap')
        comparison=m.oracle.compare(case,row['before_words'],row['after_words'],refs[row['case_id']],chosen)
        m.need(row['comparison']==comparison,'comparison differs from complete independent recomputation')
    env=m.load(out/'compiler-environment-before.json');after=m.load(out/'compiler-environment-after.json')
    m.need(after['changed']==[] and after['files_checked']==len(env['file_sha256']) and after['before_sha256']==m.sha(out/'compiler-environment-before.json'),'compiler environment drift')
    # This checks the captured before/after record. Replaying on a new host records its own actual environment.
    if r['mode']=='compile':
        m.need(r['status']=='COMPILED_ONLY' and r['exit_code']==0 and r['native_execution'] is False and r['native_launch_attempted'] is False,'offline status differs')
    else:
        m.need(r['status']=='CHARACTERIZATION' and r['exit_code']==m.selected_exit(records) and r['native_execution'] is True and r['native_launch_attempted'] is True,'selected result or status differs')
        m.need(len(archs)==1 and archs[0]==10*r['capability'][0]+r['capability'][1],'active GPU target differs')
    return {'status':'VERIFIED_'+r['status'],'records':len(records),'native_execution':r['native_execution'],'numeric_acceptance':False,'exit_code':r['exit_code']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.capture.resolve())))
