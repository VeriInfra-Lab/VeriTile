#!/usr/bin/env python3
"""Prepare exact FP4 references, compile original JIT bodies, or collect GPU words."""
from pathlib import Path
import argparse,ast,hashlib,json,os,platform,shutil,subprocess,sys,traceback,types,zipfile
HERE=Path(__file__).resolve().parent
# The comparison oracle is part of the evidence, so do not load unbound .pyc.
_oracle_bytes=(HERE/'oracle.py').read_bytes()
_ORACLE_LOADED_SHA=hashlib.sha256(_oracle_bytes).hexdigest()
oracle=types.ModuleType('fp4_checked_oracle');oracle.__file__=str(HERE/'oracle.py')
exec(compile(_oracle_bytes,oracle.__file__,'exec'),oracle.__dict__)
SOURCE='bench/tritonbench_g/fp4_to_bf16/fp4_to_bf16.py'
SOURCE_SHA='8571cb56ce932af6a7dd68e4047ee0863984812ac303d0d2959fc544a65e9958'
MODEL_SOURCE='bench/tritonbench_g/fp4_to_bf16/Proofs/Mathematical_FP4Bits.lean'
MODEL_SHA='6876d841a7f76a9ca8a636468147a6caaff5a4bb01d7d91e36f56af69b978545'
ARCHIVE='bench/review-checkpoints/2026-10-14/next-value-repair/fp4-unscaled/inputs.zip'
ARCHIVE_SHA='6e2e01d6ff053f57c93f847df75c16b1be9bd1586fa54b119317be59f1ee4ce3'
JIT_NAMES=['_fp4_packed_to_bf16','triton_f4_to_bf16_kernel']
CONSTANTS={'sign_mask_f4':8,'mantissa_mask_f4':1,'mbits_f4_e2m1':1,'ebits_f4_e2m1':2,
           'f4_e2m1_exp_bias':1,'mbits_f32':23,'ebits_f32':8,'f32_exp_bias':127,
           'zero_bits_f32':0,'zero_point_five_bits_f32':0x3f000000}
OPTIONS={'num_warps':4,'enable_fp_fusion':False}

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def load(p):return json.loads(p.read_text())
def need(ok,msg):
    if not ok:raise ValueError(msg)
def tools_binding():
    need(sha(HERE/'oracle.py')==_ORACLE_LOADED_SHA,'Oracle source changed after checked-byte loading')
    return {name:sha(HERE/name) for name in ['run.py','oracle.py','ModelExport.lean']}

def validate_producer(producer,table_sha):
    need(producer.get('status')=='lean_reference_exported' and producer.get('native_execution') is False,'Missing CPU Lean export receipt')
    need(producer.get('table_sha256')==table_sha and producer.get('exporter_sha256')==sha(HERE/'ModelExport.lean'),'Lean producer/exporter hash differs')
    need(producer.get('source')==MODEL_SOURCE and producer.get('source_sha256')==MODEL_SHA,'Lean source binding differs')

def extract_jit(data):
    text=data.decode();tree=ast.parse(text);lines=text.splitlines(keepends=True);chunks=[];locations=[]
    for name in JIT_NAMES:
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        need(len(nodes)==1,'Original JIT definition missing/duplicated: '+name);node=nodes[0]
        need(len(node.decorator_list)==1 and ast.unparse(node.decorator_list[0])=='triton.jit','Unexpected decorator')
        start=node.decorator_list[0].lineno;end=node.end_lineno
        chunk=''.join(lines[start-1:end]);chunks.append(chunk)
        locations.append({'function':name,'first_line':start,'last_line':end,'source_segment_sha256':sha_bytes(chunk.encode())})
    return ('import triton\nimport triton.language as tl\n\n'+'\n\n'.join(chunks)+'\n').encode(),locations

def prepare(repo,table_path,receipt_path,out):
    z=repo/ARCHIVE;need(sha(z)==ARCHIVE_SHA,'Frozen source archive changed')
    with zipfile.ZipFile(z) as archive:data=archive.read(SOURCE)
    need(sha_bytes(data)==SOURCE_SHA,'Original source SHA differs')
    producer=load(receipt_path)
    validate_producer(producer,sha(table_path))
    need(sha(repo/MODEL_SOURCE)==MODEL_SHA,'Lean source differs from checked producer')
    table=load(table_path)
    need(table.get('schema')==1 and len(table.get('rows',[]))==256,'Expected complete Lean 256-byte table')
    need([r['input'] for r in table['rows']]==list(range(256)),'Missing/duplicate/reordered input byte')
    for row in table['rows']:
        need(row['bf16']==row['software_bf16']==oracle.decode_byte(row['input']),'Independent oracle differs from Lean table')
    jit,locations=extract_jit(data);out.mkdir(parents=True,exist_ok=False)
    (out/'original.py').write_bytes(data);(out/'original_jit.py').write_bytes(jit)
    shutil.copyfile(table_path,out/'lean-table.json');shutil.copyfile(receipt_path,out/'lean-export-receipt.json')
    reference={'schema':1,'task':'fp4_to_bf16','job':'N21','source_sha256':SOURCE_SHA,
      'source_archive_sha256':ARCHIVE_SHA,'source_archive_member':SOURCE,'original_jit_sha256':sha_bytes(jit),
      'jit_locations':locations,'tools_sha256':tools_binding(),'lean_table_sha256':sha(out/'lean-table.json'),'lean_export_receipt_sha256':sha(out/'lean-export-receipt.json'),
      'constant_arguments':CONSTANTS,'requested_options':OPTIONS,'cases':oracle.cases(),
      'native_execution':False,'universal_native_equality_proved':False,
      'scope':'Original unscaled kernel, disjoint physical allocations, block16/512, guarded pointers, raw bf16 outputs and full allocated memory frame. Finite tests; no native alias, scheduling or compiler-refinement proof.',
      'not_observed':['internal intermediate words','device execution until run mode succeeds']}
    dump(out/'reference.json',reference);validate(out)
    return {'status':'REFERENCE_PREPARED_CPU_ONLY','cases':len(reference['cases']),'all256_packed_bytes_checked':True,'native_execution':False}

def validate(root):
    r=load(root/'reference.json');need(r.get('schema')==1 and r.get('task')=='fp4_to_bf16','Unknown reference')
    need(r['tools_sha256']==tools_binding(),'Runner/oracle sources changed; prepare a fresh reference')
    need(r['source_sha256']==sha(root/'original.py')==SOURCE_SHA,'Original source differs')
    need(r['source_archive_sha256']==ARCHIVE_SHA and r['source_archive_member']==SOURCE,'Source origin differs')
    jit,locations=extract_jit((root/'original.py').read_bytes())
    need(sha(root/'original_jit.py')==r['original_jit_sha256']==sha_bytes(jit) and r['jit_locations']==locations,'JIT body changed')
    need(sha(root/'lean-table.json')==r['lean_table_sha256'],'Lean table changed')
    need(sha(root/'lean-export-receipt.json')==r['lean_export_receipt_sha256'],'Lean export receipt changed')
    producer=load(root/'lean-export-receipt.json')
    validate_producer(producer,r['lean_table_sha256'])
    table=load(root/'lean-table.json')
    need(table.get('schema')==1 and [x['input'] for x in table['rows']]==list(range(256)),'Incomplete Lean table')
    for row in table['rows']:need(row['bf16']==row['software_bf16']==oracle.decode_byte(row['input']),'Lean reference mismatch')
    need(r['constant_arguments']==CONSTANTS and r['requested_options']==OPTIONS,'Launch profile changed')
    need(r['cases']==oracle.cases(),'Case corpus or expected values changed')
    need(r['native_execution'] is False and r['universal_native_equality_proved'] is False,'Reference falsely claims native execution or proof')
    return r

def module(root):
    r=validate(root);path=root/'original_jit.py';data=path.read_bytes()
    need(sha_bytes(data)==r['original_jit_sha256'],'JIT bytes changed before compilation')
    m=types.ModuleType('fp4_original_jit');m.__file__=str(path)
    sys.modules[m.__name__]=m
    # Execute the checked bytes directly. A timestamp-valid .pyc must never
    # substitute code that was not covered by the source binding.
    exec(compile(data,str(path),'exec'),m.__dict__)
    return m

def serial(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if hasattr(value,'_asdict'):return serial(value._asdict())
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return repr(value)

def artifacts(kernel,root,prefix):
    files={}
    for name,value in kernel.asm.items():
        if isinstance(value,str):data=value.encode()
        elif isinstance(value,bytes):data=value
        else:continue
        need(name.replace('_','').isalnum(),'Unsafe compiler artifact name')
        path=root/(prefix+'.'+name);path.write_bytes(data);files[path.name]=sha(path)
    return {'artifact_sha256':files,'resolved_metadata':serial(kernel.metadata)}

def execute(reference,out,compile_only,arch):
    r=validate(reference);out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(reference,out/'reference')
    (out/'tools').mkdir()
    for name in tools_binding():shutil.copyfile(HERE/name,out/'tools'/name)
    report={'schema':1,'task':'fp4_to_bf16','job':'N21','reference_sha256':sha(reference/'reference.json'),
      'tools_sha256':tools_binding(),'native_execution':False,'native_launch_attempted':False,
      'universal_native_equality_proved':False,'requested_options':OPTIONS,'records':[],
      'python':sys.version,'platform':platform.platform(),'mode':'offline_compile' if compile_only else 'gpu',
      'scope':r['scope'],'internal_operands_observed':False,'environment_binding_complete':False}
    code=2
    try:
        import triton
        report['triton']=triton.__version__
        m=module(reference)
        if compile_only:
            from triton.compiler import ASTSource
            from triton.backends.compiler import GPUTarget
            report['target']={'backend':'cuda','arch':arch,'warp_size':32}
            for block in (16,512):
                row={'block':block,'status':'COMPILE_FAILED','native_execution':False}
                report['records'].append(row)
                try:
                    constants={**CONSTANTS,'BLOCK_SIZE_IN':block}
                    src=ASTSource(m.triton_f4_to_bf16_kernel,signature={'x_ptr':'*u8','output_ptr':'*bf16','n_elements_in':'i32'},constexprs=constants)
                    kernel=triton.compile(src,target=GPUTarget('cuda',arch,32),options=OPTIONS)
                    row.update(artifacts(kernel,out,f'block{block}'));row['status']='COMPILED_ONLY'
                except Exception as exc:row['error']=repr(exc);row['traceback']=traceback.format_exc()
            code=0 if all(x['status']=='COMPILED_ONLY' for x in report['records']) else 1
            report['status']='COMPILED_ONLY' if code==0 else 'COMPILE_FAILED'
        else:
            import numpy as np
            import torch
            need(torch.cuda.is_available(),'CUDA unavailable; native execution was not run')
            report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),device_index=torch.cuda.current_device(),
                device_uuid=str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),'uuid','unavailable')),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),input_dtype='uint8',output_dtype='bfloat16')
            try:report['driver_query']=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
            except Exception as exc:report['driver_query_error']=repr(exc)
            report['environment_binding_complete']=(report['device_uuid']!='unavailable' and
                report['device_uuid'] in report.get('driver_query',''))
            for case in r['cases']:
                src=torch.tensor(case['input_words'],dtype=torch.uint8,device='cuda')
                initial=np.asarray(case['output_initial_words'],dtype=np.uint16).view(np.int16)
                dst=torch.from_numpy(initial.copy()).to('cuda').view(torch.bfloat16)
                row={'id':case['id'],'count':case['count'],'block':case['block'],'grid':case['grid'],'status':'LAUNCH_PENDING',
                     'input_origin':case['input_origin'],'output_origin':case['output_origin']}
                ip,op=src.data_ptr(),dst.data_ptr()
                input_end=ip+src.numel();output_end=op+2*dst.numel()
                need(input_end<=op or output_end<=ip,'Device allocations overlap outside supported profile')
                row['allocation_byte_ranges']={'input':[ip,input_end],'output':[op,output_end]}
                row['kernel_pointer_bytes']={'input':src[case['input_origin']:].data_ptr(),'output':dst[case['output_origin']:].data_ptr()}
                report['records'].append(row);report['native_launch_attempted']=True
                kernel=m.triton_f4_to_bf16_kernel[(case['grid'],)](src[case['input_origin']:],dst[case['output_origin']:],case['count'],**CONSTANTS,BLOCK_SIZE_IN=case['block'],**OPTIONS)
                torch.cuda.synchronize();report['native_execution']=True
                input_after=[int(x) for x in src.cpu().tolist()]
                output_after=[int(x) for x in dst.view(torch.int16).cpu().numpy().view(np.uint16)]
                row.update(input_after_words=input_after,output_after_words=output_after,
                           mismatches=oracle.compare(case,input_after,output_after))
                row.update(artifacts(kernel,out,case['id']))
                row['status']='MISMATCH' if row['mismatches'] else 'MATCH_ON_TESTED_INPUTS'
            code=0 if all(x['status']=='MATCH_ON_TESTED_INPUTS' for x in report['records']) else 1
            report['status']='MATCH_ON_TESTED_INPUTS' if code==0 else 'MISMATCH'
    except Exception as exc:
        report['status']='EXECUTION_ERROR' if report['native_launch_attempted'] else 'NOT_RUN'
        report['error']=repr(exc);report['traceback']=traceback.format_exc();code=2
    finally:
        validate(reference);validate(out/'reference')
        report['exit_code']=code;report['artifact_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        dump(out/'report.json',report)
    return code,{'status':report['status'],'output':str(out),'native_execution':report['native_execution'],'records':len(report['records'])}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--repo-root',type=Path,required=True);prep.add_argument('--lean-table',type=Path,required=True);prep.add_argument('--lean-receipt',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    for name in ('compile','run'):
        q=sub.add_parser(name);q.add_argument('--reference',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--arch',type=int,default=90)
    a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.repo_root,a.lean_table,a.lean_receipt,a.output)));return 0
    code,result=execute(a.reference,a.output,a.mode=='compile',a.arch);print(json.dumps(result));return code
if __name__=='__main__':raise SystemExit(main())
