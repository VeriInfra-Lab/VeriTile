#!/usr/bin/env python3
"""A minimal exact-word CHARACTERIZATION experiment, never an all-input numeric PASS."""
from pathlib import Path
import argparse,hashlib,json,sys,types,os,subprocess,traceback,struct
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def need(x,msg):
 if not x:raise ValueError(msg)
def checked(path,name,h):
 data=path.read_bytes();need(hashlib.sha256(data).hexdigest()==h,'source bytes changed')
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m;exec(compile(data,str(path),'exec'),m.__dict__);return m
# Use the frozen probe's environment serializer from bytes, not timestamp-sensitive .pyc.
PROBE_SHA=sha(P/'probe.py');probe=checked(P/'probe.py','llama_checked_tools',PROBE_SHA)
def words(t):return t.cpu().tolist()
def save(out,name,arr,width):
 dump(out/(name+'.json'),arr);(out/(name+'.bin')).write_bytes(struct.pack('<'+('H' if width==16 else 'I')*len(arr),*(x&((1<<width)-1) for x in arr)))
def compile_one(fn,signature,constants,arch,out,label):
 import triton
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 k=triton.compile(ASTSource(fn,signature=signature,constexprs=constants),target=GPUTarget('cuda',arch,32),options={'num_warps':4,'num_stages':3,'enable_fp_fusion':False})
 artifacts={}
 for kind,data in k.asm.items():
  if not isinstance(data,(str,bytes)):continue
  q=out/(label+'.'+kind);q.write_bytes(data.encode() if isinstance(data,str) else data);artifacts[q.name]=sha(q)
 return k,{'signature':signature,'constexprs':constants,'options':{'num_warps':4,'num_stages':3,'enable_fp_fusion':False},'artifact_sha256':artifacts,'metadata':probe.serial(k.metadata)}
def run(out,mode,archs,device):
 need(not out.exists(),'fresh output directory required');out.mkdir(parents=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-cache');sys.pycache_prefix=str(out/'pycache')
 bindings={'capture.py':sha(__file__),'probe.py':PROBE_SHA,'exponent_probe.py':sha(P/'exponent_probe.py'),'native/original_jit.py':sha(P/'native/original_jit.py'),'native/receipt.json':sha(P/'native/receipt.json')}
 original=json.loads((P/'native/input-binding.json').read_text());need(bindings['native/original_jit.py']==original['jit_sha256'],'original JIT not bound')
 report={'schema':1,'status':'NOT_RUN','mode':mode,'numeric_acceptance':False,'native_execution':False,'launch_attempted':False,'records':[],'tools_sha256':bindings,'environment_binding_complete':False,'purpose':'Characterize the known exponent-lowering collision and capture original full-row buffers. There is no full-row expected numerical evaluator; a successful capture is not a kernel numerical PASS.'}
 before=None;code=2
 try:
  if mode=='run':
   import torch
   report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
   if not torch.cuda.is_available():
    report['reason']='CUDA unavailable; neither original kernel nor diagnostic launched';return code
   torch.cuda.set_device(device);prop=torch.cuda.get_device_properties(torch.cuda.current_device());arch=10*prop.major+prop.minor
   need(arch in (80,90),'uncompiled architecture profile');need(not archs or archs==[arch],'arch/device mismatch');archs=[arch]
   uuid=str(getattr(prop,'uuid','unavailable'));report.update(gpu=prop.name,device_uuid=uuid,capability=[prop.major,prop.minor],device=device)
   try:
    q=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],text=True,capture_output=True)
    report['driver_query']={'command':q.args,'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr};report['environment_binding_complete']=uuid!='unavailable' and q.returncode==0 and any(x.split(',')[0].strip()==uuid for x in q.stdout.splitlines())
   except Exception as e:report['driver_query_error']=repr(e)
  need(archs and all(x in (80,90) for x in archs),'SM80/90 required')
  before=probe.environment();dump(out/'environment-before.json',before)
  diagnostic=checked(P/'exponent_probe.py','llama_checked_diagnostic',bindings['exponent_probe.py']).exponent_probe
  source=checked(P/'native/original_jit.py','llama_checked_original',bindings['native/original_jit.py'])._fwd_kernel_token_softmax
  for arch in archs:
   k,record=compile_one(diagnostic,{'Input':'*u32','Scaled':'*u32','Output':'*u32'},{'N':2},arch,out,f'sm{arch}-diagnostic');record.update(kind='DIAGNOSTIC_ONLY',arch=arch)
   if mode=='run':
    origin=8;sentinel=0xdeadbeef;a=[sentinel]*18;a[8:10]=[0xbf400006,0xbf400007]
    A=torch.tensor(a,dtype=torch.uint32,device=device);S=torch.full((18,),sentinel,dtype=torch.uint32,device=device);O=S.clone()
    report['launch_attempted']=True;k[(1,1,1)](A[origin:],S[origin:],O[origin:]);torch.cuda.synchronize();report['native_execution']=True
    aw,sw,ow=words(A),words(S),words(O)
    for n,w in [('input-before',a),('input-after',aw),('scaled',sw),('exp',ow)]:save(out,'diagnostic-'+n,w,32)
    record.update(pointer_origin=origin,input_words=a[8:10],scaled_words=sw[8:10],ex2_words=ow[8:10],input_preserved=aw==a,scaled_matches_proved_collision=sw[8:10]==[0xbf8a7fb1]*2,frame_preserved=all(sw[i]==ow[i]==sentinel for i in range(18) if i not in (8,9)),agrees_with_both_direct_correct_rounding_targets=ow[8:10]==[0x3ef1da02,0x3ef1da01],ex2_equal_on_equal_input=ow[8]==ow[9])
   report['records'].append(record)
   if mode!='run':continue
   for dtype,width,fmt,maxword,small in [('fp32',32,torch.float32,0,[0xbf400006,0xbf400007])]:
    sig={n:('*'+dtype if n in ['Logics','Prob_Out'] else '*i32' if n in ['B_Start_Loc','B_Seqlen'] else 'i32') for n in source.arg_names if n!='BLOCK_SIZE'}
    kernel,compiled=compile_one(source,sig,{'BLOCK_SIZE':32},arch,out,dtype+'-original')
    for case,x in enumerate(small):
     rawtype=torch.uint16 if width==16 else torch.uint32;sentinel=0xdead if width==16 else 0xdeadbeef;origin=8
     a=[0]*160;c=[sentinel]*180;start=[0]*12;length=[0]*12;start[10]=3;length[10]=2
     for head in (0,1):a[origin+head*64+6]=maxword;a[origin+head*64+8]=x
     A=torch.tensor(a,dtype=rawtype,device=device);C=torch.tensor(c,dtype=rawtype,device=device);ST=torch.tensor(start,dtype=torch.int32,device=device);LE=torch.tensor(length,dtype=torch.int32,device=device)
     pointers=[A.view(fmt)[origin:],ST[origin:],LE[origin:],C.view(fmt)[origin:]]
     need(all(t.data_ptr()!=b.data_ptr() for t,b in zip(pointers,[A,ST,LE,C])),'nonzero pointer origin lost')
     report['launch_attempted']=True;kernel[(3,2,1)](*pointers,64,2,64,3);torch.cuda.synchronize();report['native_execution']=True;actual=words(C);active=[origin+head*64+o for head in (0,1) for o in (9,12)]
     label=f'{dtype}-case{case}'
     for name,w,bits in [('input-before',a,width),('input-after',words(A),width),('output-before',c,width),('output-after',actual,width),('start-before',start,32),('start-after',words(ST),32),('length-before',length,32),('length-after',words(LE),32)]:save(out,label+'-'+name,w,bits)
     report['records'].append({'kind':'ORIGINAL_KERNEL_CAPTURE','case':label,'compiled':compiled,'grid':[3,2,1],'runtime_strides':[64,2,64,3],'pointer_origin':origin,'actual_pointer_values':[t.data_ptr() for t in pointers],'active_physical_output_indices':active,'raw_output_words':[actual[i] for i in active],'frame_preserved':all(actual[i]==c[i] for i in range(len(c)) if i not in active),'inputs_preserved':words(A)==a and words(ST)==start and words(LE)==length,'expected_numerical_output':'NOT_AVAILABLE; characterization only'})
  if mode=='run':
   report.update(status='CHARACTERIZATION',native_execution=True)
   fields=['input_preserved','inputs_preserved','frame_preserved','scaled_matches_proved_collision','ex2_equal_on_equal_input']
   report['structural_or_scalar_invariants_hold']=all(r[k] for r in report['records'] for k in fields if k in r)
   if not report['structural_or_scalar_invariants_hold']:
    report['status']='CHARACTERIZATION_INVARIANT_VIOLATION';code=1
    return code
  else:report['status']='COMPILED_ONLY'
  code=0
 except Exception:
  report.update(status='ERROR',error=traceback.format_exc());code=1
 finally:
  if before is not None:
   after=probe.environment();dump(out/'environment-after.json',after);report['compiler_environment_stable']=before==after
   if before!=after:report['status']='ERROR';code=1
  report['tools_stable']=all(sha(P/n)==h for n,h in bindings.items())
  if not report['tools_stable']:report['status']='ERROR';code=1
  report['artifacts_sha256']={q.relative_to(out).as_posix():sha(q) for q in out.iterdir() if q.is_file()};report['exit_code']=code;dump(out/'report.json',report)
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['compile','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--arch',type=int,action='append');p.add_argument('--device',default='cuda:0');a=p.parse_args();sys.exit(run(a.output,a.mode,a.arch,a.device))
