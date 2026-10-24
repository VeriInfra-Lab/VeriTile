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
PROBE_SHA=sha(P/'probe.py');probe=checked(P/'probe.py','logsumexp_checked_tools',PROBE_SHA)
def words(t):return t.cpu().tolist()
def save(out,name,arr,width):
 dump(out/(name+'.json'),arr);(out/(name+'.bin')).write_bytes(struct.pack('<'+('H' if width==16 else 'I')*len(arr),*(x&((1<<width)-1) for x in arr)))
def compile_one(fn,signature,constants,arch,out,label,warps=4,fusion=False):
 import triton
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 k=triton.compile(ASTSource(fn,signature=signature,constexprs=constants),target=GPUTarget('cuda',arch,32),options={'num_warps':warps,'num_stages':3,'enable_fp_fusion':fusion})
 artifacts={}
 for kind,data in k.asm.items():
  if not isinstance(data,(str,bytes)):continue
  q=out/(label+'.'+kind);q.write_bytes(data.encode() if isinstance(data,str) else data);artifacts[q.name]=sha(q)
 return k,{'signature':signature,'constexprs':constants,'options':{'num_warps':warps,'num_stages':3,'enable_fp_fusion':fusion},'artifact_sha256':artifacts,'metadata':probe.serial(k.metadata)}
def cases():
 rows=[]
 for scaled in (False,True):
  for name,x in [('A',0xbf400006),('B',0xbf400007)]:
   rows.append({'id':f'exp-{name}-scale{int(scaled)}','input_word':x,'scaled':scaled,'scale_word':0x3f800000,'grid':[3,1,1]})
 rows.append({'id':'fusion-gap','input_word':0x3f800001,'scaled':True,'scale_word':0x3f7ffffe,'grid':[3,1,1]})
 for name,s in [('zero',0),('negative-one',0xbf800000),('infinity',0x7f800000),('qnan',0x7fc00000)]:
  rows.append({'id':'scale-'+name,'input_word':0xbf400006,'scaled':True,'scale_word':s,'grid':[3,1,1]})
 rows.append({'id':'extra-all-masked-program','input_word':0xbf400006,'scaled':False,'scale_word':0x3f800000,'grid':[1,2,1],'outside_normal_host_grid':True})
 return rows

def fixture(case):
 origin=8;sentinel=0xdeadbeef;a=[0]*67;c=[sentinel]*19
 for row in range(3):a[origin+row*17+1]=case['input_word']
 active=[origin+n+d for n in range(case['grid'][0]) for d in range(case['grid'][1])]
 need(len(set(active))==len(active),'fixture output alias')
 return a,c,active

def run(out,mode,archs,device,warps,fusion):
 need(not out.exists(),'fresh output directory required');out.mkdir(parents=True);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-cache');sys.pycache_prefix=str(out/'pycache')
 bindings={'capture.py':sha(__file__),'probe.py':PROBE_SHA,'exponent_probe.py':sha(P/'exponent_probe.py'),'native/original_jit.py':sha(P/'native/original_jit.py'),'native/receipt.json':sha(P/'native/receipt.json'),'native/input-binding.json':sha(P/'native/input-binding.json')}
 original=json.loads((P/'native/input-binding.json').read_text());need(bindings['native/original_jit.py']==original['jit_sha256'],'original JIT not bound')
 need(warps in original['original_autotune_warps'],'not original autotune member')
 report={'schema':1,'status':'NOT_RUN','mode':mode,'numeric_acceptance':False,'native_execution':False,'launch_attempted':False,'records':[],'tools_sha256':bindings,'environment_binding_complete':False,'purpose':'CHARACTERIZATION of original complete Triton per-program block-LSE and a separate exponent diagnostic. No full-output numerical evaluator; no final-output mismatch inference. Host cross-block torch.logsumexp is outside the old accepted scope.'}
 before=None;code=2
 try:
  if mode=='run':
   import torch
   report.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
   if not torch.cuda.is_available():report['reason']='CUDA unavailable; no original or diagnostic launch';return code
   torch.cuda.set_device(device);prop=torch.cuda.get_device_properties(torch.cuda.current_device());arch=10*prop.major+prop.minor
   need(arch in (80,90),'architecture absent from offline profiles');need(not archs or archs==[arch],'arch/device mismatch');archs=[arch]
   uuid=str(getattr(prop,'uuid','unavailable'));report.update(gpu=prop.name,device_uuid=uuid,capability=[prop.major,prop.minor],device=device)
   try:
    q=subprocess.run(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader'],text=True,capture_output=True)
    report['driver_query']={'command':q.args,'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr};report['environment_binding_complete']=uuid!='unavailable' and q.returncode==0 and any(x.split(',')[0].strip()==uuid for x in q.stdout.splitlines())
   except Exception as e:report['driver_query_error']=repr(e)
  need(archs and all(x in (80,90) for x in archs),'SM80/90 required')
  before=probe.environment();dump(out/'environment-before.json',before)
  diagnostic=checked(P/'exponent_probe.py','logsumexp_diagnostic',bindings['exponent_probe.py']).exponent_probe
  source=checked(P/'native/original_jit.py','logsumexp_original',bindings['native/original_jit.py']).logsumexp_fwd_kernel
  for arch in archs:
   diagnostic_kernel,record=compile_one(diagnostic,{'Input':'*u32','Scaled':'*u32','Output':'*u32'},{'N':2},arch,out,f'sm{arch}-diagnostic',warps,fusion);record.update(kind='SEPARATE_EXP_DIAGNOSTIC',arch=arch)
   if mode=='run':
    origin=8;sentinel=0xdeadbeef;a=[sentinel]*18;a[8:10]=[0xbf400006,0xbf400007]
    A=torch.tensor(a,dtype=torch.uint32,device=device);S=torch.full((18,),sentinel,dtype=torch.uint32,device=device);O=S.clone()
    report['launch_attempted']=True;diagnostic_kernel[(1,1,1)](A[origin:],S[origin:],O[origin:]);torch.cuda.synchronize();report['native_execution']=True
    aw,sw,ow=words(A),words(S),words(O)
    for n,w in [('input-before',a),('input-after',aw),('scaled',sw),('exp',ow)]:save(out,'diagnostic-'+n,w,32)
    record.update(pointer_origin=origin,input_words=a[8:10],scaled_words=sw[8:10],ex2_words=ow[8:10],input_preserved=aw==a,scaled_matches_proved_collision=sw[8:10]==[0xbf8a7fb1]*2,frame_preserved=all(sw[i]==ow[i]==sentinel for i in range(18) if i not in (8,9)),agrees_with_both_direct_RN_exp_targets=ow[8:10]==[0x3ef1da02,0x3ef1da01],ex2_equal_on_equal_input=ow[8]==ow[9])
   report['records'].append(record)
   compiled={}
   for scaled in (False,True):
    sig={'x':'*fp32','z':'*fp32'};constants={'D':17,'B':32,'HAS_SCALE':scaled}
    if scaled:sig['scale']='fp32'
    else:constants['scale']=None
    k,r=compile_one(source,sig,constants,arch,out,f'sm{arch}-original-scale{int(scaled)}',warps,fusion);r.update(kind='ORIGINAL_KERNEL_COMPILE',arch=arch)
    compiled[scaled]=(k,r);report['records'].append(r)
   if mode!='run':continue
   for case in cases():
    a,c,active=fixture(case);origin=8;k,metadata=compiled[case['scaled']]
    A=torch.tensor(a,dtype=torch.uint32,device=device);C=torch.tensor(c,dtype=torch.uint32,device=device)
    ap=A.view(torch.float32)[origin:];cp=C.view(torch.float32)[origin:];need(ap.data_ptr()!=A.data_ptr() and cp.data_ptr()!=C.data_ptr(),'nonzero pointer origin lost')
    args=[ap,cp];scalar=struct.unpack('<f',struct.pack('<I',case['scale_word']))[0]
    if case['scaled']:args.append(scalar)
    report['launch_attempted']=True;k[tuple(case['grid'])](*args);torch.cuda.synchronize();report['native_execution']=True
    actual=words(C);after=words(A)
    for name,w in [('input-before',a),('input-after',after),('output-before',c),('output-after',actual)]:save(out,case['id']+'-'+name,w,32)
    report['records'].append({'kind':'ORIGINAL_KERNEL_CAPTURE','case':case,'compiled':metadata,'pointer_origin':origin,'pointer_values':[ap.data_ptr(),cp.data_ptr()],'scale_argument_encoding':'Python host float constructed from listed binary32 word; actual scalar ABI still a separate native binding, especially NaN payload handling.','active_physical_output_indices':active,'raw_output_words':[actual[i] for i in active],'frame_preserved':all(actual[i]==c[i] for i in range(len(c)) if i not in active),'input_preserved':after==a,'expected_numerical_output':'NOT_AVAILABLE; characterization only'})
  if mode=='run':
   report.update(status='CHARACTERIZATION',native_execution=True)
   fields=['input_preserved','frame_preserved','scaled_matches_proved_collision','ex2_equal_on_equal_input']
   report['structural_or_scalar_invariants_hold']=all(r[k] for r in report['records'] for k in fields if k in r)
   if not report['structural_or_scalar_invariants_hold']:report['status']='CHARACTERIZATION_INVARIANT_VIOLATION';code=1
   else:code=0
  else:report['status']='COMPILED_ONLY';code=0
 except Exception:report.update(status='ERROR',error=traceback.format_exc());code=1
 finally:
  if before is not None:
   after=probe.environment();dump(out/'environment-after.json',after);report['compiler_environment_stable']=before==after
   if before!=after:report['status']='ERROR';code=1
  report['tools_stable']=all(sha(P/n)==h for n,h in bindings.items())
  if not report['tools_stable']:report['status']='ERROR';code=1
  report['artifacts_sha256']={q.relative_to(out).as_posix():sha(q) for q in out.iterdir() if q.is_file()};report['exit_code']=code;dump(out/'report.json',report)
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['compile','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--arch',type=int,action='append');p.add_argument('--device',default='cuda:0');p.add_argument('--num-warps',type=int,default=4);p.add_argument('--fusion',type=int,choices=[0,1],default=0);a=p.parse_args();sys.exit(run(a.output,a.mode,a.arch,a.device,a.num_warps,bool(a.fusion)))
