"""Exact guard-only change and original/candidate admission captures; no kernel runs."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,json,os
from unittest.mock import patch
R=Path(__file__).resolve().parents[4]
E=R/'.veritile-cache/kernel-evidence/token_attn_llama2_grid_axes_checked'
os.environ['TMPDIR']=str(E)
os.environ['TRITON_CACHE_DIR']=str(E/'triton-cache')
os.environ['TRITON_INTERPRET']='1'
import torch
torch.set_num_threads(1)
OLD=R/'bench/repairs/tritonbench_g/token_attn_llama2/token_attn_llama2_repaired.py'
NEW=Path(__file__).with_name('token_attn_llama2_grid_axes_checked.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD)=='e2c0688906af29ede2ac51f3d15eed7c6d7828796817bc7132578ef5282c9d1c'
addition='    if heads > 65535 or triton.cdiv(max_input_len, 32) > 65535:\n        raise ValueError("CUDA grid y/z exceeds the per-axis limit")\n'
assert NEW.read_text().count(addition)==1
assert NEW.read_text().replace(addition,'')==OLD.read_text()
old_ast,new_ast=ast.parse(OLD.read_text()),ast.parse(NEW.read_text())
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
assert ast.dump(fn(old_ast,'_fwd_kernel_token_att1'))==ast.dump(fn(new_ast,'_fwd_kernel_token_att1'))
assert ast.dump(fn(old_ast,'token_att_fwd'))==ast.dump(fn(new_ast,'token_att_fwd'))

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
modules=[load(OLD,'original_llama2_axis_guard'),load(NEW,'candidate_llama2_axis_guard')]
class CudaFacade(torch.Tensor):
 @property
 def is_cuda(self):return True
f=lambda x:x.as_subclass(CudaFacade)
records=[]
for name,B,H,M,D in [
 ('ordinary',1,1,1,16),('y_limit',1,65535,1,16),('y_exceeded',1,65536,1,16),
 ('z_limit',1,1,65535*32,16),('z_exceeded',1,1,65535*32+1,16),
 ('empty_batch_both_axes_exceeded',0,65536,65535*32+1,16),
 ('empty_maximum_y_exceeded',1,65536,0,16),
 ('dimension_32',1,6,33,32),('dimension_64',1,6,33,64),('dimension_128',1,6,33,128)]:
 K=2 if H==6 else 1
 length=1 if M else 0
 T=1 if B and M else 0
 tensors=[f(torch.zeros(B,H,D)),f(torch.zeros(1,K,D)),f(torch.full((H,T),-7.)),
   f(torch.zeros(B,M,dtype=torch.int32)),f(torch.zeros(B,dtype=torch.int32)),
   f(torch.full((B,),length,dtype=torch.int32))]
 saved=[t.clone() for t in tensors]
 outcomes=[]
 for m in modules:
  calls=[]
  class Capture:
   def __getitem__(self,grid):
    def call(*args,**kw):
     assert [id(args[i]) for i in [0,1,7,3,4,5]]==[id(t) for t in tensors]
     calls.append({'grid':list(grid),'options':kw,'scale':args[2]})
    return call
  with patch.object(m,'_fwd_kernel_token_att1',Capture()),patch.object(torch.cuda,'device',lambda _:contextlib.nullcontext()):
   try:
    assert m.token_att_fwd(*tensors,M) is None
    outcomes.append({'accepted':True,'launches':calls})
   except ValueError as e:
    outcomes.append({'accepted':False,'error':str(e),'launches':calls})
  assert all(torch.equal(t,v) for t,v in zip(tensors,saved))
 expected=B==0 or M==0 or (H<=65535 and (M+31)//32<=65535)
 assert outcomes[0]['accepted']
 assert outcomes[1]['accepted']==expected
 if expected:assert outcomes[0]['launches']==outcomes[1]['launches']
 else:assert outcomes[1]['launches']==[] and outcomes[1]['error']=='CUDA grid y/z exceeds the per-axis limit'
 records.append({'case':name,'batch':B,'heads':H,'maximum':M,'dimension':D,'original':outcomes[0],'candidate':outcomes[1],
   'expected_exact_new_domain':expected,'all_buffers_unchanged':True,'same_actual_tensor_objects_forwarded':True})
report={'canonical':{'path':str(OLD),'sha256':sha(OLD)},'candidate':{'path':str(NEW),'sha256':sha(NEW)},
 'exact_change':addition,'removing_only_added_guard_restores_entire_original_file':True,
 'entire_kernel_and_launch_wrapper_asts_unchanged':True,'cases':records,'gpu_execution':False,'kernel_execution':False,
 'numerical_experiments':[],'boundary':'CPU Torch tensors with only is_cuda facade, mocked CUDA context, and captured launch. No numeric kernel execution or compiler invocation.'}
(E/'candidate-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(records),'only_live_axis_overflows_rejected':True,'empty_domain_preserved':True,'all_other_source_bytes_identical':True,'gpu_execution':False,'kernel_execution':False},indent=2))
