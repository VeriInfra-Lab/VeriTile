"""Actual Torch FakeTensor CUDA metadata and owner-context wrapper controls."""
import contextlib,importlib.util,sys,warnings
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
warnings.filterwarnings('ignore',message='Accessing the data pointer')
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('matmul_host',D/'matmul_leakyrelu_fp8_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Runner:
    def __init__(self):self.current=7;self.launches=[];self.allocations=[]
    @contextlib.contextmanager
    def device(self,owner):
        previous=self.current;self.current=owner
        try:yield
        finally:self.current=previous
    def __getitem__(self,grid):
        def invoke(*args,**kw):
            assert self.current==args[0].device
            self.launches.append((args,kw))
        return invoke

cases=0
with FakeTensorMode():
    for M,N,K in [(0,3,4),(3,0,4),(3,5,0),(3,5,67),(129,257,65)]:
        for form in ('contiguous','transpose'):
            a=torch.empty((M,K),device='cuda:2',dtype=torch.float32)
            b=torch.empty((K,N),device='cuda:2',dtype=torch.float32) if form=='contiguous' else torch.empty((N,K),device='cuda:2',dtype=torch.float32).T
            runner=Runner()
            with patch.object(m,'matmul_kernel',runner),patch.object(torch.cuda,'device',runner.device):
                out=m.matmul(a,b,'leaky_relu')
            assert out.shape==(M,N) and out.dtype==torch.float16 and out.device==a.device
            assert out.storage_offset()==0 and out.stride()==(max(N,1),1)
            assert out is not a and out is not b
            assert runner.current==7 and len(runner.launches)==int(bool(M and N and K))
            if runner.launches:
                args,kw=runner.launches[0]
                assert args[0] is a and args[1] is b and args[2] is out
                assert args[6:]==(*a.stride(),*b.stride(),*out.stride())
            cases+=1
    a=torch.empty((3,5),device='cuda:2');b=torch.empty((5,7),device='cuda:1')
    try:m._validate_inputs(a,b,'')
    except ValueError:pass
    else:raise AssertionError('cross-device')
print(f'{cases} actual CUDA metadata/owner/identity host cases; 1 cross-device rejection passed')
