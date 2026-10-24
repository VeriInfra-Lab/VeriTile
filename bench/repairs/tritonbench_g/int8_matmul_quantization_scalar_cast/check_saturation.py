"""Admitted composed-input saturation witness; symbolic analysis, no GPU run."""
import ast
import contextlib
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=HERE/'int8_matmul_quantization_scalar_cast.py'
OUTPUT=ROOT/'.veritile-cache/task-deliveries/int8_matmul_quantization_scalar_cast'

def pin(p):
    p=Path(p).resolve();return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def run():
    import torch
    import triton
    torch.set_num_threads(1)
    K=262144
    tree=ast.parse(SOURCE.read_text())
    events=[];captured={}
    class Tensor:
        def __init__(self,data):self.data=data;self.device=torch.device('cuda:0');self.is_cuda=True
        def __getattr__(self,name):return getattr(self.data,name)
    class Admitted(Exception):pass
    @contextlib.contextmanager
    def device(owner):
        assert owner==torch.device('cuda:0');events.append('device-enter')
        try:yield
        finally:events.append('device-exit')
    def allocate(shape,device,dtype):
        assert device==torch.device('cuda:0');events.append('allocation')
        return Tensor(torch.empty(shape,dtype=dtype))
    class Launch:
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                events.append('producer-launch');captured.update(grid=grid,args=args,kwargs=kwargs)
                raise Admitted()
            return call
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,int8=torch.int8,float16=torch.float16,float32=torch.float32,
        empty=allocate,cuda=SimpleNamespace(device=device),_C=SimpleNamespace(_overlaps=lambda a,b:torch._C._overlaps(a.data,b.data)))
    names={'_tensor','_check_row','_check_gemm','quantize_int8_perrow','matmul_quantize_int8','matmul_int8'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'torch':fake,'triton':triton,'quantize_int8_perrow_kernel':Launch()}
    exec(compile(ast.Module(nodes,[]),str(SOURCE),'exec'),ns)
    fpa=Tensor(torch.full((1,K),-127.,dtype=torch.float32))
    b=Tensor(torch.full((K,1),-128,dtype=torch.int8))
    bs=Tensor(torch.full((1,),2.**-31,dtype=torch.float32))
    try:ns['matmul_quantize_int8'](fpa,b,bs)
    except Admitted:pass
    else:raise AssertionError('did not reach actual producer launch')
    assert captured['grid']==(1,)
    assert captured['args'][3:]==(1,K,K,1,K,1,1,1,K)
    # PTX div.full.f32 has at most 2 ulp error. These enclosing rational
    # intervals are deliberately wider at s=1, and cover the ~-127 quotient.
    ds=Fraction(1,2**21);dq=Fraction(1,2**16)
    slo,shi=1-ds,1+ds
    qlo,qhi=Fraction(-127)/slo-dq,Fraction(-127)/shi+dq
    assert Fraction(-128)<qlo<qhi<Fraction(-126)
    # The entire live scale interval lies strictly inside the binary16
    # rounding cell of exactly 1; the lower spacing is 2^-11, upper 2^-10.
    assert Fraction(1)-Fraction(1,2**12)<slo<shi<Fraction(1)+Fraction(1,2**11)
    rows=[];limit=2**31-1
    for q in (-127,-126):
        term=q*(-128);atoms=K//32;carry=0
        for _ in range(atoms):carry=min(limit,carry+32*term)
        assert carry==limit
        cast=torch.tensor(carry,dtype=torch.int32).float()
        actual=(cast*torch.tensor(1.,dtype=torch.float16)*bs.data[0]).half().item()
        assert actual==1
        partial=term*(K//2);assert partial<limit
        split_partial=(torch.tensor(partial,dtype=torch.int32).float()*bs.data[0]).half()
        split_output=(split_partial+split_partial).half().item()
        exact=Fraction(term*K,2**31)
        assert split_output==float(exact)>1
        rows.append({'producer_code':q,'K32_product_sum':32*term,'K32_atoms_split1':atoms,
                     'unbounded_sum':term*K,'satfinite_carry':carry,'converted_fp32':cast.item(),
                     'SPLIT_K_1_output':actual,'SPLIT_K_2_partial_integer':partial,
                     'SPLIT_K_2_output':split_output,'GEMM_target_for_stored_codes':str(exact)})
    real=Fraction((-127)*(-128)*K,2**31)
    assert real==Fraction(127,64)
    record={'candidate':pin(SOURCE),'checker':pin(__file__),'actual_composed_guard_admission':True,
        'public_input':{'entry':'matmul_quantize_int8','shape':[1,K,1],'FP32_fpa':'uniform -127','INT8_B':'uniform -128',
                        'FP32_B_scale':'2^-31','out':None,'CUDA':'mocked cuda:0 with real CPU tensor storage'},
        'events':events,'producer_grid':captured['grid'],'producer_actual_scalar_arguments':list(captured['args'][3:]),
        'producer_derivation':{'maximum':127,'div_full_ulp_error_bound':2,'live_scale_interval':[str(slo),str(shi)],
            'quotient_interval':[str(qlo),str(qhi)],'signed16_toward_zero_then_low8_codes':[-127,-126],
            'stored_FP16_scale':1,'scope':'Bounds follow from documented div.full instruction accuracy, not assumed exact IEEE division. Both possible stored codes produce the failing SPLIT_K=1 result.'},
        'cases':rows,'independent_composed_Real_target':str(real),'observed_branch':'Config 10: BM32 BN32 BK128 GROUP8 SPLIT_K1; config27 is its SPLIT_K2 control.',
        'autotune_scope':'All 34 original configs remain eligible. No assertion that autotuning on a GPU selects config10; this disproves a claim covering every admitted selected configuration.',
        'evidence_sources':[pin(OUTPUT/'native/native-probe.json'),pin(OUTPUT/'large-producer/native-probe.json'),
                            pin(OUTPUT/'floating-point-instructions-div.txt'),pin(OUTPUT/'warp-level-matrix-instructions-mma.txt')],
        'GPU_execution':False,'CPU_scope':'Actual admission/wrapper AST until producer submission. Exact integer recurrence and concrete scalar conversion checks follow separately; no hand interpreter is claimed as a full quoted-source proof.',
        'classification':'Further mathematical/integer-semantic source defect after scalar-cast compilation repair; not a numerical experiment assumption.'}
    (OUTPUT/'saturation.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS actual composed guards; both allowed producer codes saturate SPLIT_K1 to output1; independent target127/64')
    print(pin(OUTPUT/'saturation.json'))
if __name__=='__main__':run()
