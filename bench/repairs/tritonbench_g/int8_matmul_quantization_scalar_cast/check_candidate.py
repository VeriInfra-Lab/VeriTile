"""Exact candidate AST with ordinary Python-int specialization; no GPU run."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / 'int8_matmul_quantization_scalar_cast.py'
OLD = HERE.parent / 'int8_matmul_quantization'
spec = importlib.util.spec_from_file_location('int8_cast_prior_probe', OLD / 'check_counterexample.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
prior.SOURCE = SOURCE

def pin(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def run(output, large_producer=False):
    output.mkdir(parents=True, exist_ok=True)
    (output/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(output/'tmp')
    os.environ['TRITON_CACHE_DIR'] = str(output/'compiler-cache')
    os.environ.pop('TRITON_INTERPRET', None)
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    torch.set_num_threads(1)
    target = GPUTarget('cuda',80,32)
    backend = CUDABackend(target)
    records=[]
    def compile_case(kind, kernel, configs, values, keywords, admitted, indices, unit, label):
        for index in indices:
            cfg=configs[index]
            signature, constants, attrs, options=prior.specialize(kernel,values,keywords,cfg,backend,unit)
            code=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants,attrs=attrs),target=target,options=options)
            artifacts=[]
            for ext in ('ttir','ttgir','llir','ptx'):
                p=output/f'{label}-config{index}.{ext}';p.write_text(code.asm[ext]);artifacts.append(pin(p))
            record={'kind':kind,'case':label,'admitted':admitted,'config_index':index,'config':cfg.kwargs,
                    'options':options,'signature':signature,'constants':{kernel.arg_names[k[0]]:v for k,v in constants.items()},
                    'attrs':{kernel.arg_names[k[0]]:v for k,v in attrs.items()},'artifacts':artifacts,
                    'status':'compile_pass','satfinite_MMA':'.satfinite.s32.s8.s8.s32' in code.asm['ptx']}
            records.append(record)
            print(label,index,'PASS',flush=True)
    if large_producer:
        kernel,configs,tree=prior.load_kernel(SOURCE,'producer','int8_scalar_cast_large_producer')
        values,keywords,admitted=prior.admitted_call(tree,configs,'producer',1,k=262144)
        values[0].fill_(-127)
        admitted['contents']={'FP32_input':'all -127','Real_target_scale':1,'Real_target_code':-127}
        compile_case('producer',kernel,configs,values,keywords,admitted,(0,),'stride_fpak','producer-K262144')
        record={'source':pin(SOURCE),'profiles':records,'GPU_execution':False,'scope':'Actual large producer specialization and offline compilation.'}
        (output/'native-probe.json').write_text(json.dumps(record,indent=2)+'\n')
        return
    for kind,indices,unit in [('producer',(3,0),'stride_fpak'),('gemm',(10,27),'stride_ak')]:
        kernel,configs,tree=prior.load_kernel(SOURCE,kind,'int8_scalar_cast_'+kind)
        for m in (1,2):
            values,keywords,admitted=prior.admitted_call(tree,configs,kind,m,n=m)
            compile_case(kind,kernel,configs,values,keywords,admitted,indices,unit,f'{kind}-M{m}')
    # The large-K raw helper case is admitted by the actual same guards and
    # wrapper. This is not claimed as producer reachability evidence.
    original_ones=torch.ones
    def payload(shape,dtype):
        t=original_ones(shape,dtype=dtype)
        return t.fill_(-128 if dtype==torch.int8 else 2.0**-15)
    kernel,configs,tree=prior.load_kernel(SOURCE,'gemm','int8_scalar_cast_largeK')
    with patch.object(torch,'ones',side_effect=payload):
        values,keywords,admitted=prior.admitted_call(tree,configs,'gemm',1,n=1,k=262144)
    admitted['contents']={'INT8_A':-128,'INT8_B':-128,'FP16_row_scale':2.0**-15,'FP32_column_scale':2.0**-15,'Real_target':4}
    compile_case('gemm',kernel,configs,values,keywords,admitted,(10,27),'stride_ak','raw-K262144')
    record={'source':pin(SOURCE),'checker':pin(__file__),'reused_probe':pin(OLD/'check_counterexample.py'),
            'triton_version':triton.__version__,'target':'CUDA SM80','profiles':records,
            'old_eight_failed_profiles_now_compile':True,'GPU_execution':False,
            'scope':'Actual assertion/wrapper AST and ordinary specialization; exact JIT body/signature, host autotune decorator omitted. Large-K raw-helper values are not a composed-producer reachability claim.'}
    (output/'native-probe.json').write_text(json.dumps(record,indent=2)+'\n')
    print('receipt',pin(output/'native-probe.json'),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'.veritile-cache/task-deliveries/int8_matmul_quantization_scalar_cast/native')
    parser.add_argument('--large-producer',action='store_true')
    args=parser.parse_args();run(args.output,args.large_producer)
