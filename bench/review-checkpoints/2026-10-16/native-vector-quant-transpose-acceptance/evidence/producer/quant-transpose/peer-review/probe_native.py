from pathlib import Path
import ast, datetime, hashlib, importlib.util, json, os, sys, traceback
sys.dont_write_bytecode = True
work = Path(__file__).resolve().parent
os.environ['TRITON_CACHE_DIR'] = str(work / 'triton-cache')
os.environ['CUDA_CACHE_PATH'] = str(work / 'cuda-cache')
import torch, triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget, BaseBackend
from triton.runtime.jit import mangle_type
from triton._C.libtriton import native_specialize_impl
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
src = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py')
text = src.read_text()
prefix = text.split('################################################################')[0]
names = {'_quantize_global_transpose', 'quantize_global_transpose'}
nodes = lambda t: [ast.dump(n) for n in ast.parse(t).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert nodes(prefix) == nodes(text)
copied = work / 'original_prefix.py'
copied.write_text(prefix)
spec = importlib.util.spec_from_file_location('independent_transpose_peer', copied)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
metadata = ['stride_am', 'stride_an', 'stride_bn', 'stride_bm', 'M', 'N']
base_signature = {'A':'*fp32','absmax_inv_ptr':'*fp32','B':'*i8',**{k:'i32' for k in metadata}}
profiles = [('mixed_stride_an_i64', {'stride_an':'i64'}, {}),
            ('mixed_stride_am_i64', {'stride_am':'i64'}, {}),
            ('mixed_M_i64', {'M':'i64'}, {}),
            ('mixed_N_i64', {'N':'i64'}, {}),
            ('all_metadata_i64', {k:'i64' for k in metadata}, {})]
views = []
for axis in [0,1]:
    for stride in [2**30,2**31]:
        strides = (stride,1) if axis==0 else (1,stride)
        x = torch.as_strided(torch.tensor([1.0], dtype=torch.float32), (1,1), strides)
        maximum = x.abs().max().unsqueeze(0)
        inverse = 1.0/maximum
        vals = dict(zip(metadata, [*x.stride(),1,1,*x.shape]))
        specialization = {k:native_specialize_impl(BaseBackend,v,False,True,True) for k,v in vals.items()}
        signature = {k:v[0] for k,v in specialization.items()}
        constexprs = {k:v[1] for k,v in specialization.items() if v[0]=='constexpr'}
        name = f'singleton_axis{axis}_stride{stride}_runtime_specialized'
        profiles.append((name, signature, constexprs))
        views.append({'name':name,'device':str(x.device),'shape':list(x.shape),'strides':list(x.stride()),
            'storage_offset':x.storage_offset(),'storage_length':x.untyped_storage().nbytes()//x.element_size(),
            'unit_stride_assertion':any(s==1 for s in x.stride()),'maximum':maximum.item(),'inverse':inverse.item(),
            'active_coordinates':[[0,0]],'active_input_offsets':[0],'active_output_offsets':[0],
            'runtime_arguments':vals,'runtime_specialization':specialization,
            'domain_stride_bounds':all(s<2**31 for s in strides),
            'domain_input_arithmetic_bound':128*sum(strides)<2**31,
            'raw_padded_i64_offset_upper':127*sum(strides),
            'raw_padded_i64_offset_fits':127*sum(strides)<2**63})
records = []
for name, overrides, constants in profiles:
    for ci, config in enumerate(module._quantize_global_transpose.configs):
        sig = {**base_signature,**overrides}
        const = {**config.kwargs,**constants}
        opts = {k:v for k,v in config.all_kwargs().items() if k not in config.kwargs}
        rec = {'name':name,'config_index':ci,'signature':sig,'constexprs':const,'options':opts}
        tag = f'{name}-config{ci}'
        try:
            kernel = triton.compile(ASTSource(module._quantize_global_transpose.fn, sig, constexprs=const),target=GPUTarget('cuda',90,32),options=opts)
            rec['status']='compiled'
            rec['artifacts']={}
            for stage in ['ttir','ttgir','llir','ptx']:
                f=work/f'{tag}.{stage}'
                f.write_text(kernel.asm[stage])
                rec['artifacts'][stage]={'path':str(f),'sha256':sha(f)}
            rec['typed_integer_operations']=[l.strip() for l in kernel.asm['ttir'].splitlines() if any(k in l for k in ['arith.addi','arith.muli','arith.extsi','arith.trunci','arith.divsi','arith.remsi','arith.cmpi','tt.addptr'])]
        except Exception:
            rec['status']='failed'
            rec['error']=traceback.format_exc()
            (work/f'{tag}.log').write_text(rec['error'])
        records.append(rec)
        print(name,ci,rec['status'],flush=True)
out={'reviewer':'peer:/root/rotary_grid_recover','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'original_source':str(src),'original_source_sha256':sha(src),'production_AST_unchanged':True,
     'copied_prefix_sha256':sha(copied),'probe_sha256':sha(__file__),'torch_version':torch.__version__,
     'torch_commit':torch.version.git_version,'triton_version':triton.__version__,'gpu_executions':0,
     'metadata_probe_device':'CPU; metadata construction also applies to ordinary CUDA dense-strided tensors',
     'target':{'backend':'cuda','arch':90,'warp_size':32},'views':views,'records':records}
receipt=work/'native-review-receipt.json'
receipt.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'receipt':str(receipt),'sha256':sha(receipt),'successes':sum(r['status']=='compiled' for r in records)}))
