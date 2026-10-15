"""Replay the exact original wrapper AST on CPU tensors with a launch spy.

No Triton kernel, native compilation, or GPU execution is performed.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch._subclasses.fake_impls import _compute_stride

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
WIP = Path('/tmp/veritile-standardization-20261015/kcache-host-wip')
SOURCE = ROOT / 'bench/tritonbench_g/kcache_copy_triton/kcache_copy_triton.py'


def tensor(shape, strides=None, dtype=torch.float32):
    if strides is None:
        return torch.empty(shape, dtype=dtype)
    return torch.empty_strided(shape, strides, dtype=dtype)


def meta(t):
    return {'shape': list(t.shape), 'strides': list(t.stride()), 'dtype': str(t.dtype), 'offset': t.storage_offset()}


class Spy:
    def __init__(self, original):
        self.original = original
        self.launches = []

    def __getitem__(self, grid):
        def launch(k, cache, tables, lengths, kt, kh, kd, cb, ch, split, cs, cd, bts, btb, block_size, **kwargs):
            assert (kt, kh, kd) == k.stride()
            assert cb == cache.stride(0) and ch == cache.stride(1)
            assert (bts, btb) == tables.stride()[:2]
            self.launches.append({
                'grid': list(grid), 'key': meta(k),
                'reshape_copied': k.untyped_storage().data_ptr() != self.original.untyped_storage().data_ptr(),
                'strides': [kt, kh, kd, cb, ch, split, cs, cd, bts, btb],
                'block_size': block_size, 'n_tokens': kwargs['n_tokens'],
                'head_dim': kwargs['HEAD_DIM'], 'width': kwargs['KCACHE_X'], 'num_warps': kwargs['num_warps'],
            })
        return launch


def make(name, key=(2,3,8), cache=(4,3,4,8), n=1, new=False, key_strides=None, cache_strides=None,
         lengths=(2,), tables=(2,4), lengths_strides=None, dtype=torch.float32, cache_dtype=None):
    return name, {'k': tensor(key,key_strides,dtype), 'k_cache': tensor(cache,cache_strides,cache_dtype or dtype),
                 'kv_lengths': tensor(lengths,lengths_strides,torch.int32), 'block_tables': tensor(tables,dtype=torch.int32),
                 'n':n,'use_new_kcache_layout':new}


cases = [
    make('3d_old_n1'),
    make('4d_old_n1',key=(2,1,3,8)),
    make('3d_old_n2',key=(4,3,8),n=2),
    make('4d_old_n2',key=(2,2,3,8),n=2),
    make('4d_noncontiguous_copy',key=(2,2,3,8),key_strides=(24,48,8,1),n=2),
    make('4d_rowgap_view',key=(2,2,3,8),key_strides=(80,40,10,1),n=2),
    make('4d_leading_singleton_view',key=(1,2,3,8),key_strides=(100,24,8,1)),
    make('4d_tail_singletons',key=(2,1,1,1),key_strides=(17,13,7,3),cache=(4,1,4,1)),
    make('4d_zero_prefix',key=(0,2,3,8),lengths=(0,),tables=(0,4)),
    make('4d_zero_head_ambiguous',key=(2,1,0,8),cache=(4,0,4,8)),
    make('3d_zero_width',key=(2,3,0),cache=(4,3,4,0)),
    make('new_zero_width',key=(2,3,0),cache=(4,3,0,4,0),new=True),
    make('zero_tokens_n0',n=0),
    make('negative_n',n=-2),
    make('n3_indivisible',key=(4,3,8),n=3),
    make('batch_mismatch',lengths=(3,)),
    make('3d_new_n1',cache=(4,3,2,4,4),new=True),
    make('3d_new_n2',key=(4,3,8),cache=(4,3,2,4,4),new=True,n=2),
    make('4d_new_n2',key=(2,2,3,8),cache=(4,3,2,4,4),new=True,n=2),
    make('new_wrong_rank',new=True),
    make('new_wrong_heads',cache=(4,2,2,4,4),new=True),
    make('new_wrong_split_product',cache=(4,3,3,4,4),new=True),
    make('legacy_accepts_rank5',cache=(4,3,2,4,4)),
    make('legacy_accepts_wrong_heads',cache=(4,1,4,8)),
    make('noncontiguous_lengths',lengths_strides=(2,)),
    make('extra_length_dimension',lengths=(2,2)),
    make('extra_table_dimension',tables=(2,4,2)),
    make('warps8_nonpower129',key=(2,3,129),cache=(4,3,4,129)),
    make('warps8_power256',key=(2,3,256),cache=(4,3,4,256)),
    make('old_last_stride2',cache_strides=(192,64,16,2)),
    make('new_last_stride2',cache=(4,3,2,4,4),cache_strides=(192,64,32,8,2),new=True),
    make('key_rank2',key=(2,8)),
    make('cache_rank3',cache=(4,3,8),new=True),
    make('lengths_rank0',lengths=()),
    make('tables_rank1',tables=(2,)),
    make('dtype_mismatch',cache_dtype=torch.float16),
]

tree = ast.parse(SOURCE.read_text())
wrapper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'copy_k_to_blocked_cache')
module = ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
rows=[]
for name,args in cases:
    spy=Spy(args['k'])
    env={'torch':torch,'_copy_to_kcache_seqlen_n_kernel':spy}
    exec(compile(module,str(SOURCE),'exec'),env)
    row={'name':name,'inputs':{k:meta(v) if isinstance(v,torch.Tensor) else v for k,v in args.items()}}
    try:
        result=env['copy_k_to_blocked_cache'](**args)
    except Exception as exc:
        assert not spy.launches
        row.update({'status':'host_error','exception_type':type(exc).__name__,'exception':str(exc)})
    else:
        assert result is None and len(spy.launches)==1
        row.update({'status':'launch_captured','return_none':True,'launch':spy.launches[0]})
        k=args['k']
        if k.dim()==4:
            shape=[k.shape[0]*k.shape[1],k.shape[2],k.shape[3]]
            inferred=_compute_stride(list(k.shape),list(k.stride()),shape)
            expected_copied=inferred is None
            # Empty tensors have the same zero storage pointer; all these are views.
            assert row['launch']['reshape_copied']==expected_copied
            if inferred is not None:
                assert row['launch']['key']['strides']==inferred
            row['reshape_inference']={'inferred_view_strides':inferred,'copy_required':expected_copied}
    rows.append(row)

source_kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_copy_to_kcache_seqlen_n_kernel')
body_names={n.id for stmt in source_kernel.body for n in ast.walk(stmt) if isinstance(n,ast.Name)}
assert 'stride_kcx' not in body_names and 'HEAD_DIM' not in body_names

receipt={
 'status':'pass','cases':rows,'case_count':len(rows),
 'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'wrapper_ast_sha256':hashlib.sha256(ast.dump(wrapper,include_attributes=False).encode()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'torch_version':torch.__version__,
 'torch_stride_primitive_sha256':hashlib.sha256(inspect.getsource(_compute_stride).encode()).hexdigest(),
 'gpu_execution':False,'native_compilation':False,
 'execution':'Exact original wrapper AST; real CPU tensors and actual Torch reshape; Triton kernel launch spy.',
 'source_findings':[
  {'kind':'ignored_last_stride','detail':'stride_kcx is passed by both layout branches but not read by kernel body; a last-dimension stride of 2 still addresses lane+1 instead of lane+2.'},
  {'kind':'ignored_lengths_stride','detail':'Only lengths.shape[0] is checked; kernel addresses seq_lengths+seq and does not receive a length stride. Noncontiguous or extra-dimensional lengths can pass host checks.'},
  {'kind':'legacy_shape_unchecked','detail':'Old-layout branch accepts rank5 caches and mismatched head counts; model must retain actual shape[-2], stride(2), stride(3) rather than imposing new-layout assertions.'},
  {'kind':'zero_or_negative_n','detail':'No positive-n check. n=0 produces grid dimension0; negative n reaches negative grid tuple in spy. Actual Triton launch behavior is not exercised.'},
  {'kind':'native_arange_domain','detail':'head_dim=129 requests num_warps8 and KCACHE_X129; host accepts it although original tl.arange requires a power-of-two width.'},
 ],
 'limitations':['No native launch or CUDA allocator/device behavior validated.','Finite probes do not prove complete Torch reshape refinement.','Input tensor offsets in this probe are zero; arbitrary storage-base alias/offset mapping remains a native obligation.']}
out=WIP/'evidence/wrapper-probe.json'
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'pass','cases':len(rows),'launches':sum(r['status']=='launch_captured' for r in rows),'errors':sum(r['status']=='host_error' for r in rows),'output':str(out)}))
