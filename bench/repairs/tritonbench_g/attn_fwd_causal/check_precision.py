"""Inspect repaired causal masks and actual source precision in saved lowering."""
from pathlib import Path
import argparse,ast,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-attn-causal-lowering'));args=p.parse_args()
A=Path(__file__).resolve().parent;source=A/'attn_fwd_causal_repaired.py';text=source.read_text();tree=ast.parse(text)
functions={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
inner=functions['_attn_fwd_inner'];outer=functions['_attn_fwd']
assert 'tl.minimum((start_m + 1) * BLOCK_M, N_CTX)' in inner
assert "tl.where(mask, qk, float('-inf'))" in inner
assert 'offs_m[:, None] >= start_n + offs_n[None, :]' in inner
assert 'offs_n[None, :] < N_CTX - start_n' in inner and 'HEAD_DIM' in inner
assert 'tl.dot(q, k).to(tl.float32) * q_scale * k_scale' in inner
assert 'other=0.0).to(tl.float16)' in inner and 'p = p.to(tl.float16)' in inner
assert 'tl.dot(p, v, out_dtype=tl.float16)' in inner
assert outer.count('_attn_fwd_inner(')==2 and 'acc.to(Out.type.element_ty)' in outer
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('contiguous','clone','resolve_neg','resolve_conj') for n in ast.walk(tree))
lean=(A/'Kernels.lean').read_text();assert 'import bench.' not in lean
native=(A/'Proofs/Native.lean').read_text();assert 'Fin 8 → Graph' in native and 'Fin 16 → Graph' in native
report=json.loads((args.output_dir/'report.json').read_text());assert report['status']=='PASS'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in report['results']:
 assert row['source_sha256']==sha(source)
 for artifact in row['artifacts'].values():assert sha(Path(artifact['path']))==artifact['sha256']
 ptx=Path(row['artifacts']['ptx']['path']).read_text();ir=Path(row['artifacts']['ttir']['path']).read_text()
 assert 'tensor<128x128xf32> * tensor<128x64xf32> -> tensor<128x64xf32>' in ir
 assert 'tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf16>' in ir
 assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
 assert 'mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16' in ptx
 assert 'ex2.approx.ftz.f32' in ptx and 'fma.rn.f32' in ptx
 assert 'cvt.f32.f16' in ptx and ('cvt.rn.f16.f32' in ptx or 'cvt.rn.f16x2.f32' in ptx)
 assert row['metadata']['shared']<=163840
print('PASS: two causal phases, 128 features, TF32 QK, FP16 P/V/PV, FP32 state, final FP16;',len(report['results']),'profiles')
