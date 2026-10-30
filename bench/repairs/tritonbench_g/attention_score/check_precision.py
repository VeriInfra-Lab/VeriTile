"""Inspect the source, typed source, and saved current-source SM80 artifacts."""
from pathlib import Path
import argparse,ast,hashlib,json,re
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-attention-score-lowering'));args=p.parse_args()
A=Path(__file__).resolve().parent;source=A/'attention_score_repaired.py';text=source.read_text();tree=ast.parse(text)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_score_kernel');kernel=ast.unparse(fn)
assert 'tl.dot(q, k)' in kernel and 'qk_scale *= 1.4426950408889634' in kernel
assert 'qk = qk * qk_scale' in kernel and 'qk = qk - m[:, None]' in kernel
assert 'tl.math.exp2(qk)' in kernel and 'tl.sum(p, axis=0)' in kernel
assert 'if not IS_EVEN_M:' in kernel and 'hi = tl.cast(N_CTX, tl.int64)' in kernel
assert 'tl.cast(sliding_window_offset, tl.int64)' in kernel and 'tl.cast(sliding_window_size, tl.int64)' in kernel
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('contiguous','clone','resolve_neg','resolve_conj') for n in ast.walk(tree))
lean=(A/'Kernels.lean').read_text();assert 'import bench.' not in lean
assert '($(N_CTX)).to(tl.int64)' in lean and '($((sliding_window_offset : Int))).to(tl.int64)' in lean
# tl.cast accepts both constexpr and tensor scalars; the typed source's .to node has the same cast semantics.
staging=(A/'Proofs/NativeStaging.lean').read_text();formula=(A/'Proofs/NativeFormula.lean').read_text()
assert '.fmaRN' in staging and '.negateSign' in staging and '.ex2ApproxFTZ' in formula
report=json.loads((args.output_dir/'report.json').read_text());assert report['status']=='PASS'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in report['results']:
 assert row['source_sha256']==sha(source)
 for artifact in row['artifacts'].values():assert sha(Path(artifact['path']))==artifact['sha256']
 ptx=Path(row['artifacts']['ptx']['path']).read_text();ir=Path(row['artifacts']['ttir']['path']).read_text()
 assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
 assert 'mul.f32' in ptx and 'fma.rn.f32' in ptx and 'ex2.approx.ftz.f32' in ptx
 assert '0f3FB8AA3B' in ptx and 'shfl.sync.bfly.b32' in ptx
 assert 'inputPrecision = tf32' in ir and 'xf16>' not in ir
 assert not re.search(r'\bld\.global[^;]*\.b(?:64|128)\b',ptx)
print('PASS: current source casts, TF32 QK, scale-first FMA, exp2, column reductions, FP32 stores;',len(report['results']),'profiles')
