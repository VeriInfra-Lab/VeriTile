from pathlib import Path
import ast,hashlib,json,re
A=Path(__file__).resolve().parent
S=A
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path,default=Path('/tmp/veritile-attention-aligned-lowering'))
args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=S/'attention_kernel_aligned_repaired.py';code=source.read_text();tree=ast.parse(code)
assert not any(isinstance(n,ast.Attribute) and n.attr in ('contiguous','clone','resolve_conj','resolve_neg') for n in ast.walk(tree))
assert code.count('qk += (b0 + b1)')==2
assert 'qk += tl.dot(q, k, out_dtype=OUT_DTYPE)' in code
assert 'qk += tl.dot(q, k, out_dtype=tl.float32).to(OUT_DTYPE)' in code
lean=(S/'Kernels.lean').read_text()
assert 'out_dtype=tl.float16' in lean and 'out_dtype=tl.float32' in lean
report=json.loads((args.artifacts/'report.json').read_text());rows=[]
for r in report['results']:
 assert sha(source)==r['source_sha256']
 p=Path(r['artifacts']['ttir']['path']);ir=p.read_text();assert sha(p)==r['artifacts']['ttir']['sha256']
 m=int(r['constants']['BLOCK_M']);n=int(r['constants']['BLOCK_N']);d=int(r['constants']['BLOCK_DMODEL'])
 t='bf16' if r['name'].startswith('bfloat16') else 'f16'
 pats={
  'query_widen':rf'arith.extf .* : tensor<{m}x{d}x{t}> to tensor<{m}x{d}xf32>',
  'query_narrow':rf'arith.truncf .* : tensor<{m}x{d}xf32> to tensor<{m}x{d}x{t}>',
  'bias_storage_add':rf'arith.addf .* : tensor<{m}x{n}x{t}>',
  'bias_widen':rf'arith.extf .* : tensor<{m}x{n}x{t}> to tensor<{m}x{n}xf32>',
  'QK_precision':rf'tt.dot .* : tensor<{m}x{d}x{t}> \* tensor<{d}x{n}x{t}> -> tensor<{m}x{n}x'+('f32' if t=='bf16' else 'f16')+'>',
  'P_storage_cast':rf'arith.truncf .* : tensor<{m}x{n}xf32> to tensor<{m}x{n}x{t}>',
  'PV_default_FP32':rf'tt.dot .* : tensor<{m}x{n}x{t}> \* tensor<{n}x{d}x{t}> -> tensor<{m}x{d}xf32>',
  'output_storage_cast':rf'arith.truncf .* : tensor<{m}x{d}xf32> to tensor<{m}x{d}x{t}>',
  'output_storage_store':rf'tt.store .* : tensor<{m}x{d}x!tt.ptr<{t}>>',
 }
 checks={k:bool(re.search(v,ir)) for k,v in pats.items()};assert all(checks.values()),(r['name'],checks)
 assert r['metadata']['enable_fp_fusion']
 rows.append({'profile':r['name'],'ttir':str(p),'sha256':sha(p),'checks':checks})
result={'status':'PASS','source':str(source),'source_sha256':sha(source),'typed_source':str(S/'Kernels.lean'),'typed_source_sha256':sha(S/'Kernels.lean'),'script_sha256':sha(Path(__file__)),'profiles':rows,
 'typed_source_annotations':['FP16 QK has explicit FP16 output; BF16 uses FP32 QK followed by BF16 narrowing. Bias arithmetic has no log2(e) multiplier.'],
 'boundary':'Ordinary Python-to-typed-source, compiler/layout and runtime contracts. This check pins the complete candidate and both typed bodies and validates precision in all eight offline profiles; it is not a compiler implementation proof.',
 'GPU_execution':False,'FP_to_Real_accuracy_claimed':False}
(args.artifacts/'source-binding.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(rows),'profiles')
