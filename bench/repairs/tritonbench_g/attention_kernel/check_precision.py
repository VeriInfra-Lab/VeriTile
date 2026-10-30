from pathlib import Path
import ast,hashlib,json,re
A=Path(__file__).resolve().parent
S=A
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path,default=Path('/tmp/veritile-attention-kernel-lowering'))
args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=S/'attention_kernel_repaired.py';code=source.read_text();tree=ast.parse(code)
assert not any(isinstance(n,ast.Attribute) and n.attr in ('contiguous','clone','resolve_conj','resolve_neg') for n in ast.walk(tree))
assert 'start_m = tl.program_id(0)\n' in code and 'off_hz = tl.program_id(1).to(tl.int64)' in code
assert 'off_z = off_hz // H' in code and 'off_h = off_hz % H' in code
for tensor in ['q','k','v','o']:
 assert f'{tensor}_offset = off_z * stride_{tensor}z + off_h * stride_{tensor}h' in code
for line in ['qk += tl.dot(q, k)','qk += ((b0 + b1) * 1.44269504)','acc += tl.dot(p.to(OUT_DTYPE), v)','acc = acc / l_i[:, None]']:
 assert line in code
lean=(S/'Kernels.lean').read_text()
assert lean.count('qk += (tl.dot(q, k)).to(tl.float32)')==2
assert lean.count(')).to(tl.float32)')==4
assert '(1.44269504).to(tl.float16)' in lean and '(1.44269504).to(tl.bfloat16)' in lean
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
  'bias_storage_multiply':rf'arith.mulf .* : tensor<{m}x{n}x{t}>',
  'bias_widen':rf'arith.extf .* : tensor<{m}x{n}x{t}> to tensor<{m}x{n}xf32>',
  'QK_default_FP32':rf'tt.dot .* : tensor<{m}x{d}x{t}> \* tensor<{d}x{n}x{t}> -> tensor<{m}x{n}xf32>',
  'P_storage_cast':rf'arith.truncf .* : tensor<{m}x{n}xf32> to tensor<{m}x{n}x{t}>',
  'PV_default_FP32':rf'tt.dot .* : tensor<{m}x{n}x{t}> \* tensor<{n}x{d}x{t}> -> tensor<{m}x{d}xf32>',
  'output_storage_cast':rf'arith.truncf .* : tensor<{m}x{d}xf32> to tensor<{m}x{d}x{t}>',
  'output_storage_store':rf'tt.store .* : tensor<{m}x{d}x!tt.ptr<{t}>>',
 }
 checks={k:bool(re.search(v,ir)) for k,v in pats.items()};assert all(checks.values()),(r['name'],checks)
 assert r['metadata']['enable_fp_fusion']
 rows.append({'profile':r['name'],'ttir':str(p),'sha256':sha(p),'checks':checks})
result={'status':'PASS','source':str(source),'source_sha256':sha(source),'typed_source':str(S/'Kernels.lean'),'typed_source_sha256':sha(S/'Kernels.lean'),'script_sha256':sha(Path(__file__)),'profiles':rows,
 'typed_source_annotations':[
  'Explicit .to(tl.float32) on already-FP32 default QK/PV dot results only bridges the DSL explicit fp32 carrier to its FP32 compute real carrier. It is the identity in Real and Native staging, and creates no native rounding instruction.',
  'Explicit bias scalar .to(storage) states Triton implicit floating-scalar conversion; storage bias add/multiply and actual constants 0x3DC5/0x3FB9 are retained.',
  'Lean natural-number indexing is connected to i32/i64/u64 by Machine.closed; Python start_m remains i32 to satisfy the block-pointer offset ABI.'],
 'boundary':'Ordinary Python-to-typed-source, compiler/layout and runtime contracts. This check pins the complete candidate and both typed bodies and validates precision in all eight offline profiles; it is not a compiler implementation proof.',
 'GPU_execution':False,'FP_to_Real_accuracy_claimed':False}
(args.artifacts/'source-binding.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(rows),'profiles')
