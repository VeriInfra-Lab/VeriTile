"""Hash-bound source and PTX dataflow audit; no native numerical conformance."""
from pathlib import Path
import argparse,ast,hashlib,json,re
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-fused-rotary-lowering'));args=p.parse_args()
A=Path(__file__).resolve().parent;source=A/'fused_rotary_embedding_repaired.py';tree=ast.parse(source.read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
original=A.parents[2]/'tritonbench_g/fused_rotary_embedding/fused_rotary_embedding.py';old={n.name:n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
kernel=functions['decoding_fused_rotary_embedding_kernel'];original_kernel=old[kernel.name]
def assignments(fn):return {n.targets[0].id:n for n in ast.walk(fn) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
assign=assignments(kernel);oldassign=assignments(original_kernel)
for name in ['out_q0','out_q1','out_k0','out_k1']:assert ast.dump(assign[name].value)==ast.dump(oldassign[name].value)
assert '.contiguous(' not in ast.unparse(functions['decoding_fused_rotary_embedding'])
calls=[n for n in ast.walk(functions['decoding_fused_rotary_embedding']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to']
assert len(calls)==2 and {ast.unparse(n.func.value) for n in calls}=={'kv_lengths','block_tables'}
for name in ['cur_head_idx','cur_token_idx','dim_range','dim_range0']:
 assert '.to(tl.int64)' in ast.unparse(assign[name].value)
for name in ['past_kv_seq_len','block_ids']:assert '.to(tl.int64)' in ast.unparse(assign[name].value)
seen=set()
def audit(path):
 if path in seen:return
 seen.add(path)
 for imp in re.findall(r'^import (\S+)',path.read_text(),re.M):
  if imp.startswith('bench.repairs.tritonbench_g.fused_rotary_embedding.'):
   suffix=imp.split('fused_rotary_embedding.',1)[1];assert suffix not in ['Correct','Proofs.Real','Proofs.HistoricalCorrect']
   audit(A/(suffix.replace('.','/')+'.lean'))
audit(A/'FPEquiv.lean')
leaf_names={'loaded_q0':'q0','loaded_q1':'q1','loaded_cos':'c','loaded_sin':'s','loaded_k0':'k0','loaded_k1':'k1','loaded_v':'v'}
leaves={assign[n].lineno:value for n,value in leaf_names.items()}
def mul(a,b):return ('mul',a,b)
def neg(a):return ('neg',a)
def fma(a,b,c):return ('fma',a,b,c)
expected={'out_q0':fma('q0','c',neg(mul('q1','s'))),'out_q1':fma('q1','c',mul('q0','s')),'out_k0':fma('c','k0',neg(mul('s','k1'))),'out_k1':fma('s','k0',mul('c','k1')),'loaded_v':'v'}
stores={n.lineno:expected[ast.unparse(n.value.args[1])] for n in ast.walk(kernel) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='store'}
report=json.loads((args.output_dir/'report.json').read_text());assert report['status']=='PASS';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();graphs=0
for row in report['results']:
 assert row['source_sha256']==sha(source)
 for artifact in row['artifacts'].values():assert sha(Path(artifact['path']))==artifact['sha256']
 ptx=Path(row['artifacts']['ptx']['path']).read_text();env={};line=None;observed=[]
 for raw in ptx.splitlines():
  loc=re.match(r'\s*\.loc\s+1\s+(\d+)\s',raw)
  if loc:line=int(loc[1]);continue
  load=re.search(r'ld\.global(?:\.v\d+)?\.b32\s*\{\s*([^}]+)\s*\}',raw)
  if load and line in leaves:
   regs=[r.strip() for r in load[1].split(',')];assert len(regs)==1,(row['name'],raw)
   env[regs[0]]=leaves[line];continue
  op=re.search(r'\b(mul\.f32|neg\.f32|fma\.rn\.f32|mov\.b32)\s+([^;]+);',raw)
  if op:
   operands=[r.strip() for r in op[2].split(',')];dest=operands[0]
   if all(v in env for v in operands[1:]):
    vals=[env[v] for v in operands[1:]]
    env[dest]=vals[0] if op[1]=='mov.b32' else ({'mul.f32':mul,'neg.f32':neg,'fma.rn.f32':fma}[op[1]])(*vals)
  store=re.search(r'st\.global(?:\.v\d+)?\.b32\s*\[[^]]+\],\s*\{\s*([^}]+)\s*\}',raw)
  if store and line in stores:
   regs=[r.strip() for r in store[1].split(',')];assert len(regs)==1
   assert env.get(regs[0])==stores[line],(row['name'],line,env.get(regs[0]),stores[line]);observed.append(line)
 assert set(observed)==set(stores),(row['name'],observed,stores);graphs+=len(observed)
 assert '.ftz.f32' not in ptx and 'add.f32' not in ptx and 'sub.f32' not in ptx
 assert 'mul.lo.s64' in ptx and ('cvt.u64.u32' in ptx or 'cvt.s64.s32' in ptx)
print('PASS:',len(report['results']),'SM80/SM90 actual-JIT profiles;',graphs,'exact ordered Q/K/V store dataflows; unchanged arithmetic AST, metadata-only copies, independent FP proof closure')
print('Original SHA256:',sha(original))
