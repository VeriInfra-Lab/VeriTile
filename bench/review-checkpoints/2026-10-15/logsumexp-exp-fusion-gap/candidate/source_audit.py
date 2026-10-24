from pathlib import Path
import ast,json,hashlib
P=Path(__file__).resolve().parent;N=P/'native'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=(N/'original.py').read_text();tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='logsumexp_fwd_kernel');host=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='logsumexp_fwd')
jit=next(n for n in ast.parse((N/'original_jit.py').read_text()).body if isinstance(n,ast.FunctionDef));assert ast.dump(fn.args)==ast.dump(jit.args) and [ast.dump(n) for n in fn.body]==[ast.dump(n) for n in jit.body]
assert ast.unparse(fn.decorator_list[1])=="triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None})"
assert [ast.unparse(a.annotation) for a in fn.args.args]==['tl.tensor','tl.tensor','tl.float32','tl.constexpr','tl.constexpr','tl.constexpr']
assignments={t.id:ast.unparse(n.value) for n in host.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
assert assignments['B']=='min(triton.next_power_of_2(D), 64 * 1024)' and assignments['ND']=='triton.cdiv(D, B)'
launches=[n for n in ast.walk(host) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and ast.unparse(n.func.value)=='logsumexp_fwd_kernel'];assert len(launches)==1 and ast.unparse(launches[0].func.slice)=='(N, ND)'
assert {x.arg:ast.unparse(x.value) for x in launches[0].keywords}=={'x':'x','z':'z','scale':'scale','D':'D','B':'B'}
receipt=json.loads((N/'receipt.json').read_text());matrix={(r['arch'],r['options']['num_warps'],r['constexprs']['HAS_SCALE'],r['options']['enable_fp_fusion']) for r in receipt['records']}
assert matrix=={(a,w,s,f) for a in [80,90] for w in [1,2,4,8,16,32] for s in [False,True] for f in [False,True]}
assert all(r['constexprs']['D']==17 and r['constexprs']['B']==32 for r in receipt['records'])
report={'schema':1,'original_sha256':sha(N/'original.py'),'extracted_jit_sha256':sha(N/'original_jit.py'),'kernel_signature_and_body_AST_equal':True,'autotune_members':[1,2,4,8,16,32],'heuristic':'HAS_SCALE iff scale is not None','host_B':'min(next_power_of_2(D),65536)','host_grid':'N,cdiv(D,B)','tested_profile_D_B':[17,32],'complete_cross_product_profiles':48,'compiler_options':'num_stages=3 is explicit captured default; fusion variants are compiler characterization, not an original host keyword','native_execution':False,'accepted_scope_boundary':'Triton body only; host torch z.logsumexp(-1), host dtype conversion and other storage dtypes are not added to accepted fp32 scope.'}
(P/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS exact original body/signature and 48 member profiles')
