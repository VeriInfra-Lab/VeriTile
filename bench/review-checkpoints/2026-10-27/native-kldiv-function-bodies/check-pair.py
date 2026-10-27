from pathlib import Path
import ast,copy,hashlib,json,subprocess
r=Path.cwd();t=Path('bench/repairs/tritonbench_g/kldiv_ops');parent=Path('/tmp/isfinite79-receipt.txt').read_text().strip()
source=(r/t/'kldiv_ops_repaired.py').read_bytes();assert source==subprocess.check_output(['git','show',parent+':'+str(t/'kldiv_ops_repaired.py')],cwd=r)
old=ast.parse(source);new=ast.parse((r/t/'candidate_kernels.py').read_text());pairs=[]
for suffix,pointer,stride,count in [('forward','loss_ptr','loss_stride',12),('backward','new_grads_ptr','new_grads_stride',9)]:
 name='_kldiv_kernel_'+suffix;original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);candidate=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name+'_index_commuted')
 assert len(original.body)==len(candidate.body)==count
 expected=copy.deepcopy(original);expected.name=candidate.name
 sites=[n for n in expected.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id==pointer]
 assert len(sites)==1;n=sites[0];assert isinstance(n.op,ast.Add) and isinstance(n.value,ast.BinOp) and isinstance(n.value.op,ast.Mult)
 assert isinstance(n.value.left,ast.Name) and n.value.left.id=='pid' and isinstance(n.value.right,ast.Name) and n.value.right.id==stride
 n.value.left,n.value.right=n.value.right,n.value.left
 assert ast.dump(expected,include_attributes=False)==ast.dump(candidate,include_attributes=False)
 pairs.append({'source':name,'candidate':candidate.name,'only_body_delta':pointer+' += '+stride+' * pid','top_level_statements':count})
print(json.dumps({'status':'PASS','source_and_all_wrappers_byte_unchanged':True,'source_sha256':hashlib.sha256(source).hexdigest(),'pairs':pairs,'candidate_provenance':'Python spelling of the existing Lean output-row integer-commutation candidate; no floating rewrite.'},indent=2))
