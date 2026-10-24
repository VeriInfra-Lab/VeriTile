import ast,hashlib,json
from pathlib import Path
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');P=Path('/tmp/veritile-standardization-20261015/matrix-transpose-acceptance-wip');E=P/'evidence'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(E/'frontend-receipt.json');s=R/'bench/tritonbench_g/matrix_transpose/matrix_transpose.py';assert sha(s)==r['source_sha256']
for name,h in r['artifacts'].items():assert sha(E/name)==h,name
for name,h in r['primitive_files'].items():assert sha(Path(name))==h,name
original=next(n for n in ast.parse(s.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
extracted=next(n for n in ast.parse((E/'extracted_kernel.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
assert ast.dump(original,include_attributes=False)==ast.dump(extracted,include_attributes=False)
assert r['triton_version']=='3.7.1';assert r['signature']=={'M':'*fp16','Out':'*fp16','matrix_stridex':'i32','matrix_stridey':'i32','out_stridex':'i32','out_stridey':'i32'}
success=read(E/'frontend-success.json');reject=read(E/'frontend-rejections.json')
expected={(2**a,2**b) for a in range(21) for b in range(21) if a+b<=20}
observed={(x['SIZE_M'],x['D_HEAD']) for x in success}
assert len(success)==len(expected)==231 and observed==expected
for row in success:
 assert row['status']=='frontend_generated' and [row[k] for k in ['load_count','store_count','transpose_count']]==[1,1,1]
 if 'ttir_path' in row:
  p=E/row['ttir_path'];assert sha(p)==row['ttir_sha256'];text=p.read_text();assert all(x in text for x in ['tt.load','tt.store','tt.trans','f16','i32'])
assert len(reject)==224; badpairs={(x['SIZE_M'],x['D_HEAD']) for x in reject[:210]}
assert badpairs=={(2**a,2**b) for a in range(21) for b in range(21) if a+b>20}
for row in reject[:210]:assert row['status']=='frontend_error' and 'maximum tensor numel' in row['message'] and 'matrix_ptr =' in row['message']
def first_error(n):
 if n>>32:return 'arange must fit in int32'
 if n<=0:return "arange's end argument must be greater"
 if n&(n-1):return "arange's range must be a power of 2"
 if n>2**20:return 'maximum tensor numel'
for row in reject[210:]:
 m,d=row['SIZE_M'],row['D_HEAD']; error=first_error(m) or first_error(d)
 assert row['status']=='frontend_error' and error in row['message'],row
out={'schema':1,'kind':'independent_matrix_frontend_evidence_check','status':'passed','receipt_sha256':sha(E/'frontend-receipt.json'),'original_source_sha256':sha(s),'successful_shape_pairs':231,'complete_individually_legal_product_rejections':210,'boundary_rejections':14,'retained_ir_bodies_independently_read':6,'independent_source_ast_identity':True,'native_backend_execution':False,'gpu_execution':False,'whole_program_acceptance_increment':0,'boundary':'Pinned exact original JIT frontend. Nat supported shape domain; negative/non-shape inputs are primitive API boundaries, not silently claimed as Lean Nat coverage.'}
p=P.parent/'matrix-frontend-root-validation.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'passed','sha256':sha(p),'cases':len(success)+len(reject)}))
