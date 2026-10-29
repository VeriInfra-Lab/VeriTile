"""Frozen source-shape and observation audit; not a Python semantics proof."""
import ast,hashlib,json,shutil,textwrap
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parent
old=root.parent/'vector-dtype-rejection-assessment'
audit=root/'evidence/source-audit';audit.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sha_text=lambda s:hashlib.sha256(s.encode()).hexdigest()
def save(p,data):
 b=(json.dumps(data,indent=2)+'\n').encode() if not isinstance(data,bytes) else data
 if p.exists():assert p.read_bytes()==b,str(p)
 else:p.write_bytes(b)
bindings=json.loads((old/'bindings-and-partition.json').read_text())
# Exact previously delivered hashes are retained independently of new audit output.
assert sha(old/'bindings-and-partition.json')=='d08eb0b7c2e852b18f1cbc6a67cb875383fac8f13fe43b9a6777fa55d501e777'
assert sha(old/'ASSESSMENT.md')=='d7518a5055b64cc535f6373d0b1b49059c56cdbce8356ddfc01cfe45d48610e2'
save(audit/'prior-bindings-and-partition.json',(old/'bindings-and-partition.json').read_bytes())
save(audit/'prior-assessment.md',(old/'ASSESSMENT.md').read_bytes())
texts={};trees={}
for key,r in bindings['files'].items():
 p=Path(r['path']);assert sha(p)==r['sha256'],str(p)
 if p.suffix=='.py':texts[key]=p.read_text();trees[key]=ast.parse(texts[key])

def get(key,q):
 node=trees[key]
 for part in q.split('.'):
  def name(x):
   if hasattr(x,'name'):return x.name
   if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Name):return x.targets[0].id
   if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name):return x.target.id
   return None
  node=next(x for x in node.body if name(x)==part)
 return node

def unit(key,q):
 n=get(key,q);start=n.lineno;end=n.end_lineno
 s='\n'.join(texts[key].splitlines()[start-1:end])+'\n'
 filename=key+'__'+q.replace('.','_')+'.txt';save(audit/filename,s.encode())
 return {'source':key,'qualified_name':q,'line_start':start,'line_end':end,'text_file':filename,
  'text_sha256':sha_text(s),'ast_sha256':sha_text(ast.dump(n,include_attributes=False))}
units=[]
for u in bindings['source_units']:
 p=old/u['text_file'];assert sha(p)==u['text_sha256']
 got=unit(u['source'],u['qualified_name']);assert got==u,(got,u);units.append(got)
extras=[('triton_codegen','CodeGenerator.__init__'),('triton_backend','CUDAOptions'),
 ('triton_canonicalization','type_canonicalisation_dict'),('triton_core','tensor.__add__'),
 ('triton_core','load'),('triton_core','add'),('triton_semantic','TritonSemantic.tensor'),
 ('triton_semantic','TritonSemantic._load_legacy'),('triton_semantic','TritonSemantic.load'),
 ('triton_codegen','CodeGenerator.visit_BinOp'),('triton_codegen','CodeGenerator._method_name_for_bin_op')]
units.extend(unit(*x) for x in extras)
checks=[]
def check(label,condition):
 assert condition,label
 checks.append({'check':label,'passed':True})
def code(key,q):return ast.unparse(get(key,q))
def contains(key,q,*snippets):
 s=ast.dump(get(key,q),include_attributes=False)
 return all(ast.dump(n,include_attributes=False) in s for x in snippets for n in ast.parse(textwrap.dedent(x)).body)
# Original host ordering and kernel entry shape are checked, not assumed as new guards.
k=get('original_python','add_kernel')
check('original kernel has x,y,output,n_elements,BLOCK_SIZE order and no defaults',
 [a.arg for a in k.args.args]==['x_ptr','y_ptr','output_ptr','n_elements','BLOCK_SIZE'] and not k.args.defaults)
check('original kernel extraction AST unchanged',ast.dump(k,include_attributes=False)==ast.dump(get('kernel_extract','add_kernel'),include_attributes=False))
check('original kernel loads tensor operands before x+y and masked store',contains('original_python','add_kernel',
 'x = tl.load(x_ptr + offsets, mask=mask)','y = tl.load(y_ptr + offsets, mask=mask)',
 'output = x + y','tl.store(output_ptr + offsets, output, mask=mask)'))
h=get('original_python','add')
check('host allocates before CUDA assertion and launches source with output dtype inherited from x',
 isinstance(h.body[0],ast.Assign) and ast.unparse(h.body[0].value)=='torch.empty_like(x)' and
 isinstance(h.body[1],ast.Assert) and ast.unparse(h.body[1].test)=='x.is_cuda and y.is_cuda and output.is_cuda' and
 ast.unparse(h.body[2])=='n_elements = output.numel()' and
 'add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)' in code('original_python','add'))
canonical=ast.literal_eval(get('triton_canonicalization','type_canonicalisation_dict').value)
expected={'bool':'u1','float8_e4m3fnuz':'fp8e4b8','float8_e5m2fnuz':'fp8e5b16'}
check('canonical bool/fnuz spellings preserved',all(canonical[x]==y for x,y in expected.items()))
check('pointer signature parsing recurses into element dtype',contains('triton_signature_parser','str_to_ty',
 'ty = str_to_ty(name, c)','return pointer_type(element_ty=ty, const=const)'))
parser=get('triton_signature_parser','str_to_ty');tys=next(n.value for n in parser.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='tys')
type_names={ast.literal_eval(k):ast.unparse(v) for k,v in zip(tys.keys,tys.values)}
check('fnuz and bool parser entries preserve scalar types',all(type_names[k]==v for k,v in {'fp8e4b8':'float8e4b8','fp8e5b16':'float8e5b16','u1':'int1'}.items()))
check('prototype preserves source argument index',contains('triton_codegen','ast_to_ttir',
 'idx = fn.arg_names.index(k)','arg_types[idx] = str_to_ty(v, None)','prototype = ASTFunction([], arg_types, src.attrs)'))
check('CodeGenerator receives options and writes builder.options',
 'options=options' in code('triton_codegen','ast_to_ttir') and
 contains('triton_codegen','CodeGenerator.__init__','self.builder.options = options'))
v=get('triton_codegen','CodeGenerator.visit_FunctionDef')
serial=next(i for i,n in enumerate(v.body) if isinstance(n,ast.Assign) and ast.unparse(n.value)=='self.prototype.serialize(self.builder)')
visit=next(i for i,n in enumerate(v.body) if isinstance(n,ast.Expr) and ast.unparse(n.value)=='self.visit_compound_statement(node.body)')
check('signature serialization occurs before visiting kernel body',serial<visit)
check('serialization flattens input signatures first',ast.unparse(get('triton_codegen','ASTFunction.serialize').body[0])=='arg_types_ir = self.flatten_ir_types(builder, self.arg_types)')
check('flattening loops argument types in input order',contains('triton_codegen','ASTFunction.flatten_ir_types',
 'for ty in types:\n    if ty is None:\n        continue\n    ty._flatten_ir_types(builder, ir_types)'))
check('pointer IR recursively serializes scalar element',contains('triton_core','pointer_type.to_ir','return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)'))
check('scalar flatten invokes to_ir',contains('triton_core','dtype._flatten_ir_types','out.append(self.to_ir(builder))'))
guard=get('triton_core','dtype.to_ir').body[0]
check('fp8 admission guard checks options and raises ValueError before type construction',isinstance(guard,ast.If) and
 ast.unparse(guard.test)=="self.name.startswith('fp8')" and
 isinstance(guard.body[0],ast.If) and ast.unparse(guard.body[0].test)=="hasattr(builder, 'options') and self.name not in builder.options.supported_fp8_dtypes" and
 isinstance(guard.body[0].body[0],ast.Raise) and ast.unparse(guard.body[0].body[0].exc.func)=='ValueError')
opts=get('triton_backend','CUDAOptions');default=next(ast.literal_eval(n.value) for n in opts.body if isinstance(n,ast.AnnAssign) and ast.unparse(n.target)=='supported_fp8_dtypes')
check('default CUDA fp8 set plus capability>=89 addition yields frozen sm90 tuple',default==('fp8e5','fp8e4b15') and
 contains('triton_backend','CUDABackend.parse_options',"if 'supported_fp8_dtypes' not in args:\n    supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)\n    if capability >= 89:\n        supported_fp8_dtypes.add('fp8e4nv')\n    args['supported_fp8_dtypes'] = tuple(sorted(supported_fp8_dtypes))") and
 sorted(set(default)|{'fp8e4nv'})==bindings['target']['supported_fp8_dtypes'])
check('language load delegates to semantic load', '_semantic.load(' in code('triton_core','load'))
check('semantic load returns typed tensor and restores bool to int1',
 ast.unparse(get('triton_semantic','TritonSemantic.tensor').value)=='tl.tensor' and
 contains('triton_semantic','TritonSemantic._load_legacy','elt_ty = ptr_ty.element_ty','is_bool = elt_ty == tl.int1',
 'if is_bool:\n    ret = self.cast(ret, tl.int1)','return ret') and
 'self.tensor(' in code('triton_semantic','TritonSemantic._load_legacy') and
 'self._load_legacy(' in code('triton_semantic','TritonSemantic.load'))
check('tensor add delegates to semantic add and typechecking',
 contains('triton_core','tensor.__add__','return add(self, other, sanitize_overflow=True, _semantic=_semantic)') and
 '_semantic.add(' in code('triton_core','add') and
 ast.unparse(get('triton_semantic','TritonSemantic.add').body[0])=='input, other = self.binary_op_type_checking_impl(input, other, True, True)')
bt=get('triton_semantic','TritonSemantic.binary_op_type_checking_impl')
check('tensor scalar flags and default non-division promotion call preserved',contains('triton_semantic','TritonSemantic.binary_op_type_checking_impl',
 'lhs_is_scalar = isinstance(lhs, numbers.Number)','rhs_is_scalar = isinstance(rhs, numbers.Number)',
 'ret_sca_ty = self.computation_type_impl(lhs_sca_ty, lhs_is_scalar, rhs_sca_ty, rhs_is_scalar, div_or_mod)') and
 ast.literal_eval(bt.args.defaults[-1]) is False)
ct=get('triton_semantic','TritonSemantic.computation_type_impl')
tests=[ast.unparse(n.test) for n in ct.body if isinstance(n,ast.If)]
check('promotion branch priority and integer/fp8 rejection condition preserved',tests==[
 'a_is_scalar != b_is_scalar','a_ty.is_fp64() or b_ty.is_fp64()',
 'a_ty.is_fp32() or b_ty.is_fp32()','a_ty.is_fp16() or b_ty.is_fp16()',
 'a_ty.is_bf16() and b_ty.is_bf16()','a_ty.is_bf16() or b_ty.is_bf16()',
 'a_ty.is_fp8() and b_ty.is_fp8()','not a_ty.is_int() or not b_ty.is_int()',
 'div_or_mod and a_ty.int_signedness != b_ty.int_signedness'])
reject=next(n for n in ct.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not a_ty.is_int() or not b_ty.is_int()')
check('remaining noninteger promotion branch raises TypeError',isinstance(reject.body[0],ast.Raise) and ast.unparse(reject.body[0].exc.func)=='TypeError')
# Imported historical probes are preserved byte-for-byte and never re-executed.
for label in ('probe_receipt','promotion_matrix','frontend_matrix','probe','kernel_extract'):
 p=Path(bindings['files'][label]['path']);save(audit/(label+p.suffix),p.read_bytes())
model={}
for line in (root/'evidence/model-rows.tsv').read_text().splitlines():
 x,y,p,stage,detail,store=line.split('|');assert (x,y) not in model
 model[x,y]={'promotion_result':None if p=='none' else p,'stage':stage,'detail':detail,'store':store}
check('export contains all 289 unique model pairs',len(model)==289)
backend=lambda sig:'int1' if sig=='u1' else 'int'+sig[1:] if sig.startswith('i') else 'uint'+sig[1:] if sig.startswith('u') else sig
ints={'bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64'}
oldfloats={'float16','bfloat16','float32','float8_e4m3fn','float8_e5m2'}
stages={'passes_selected_dtype_gates':'admitted','signature_fp8_rejection':'signature','addition_promotion_rejection':'addition'}
rows=[]
for r in bindings['rows']:
 x,y=r['x'],r['y'];m=model[backend(canonical[x]),backend(canonical[y])]
 assert m['stage']==stages[r['stage_model_prediction']]
 assert m['promotion_result']==r['promotion_result']
 assert (r['promotion_status']=='promoted')==(m['promotion_result'] is not None)
 assert (m['stage']=='admitted')==(r['observed_frontend_status']=='frontend_generated')
 if m['stage']=='signature':
  first='x' if 'fnuz' in x else 'y';bad=x if first=='x' else y
  assert m['detail'].endswith('.'+first) and m['store']==canonical[bad]
  assert 'ValueError' in r['error'] and 'not supported in this architecture' in r['error']
 elif m['stage']=='addition':assert 'TypeError' in r['error'] and 'unexpected type' in r['error']
 if x in ints and y in ints:v='inherited_integer_bool_81'
 elif x in oldfloats and y in oldfloats:v='inherited_floating_25'
 elif (x=='float32' and y in ints) or (y=='float32' and x in ints):v='inherited_mixed_fp32_18'
 elif (x in {'float16','bfloat16'} and y in ints) or (y in {'float16','bfloat16'} and x in ints):v='inherited_mixed_small_36'
 elif m['stage']=='admitted':v='outside_pinned_inherited_160_inventory'
 else:v='rejection_model_only_not_value_execution'
 rows.append({**r,'finite_lean_model':m,'frozen_source_branch_audit':'passed_audit_not_formal_refinement',
  'value_contract_inventory':v,'new_value_execution_profile':False,'source_native_refinement_proved':False})
counts=dict(Counter(r['finite_lean_model']['stage'] for r in rows))
cross=dict(Counter(r['finite_lean_model']['stage']+':'+r['promotion_status'] for r in rows))
check('partition and cross classification match all retained observations',counts=={'admitted':189,'addition':36,'signature':64} and cross=={'admitted:promoted':189,'addition:promotion_error':36,'signature:promotion_error':36,'signature:promoted':28})
save(audit/'rows.json',rows)
value_anchors={}
for dirname in ('vector-addition-wip','vector-addition-dtype-extension-wip','vector-addition-mixed-fp32-revision1-wip','vector-addition-mixed-half-bfloat-wip','vector-pointwise-schedule-wip'):
 p=root.parent/dirname/'evidence/receipt.json';assert p.is_file();value_anchors[dirname]={'path':str(p),'sha256':sha(p)}
result={'kind':'producer_frozen_source_shape_audit_and_observation_agreement_not_formal_python_semantics',
 'prior_assessment_sha256':sha(old/'ASSESSMENT.md'),'prior_bindings_sha256':sha(old/'bindings-and-partition.json'),
 'files':bindings['files'],'source_units':units,'checks':checks,'target':bindings['target'],
 'stage_counts':counts,'stage_promotion_cross_counts':cross,'rows_sha256':sha(audit/'rows.json'),
 'pinned_inherited_value_inventory_anchors':value_anchors,'pinned_inherited_value_inventory_counts':dict(Counter(r['value_contract_inventory'] for r in rows)),
 'value_inventory_note':'A fixed reference to prior 160 value/grid profiles, not a claim about later fp64 packages or current HEAD. This package adds zero value profiles.',
 'model_rows_export_is_proof':False,'observations_are_proof_premises':False,
 'formal_python_to_model_refinement':False,'new_frontend_runs':0,'native_backend_runs':0,'gpu_runs':0,
 'structural_remaining':['Python/AST-to-model semantic refinement','host allocation/assertions and original module/harness effects','fresh-ASTSource versus JIT/cache/launch path composition','unsupported/noncanonical dtype families and overrides','exact exception wrapping','native lowering/runtime/debug/sanitizer behavior','physical memory and GPU execution'],
 'independent_review':'pending external reviewer'}
save(audit/'audit-result.json',result)
print(json.dumps({'source_units':len(units),'branch_checks':len(checks),'rows':len(rows),'stage_counts':counts,'cross_counts':cross,'audit_sha256':sha(audit/'audit-result.json')}))
