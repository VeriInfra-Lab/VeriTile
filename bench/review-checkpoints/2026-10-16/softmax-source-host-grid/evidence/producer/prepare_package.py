from pathlib import Path
import ast,hashlib,json,re,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import declarations,imports
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):(W/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
mapping=json.loads((W/'evidence/staging-map.json').read_text())
paths=list(mapping.values());source_path='bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py';source=R/source_path;tree=ast.parse(source.read_text());sh=sha(source)
functions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
stages=[]
for n in functions:
 calls=[ast.unparse(c.func) for c in ast.walk(n) if isinstance(c,ast.Call)]
 stages.append({'name':n.name,'line':n.lineno,'end_line':n.end_lineno,'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'calls':calls,'classification':'test_harness' if n.name=='test_softmax' else ('unused_jit_helper_rejects_on_direct_call' if n.name=='next_multiple_of' else 'production')})
write('evidence/source-bindings.json',{'source':source_path,'sha256':sh,'stages':stages,'unchanged_source':True,'source_model':'existing FullSource kernels, exact AST translation and existing original_all_io; new host separately authored and source-probed'})
# Exact project import closure, with library boundary and observed compiled object hashes.
seen={};external=set();todo=[p[:-5].replace('/','.') for p in paths];edges={}
while todo:
 mod=todo.pop()
 if mod in seen:continue
 rel=mod.replace('.','/')+'.lean';p=W/'after'/rel
 where='proposed'
 if not p.exists():p=R/rel;where='repository'
 if not p.exists():external.add(mod);continue
 deps=imports(p.read_text());edges[mod]=deps
 item={'path':rel,'source_sha256':sha(p),'origin':where}
 o=(W/'build-overlay'/rel if where=='proposed' else R/'.lake/build/lib/lean'/rel).with_suffix('.olean')
 if o.exists():item['observed_olean_sha256']=sha(o)
 seen[mod]=item;todo.extend(deps)
policy=['bench/original_program_acceptance.md','bench/adapter_common.py','bench/current_source_catalog.py','bench/check_original_program_scope.py','lean-toolchain','lake-manifest.json','lakefile.toml']
write('evidence/repo-dependencies.json',{'project_import_closure':list(sorted(seen.values(),key=lambda x:x['path'])),'external_import_boundary':sorted(external),'import_edges':edges,'policy_and_build_files':{p:sha(R/p) for p in policy if (R/p).exists()},'note':'Exact source and observed object hashes; imported existing project objects were reused. This manifest alone does not prove their source-to-object freshness; independent integration review retains that check.'})
public=[]
for rel in paths:
 for name,d in declarations(W/'after'/rel).items():
  if d.get('kind')=='specification':
   public.append({'declaration':name,'file':rel,'statement_sha256':d['statement_sha256'],'source_sha256':sha(W/'after'/rel),'line':next(i+1 for i,line in enumerate((W/'after'/rel).read_text().splitlines()) if line.startswith('specification '+name)),'semantic_model':'mathematical_dtype_erased_original_execution','native_fp_value_claim':False})
write('evidence/public-statements.json',public)
# Native/numerical remaining work is finite and exact-source-bound; no invented waiver.
num=[]
for n in functions:
 if not n.name.startswith('softmax_') or 'kernel' not in n.name:continue
 operations=[]
 for op in ast.walk(n):
  if isinstance(op,ast.Call) and (ast.unparse(op.func) in ['tl.exp','tl.sum','tl.max','tl.maximum','tl.load','tl.store','tl.full','tl.zeros'] or isinstance(op.func,ast.Attribute) and op.func.attr=='to'):
   operations.append({'line':op.lineno,'operation':ast.unparse(op.func),'expression':ast.unparse(op),'ast_sha256':hashlib.sha256(ast.dump(op,include_attributes=False).encode()).hexdigest()})
 num.append({'source_kernel':n.name,'source_function_ast_sha256':next(s['ast_sha256'] for s in stages if s['name']==n.name),'statement_binding':'softmax_autograd_source_admitted_real','profiles':'backward: saved output/gradient fp16,bf16,fp32,fp64; forward: all 16 fp16,bf16,fp32,fp64 input/output pairs; fallback fp64 input/storage rejected','configurations':'exact forwardConfig or six backwardConfig choices, oneTile=N<=B, both inner and noninner branches','operations':operations,'work':'Provide typed source execution/reference with actual promotion, load/store and explicit cast semantics, finite rounded add/sub/mul/div/exp and valid reduction order. Handle NaN/infinity, invalid arithmetic and undefined padding explicitly. Do not assert exact equality of finite floating arithmetic to the real-erased reference.','registered_numeric_assumption_ids':[]})
remaining=[
 {'id':'SOFTMAX-HOST-NATIVE-DOMAIN','kind':'structural_finite_domain','source':source_path,'source_sha256':sh,'finding':'Natural-number element indexing and abstract disjoint regions do not yet bind concrete source scalar argument widths, signed address intermediates, storage byte widths/base offsets and launch extent limits.','required_action':'State and check a source-grounded finite CUDA API/target domain; bind M,N,K, every row/column/tail offset, 2/4/8-byte element widths and allocation/view extents. Retain K=0 unsafe and observed frontend rejects. Do not require proofs of compiler, allocator, MMU, or scheduler implementation.','can_defer_as_numeric_assumption':False},
 {'id':'SOFTMAX-HOST-TYPED-NUMERICS','kind':'numerical_and_typed_reference','source':source_path,'source_sha256':sh,'finding':'Current new values compose original dtype-erased real source semantics. Existing backend equivalence is not native softmax value correctness.','exact_kernel_bindings':num,'can_defer_as_numeric_assumption':'Only valid, exact operation/profile/statement laws after registration; none registered here. False casts/zero padding cannot be deferred.'},
 {'id':'SOFTMAX-HOST-OUTCOME-SCHEMA','kind':'integration_and_scope','finding':'Exact schema2 rows must bind new public statements and the retained compiler, zero-tail and unused-helper outcomes. Existing accepted rows must remain intact.','required_action':'Independent root source/proof review; update stage/dtype/branch coverage and calibrated finite gaps, then run unchanged schema/catalog/official comparator checks.','can_defer_as_numeric_assumption':False},
 {'id':'SOFTMAX-HOST-FRONTEND-TARGET','kind':'source_target_evidence_boundary','finding':'Probes target CUDA SM90 Triton3.7.1 frontend only. Native backend resource/compiler outcomes and GPU execution were not run.','required_action':'Decide and record the concrete target boundary. If target-dependent backend outcomes are required by that boundary, add finite configuration evidence; do not promote frontend IR generation to full native compilation.','can_defer_as_numeric_assumption':False}
]
write('remaining-scope.json',{'task':'softmax_flaggems','accepted':False,'acceptance_increment':0,'registered_new_numerical_assumptions':[],'remaining':remaining})
# Coverage proposal records evidence without marking acceptance.
baseline=json.loads((W/'evidence/current-scope-task.json').read_text());coverage=[]
for stage in baseline['source_inventory']['stages']:
 identifier=stage['id'];name=identifier.split(':')[-1]
 disposition='mathematical_source_stage_composed; typed/native follow-up retained'
 if name=='test_softmax':disposition='proposed test_harness: non-JIT tests only; no production caller'
 elif name=='next_multiple_of':disposition='unused declared JIT helper; direct-call compiler rejection retained'
 elif name not in ['softmax_kernel_inner','softmax_kernel_non_inner','softmax_backward_kernel_inner','softmax_backward_kernel_non_inner']:disposition='host/heuristic source wiring composed and exact AST metadata checked'
 coverage.append({'id':identifier,'proposed_disposition':disposition,'evidence':['evidence/source-bindings.json','evidence/host-probe-scope-addendum.json','evidence/autotune-probe-receipt.json','evidence/helper-probe-receipt.json'],'status':'pending_independent_root_review'})
write('scope-refinement-proposal.json',{'task':'softmax_flaggems','baseline_gap':'SOFTMAX_FLAGGEMS-S1','proposal':'Replace the generic host/grid/autograd gap with the exact remaining finite obligations in remaining-scope.json after independent root review. Do not mark native value acceptance from erased execution.','stages':coverage,'dtype_scope':'Existing four Storage constructors retained; all16 input/output pairs plus dtype=None. Source-supported success and compiler rejection partition is explicit; no arbitrary Python-object/dtype-family claim.','branches':'All one/fallback and inner/noninner kernel branches, source heuristics, actual trial sequence, cache-hit route, dimension assertion, zero divisors and K=0 unsafe path retained.','acceptance_increment':0})
write('integration-proposal.json',{'task':'softmax_flaggems','status':'pending_independent_root_review','repository_changes':'add only these nine Lean sources; no existing source or policy edit proposed by this package','additive_files':{p:sha(W/'after'/p) for p in paths},'public_declarations':public,'proof_modules':mapping,'trusted_metadata_primitives':['Torch contiguous(shape,inputStrides)','Torch empty_like(shape,contiguousResultStrides)','Torch empty_like(shape,producedOutputStrides)'],'native_functional_correctness':'not_established','acceptance_increment':0})
# Select successful exact-hash compile receipt for each final staged module.
receipts=[]
for rel in paths:
 key='after/'+rel;digest=sha(W/key)
 matches=[]
 for p in (W/'evidence/attempts').glob('*.json'):
  d=json.loads(p.read_text())
  if d.get('module')==key and d.get('exit_code')==0 and d.get('source_sha256')==digest:matches.append((p,d))
 assert matches,key
 p,d=sorted(matches)[-1];log=p.with_suffix('.log');text=log.read_text()
 assert 'error:' not in text and 'DISALLOWED' not in text and 'declaration uses `sorry`' not in text
 receipts.append({'source':key,'source_sha256':digest,'receipt':str(p.relative_to(W)),'receipt_sha256':sha(p),'log':str(log.relative_to(W)),'log_sha256':sha(log),'exit_code':0,'named_axiom_checks':text.count('axiom footprint ⊆ standard base'),'noncircular_checks':text.count('non-circular ✓')})
write('evidence/final-compiles.json',{'status':'passed','modules':receipts,'named_axiom_checks':sum(d['named_axiom_checks'] for d in receipts),'noncircular_checks':sum(d['noncircular_checks'] for d in receipts),'all_module_axiom_audits':'passed via #auditModuleAxioms in each final source','imported_dependencies':'Reused existing repository objects; exact hashes in repo-dependencies.json.'})
print(json.dumps({'modules':len(paths),'project_dependencies':len(seen),'public_statements':len(public),'final_compiles':len(receipts),'axiom_checks':sum(d['named_axiom_checks'] for d in receipts)}))
