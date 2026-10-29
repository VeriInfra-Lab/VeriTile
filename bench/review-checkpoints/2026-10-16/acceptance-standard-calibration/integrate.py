"""Apply independently checked finite source-scope calibration without acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile'); T=Path('/tmp/veritile-standardization-20261015'); P=T/'acceptance-standard-calibration'; B=R/'bench/review-checkpoints/2026-10-16/acceptance-standard-calibration'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def git(*a):return subprocess.check_output(['git',*a],cwd=R)
assert not B.exists(); assert not git('status','--porcelain')
base=git('rev-parse','HEAD').decode().strip(); check=read(T/'acceptance-calibration-root-validation.json'); cal=read(P/'calibration.json'); receipt=read(P/'receipt.json')
assert sha(T/'acceptance-calibration-root-validation.json')=='eb562b17af11c0c8739dfa29aa1fc0c6d5bdaba2f48c3822805ed33c2c94b8d4'
for p,h in receipt['artifact_sha256'].items():assert sha(P/p)==h,p
ledgerPath=R/'bench/original_program_scope.json'; before=read(ledgerPath); ledger=copy.deepcopy(before); rows={r['task']:r for r in ledger['tasks']}
snap=read(P/'snapshot-selected-rows.json'); selected=snap['ledger_rows']; selected={r['task']:r for r in selected} if isinstance(selected,list) else selected
for name in ['relu_triton_kernel','vector_addition','quantize_global','quant_transpose_kernel']:assert rows[name]==selected[name],name
for path,h in check['input_sha256'].items():
 if Path(path)==ledgerPath:
  old=git('show',check['base_commit']+':bench/original_program_scope.json');assert hashlib.sha256(old).hexdigest()==h
 else:assert sha(Path(path))==h,path
for item in cal['gap_classification']:assert rows[item['task']]['structural_gaps'][item['original_gap_index']]['finding']==item['original_finding']

def gap(task,ident,lines,finding):
 path=f'bench/tritonbench_g/{task}/{task}.py'
 return {'id':ident,'kind':'remaining_original_program_scope','source_path':path,'source_lines':lines,'finding':finding,'can_defer_as_numeric_assumption':False}
new={}
new['relu_triton_kernel']=[
 gap('relu_triton_kernel','RELU-FINITE-1',[6,36],'Review exact source-to-model and wrapper linkage for all 13 admitted routes: preserve the 12 existing logical routes and add f64 load/ordered-compare/select followed by f32 narrowing store. Every pid executes the pre-gate load/compute; only pid zero stores.'),
 gap('relu_triton_kernel','RELU-FINITE-2',[23,36],'Bind empty_like fp32/device/shape/freshness, output.numel, block1024, complete ceil grid and returned tensor to the actual typed execution. Existing HostWrapper.relu_logical_correct covers 12 routes; extend this finite host composition to fp64.'),
 gap('relu_triton_kernel','RELU-FINITE-3',[8,21],'Bind tensor view bases, physical element offsets, extent and admitted finite index arithmetic; justify fresh-output read/write independence and the unchanged tail. Retain all high-pid loads; no silent contiguity or arbitrary-shape safety assumption.'),
 gap('relu_triton_kernel','RELU-FINITE-4',[6,36],'Pin the supported target/API domain and the 17 canonical input cases, with 13 observed frontend successes and four fp8 rejections at their actual first stages. Compose rejection and success outcomes rather than demanding successful rejected kernels.'),
 gap('relu_triton_kernel','RELU-FINITE-5',[17,36],'Finish exact whole-source schema2 coverage and numerical callback/case bindings. Preserve the known N>1024 tail counterexample: complete as-written execution is not an unconditional whole-tensor mathematical ReLU theorem.')]
new['vector_addition']=[
 gap('vector_addition','VA-FINITE-1',[6,27],'Review and compose the 160 shared-interpreter and 29 fp64 successful source routes through exact dtype dispatch and a common external typed memory/result relation; both interpreters may remain separate.'),
 gap('vector_addition','VA-FINITE-2',[21,27],'Connect source-order empty_like metadata/freshness, CUDA assertion, output.numel, block1024, ceil grid and returned output to actual typed execution. The existing real-only logicalLaunch is not the required typed composition.'),
 gap('vector_addition','VA-FINITE-3',[6,27],'Compose the canonical 17x17 domain with source-bound outcomes: 189 dtype-admitted pairs, 64 signature rejections and 36 addition rejections. The 28 promoted fnuz pairs are signature rejections. Admission protocol tables alone are not executing host contracts.'),
 gap('vector_addition','VA-FINITE-4',[12,19],'Bind actual tensor view offsets/extents and finite index arithmetic; justify source load-all/compute/store correspondence from read-only inputs and a fresh disjoint output. Retain actual unsafe-memory cases rather than inventing a universal native scheduler obligation.'),
 gap('vector_addition','VA-FINITE-5',[16,19],'Treat the nine fp64-to-integer/bool OutputDefined restrictions as an explicit justified defined-value domain or model complementary outcomes faithfully. Do not numerically defer an undefined conversion domain or infer native rollback from custom interpreter errors.'),
 gap('vector_addition','VA-FINITE-6',[6,27],'Bind all source stages, supported/rejected dtype cases and host branch outcomes through current schema2 public execution evidence, with exact numerical ports registered for each claimed case. Preserve actual rejection stages without proving native diagnostic wording.')]
new['quantize_global']=[
 gap('quantize_global','QG-FINITE-1',[31,39],'Compose x.abs().max().unsqueeze(0) and reciprocal with the loaded inverse scale and returned absmax. Numerical operation laws may be registered separately; producer/consumer wiring cannot be deferred.'),
 gap('quantize_global','QG-FINITE-2',[31,39],'Bind exact output shape/int8 allocation, CUDA assertion, numel and returned output/absmax tuple under stated Torch primitive metadata/freshness contracts.'),
 gap('quantize_global','QG-FINITE-3',[7,29],'Compose complete ceil grids for both source-declared block sizes 1024 and 2048, preserving masks, finite address domain and disjoint output ownership.'),
 gap('quantize_global','QG-FINITE-4',[15,39],'Establish source-grounded input/scale dtype admission and rejection coverage beyond the existing selected fp32 stage; bind all actual stages and cases through schema2 execution evidence.'),
 gap('quantize_global','QG-FINITE-5',[26,29],'Register exact floating multiply, reduction/reciprocal, llrint64 callback and native store-conversion correspondence for the claimed dtype/configuration cases. Signed int8 storage is already structurally implemented; no generic physical-memory implementation proof is added.')]
new['quant_transpose_kernel']=[
 gap('quant_transpose_kernel','QT-FINITE-1',[57,79],'Compose input abs/max/unsqueeze/reciprocal with loaded inverse scale and returned absmax; preserve 2D shape unpacking, allocation and actual assertions.'),
 gap('quant_transpose_kernel','QT-FINITE-2',[30,55],'Prove the actual grouped one-dimensional pid mapping, including the last partial group, and compose all source tiles over the complete launch grid. A supplied-coordinate tile is partial progress.'),
 gap('quant_transpose_kernel','QT-FINITE-3',[40,79],'Bind source input/output stride argument order, tensor view bases/extents, finite address domain, masks and full output coverage under fresh allocation metadata contracts.'),
 gap('quant_transpose_kernel','QT-FINITE-4',[16,79],'Establish source-grounded input/scale dtype admission and rejection coverage beyond the selected fp32 stage and bind complete host/kernel/configuration cases through schema2 execution evidence.'),
 gap('quant_transpose_kernel','QT-FINITE-5',[44,55],'Register exact arithmetic/llrint64/store-conversion numerical correspondence per actual dtype and configuration. Keep missing producers, grouped scheduling and admitted dtype routes structural; generic native runtime/launch implementation is trusted.')]
B.mkdir(parents=True); (B/'evidence').mkdir()
for name in receipt['artifact_sha256']:shutil.copyfile(P/name,B/'evidence'/name)
shutil.copyfile(P/'receipt.json',B/'evidence/receipt.json');shutil.copyfile(T/'acceptance-calibration-root-validation.json',B/'root-input-validation.json');shutil.copyfile(Path(__file__),B/'integrate.py')
policy=R/'bench/original_program_acceptance.md';assert not policy.exists()
policy.write_text('''# Finite acceptance boundary for the original programs

The user standard is all source-declared stages, dtypes and branches. Unverified
numerical assumptions may be postponed only with exact statement, operation,
dtype and configuration bindings. This boundary does not accept any program.

Production scope includes source host wiring and kernel execution, all admitted
cases, actual rejected cases, loops/masks/casts, return metadata, memory effects
and frames. State a source-grounded API/target domain; a missing Python assertion
does not imply support for arbitrary Python objects or unknown dtype families.
An already admitted route cannot be removed by narrowing that domain.

Python/Torch metadata, fresh allocation, scalar/tensor primitive operations and
Triton launch/load/store semantics form an explicit trusted boundary. The task
proof must still connect the actual producer/consumer calls, arguments, selected
configuration and results. An unconstrained final-value or launch-success oracle
cannot replace execution. Source-visible view offsets, widths, integer overflow,
unsafe accesses and ownership need exact treatment. A per-task correctness proof
of the compiler, allocator, MMU, scheduler, cache or sanitizer implementation is
outside this acceptance standard. Native correctness is not thereby established.

Test functions remain inventoried and can be marked test_harness only after the
existing non-JIT/production-call review. Import-time random tests do not require
proofs of the entire RNG or module/runtime implementation. Keep genuine
production assertions and first rejection stages; native diagnostic spelling is
not an extra source branch.

Separate faithful typed interpreters are allowed when exact dispatch and the
external memory/result relation compose them. Existing successful logical
execution is retained without requiring a redundant byte-level implementation.
Undefined numerical conversions and known false conclusions cannot be waived as
numerical assumptions. As-written source behavior must remain distinct from an
intended mathematical specification, including ReLU's pid-zero-only stores.

Acceptance still requires exact schema2 stage/dtype/branch coverage, current
public execution statements, numerical registrations where applicable, and an
independent substantive root review. Build, trust audit and official comparator
checks retain their own results. The checker has not been weakened.

The first calibration applies reviewed finite gap replacements only to ReLU,
vector addition and the two quantization tasks. All other current gap records
remain unchanged and must be read under this boundary and refined against
source evidence before acceptance; no source obligations are deleted in bulk.
''')
review={'schema':1,'kind':'finite_acceptance_boundary_root_review','version':1,'reviewer':'root','status':'reviewed_no_program_acceptance','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':base,'producer_receipt_sha256':sha(P/'receipt.json'),'root_input_validation_sha256':sha(B/'root-input-validation.json'),'policy':{'path':policy.relative_to(R).as_posix(),'sha256':sha(policy)},'finite_requirements':cal['finite_acceptance_definition'],'previous_structural_gaps':{n:copy.deepcopy(rows[n]['structural_gaps']) for n in new},'current_structural_gaps':new,'source_sha256':{p:sha(Path(p)) for p in check['input_sha256'] if str(R) in p and Path(p)!=ledgerPath},'semantic_review':'Root checked all 37 input bindings and 31 exact excerpts before the final98 ledger mutation, read all 19 ReLU/vector gap classifications and finite policy, independently inspected source host/kernel operations, and verified the selected four rows remain byte-equivalent as JSON. Universal compiler/runtime/harness requirements are not source gaps. Real source coverage, admission outcomes, undefined conversion domains and exact numerical bindings remain mandatory. No proof source or acceptance status changes.','accepted_programs':0}
write(B/'root-review.json',review)
reference={'version':1,'path':(B/'root-review.json').relative_to(R).as_posix(),'sha256':sha(B/'root-review.json'),'status':'finite_source_boundary_reviewed_no_acceptance'}
ledger['acceptance_boundary_review']=reference
for name,gaps in new.items():
 rows[name]['structural_gaps']=gaps; rows[name]['acceptance_boundary_review']=reference
 rows[name]['trusted_semantics']={'policy_path':policy.relative_to(R).as_posix(),'policy_sha256':sha(policy),'not_a_numerical_deferral':True,'implementation_correctness_established':False}
 assert rows[name]['status']=='pending' and rows[name]['review'] is None
assert all(t=={r['task']:r for r in before['tasks']}[t['task']] for t in ledger['tasks'] if t['task'] not in new)
write(ledgerPath,ledger)
md=R/'bench/original_program_scope.md';text=md.read_text();anchor='The checker rejects omitted tasks';assert text.count(anchor)==1
text=text.replace(anchor,'The [finite acceptance boundary](original_program_acceptance.md) separates\nsource-specific obligations from trusted Python/Torch/Triton primitive semantics.\nIt preserves real coverage requirements without adding universal compiler, runtime\nor test-harness implementation proofs. The reviewed first calibration changes\nfour gap lists and leaves every task pending.\n\n'+anchor);md.write_text(text)
cat=R/cal['checked_current_catalog']; cmd=[sys.executable,'-B',str(R/'bench/check_original_program_scope.py'),'--root',str(R),'--catalog',str(cat),'--ledger',str(ledgerPath),'--output',str(B/'scope-gate.json')]
with (B/'scope-gate.log').open('w') as f: result=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
gate=read(B/'scope-gate.json');assert result.returncode==1 and gate['errors']==['183 original-program scopes remain unaccepted'];assert all(t['error']=='scope remains pending' for t in gate['tasks'])
(B/'README.md').write_text('''# Reviewed finite acceptance boundary

All 183 programs remain pending; acceptance increment is zero. This checkpoint
preserves the independent calibration and exact old gap strings, records root
validation of 37 inputs and 31 quoted ranges, and replaces four current gap lists
with finite source-specific closure actions. The other 179 rows are unchanged.
No Lean, Python production, checker, catalog or numerical queue is changed.

The original calibration binds HEAD 31893d3e; its ledger hash was validated before
the final98 review integration. Integration rechecks that historical ledger via
git and checks all four selected current rows against their frozen snapshot.
The 183-task scope checker exits 1 solely because all scopes remain pending.
No new Lean/frontend/GPU run or numerical validation is claimed.
''')
paths={ledgerPath,md,policy}|{p for p in B.rglob('*') if p.is_file()}
write(B/'validation.json',{'schema':1,'kind':'finite_acceptance_calibration_integration','base_commit':base,'status':'reviewed_pending_unchanged','scope':{'requested':183,'accepted':0,'pending':183,'reviewed':183},'changed_scope_rows':sorted(new),'unchanged_scope_rows':179,'scope_gate_exit_code':result.returncode,'scope_gate_errors':gate['errors'],'checked_input_sha256':{p.relative_to(R).as_posix():sha(p) for p in sorted(paths)},'preserved_source_sha256':review['source_sha256'],'no_proof_or_checker_edits':True})
print(json.dumps({'checkpoint':str(B),'validation_sha256':sha(B/'validation.json'),'changed_rows':list(new),'scope':{'accepted':0,'pending':183}},indent=2))
