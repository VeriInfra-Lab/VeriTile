from pathlib import Path
import json,re,hashlib,collections
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path('/tmp/veritile-fp-adequacy-20261014')
def load(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def ref(p,line,description):return {'path':p,'line':line,'sha256':sha(p),'description':description}
def line_for(p,needle):
 for i,l in enumerate((ROOT/p).read_text().splitlines(),1):
  if needle in l:return i
 raise ValueError((p,needle))
review=load('bench/tritonbench_g/coverage_review.json')
nonfp=load('bench/review-checkpoints/2026-10-14/numerical-assumption-audit/nonfp/inventory.json')
num=load('bench/tritonbench_g/numerical_obligations.json')
fp=load('bench/review-checkpoints/2026-10-14/numerical-assumption-audit/fp/audit.json')
nonmap={(r['file'],r['declaration']):r for r in nonfp}
fpmap={(r['file'],r['declaration']):r for r in fp['rows']}
nummap={(r['file'],r['declaration']):r for r in num['declarations']}
base='bench/tritonbench_g/'
struct={
'fused_recurrent_retention':['fused_recurrent_retention_forward_equiv','fused_recurrent_retention_backward_equiv'],
'isfinite_kernel':['isfinite_fp32_equiv','isfinite_fp64_equiv'],
'ksoftmax_triton':['ksoftmax_backward_index_equiv','ksoftmax_forward_index_equiv'],
'layer_norm_fwd':['layer_norm_fwd_fp32_equiv'],
'reversed_cumsum':['reversed_cumsum_compare_right_equiv'],
'reversed_cumsum_scalar':['reversed_cumsum_scalar_equiv'],
 'triton_softmax':['softmax_index_equiv'],'uniform_sampling':['uniform_sampling_equiv']}
structkeys={(base+t+'/FPEquiv.lean',d) for t,ds in struct.items() for d in ds}
bitkeys={(base+'isfinite_kernel/Correct.lean',d) for d in ['original_fp32_correctness','original_fp64_correctness','index_commuted_fp32_correctness','index_commuted_fp64_correctness']}
bgmvkey=(base+'bgmv_expand_slice/FPEquiv.lean','bgmv_output_first_equiv')
roundkeys={(r['file'],r['declaration']) for r in num['declarations'] if r.get('model_requirement_ids')}
assert len(roundkeys)==33
common={
'wrapper':ref('VeriTile/Spec.lean',13,'Spec.Real is identity on Prop; name alone does not determine execution semantics.'),
'math_scalar':ref('VeriTile/Triton/Semantics/Scalar.lean',321,'Floating dtype carriers are WithBot real and cast preserves real value in mathematical evaluator.'),
'math_frame':ref('VeriTile/Triton/Correctness.lean',526,'RealizesFrame_without_Rounding is projected mathematical realization plus frame.'),
'round':ref('VeriTile/Triton/Float/RoundingModel.lean',33,'round : FloatDType -> real -> real; real identity and dtype-wise idempotence.'),
'round_eval':ref('VeriTile/Triton/Float/EvalOpR.lean',73,'Precision wrapper erased; casts call R.cast; add/sub/mul/div retain exact mathematical arithmetic.'),
'round_store':ref('VeriTile/Triton/Float/StepR.lean',line_for('VeriTile/Triton/Float/StepR.lean','def writeMemAsR'),'Rounding model modifies stores, not all intermediate arithmetic.'),
'algebra':ref('VeriTile/Triton/Float/Structural.lean',60,'Arbitrary backend operation callbacks, indexed by operation, dtype and precision.'),
'equivalent':ref('VeriTile/Triton/Float/ProfiledRewrite.lean',138,'Equivalent quantifies arbitrary carrier/backend, assumption models, domain and initial state; compares original/candidate exec results.'),
'bit_contract':ref(base+'isfinite_kernel/Correct.lean',25,'Actual goal uses engine(backend32/64 rest) Structural.exec; successful execution, independent bit Post and frame.'),
'bit_backend':ref(base+'isfinite_kernel/Proofs/BitBackend.lean',14,'Same-precision bit-preserving loads/casts and exponent finite classifier; unused operations arbitrary.'),
'bit_formula':ref(base+'isfinite_kernel/Proofs/BitClassification.lean',29,'fp32 exponent != 255; fp64 exponent != 2047; all bit patterns, including NaN/infinity.')}
rows=[]
for port in review['ports']:
 for d in port['declarations']:
  key=(port['file'],d['declaration']);nm=d['numeric_model']; t=port['file'].split('/')[2]
  old=nummap[key]; evidence=[]
  if key in nonmap:
   inv=nonmap[key]; statement=inv['statement']; ln=inv['line']
  else:
   inv=fpmap[key];statement=inv['statement_source'];ln=inv['statement_ref']['line']
  evidence=[ref(key[0],ln,'Actual public statement; statement_sha256 retained in this row.')]
  if nm=='admitted_fp':
   f=fpmap[key];kind='backend_equivalence'; basis='Actual ProfiledRewrite.Program equivalence with proof-used atoms from archived instantiated proof traversal; not inferred from admitted_fp label.'
   evidence += [common['equivalent'],common['algebra']]
   extra=f['obligation_ids'];status='relative_equivalence_under_reported_numeric_laws' if extra else 'relative_equivalence_without_extra_numeric_law'
  elif key in structkeys:
   kind='backend_equivalence';basis='Resolved ProgramSyntax numerical relation to ProfiledRewrite.Equivalent; actual assumptions and domain are empty, proof is arbitrary-backend structural equality.'
   evidence += [common['equivalent'],common['algebra']];extra=[];status='relative_equivalence_without_extra_numeric_law'
  elif key==bgmvkey:
   kind='backend_equivalence';basis='Actual statement quantifies arbitrary Algebra and compares Structural.exec, conditioned on NativeAdditionCommutes and finite operands.'
   evidence += [common['algebra'],common['equivalent']];extra=old.get('extra_numeric_ids',old.get('explicit_law_ids',[]));status='relative_equivalence_under_explicit_native_law'
  elif key in roundkeys:
   kind='rounding_model_contract';basis='Actual RoundingModel parameter/realization target; RoundingModel, evalOpR and store interpreter inspected. Only explicit casts/stores round, intermediate arithmetic stays mathematical.'
   evidence += [common['round'],common['round_eval'],common['round_store']];extra=old.get('extra_numeric_ids',[]);status='cast_store_rounding_contract_not_full_fp_execution'
  elif key in bitkeys:
   kind='bit_interpreter_contract';basis='Unfolded Contract32/64 and correct_target to typed bit Structural.exec plus independent exponent-classification Post; Spec.Real is only Prop wrapper.'
   evidence += [common['wrapper'],common['bit_contract'],common['bit_backend'],common['bit_formula']];extra=[];status='bit_value_correctness_with_native_binding_pending'
  else:
   assert nm=='mathematical',key
   kind='mathematical_contract';basis='Actual statement and named IO/contract references from archived source audit; mathematical evaluator/projection. Full 429-statement scan and imported wrapper spot checks found only four isfinite bit-interpreter exceptions.'
   evidence += [common['math_scalar'],common['math_frame']]
   for x in nonmap[key].get('references',[]):
    evidence.append(ref(key[0],x['line'],'Referenced local predicate: '+x['name']))
   extra=old.get('extra_numeric_ids',[]);status='mathematical_contract_not_full_fp_execution'
  row={'file':key[0],'declaration':key[1],'key':'::'.join(key),'task':t,'legacy_numeric_model':nm,'statement_sha256':hashlib.sha256(statement.encode()).hexdigest(),'statement':statement,'scope':d.get('scope',''),'fp_model_review':{'kind':kind,'review_basis':basis,'evidence':evidence},'kind':kind,'guarantee_level':status,'numeric_obligations':extra,'native_hardware_refinement_established':False}
  if key in fpmap:row['proof_used_atoms']=fpmap[key]['atoms']
  rows.append(row)
rows.sort(key=lambda x:x['key'])
c=collections.Counter(x['kind'] for x in rows)
assert c=={'mathematical_contract':425,'rounding_model_contract':33,'backend_equivalence':167,'bit_interpreter_contract':4},c
bytask=collections.defaultdict(list)
for r in rows:bytask[r['task']].append(r)
assert len(bytask)==178
original_task_status={x['task']:x['numerical_status'] for x in num['tasks']}
tasks=[]
for t,rs in sorted(bytask.items()):
 kinds=sorted({x['kind'] for x in rs}); oldstat=original_task_status[t]
 if oldstat!='no_extra_numeric_law':result=oldstat
 elif 'backend_equivalence' in kinds:result='backend_relative_equivalence_without_extra_law'
 elif 'bit_interpreter_contract' in kinds:result='bit_interpreter_native_binding_pending'
 elif 'rounding_model_contract' in kinds:result='cast_store_rounding_only'
 else:result='mathematical_only'
 tasks.append({'task':t,'kinds':kinds,'legacy_numerical_status':oldstat,'semantic_result':result,'declaration_keys':[x['key'] for x in rs],'native_hardware_refinement_established':False})
noatom=[x for x in rows if x['kind']=='backend_equivalence' and x['guarantee_level']=='relative_equivalence_without_extra_numeric_law']
math_files=sorted({x['file'] for x in rows if x['legacy_numeric_model']=='mathematical'})
scan_pat=re.compile(r'FP\.Structural|\bAlgebra(?:\s|\.)|Float32Bits|Float64Bits|BitClassification|bit-interpreter|backend32|backend64')
scans=[]
for f in math_files:
 hits=[{'line':i,'source':l} for i,l in enumerate((ROOT/f).read_text().splitlines(),1) if scan_pat.search(l)]
 scans.append({'path':f,'sha256':sha(f),'backend_marker_hits':hits})
assert [x['path'] for x in scans if x['backend_marker_hits']]==[base+'isfinite_kernel/Correct.lean']
wrapper_checks=[]
for t in ['llama_ff_triton','rms_matmul_rbe','rms_rbe_matmul','kldiv_ops','kldiv_triton','pow_scalar_tensor','softmax_optimize']:
 p=base+t+'/Proofs/Contracts.lean'
 wrapper_checks.append({'task':t,'result':'mathematical_contract','evidence':ref(p,line_for(p,'eraseDType'),'Contract execution explicitly eraseDType.toAlgKernel; typed input/byte decode does not make outer arithmetic IEEE.')})
wrapper_checks += [
{'task':'flash_attn','result':'mathematical_contract','evidence':ref(base+'flash_attn/Proofs/Contracts.lean',11,'RegionIO/FlatIO execute mathematical Kernel and BlockState, not Structural.State/Algebra.')},
{'task':'uniform_sampling','result':'mathematical_contract_with_exact_random_primitive','evidence':ref(base+'uniform_sampling/Correct.lean',6,'Canonical Philox and rounded fp32 uniform decoder retained, outer arithmetic and destination conversions erase precision.')},
{'task':'f8_conversion_utils','result':'mathematical_contract','evidence':ref(base+'f8_conversion_utils/Correct.lean',14,'Input/output precision erased in four value contracts; unchanged-real-value copy does not prove fp8 quantization.')},
{'task':'reversed_cumsum_scalar','result':'mathematical_contract','evidence':ref(base+'reversed_cumsum_scalar/Correct.lean',14,'Contract receives eraseDType kernel and mathematical BlockState; separate FPEquiv is arbitrary-backend relative equivalence.')},
{'task':'layer_norm_fwd','result':'mathematical_contract','evidence':ref(base+'layer_norm_fwd/Correct.lean',19,'Correctness target uses erased kernels and RealizesFrame_without_Rounding; separate FPEquiv has arbitrary backend equality.')}
]
abstract_no_law=[t for t,rs in bytask.items() if original_task_status[t]=='no_extra_numeric_law' and {x['legacy_numeric_model'] for x in rs}=={'abstract_rounding','mathematical'}]
admitted_no_law=[t for t,rs in bytask.items() if original_task_status[t]=='no_extra_numeric_law' and any(x['legacy_numeric_model']=='admitted_fp' for x in rs)]
assert len(abstract_no_law)==31 and len(admitted_no_law)==57
principles=[
'Classify the instantiated proposition and interpreter reached by its named contracts. Spec.Real is Prop identity; numeric_model tags and theorem filenames do not establish semantics.',
'backend_equivalence means candidate execution equals original execution under the same algebra/model and displayed domains. Empty proof-used atoms remove extra arithmetic-law debt, not native refinement debt.',
'An equality of Option execution results can hold when both sides fail. Separate successful-execution witnesses and their exact profiles must not be generalized beyond their theorem domains.',
'RoundingModel is a real-valued cast/store model, with round_real and round_idem. It does not implement per-operation IEEE rounding, NaN/infinity/signed zero or concrete native dot/FMA/reduction lowering.',
'Exact bit or integer primitives inside a real arithmetic contract do not upgrade the whole result to a bit interpreter. Only the four isfinite contracts have complete source execution against a typed bit-classification backend here.',
'Task aggregation keeps all model kinds, scope and numerical-law requirements as independent axes. A task with a backend relative-equivalence theorem is stronger than math-only, but still has no absolute GPU error bound or compiler refinement by that fact.',
'Unreviewed/changed source must be unknown or require explicit review, never silently no-extra/accepted. Preserve per-declaration hashes and resolve aliases/ProgramSyntax before any automatic promotion.'
]
report={
'schema':'veritile-fp-model-adequacy-audit-v1','reviewed_on':'2026-10-14','root':str(ROOT),
'summary':{'public_declarations':len(rows),'sourced_tasks':len(bytask),'model_kind_counts':dict(c),'legacy_mathematical_declarations':429,'mathematical_bit_interpreter_overrides':4,'legacy_abstract_rounding_declarations':45,'abstract_true_rounding':33,'abstract_backend_no_law':11,'abstract_backend_explicit_native_law':1,'admitted_noatom_declarations':92,'admitted_noatom_tasks_including_numeric_overlap':58,'admitted_wholly_noextra_tasks':57,'all_backend_noextra_declarations':len(noatom),'all_backend_noextra_tasks_including_numeric_overlap':len({x['task'] for x in noatom}),'task_semantic_counts':dict(collections.Counter(t['semantic_result'] for t in tasks))},
'classification_principles':principles,
'coverage':{'method':'Read-only source audit; all 629 source entries classified using archived proof-used-atom audit (155) and statement/reference inventory (474); all 429 mathematical declarations/178 containing files marker-scanned; suspicious imported contract wrappers and 8 abstract structural task families examined. No new Lean elaboration, no GPU/native experiment.',
'limits':['This is a source-model classification, not a new machine proof that all possible transitive definitions lack hardware semantics. The direct-file negative scan is supporting evidence only; explicit wrappers were inspected separately.','Native successful profiles, experiment acceptance and source translation are separate axes and remain owned by their existing evidence. No numerical-law report was invalidated by this audit.','Do not interpret native_hardware_refinement_established=false as a counterexample; it records absence of a full binding in these reviewed contracts.']},
'inputs':[ref(p,1,'Archived review input, not modified by this audit.') for p in ['bench/tritonbench_g/coverage_review.json','bench/review-checkpoints/2026-10-14/numerical-assumption-audit/nonfp/inventory.json','bench/review-checkpoints/2026-10-14/numerical-assumption-audit/fp/audit.json']],
'common_semantic_evidence':common,
'explicit_overrides':[r for r in rows if r['legacy_numeric_model']!='admitted_fp' and r['kind']!='mathematical_contract'],
'rows':rows,'tasks':tasks,
'abstract_mathematical_31':{'tasks':sorted(abstract_no_law),'backend_relative_equivalence_tasks':sorted(set(abstract_no_law)&set(struct)),'cast_store_only_tasks':sorted(set(abstract_no_law)-set(struct))},
'admitted_noextra_57':{'tasks':sorted(admitted_no_law),'interpretation':'All have relative candidate-vs-original backend equivalence without proof-used numerical atoms; no full native binding or absolute numeric accuracy follows.'},
'mathematical_source_scan':scans,'imported_wrapper_checks':wrapper_checks,
'required_explicit_bindings':[
{'kind':'backend_equivalence','required':'Bind exact original/candidate typed source, full dtype/precision/context, and callback semantics to compiled execution; for numeric atoms additionally bind the exact rule/context evidence. Prove successful execution in the intended domain separately if acceptance needs it.'},
{'kind':'rounding_model_contract','required':'Supply per-operation concrete FP interpreter or sound error/refinement theorem, including intermediate add/mul/div/dot/reduction/transcendentals and casts/stores; existing real-valued R alone is insufficient.'},
{'kind':'mathematical_contract','required':'Add a retained-dtype actual FP contract or numeric approximation/error relation to this mathematical target. A sampled experiment can characterize domain/error but is not an all-input proof by itself.'},
{'kind':'bit_interpreter_contract','required':'isfinite: bind fp32/fp64 bit-preserving load/cast/store and classifier behavior to selected Triton/libdevice/compiler/device profiles. Current exponent-bit proof is useful but is not this binding.'}
]
}
OUT.mkdir(exist_ok=True)
(OUT/'model-level-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# FP 模型层级复核（只读）','', '629 条公开声明的实际保证层级为：425 条 mathematical_contract、33 条 rounding_model_contract、167 条 backend_equivalence、4 条 bit_interpreter_contract。numeric_model 标签不能替代目标谓词与解释器复核。','', '31 项 mathematical + abstract_rounding 任务中，8 项已有任意后端的候选/原始执行等价，23 项只有 cast/store 舍入模型。57 项带 admitted_fp 且 proof-used atoms 为空的任务都有后端相对等价；不能把它们归为数学 only，也不能称为 GPU 绝对数值验收。','', '## 8 项结构等价的 abstract_rounding 特例','']
for t in sorted(struct):
 for d in struct[t]:lines.append('- `'+base+t+'/FPEquiv.lean::'+d+'`')
lines += ['','另有 `'+ '::'.join(bgmvkey)+'`：任意 Algebra 的执行相等，但显式需要 NativeAdditionCommutes 与 finite_operands。','', '## 429 条 mathematical 中的特殊 wrapper','', '仅发现 isfinite 的四条公开声明是完整 bit-interpreter 合同；Spec.Real 只是 Prop 包装：','']
for k in sorted(bitkeys):lines.append('- `'+ '::'.join(k)+'`')
lines += ['','实际目标在 Correct.lean:25，Contract32/64 在 32/35；BitBackend.lean:14/24 安装位保持与指数分类；BitClassification.lean:29/30 给出独立有限性公式。它们证明成功执行、输出及 frame，但 native libdevice/compiler 对应仍未绑定。','', 'llama_ff_triton、rms_matmul_rbe、rms_rbe_matmul 的 ByteIO 虽含精确字节解码，外层执行明确为 eraseDType.toAlgKernel。uniform_sampling 保留 Philox/fp32 decoder，但外层算术与输出转换擦除精度；这些仍是 mathematical_contract。f8_conversion_utils 的数学合同是 real copy，不是 FP8 量化正确性。','', '## 舍入模型缺口','', 'RoundingModel 的 round 输入输出为实数，要求 real identity 与 dtype-wise idempotence。evalOpR 擦除 precision wrapper，只在 cast/store 应用舍入；中间 add/sub/mul/div 保留数学算术。因此 33 条这类声明、其中 23 个仅有这种 FP 层级的任务仍欠实际逐运算 FP 语义/误差保证，不能因没有额外数值公理而完成验收。','', '## 验收原则','']
lines += ['- '+p for p in principles]
lines += ['','## 覆盖边界','',report['coverage']['method'],'','所有 629 条的完整键、真实 statement、kind、review_basis、证据路径/行号/hash 均在 JSON 的 rows；49 条非 admitted 标签特殊声明在 explicit_overrides。429 条数学声明文件扫描与隐藏 wrapper 复核分别保留在 mathematical_source_scan/imported_wrapper_checks。此审计没有重跑 Lake/GPU，也不把源码负向扫描冒充内核验证。','']
(OUT/'model-level-audit.md').write_text('\n'.join(lines))
print(json.dumps(report['summary'],indent=2));print('overrides',len(report['explicit_overrides']));print('json_bytes',(OUT/'model-level-audit.json').stat().st_size)
